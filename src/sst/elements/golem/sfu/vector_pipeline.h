#ifndef SST_GOLEM_VECTOR_PIPELINE_H
#define SST_GOLEM_VECTOR_PIPELINE_H

#include <array>
#include <functional>
#include <map>
#include <stdexcept>
#include <utility>
#include <sst/elements/golem/sfu/hardware_pipeline.h>
#include <sst/elements/golem/sfu/vector_math.h>

namespace SST { namespace Golem {

// Cycle-driven scheduler used by SFU, also executable without an SST runtime.
// Input snapshots represent the already-timed GM reads. Only interstage FIFO
// elements and accepted hardware tokens are live internal pipeline state.
class SFUVectorPipeline {
public:
    enum Stage : uint8_t { Negate, Exp, TailMultiply, AddOne, Reciprocal, Multiply, Direct, GateMultiply };
    struct Issue {
        uint64_t job, issueCycle, completionCycle;
        Stage stage;
        size_t first, elements;
    };
    struct Completion {
        uint64_t job, submittedCycle, completionCycle;
        SFUVectorOp op;
        std::vector<double> values;
    };
    using Trace = std::function<void(const Issue&)>;

    SFUVectorPipeline(TensorHardwarePipeline& vector, TensorHardwarePipeline& exp,
                      TensorHardwarePipeline& reciprocal, uint32_t vectorLanes,
                      uint32_t expLanes, size_t fifoElements, size_t maxRequests,
                      size_t maxElements = 8192,
                      SFUInternalPrecision precision = SFUInternalPrecision::FP16,
                      uint32_t reciprocalLanes = 16,
                      SFUNonlinearBackend backend = SFUNonlinearBackend::Libm)
        : vector_(vector), exp_(exp), reciprocal_(reciprocal),
          vectorLanes_(vectorLanes), expLanes_(expLanes), fifoElements_(fifoElements),
          maxRequests_(maxRequests), maxElements_(maxElements), reciprocalLanes_(reciprocalLanes), arithmetic_(precision,backend) {
        if (!vectorLanes || !expLanes || !reciprocalLanes || !maxRequests || !maxElements ||
            fifoElements < std::max({vectorLanes, expLanes, reciprocalLanes}))
            throw std::invalid_argument("invalid SFU vector pipeline capacity");
    }

    // A rejected submission does not borrow operands or mutate a result.
    bool submit(const SFUVectorRequest& request, uint64_t now, uint64_t* id) {
        if (!id || !request.input || !request.coefficients || !request.rowWidth ||
            request.input->empty() || request.input->size() % request.rowWidth ||
            request.input->size() > maxElements_ || jobs_.size() >= maxRequests_ ||
            (request.op != SFUVectorOp::Add && request.op != SFUVectorOp::Multiply &&
             request.op != SFUVectorOp::Silu && request.op != SFUVectorOp::SiluMultiply) ||
            (request.op != SFUVectorOp::Silu &&
             request.coefficients->size() != request.input->size())) return false;
        Job job;
        job.op = request.op; job.submitted = now;
        for (double x : *request.input) job.input.push_back(vectorFp16Value(x));
        job.original = job.input;
        if (request.op != SFUVectorOp::Silu)
            for (double x : *request.coefficients) job.coefficients.push_back(vectorFp16Value(x));
        job.output.resize(job.input.size());
        *id = ++sequence_;
        jobs_.emplace(*id, std::move(job));
        requestHighWater = std::max<uint64_t>(requestHighWater, jobs_.size());
        return true;
    }

    bool active() const { return !jobs_.empty(); }
    size_t fifoCapacity() const { return fifoElements_; }
    std::vector<Completion> takeCompleted() { return std::exchange(completed_, {}); }

    void step(uint64_t now, const Trace& trace = {}) {
        vector_.retire(now); exp_.retire(now); reciprocal_.retire(now);
        while (!pending_.empty() && pending_.begin()->first <= now) {
            auto token = std::move(pending_.begin()->second);
            const uint64_t done = pending_.begin()->first;
            pending_.erase(pending_.begin());
            complete(token, done);
        }
        issueResource(vector_, now, vectorCursor_, {Negate, TailMultiply, AddOne, Multiply, Direct, GateMultiply}, trace);
        issueResource(exp_, now, expCursor_, {Exp}, trace);
        issueResource(reciprocal_, now, reciprocalCursor_, {Reciprocal}, trace);
    }

    uint64_t fifoBackpressureEvents = 0;
    uint64_t resourceBackpressureEvents = 0;
    uint64_t fifoHighWater = 0; // Includes reserved destination credits.
    uint64_t requestHighWater = 0;

private:
    struct Element { size_t index; float value; };
    struct Job {
        SFUVectorOp op = SFUVectorOp::Add;
        uint64_t submitted = 0;
        size_t consumed = 0;
        std::vector<float> input, original, coefficients;
        std::vector<double> output;
        std::array<std::deque<Element>, 6> fifo;
        std::array<size_t, 6> reserved = {};
        std::array<size_t, 8> done = {};
    };
    struct Token { uint64_t job; Stage stage; std::vector<Element> elements; };
    TensorHardwarePipeline &vector_, &exp_, &reciprocal_;
    uint32_t vectorLanes_, expLanes_;
    size_t fifoElements_, maxRequests_, maxElements_;
    uint32_t reciprocalLanes_;
    SFUArithmetic arithmetic_;
    uint64_t sequence_ = 0, vectorCursor_ = 0, expCursor_ = 0, reciprocalCursor_ = 0;
    std::map<uint64_t, Job> jobs_;
    std::multimap<uint64_t, Token> pending_;
    std::vector<Completion> completed_;

    static size_t inputQueue(Stage stage) { return stage == GateMultiply ? 5 : size_t(stage) - 1; }
    static size_t outputQueue(Stage stage) { return size_t(stage); }
    static bool producesFifo(const Job& job, Stage stage) {
        return stage < Multiply || (stage == Multiply && job.op == SFUVectorOp::SiluMultiply);
    }
    bool applicable(const Job& job, Stage stage) const {
        if (job.op == SFUVectorOp::SiluMultiply) return stage != Direct;
        return job.op == SFUVectorOp::Silu ? stage <= Multiply : stage == Direct;
    }

    size_t available(const Job& job, Stage stage) const {
        const size_t lanes = stage == Exp ? expLanes_ :
                             stage == Reciprocal ? reciprocalLanes_ : vectorLanes_;
        if (stage == Negate || stage == Direct)
            return std::min(lanes, job.input.size() - job.consumed);
        const auto& queue = job.fifo[inputQueue(stage)];
        // Pack full hardware beats, except for the final tail. This avoids
        // turning a scalar reciprocal output into one vector instruction/item.
        if (queue.size() < lanes && job.done[stage == GateMultiply ? Multiply : stage - 1] != job.input.size()) return 0;
        return std::min(lanes, queue.size());
    }

    void issueResource(TensorHardwarePipeline& pipeline, uint64_t now,
                       uint64_t& cursor, std::initializer_list<Stage> stages,
                       const Trace& trace) {
        if (now < pipeline.nextIssueCycle) return;
        // Round robin over (request, stage), including competing vector stages.
        std::vector<std::pair<uint64_t, Stage>> candidates;
        for (const auto& entry : jobs_)
            for (Stage stage : stages)
                if (applicable(entry.second, stage) && available(entry.second, stage))
                    candidates.emplace_back(entry.first, stage);
        const auto ticket = [](const std::pair<uint64_t, Stage>& c) { return c.first * 7 + c.second; };
        auto first = std::find_if(candidates.begin(), candidates.end(),
                                 [&](const auto& c) { return ticket(c) > cursor; });
        std::rotate(candidates.begin(), first, candidates.end());
        for (const auto& candidate : candidates) {
            Job& job = jobs_.at(candidate.first);
            const Stage stage = candidate.second;
            const size_t count = available(job, stage);
            if (producesFifo(job, stage)) {
                const size_t occupied = job.fifo[outputQueue(stage)].size() + job.reserved[outputQueue(stage)];
                if (occupied + count > fifoElements_) { ++fifoBackpressureEvents; continue; }
            }
            uint64_t done = 0;
            if (!pipeline.reserve(now, &done)) { ++resourceBackpressureEvents; return; }
            Token token{candidate.first, stage, {}};
            for (size_t i = 0; i < count; ++i) {
                if (stage == Negate || stage == Direct) {
                    const size_t index = job.consumed++;
                    token.elements.push_back({index, job.input[index]});
                } else {
                    token.elements.push_back(job.fifo[inputQueue(stage)].front());
                    job.fifo[inputQueue(stage)].pop_front();
                }
            }
            if (producesFifo(job, stage)) {
                job.reserved[outputQueue(stage)] += count;
                fifoHighWater = std::max<uint64_t>(fifoHighWater,
                    job.fifo[outputQueue(stage)].size() + job.reserved[outputQueue(stage)]);
            }
            if (trace) trace({candidate.first, now, done, stage, token.elements.front().index, count});
            pending_.emplace(done, std::move(token));
            cursor = ticket(candidate);
            return;
        }
    }

    void complete(const Token& token, uint64_t now) {
        Job& job = jobs_.at(token.job);
        for (const auto& element : token.elements) {
            float y = element.value;
            switch (token.stage) {
            case Negate: y = arithmetic_.fp16() ? -std::abs(y) : -y; break;
            case TailMultiply: if(arithmetic_.fp16() && job.input[element.index]<0) job.input[element.index]=arithmetic_.mul(job.input[element.index],y); break;
            case Exp: y = arithmetic_.exp(y); break;
            case AddOne: y = arithmetic_.add(1.0f, y); break;
            case Reciprocal: y = arithmetic_.reciprocal(y); break;
            case Multiply:
                y = arithmetic_.fp16() && arithmetic_.backend()==SFUNonlinearBackend::Lut
                    ? arithmetic_.silu(job.original[element.index])
                    : arithmetic_.mul(y, job.input[element.index]);
                break;
            case GateMultiply: y = arithmetic_.mul(y, job.coefficients[element.index]); break;
            case Direct:
                y = job.op == SFUVectorOp::Add ? arithmetic_.add(y, job.coefficients[element.index])
                                             : arithmetic_.mul(y, job.coefficients[element.index]);
                break;
            }
            if (producesFifo(job, token.stage)) job.fifo[outputQueue(token.stage)].push_back({element.index, token.stage == Multiply ? vectorFp16Value(y) : y});
            else job.output[element.index] = vectorFp16Value(y);
        }
        job.done[token.stage] += token.elements.size();
        if (producesFifo(job, token.stage)) job.reserved[outputQueue(token.stage)] -= token.elements.size();
        if (((token.stage == Multiply && job.op == SFUVectorOp::Silu) || token.stage == GateMultiply || token.stage == Direct) &&
            job.done[token.stage] == job.input.size()) {
            completed_.push_back({token.job, job.submitted, now, job.op, std::move(job.output)});
            jobs_.erase(token.job);
        }
    }
};

} }
#endif
