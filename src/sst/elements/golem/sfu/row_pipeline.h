#ifndef SST_GOLEM_SFU_ROW_PIPELINE_H
#define SST_GOLEM_SFU_ROW_PIPELINE_H

#include <algorithm>
#include <functional>
#include <map>
#include <stdexcept>
#include <utility>
#include <sst/elements/golem/sfu/hardware_pipeline.h>
#include <sst/elements/golem/sfu/vector_math.h>

namespace SST { namespace Golem {

// RoPE/Norm micro-op controller. Input/output snapshots are bounded descriptor
// staging, distinct from the four FP16 row banks and finite beat registers.
// A shared finite beat-register pool overlaps reads/compute/writes. Phase
// barriers retain FP16 reduction order; serial mode uses the identical pool.
class SFURowPipeline {
public:
    enum class Phase { Load, NormMax, InputScale, Square, Reduce, Mean, EpsilonScaleA, EpsilonScaleB, Epsilon, Rsqrt,
                       Scale, Gamma, RopeLeft, RopeRight, RopeMerge, Done };
    enum class Port { Read, Compute, Write };
    struct Issue {
        uint64_t job, context, issueCycle, completionCycle;
        Phase phase; Port port;
        size_t first, elements, bytes;
        const char* unit; size_t micro;
    };
    struct Completion {
        uint64_t job, submittedCycle, completionCycle;
        SFUVectorOp op;
        std::vector<double> values;
    };
    using Trace = std::function<void(const Issue&)>;
    SFURowPipeline(TensorHardwarePipeline& vector, TensorHardwarePipeline& reduction,
                   TensorHardwarePipeline& rsqrt, TensorHardwarePipeline& read,
                   TensorHardwarePipeline& write, size_t lanes, size_t readElements,
                   size_t writeElements, size_t rowBytes, size_t stagingBytes,
                   size_t contexts, size_t maxRequests, size_t fifoElements,
                   SFUInternalPrecision precision = SFUInternalPrecision::FP16,
                   bool tree = true, TensorHardwarePipeline* divide = nullptr,
                   SFUNonlinearBackend backend = SFUNonlinearBackend::Libm, bool overlap = true)
        : vector_(vector), reduction_(reduction), rsqrt_(rsqrt), read_(read), write_(write),
          lanes_(lanes), readElements_(readElements), writeElements_(writeElements),
          rowBytes_(rowBytes), stagingBytes_(stagingBytes),
          contexts_(lanes ? std::min(contexts, fifoElements / lanes) : 0),
          maxRequests_(maxRequests), arithmetic_(precision,backend), tree_(tree), divide_(divide),
          beatCapacity_(lanes ? fifoElements / lanes : 0), overlap_(overlap) {
        if (!lanes || readElements < 2 || writeElements < 2 || !rowBytes ||
            !stagingBytes || !contexts_ || !maxRequests)
            throw std::invalid_argument("invalid SFU row pipeline capacity");
    }
    bool submit(const SFUVectorRequest& r, uint64_t now, uint64_t* id) {
        if (!id || !r.input || !r.coefficients || !r.rowWidth || r.input->empty() ||
            r.input->size() % r.rowWidth || r.input->size() > 8192 ||
            (r.op != SFUVectorOp::Rope && r.op != SFUVectorOp::RmsNorm) ||
            uint64_t(r.rowWidth) * 8 > rowBytes_ || jobs_.size() >= maxRequests_) return false;
        const size_t rows = r.input->size() / r.rowWidth;
        if (r.op == SFUVectorOp::RmsNorm) {
            if (r.coefficients->size() != r.rowWidth || !std::isfinite(r.epsilon) ||
                r.epsilon <= 0) return false;
        } else {
            if (!r.positions || r.positions->size() != rows || !r.activeWidth ||
                r.activeWidth > r.rowWidth || (r.activeWidth & 1) || !r.coefficientStride ||
                uint64_t(r.coefficientOffset) + r.activeWidth > r.coefficientStride ||
                r.coefficients->size() % r.coefficientStride) return false;
            for (auto p : *r.positions)
                if (p >= r.coefficients->size() / r.coefficientStride) return false;
        }
        // FP16 input + output + coefficient snapshots, plus 32-bit positions.
        const size_t bytes = r.input->size() * 4 + r.coefficients->size() * 2 +
                             (r.op == SFUVectorOp::Rope ? rows * 4 : 0);
        if (bytes > stagingBytes_ - stagingUsed_) { ++admissionBackpressure; return false; }
        Job job; job.op = r.op; job.submitted = now; job.width = r.rowWidth;
        job.activeWidth = r.activeWidth; job.stride = r.coefficientStride;
        job.offset = r.coefficientOffset; job.epsilon = r.epsilon; job.bytes = bytes;
        job.input = *r.input; job.coefficients = *r.coefficients;
        if (r.op == SFUVectorOp::Rope) job.positions = *r.positions;
        job.output = r.op == SFUVectorOp::Rope ? job.input : std::vector<double>(job.input.size());
        *id = ++sequence_; jobs_.emplace(*id, std::move(job)); stagingUsed_ += bytes;
        stagingHighWater = std::max(stagingHighWater, stagingUsed_);
        requestHighWater = std::max(requestHighWater, jobs_.size()); return true;
    }
    bool active() const { return !jobs_.empty(); }
    size_t rowCapacity() const { return rowBytes_; }
    size_t stagingCapacity() const { return stagingBytes_; }
    size_t contexts() const { return contexts_; }
    size_t beatCapacity() const { return beatCapacity_; }
    size_t beatUsed() const { return beatUsed_; }
    bool overlaps() const { return overlap_; }
    size_t rowUsed() const { return rowUsed_; }
    size_t stagingUsed() const { return stagingUsed_; }
    std::vector<Completion> takeCompleted() { return std::exchange(completed_, {}); }
    static const char* name(Phase p) {
        switch (p) {
        case Phase::NormMax:return "norm_max"; case Phase::InputScale:return "norm_input_scale";
        case Phase::EpsilonScaleA:return "norm_epsilon_scale_a"; case Phase::EpsilonScaleB:return "norm_epsilon_scale_b";
        case Phase::Load:return "row_load"; case Phase::Square:return "norm_square";
        case Phase::Reduce:return "norm_reduce"; case Phase::Mean:return "norm_mean";
        case Phase::Epsilon:return "norm_epsilon"; case Phase::Rsqrt:return "norm_root";
        case Phase::Scale:return "norm_scale"; case Phase::Gamma:return "norm_gamma";
        case Phase::RopeLeft:return "rope_left"; case Phase::RopeRight:return "rope_right";
        case Phase::RopeMerge:return "rope_merge"; case Phase::Done:return "done";
        } return "invalid";
    }
    void step(uint64_t now, const Trace& trace = {}) {
        for (auto* resource : {&vector_, &reduction_, &rsqrt_, &read_, &write_}) resource->retire(now);
        if(divide_) divide_->retire(now);
        while (!pending_.empty() && pending_.begin()->first <= now) {
            const auto token = pending_.begin()->second; const auto done = pending_.begin()->first;
            pending_.erase(pending_.begin());
            auto& c = rows_.at(token.context);
            auto& b = c.beats.at(token.beat); b.busy = false;
            complete(c, b, token.port);
            if (b.finished) {
                c.completed += b.count; c.beats.erase(token.beat); --beatUsed_;
                if (c.completed == width(c)) nextPhase(c);
            }
            if (c.phase == Phase::Done) {
                auto& job = jobs_.at(c.job); ++job.finished;
                rowUsed_ -= job.width * 8;
                const auto jobId = c.job; rows_.erase(token.context);
                if (job.finished == job.input.size() / job.width) {
                    completed_.push_back({jobId, job.submitted, done, job.op, std::move(job.output)});
                    stagingUsed_ -= job.bytes; jobs_.erase(jobId);
                }
            }
        }
        admitRows();
        if(active()) ++activeCycles;
        // Each credit covers a full lane beat for its entire read/compute/write
        // lifetime. All row contexts compete for the same fixed register pool.
        std::vector<uint64_t> allocation;
        for(const auto& entry:rows_) allocation.push_back(entry.first);
        auto allocationFirst=std::upper_bound(allocation.begin(),allocation.end(),beatCursor_);
        std::rotate(allocation.begin(),allocationFirst,allocation.end());
        for(auto key:allocation) {
            auto& c=rows_.at(key);
            if(c.cursor==width(c) || (!overlap_ && !c.beats.empty()) ||
               ((scalar(c.phase) || (c.phase==Phase::Reduce && !tree_)) && !c.beats.empty())) continue;
            if(beatUsed_==beatCapacity_){++beatBackpressure;continue;}
            Beat b; b.cursor=c.cursor; b.count=count(c);
            b.port=c.phase==Phase::Load ? Port::Write : (scalar(c.phase) && !(compensated(c) && c.phase==Phase::Mean)) ? Port::Compute : Port::Read;
            c.cursor+=b.count; c.beats.emplace(++beatSequence_,std::move(b));
            ++beatUsed_; beatCursor_=key; ++issuedBeats;
            beatHighWater=std::max(beatHighWater,beatUsed_);
            rowBeatHighWater=std::max(rowBeatHighWater,c.beats.size());
        }
        // Independent physical ports; all arithmetic contends for the same
        // vector object as Add/Multiply/SiLU. Each port issues at most once/cycle.
        for (auto port : {Port::Read, Port::Compute, Port::Write}) {
            std::vector<uint64_t> order;
            for (const auto& entry : rows_) order.push_back(entry.first);
            const size_t pi = static_cast<size_t>(port);
            auto first = std::upper_bound(order.begin(), order.end(), cursors_[pi]);
            std::rotate(order.begin(), first, order.end());
            for (auto key : order) {
                auto& c = rows_.at(key);
                // Preserve per-port beat order within a phase. In-place tree
                // writes can never pass earlier unread operands this way.
                auto selected=c.beats.end();
                for(auto it=c.beats.begin();it!=c.beats.end();++it)
                    if(it->second.port==port && !it->second.busy){selected=it;break;}
                if(selected==c.beats.end() || selected->second.busy) continue;
                auto& b=selected->second;
                auto& resource = pipeline(c,port,b);
                if (now < resource.nextIssueCycle) { ++resourceBackpressure; ++portWaitCycles[pi]; continue; }
                uint64_t done = 0;
                if (!resource.reserve(now, &done)) { ++resourceBackpressure; ++portWaitCycles[pi]; continue; }
                b.busy = true; pending_.emplace(done, Token{key,selected->first,port}); cursors_[pi] = key;
                size_t bytes = 0;
                if (port == Port::Read) { bytes = reads(c,b) * 2; readBytes += bytes; }
                if (port == Port::Write) { bytes = b.count * 2 * (c.phase == Phase::Load || pairOutput(c) ? 2 : 1); writeBytes += bytes; }
                ++portIssues[pi];
                if (trace) trace({c.job, key, now, done, c.phase, port, b.cursor, b.count, bytes,
                    &resource==&read_ ? "read" : &resource==&write_ ? "write" : &resource==&vector_ ? "vector" : &resource==&reduction_ ? "reduction" : &resource==&rsqrt_ ? "root" : "divide", b.micro});
                // Compute has distinct vector/reduction/rsqrt pipelines, so
                // another context may use a different unit in this cycle.
            }
        }
    }
    size_t rowHighWater = 0, stagingHighWater = 0, requestHighWater = 0;
    size_t beatHighWater = 0, rowBeatHighWater = 0;
    uint64_t activeCycles=0, issuedBeats=0, beatBackpressure=0;
    uint64_t portWaitCycles[3]={}, portIssues[3]={};
    uint64_t resourceBackpressure = 0, admissionBackpressure = 0, readBytes = 0, writeBytes = 0;
private:
    struct Job {
        SFUVectorOp op{}; uint64_t submitted = 0;
        size_t width = 0, activeWidth = 0, stride = 0, offset = 0;
        size_t nextRow = 0, finished = 0, bytes = 0; float epsilon = 0;
        std::vector<double> input, coefficients, output;
        std::vector<uint32_t> positions;
    };
    struct Beat {
        size_t cursor=0,count=0,micro=0; Port port=Port::Write; bool busy=false,finished=false;
        std::vector<float> values, residuals;
    };
    struct Row {
        uint64_t job = 0; size_t row = 0, cursor = 0, completed = 0, reduceSize = 0;
        Phase phase = Phase::Load;
        std::map<uint64_t,Beat> beats;
        float scalar = 0, scalarLow = 0, accumulator = 0, maximum = 0, inputScale = 1, epsilon = 0;
        std::vector<float> x, gamma, work, other;
    };
    struct Token { uint64_t context,beat; Port port; };
    TensorHardwarePipeline &vector_, &reduction_, &rsqrt_, &read_, &write_;
    size_t lanes_, readElements_, writeElements_, rowBytes_, stagingBytes_, contexts_, maxRequests_;
    SFUArithmetic arithmetic_; bool tree_; TensorHardwarePipeline* divide_;
    size_t beatCapacity_,beatUsed_=0; bool overlap_;
    uint64_t beatSequence_=0,beatCursor_=0;
    uint64_t sequence_ = 0, contextSequence_ = 0, admissionCursor_ = 0, cursors_[3] = {};
    size_t rowUsed_ = 0, stagingUsed_ = 0;
    std::map<uint64_t, Job> jobs_; std::map<uint64_t, Row> rows_;
    std::multimap<uint64_t, Token> pending_; std::vector<Completion> completed_;
    static bool scalar(Phase p) { return p == Phase::Mean || p == Phase::EpsilonScaleA || p == Phase::EpsilonScaleB || p == Phase::Epsilon || p == Phase::Rsqrt; }
    void admitRows() {
        while (rows_.size() < contexts_) {
            std::vector<uint64_t> order; for (const auto& j : jobs_) order.push_back(j.first);
            auto first = std::upper_bound(order.begin(), order.end(), admissionCursor_);
            std::rotate(order.begin(), first, order.end()); bool admitted = false;
            for (auto id : order) {
                auto& j = jobs_.at(id);
                if (j.nextRow == j.input.size() / j.width || j.width * 8 > rowBytes_ - rowUsed_) continue;
                Row c; c.job = id; c.row = j.nextRow++; c.reduceSize = j.width;
                for (auto* v : {&c.x, &c.gamma, &c.work, &c.other}) v->resize(j.width);
                rows_.emplace(++contextSequence_, std::move(c)); rowUsed_ += j.width * 8;
                rowHighWater = std::max(rowHighWater, rowUsed_); admissionCursor_ = id; admitted = true; break;
            }
            if (!admitted) break;
        }
    }
    size_t width(const Row& c) const {
        const auto& j = jobs_.at(c.job);
        if (c.phase == Phase::Reduce) return tree_ ? (c.reduceSize + 1) / 2 : j.width;
        if (scalar(c.phase)) return 1;
        if (c.phase == Phase::RopeLeft || c.phase == Phase::RopeRight || c.phase == Phase::RopeMerge)
            return j.activeWidth;
        return j.width;
    }
    bool compensated(const Row& c) const {return arithmetic_.fp16() && tree_ && jobs_.at(c.job).op==SFUVectorOp::RmsNorm;}
    bool pairOutput(const Row& c) const {return compensated(c) && (c.phase==Phase::Square || c.phase==Phase::Reduce || c.phase==Phase::Scale);}
    size_t microCount(const Row& c) const {
        if(!compensated(c))return 1;
        switch(c.phase) {
        case Phase::Square:return 2;
        case Phase::Reduce:case Phase::Epsilon:return 8;
        case Phase::Mean:return 2;
        case Phase::Rsqrt:return 5;
        case Phase::Scale:case Phase::Gamma:return 4;
        default:return 1;
        }
    }
    size_t readFactor(const Row& c) const {
        if(compensated(c) && c.phase==Phase::Mean)return 2;
        if(compensated(c) && c.phase==Phase::Reduce)return 4;
        if(compensated(c) && c.phase==Phase::Gamma)return 3;
        return c.phase == Phase::Gamma || c.phase == Phase::RopeLeft || c.phase == Phase::RopeRight ||
               c.phase == Phase::RopeMerge || (c.phase == Phase::Reduce && tree_) ? 2 : 1;
    }
    size_t reads(const Row& c,const Beat& b) const {
        if (c.phase == Phase::Reduce && tree_)
            return std::min(b.count * 2, c.reduceSize - b.cursor * 2) * (compensated(c) ? 2 : 1);
        return b.count * readFactor(c);
    }
    size_t count(const Row& c) const {
        if (scalar(c.phase) || (c.phase == Phase::Reduce && !tree_)) return 1;
        const size_t portLimit = c.phase == Phase::Load ? writeElements_ / 2 : readElements_ / readFactor(c);
        return std::min({lanes_, portLimit, writeElements_ / (pairOutput(c) ? 2 : 1), width(c) - c.cursor});
    }
    TensorHardwarePipeline& pipeline(const Row& c,Port port,const Beat& b) {
        if (port == Port::Read) return read_;
        if (port == Port::Write) return write_;
        if(compensated(c)) {
            if((c.phase==Phase::Mean || c.phase==Phase::InputScale || c.phase==Phase::EpsilonScaleA || c.phase==Phase::EpsilonScaleB) && divide_)return *divide_;
            if(c.phase==Phase::Rsqrt) {
                if(b.micro==0)return rsqrt_;
                if(b.micro==4 && divide_)return *divide_;
                return vector_;
            }
            if(c.phase==Phase::Scale) {
                if((b.micro==0 || b.micro==3) && divide_)return *divide_;
                return vector_;
            }
        }
        if (c.phase == Phase::Reduce) return reduction_;
        if (c.phase == Phase::Rsqrt) return rsqrt_;
        if (c.phase == Phase::Scale && arithmetic_.fp16() && divide_) return *divide_;
        return vector_;
    }
    void start(Row& c, Phase p) {
        c.phase = p; c.cursor = 0; c.completed=0;
    }
    void nextPhase(Row& c) {
        switch (c.phase) {
        case Phase::Load:start(c, jobs_.at(c.job).op == SFUVectorOp::RmsNorm ? (arithmetic_.fp16() ? Phase::NormMax : Phase::Square) : Phase::RopeLeft);break;
        case Phase::NormMax: {
            int exponent=0; std::frexp(c.maximum,&exponent);
            c.inputScale=std::ldexp(1.f,std::max(0,exponent-1));
            start(c,Phase::InputScale);break;
        }
        case Phase::InputScale:start(c,Phase::Square);break;
        case Phase::Square:start(c, c.reduceSize == 1 ? Phase::Mean : Phase::Reduce);break;
        case Phase::Reduce:
            if (tree_) c.reduceSize = (c.reduceSize + 1) / 2;
            start(c, !tree_ || c.reduceSize == 1 ? Phase::Mean : Phase::Reduce);break;
        case Phase::Mean:c.epsilon=arithmetic_.round(jobs_.at(c.job).epsilon);start(c, arithmetic_.fp16() ? Phase::EpsilonScaleA : Phase::Epsilon);break;
        case Phase::EpsilonScaleA:start(c,Phase::EpsilonScaleB);break;
        case Phase::EpsilonScaleB:start(c,Phase::Epsilon);break;
        case Phase::Epsilon:start(c, Phase::Rsqrt);break;
        case Phase::Rsqrt:start(c, Phase::Scale);break;
        case Phase::Scale:start(c, Phase::Gamma);break;
        case Phase::Gamma:case Phase::RopeMerge:c.phase = Phase::Done;break;
        case Phase::RopeLeft:start(c, Phase::RopeRight);break;
        case Phase::RopeRight:start(c, Phase::RopeMerge);break;
        case Phase::Done:break;
        }
    }
    void complete(Row& c, Beat& b, Port port) {
        auto& j = jobs_.at(c.job); const size_t base = c.row * j.width;
        if (port == Port::Read) { b.port = Port::Compute; return; }
        if (port == Port::Compute) {
            // A micro-op reserves the real shared unit for every primitive.
            // The functional result is committed after the last reservation;
            // operands stay behind the phase barrier throughout the sequence.
            if(++b.micro < microCount(c))return;
            if(compensated(c) && scalar(c.phase)) {
                if(c.phase==Phase::Mean) {c.scalar=arithmetic_.div(c.work[0],float(j.width));c.scalarLow=arithmetic_.div(c.other[0],float(j.width));}
                else if(c.phase==Phase::EpsilonScaleA || c.phase==Phase::EpsilonScaleB)c.epsilon=arithmetic_.div(c.epsilon,c.inputScale);
                else if(c.phase==Phase::Epsilon) {auto p=normAdd(arithmetic_,{c.scalar,c.scalarLow},{c.epsilon,0});c.scalar=p.hi;c.scalarLow=p.lo;}
                else {auto p=normRoot(arithmetic_,{c.scalar,c.scalarLow});c.scalar=p.hi;c.scalarLow=p.lo;}
                b.finished=true;return;
            }
            if (scalar(c.phase)) {
                if (c.phase == Phase::Mean) c.scalar = arithmetic_.div(
                    tree_ || j.width == 1 ? c.work[0] : c.accumulator, float(j.width));
                else if (c.phase == Phase::EpsilonScaleA || c.phase == Phase::EpsilonScaleB) c.epsilon=arithmetic_.div(c.epsilon,c.inputScale);
                else if (c.phase == Phase::Epsilon) c.scalar = arithmetic_.add(c.scalar, c.epsilon);
                else c.scalar = arithmetic_.fp16() ? arithmetic_.sqrt(c.scalar) : arithmetic_.rsqrt(c.scalar);
                b.finished=true; return;
            }
            b.values.resize(b.count);b.residuals.resize(b.count);
            for (size_t k = 0; k < b.count; ++k) {
                const size_t i = b.cursor + k; float value = 0;
                if(compensated(c) && (pairOutput(c) || c.phase==Phase::Gamma)) {
                    SFUNormPair p;
                    if(c.phase==Phase::Square)p=normSquare(arithmetic_,c.x[i]);
                    else if(c.phase==Phase::Reduce)p=2*i+1<c.reduceSize ? normAdd(arithmetic_,{c.work[2*i],c.other[2*i]},{c.work[2*i+1],c.other[2*i+1]}) : SFUNormPair{c.work[2*i],c.other[2*i]};
                    else if(c.phase==Phase::Scale)p=normDivide(arithmetic_,c.x[i],{c.scalar,c.scalarLow});
                    else p={normGamma(arithmetic_,{c.work[i],c.other[i]},c.gamma[i]),0};
                    b.values[k]=p.hi;b.residuals[k]=p.lo;continue;
                }
                switch (c.phase) {
                case Phase::NormMax:value=std::abs(c.x[i]);c.maximum=std::max(c.maximum,value);break;
                case Phase::InputScale:value=arithmetic_.div(c.x[i],c.inputScale);break;
                case Phase::Square:value = arithmetic_.mul(c.x[i], c.x[i]);break;
                case Phase::Reduce:
                    if (!tree_) value = arithmetic_.add(c.accumulator, c.work[i]);
                    else value = 2*i+1 < c.reduceSize ? arithmetic_.add(c.work[2*i], c.work[2*i+1]) : c.work[2*i];
                    break;
                case Phase::Scale:value = arithmetic_.fp16() ? arithmetic_.div(c.x[i], c.scalar) : arithmetic_.mul(c.x[i], c.scalar);break;
                case Phase::Gamma:value = arithmetic_.mul(c.work[i], c.gamma[i]);break;
                case Phase::RopeLeft:value = arithmetic_.mul(c.x[i & ~size_t(1)], c.gamma[i]);break;
                case Phase::RopeRight:value = arithmetic_.mul(c.x[i | size_t(1)], c.gamma[i ^ size_t(1)]);break;
                case Phase::RopeMerge:value = (i & 1) ? arithmetic_.add(c.work[i], c.other[i]) : arithmetic_.sub(c.work[i], c.other[i]);break;
                default:throw std::logic_error("invalid SFU row compute phase");
                }
                b.values[k] = value;
            }
            if (c.phase == Phase::Reduce && !tree_) {
                c.accumulator = b.values[0]; b.finished=true;
            } else b.port = Port::Write;
            return;
        }
        for (size_t k = 0; k < b.count; ++k) {
            const size_t i = b.cursor + k;
            if (c.phase == Phase::Load) {
                c.x[i] = vectorFp16Value(j.input[base+i]);
                const size_t coefficient = j.op == SFUVectorOp::Rope ?
                    size_t(j.positions[c.row])*j.stride+j.offset+i : i;
                c.gamma[i] = j.op == SFUVectorOp::Rope && i >= j.activeWidth ? 0 : vectorFp16Value(j.coefficients[coefficient]);
            } else if (c.phase == Phase::InputScale) c.x[i]=b.values[k];
            else if (c.phase == Phase::Gamma || c.phase == Phase::RopeMerge)
                j.output[base+i] = vectorFp16Value(b.values[k]);
            else if (c.phase == Phase::RopeRight) c.other[i] = b.values[k];
            else {c.work[i] = b.values[k];if(pairOutput(c))c.other[i]=b.residuals[k];}
        }
        b.finished=true;
    }
};

} }
#endif
