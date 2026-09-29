#include <cassert>
#include <cmath>
#include <vector>

#include <sst/elements/golem/sfu/vector_math.h>

int main() {
    using namespace SST::Golem;

    const std::vector<double> ropeInput = {1.0, 2.0, -3.0, 4.0};
    const std::vector<double> table = {1.0, 0.0, 0.0, 1.0};
    const std::vector<uint32_t> positions = {0, 1};
    SFUVectorRequest request{};
    request.op = SFUVectorOp::Rope;
    request.input = &ropeInput;
    request.coefficients = &table;
    request.positions = &positions;
    request.rowWidth = 2;
    request.activeWidth = 2;
    request.coefficientStride = 2;
    std::vector<double> output;
    assert(evaluateSFUVectorOp(request, &output));
    assert((output == std::vector<double>{1.0, 2.0, -4.0, -3.0}));
    request.activeWidth = 1;
    assert(!evaluateSFUVectorOp(request, &output));

    const std::vector<double> normInput = {
        1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
    };
    const std::vector<double> gamma = {1.0, 2.0, 0.5, 1.0};
    request = {};
    request.op = SFUVectorOp::RmsNorm;
    request.input = &normInput;
    request.coefficients = &gamma;
    request.rowWidth = 4;
    assert(evaluateSFUVectorOp(request, &output));
    assert((output == std::vector<double>{
        1.0, 2.0, 0.5, 1.0,
        0.0, 0.0, 0.0, 0.0,
    }));
    const std::vector<double> variedInput = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> variedGamma = {1.0, 0.5, -1.0, 2.0};
    request.input = &variedInput;
    request.coefficients = &variedGamma;
    assert(evaluateSFUVectorOp(request, &output));
    const double inverseRms = 1.0 / std::sqrt(7.5 + 1.0e-6);
    for (size_t dim = 0; dim < variedInput.size(); ++dim) {
        assert(std::fabs(output[dim] -
            variedInput[dim] * inverseRms * variedGamma[dim]) < 0.001);
    }
    request.epsilon = 0.0f;
    assert(!evaluateSFUVectorOp(request, &output));
    return 0;
}
