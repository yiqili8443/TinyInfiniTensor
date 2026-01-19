#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    Shape dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();

    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业 ===================================
    if (inputs.size() < 0) {
        IT_ASSERT(false, "Concat operator requires at least one input tensor.");
    }

    for (auto &input : inputs) {
        IT_ASSERT(input->getRank() == rank,
                 "All input tensors must have the same rank.");
    }

    size_t concat_dim = getDim();
    for (auto &input : inputs) {
        auto input_dims = input->getDims();
        for (size_t i = 0; i < rank; i++) {
            if (i != concat_dim) {
                IT_ASSERT(input_dims[i] == dims[i],
                         "All input tensors must have the same shape except for the concat dimension.");
            }
        }
    }

    int total_size = 0;
    for (auto &input : inputs) {
        total_size += input->getDims()[concat_dim];
    }
    dims[concat_dim] = total_size;

    return {{dims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
