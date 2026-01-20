#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        if (inputs.size() != 2) {
            IT_ASSERT(inputs.size() == 2, "Matmul requires two inputs.");
        }
        
        const auto A = inputs[0], B = inputs[1];
        auto shapeA = A->getDims();
        auto shapeB = B->getDims();

        // A[m, k] * B[k, n] = C[m, n]
        int m, n, k;
        if (transA) {
            m = shapeA[shapeA.size() - 1];
            k = shapeA[shapeA.size() - 2];
        } else {
            m = shapeA[shapeA.size() - 2];
            k = shapeA[shapeA.size() - 1];
        }

        if (transB) {
            n = shapeB[shapeB.size() - 2];
            IT_ASSERT(k == shapeB[shapeB.size() - 1]);
        } else {
            n = shapeB[shapeB.size() - 1];
            IT_ASSERT(k == shapeB[shapeB.size() - 2]);
        }

        Shape outputShape;

        if (shapeA.size() > 2) {
            outputShape.insert(outputShape.end(), shapeA.begin(), shapeA.end() - 2);
        }

        outputShape.push_back(m);
        outputShape.push_back(n);

        return {{outputShape}};
    }

} // namespace infini