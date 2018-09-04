#include "caffe2/operators/matmul_op.h"

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(MatMul, MatMulOp<float, HIPContext>);

}
