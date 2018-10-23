#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/hip/operator_fallback_hip.h"
#include "caffe2/operators/sparse_normalize_op.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(
    SparseNormalize,
    GPUFallbackOp);
}
