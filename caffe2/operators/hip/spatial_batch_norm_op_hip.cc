#include "caffe2/operators/spatial_batch_norm_op.h"

#include "caffe2/operators/hip/spatial_batch_norm_op_hip_impl.cuh"

namespace caffe2 {

REGISTER_HIP_OPERATOR(SpatialBN, SpatialBNOp<HIPContext>);
REGISTER_HIP_OPERATOR(SpatialBNGradient, SpatialBNGradientOp<HIPContext>);

} // namespace caffe2
