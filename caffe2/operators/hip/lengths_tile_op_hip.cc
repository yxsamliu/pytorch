#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/lengths_tile_op.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(LengthsTile, LengthsTileOp<HIPContext>);
} // namespace caffe2
