#include "caffe2/operators/lengths_pad_op.h"

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(LengthsPad, LengthsPadOp<HIPContext>);
} // namespace caffe2
