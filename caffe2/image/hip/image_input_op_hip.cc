#include "caffe2/core/hip/common_hip.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/image/image_input_op.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(ImageInput, ImageInputOp<HIPContext>);

}  // namespace caffe2
