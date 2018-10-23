#include <caffe2/core/hip/common_hip.h>
#include <caffe2/core/hip/context_hip.h>
#include <caffe2/video/video_input_op.h>

namespace caffe2 {

REGISTER_HIP_OPERATOR(VideoInput, VideoInputOp<HIPContext>);

} // namespace caffe2
