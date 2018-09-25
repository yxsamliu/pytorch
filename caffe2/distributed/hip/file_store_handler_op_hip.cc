#include "caffe2/distributed/file_store_handler_op.h"

#include <caffe2/core/hip/context_hip.h>

namespace caffe2 {

REGISTER_HIP_OPERATOR(
    FileStoreHandlerCreate,
    FileStoreHandlerCreateOp<HIPContext>);

} // namespace caffe2
