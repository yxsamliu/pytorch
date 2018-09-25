#include "caffe2/distributed/redis_store_handler_op.h"

#include <caffe2/core/hip/context_hip.h>

namespace caffe2 {

REGISTER_HIP_OPERATOR(
    RedisStoreHandlerCreate,
    RedisStoreHandlerCreateOp<HIPContext>);

} // namespace caffe2
