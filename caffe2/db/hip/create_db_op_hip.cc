#include "caffe2/core/hip/context_hip.h"
#include "caffe2/db/create_db_op.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(CreateDB, CreateDBOp<HIPContext>);
} // namespace caffe2
