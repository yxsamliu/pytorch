#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/data_couple.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(DataCouple, DataCoupleOp<HIPContext>);
}
