#include "caffe2/utils/math.h"
#include "caffe2/queue/queue_ops.h"

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(CreateBlobsQueue, CreateBlobsQueueOp<HIPContext>);
REGISTER_HIP_OPERATOR(EnqueueBlobs, EnqueueBlobsOp<HIPContext>);
REGISTER_HIP_OPERATOR(DequeueBlobs, DequeueBlobsOp<HIPContext>);
REGISTER_HIP_OPERATOR(CloseBlobsQueue, CloseBlobsQueueOp<HIPContext>);

REGISTER_HIP_OPERATOR(SafeEnqueueBlobs, SafeEnqueueBlobsOp<HIPContext>);
REGISTER_HIP_OPERATOR(SafeDequeueBlobs, SafeDequeueBlobsOp<HIPContext>);

}
