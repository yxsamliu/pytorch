#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/rnn/recurrent_network_blob_fetcher_op.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(
    RecurrentNetworkBlobFetcher,
    RecurrentNetworkBlobFetcherOp<HIPContext>);
} // namespace caffe2
