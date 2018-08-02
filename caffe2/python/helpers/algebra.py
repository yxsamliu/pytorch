## @package algebra
# Module caffe2.python.helpers.algebra
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import workspace

def transpose(model, blob_in, blob_out, use_gpu_engine=False, **kwargs):
    """Transpose."""
    if use_gpu_engine:
        kwargs['engine'] = 'MIOPEN' if workspace.has_hip_support else 'CUDNN'
    return model.net.Transpose(blob_in, blob_out, **kwargs)


def sum(model, blob_in, blob_out, **kwargs):
    """Sum"""
    return model.net.Sum(blob_in, blob_out, **kwargs)


def batch_mat_mul(model, blob_in, blob_out,
                  enable_tensor_core=False, **kwargs):
    if enable_tensor_core:
        kwargs['engine'] = 'TENSORCORE'

    return model.net.BatchMatMul(blob_in, blob_out, **kwargs)
