# Originally provided by https://github.com/GaParmar/clean-fid
# Modified to remove the normalization so that the input tensors are expected
# to be provided in [-1, 1] to the forward method

import os
import urllib.request
import contextlib
import requests
import shutil
import torch
import torch.nn as nn

inception_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"


"""
Download the pretrined inception weights if it does not exists
ARGS:
    fpath - output folder path
"""


def check_download_inception(fpath="./"):
    inception_path = os.path.join(fpath, "inception-2015-12-05.pt")
    if not os.path.exists(inception_path):
        # download the file
        with urllib.request.urlopen(inception_url) as response, open(
            inception_path, "wb"
        ) as f:
            shutil.copyfileobj(response, f)
    return inception_path


@contextlib.contextmanager
def disable_gpu_fuser_on_pt19():
    # On PyTorch 1.9 a CUDA fuser bug prevents the Inception JIT model to run. See
    #   https://github.com/GaParmar/clean-fid/issues/5
    #   https://github.com/pytorch/pytorch/issues/64062
    if torch.__version__.startswith("1.9."):
        old_val = torch._C._jit_can_fuse_on_gpu()
        torch._C._jit_override_can_fuse_on_gpu(False)
    yield
    if torch.__version__.startswith("1.9."):
        torch._C._jit_override_can_fuse_on_gpu(old_val)


class InceptionV3W(nn.Module):
    """
    Wrapper around Inception V3 torchscript model provided here
    https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt

    path: locally saved inception weights
    """

    def __init__(self, path="./", download=True):
        super(InceptionV3W, self).__init__()
        # download the network if it is not present at the given directory
        # use the current directory by default
        if download:
            check_download_inception(fpath=path)
        path = os.path.join(path, "inception-2015-12-05.pt")
        self.base = torch.jit.load(path).eval()
        self.layers = self.base.layers

    """
    Get the inception features without resizing
    x: Image with values in range [0,255]
    """

    def forward(self, x):
        with disable_gpu_fuser_on_pt19():
            bs = x.shape[0]
            # make sure it is resized already
            assert (x.shape[2] == 299) and (x.shape[3] == 299)
            features = self.layers.forward(
                x,
            ).view((bs, 2048))
            return features
