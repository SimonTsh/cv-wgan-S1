# coding: utf-8

from threading import Lock
from scipy import linalg
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

import inception_torchscript

"""
Numpy implementation of the Frechet Distance.
The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
Stable version by Danica J. Sutherland.
Params:
    mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    mu2   : The sample mean over activations, precalculated on an
            representative data set.
    sigma1: The covariance matrix over activations for generated samples.
    sigma2: The covariance matrix over activations, precalculated on an
            representative data set.
From https://github.com/GaParmar/clean-fid/ 
"""


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def torch_frechet_distance(
    mu_x: torch.Tensor, sigma_x: torch.Tensor, mu_y: torch.Tensor, sigma_y: torch.Tensor
) -> torch.Tensor:
    a = (mu_x - mu_y).square().sum(dim=-1)
    b = sigma_x.trace() + sigma_y.trace()
    c = torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum(dim=-1)

    return a + b - 2 * c


class FIDSingletonMeta(type):
    _instance = None
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                instance = super().__call__(*args, **kwargs)
                cls._instance = instance
        return cls._instance


def resize_single_channel(
    tensor_channel: torch.Tensor, output_size=(299, 299), resample=Image.BICUBIC
):
    # Tensor is expected to have the following shape
    # (1, H, W)

    # Convert the pytorch tensor to a numpy array
    # and squeeze the C=1 dim
    x_np = tensor_channel.squeeze().numpy()

    # Convert as a PIL image
    img = Image.fromarray(x_np.astype(np.float32), mode="F")

    # Resize and interpolate
    img = img.resize(output_size, resample=resample)

    # Convert back as ndarray
    x_np_resized = np.asarray(img)

    # Convert back to pytorch tensor
    # And permute to (C, H, W)
    tensor_resize = torch.from_numpy(x_np_resized.copy()).unsqueeze(axis=0)
    return tensor_resize


def resize(
    tensor: torch.Tensor, output_size=(299, 299), resample=Image.BICUBIC
) -> torch.Tensor:
    if len(tensor.shape) == 4:
        # B, C, H, W
        # We iterate over all the single samples
        # resize them individually
        return torch.stack(
            [resize(sample_i, output_size, resample) for sample_i in tensor], axis=0
        )
    elif len(tensor.shape) == 3:
        # C, H, W
        C, _, _ = tensor.shape
        if C == 1:
            return resize_single_channel(tensor, output_size, resample)
        else:
            # We resize each channel individually and re-stack them
            return torch.stack(
                [
                    resize_single_channel(channel_i, output_size, resample)
                    for channel_i in tensor
                ],
                axis=0,
            )
    else:
        raise RuntimeError(f"Cannot a {len(tensor.shape)}-d tensor")


class FIDEvaluator(metaclass=FIDSingletonMeta):
    """
    The dataset and generator are both expected to give data
    in the range [0, 1], dtype float
    """

    def __init__(
        self,
        real_dataset,
        z_dim,
        device,
        dtype,
        num_gen=50_000,
        batch_size=128,
        num_workers=8,
        verbose=True,
        postprocess=None,
    ) -> None:
        self.z_dim = z_dim
        self.dtype = dtype
        self.device = device
        self.num_gen = num_gen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose
        self.resize_fn = resize
        self.postprocess = postprocess

        # The first time we create the instance
        # we preload the Inception Network
        self.feature_extractor = inception_torchscript.InceptionV3W().to(device)
        self.feature_extractor.eval()

        # And precompute the statistics over the dataset
        self.dataset_mu, self.dataset_sigma = self.precompute_dataset_statistics(
            real_dataset, num_gen
        )

    def expand_if_needed(self, X):
        # X is expected to be [B, C, H, W]
        # for feeding inception, we need C=3
        B, C, H, W = X.shape
        # if C != 3:
        if C == 1:
            return X.expand(B, 3, H, W)
        elif C == 3:
            return X
        else:
            raise RuntimeError(
                f"Got an input tensor with C={C}. I do not know how to expand it to match the C=3 constraints of inception "
            )

    def postprocess_if_needed(self, X):
        if self.postprocess:
            return self.postprocess(X)
        else:
            return X

    def precompute_dataset_statistics(self, dataset, num_samples):
        # Build a dataloader
        if isinstance(dataset, torch.utils.data.DataLoader):
            dataloader = dataset
            # the batch size for these iterations is given by the
            # dataloadr
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
            )
        local_batch_size = dataloader.batch_size
        features = []
        tot_num_samples = 0
        if self.verbose:
            print(
                f"I will be using at most { num_samples // local_batch_size } minibatches"
            )
        with torch.no_grad():
            pbar = (
                tqdm(dataloader, desc="Dataset stats") if self.verbose else dataloader
            )
            for X in pbar:
                # X might be a list if the datasets has
                # input, labels, ...
                # We consider the input tensors are the first entry
                if isinstance(X, tuple) or isinstance(X, list):
                    X = X[0]
                X = self.postprocess_if_needed(X)
                X = self.resize_fn(X)
                X = self.expand_if_needed(X)
                feat = self.feature_extractor(X.to(self.device))
                feat = feat.detach().cpu().numpy()
                features.append(feat)
                # We count the total number of samples to use at
                # most num_gen samples
                # This leads to a tqdm bar which stops before finishing
                # but that may be tricky to implement this limit on the
                # dataloader above
                tot_num_samples += X.shape[0]
                if tot_num_samples >= num_samples:
                    break
        np_feats = np.concatenate(features)

        dataset_mu = np.mean(np_feats, axis=0)
        dataset_sigma = np.cov(np_feats, rowvar=False)
        return dataset_mu, dataset_sigma

    def compute_generator_statistics(self, gen, num_gen):
        if self.verbose:
            print(
                f"Computing the generator statistics using a batch size of {self.batch_size}"
            )
        features = []
        with torch.no_grad():
            rng = range(num_gen // self.batch_size)
            pbar = tqdm(rng, desc="Generator stats") if self.verbose else rng
            for _ in pbar:
                z = torch.randn((self.batch_size, self.z_dim), dtype=self.dtype).to(
                    self.device
                )
                X = gen(z)
                # Bring X back to the CPU for resizing
                X = X.cpu()
                # X is [B, C, H, W]
                # we resize all the channels of all the samples to
                # the expected dimension
                X = self.postprocess_if_needed(X)
                X = self.resize_fn(X)
                X = self.expand_if_needed(X)

                # Bring X back to the right device
                X = X.to(self.device)

                # x is [B, C, 299, 299]
                # If needed, we expand the channel size to match the expected
                # channel size of the feature extractor
                feat = self.feature_extractor(X)
                feat = feat.detach().cpu().numpy()
                features.append(feat)
        np_feats = np.concatenate(features)

        gen_mu = np.mean(np_feats, axis=0)
        gen_sigma = np.cov(np_feats, rowvar=False)
        return gen_mu, gen_sigma

    def compute(self, gen):
        gen_mu, gen_sigma = self.compute_generator_statistics(gen, self.num_gen)

        return frechet_distance(self.dataset_mu, self.dataset_sigma, gen_mu, gen_sigma)


def evaluate_fid(dataset, gen, device, z_dim, dtype, postprocess, batch_size):
    FID_evaluator = FIDEvaluator(
        real_dataset=dataset,
        z_dim=z_dim,
        device=device,
        dtype=dtype,
        postprocess=postprocess,
        batch_size=batch_size,
    )
    score = FID_evaluator.compute(gen)
    return score
