import torch
from torchvision import transforms
from torchvision.transforms import GaussianBlur
import numpy as np
from skimage import exposure

_EPS = 1e-5
_NEPS = 1e-2


def fft_shift(x):
    x = transforms.ToTensor()(x)
    x = transforms.Pad(2)(x)
    x = torch.fft.fft2(x, norm="ortho")
    x = torch.fft.fftshift(x)
    return x


def fft_shift_tensor(x):
    x = torch.fft.fft2(x, norm="ortho")
    x = torch.fft.fftshift(x)
    return x


def fft_shift_to_db(x):
    x = torch.log10(x.abs() + _EPS) * torch.exp(1j * x.angle())
    x = torch.fft.fft2(x, norm="ortho")
    x = torch.fft.fftshift(x)
    return x


def ifft_shift_to_db(x):
    x = torch.fft.ifft2(x, norm="ortho")
    return x


def inverse_fft(x):
    x = torch.fft.ifft2(x, norm="ortho")
    x = x.abs()
    return x


def inverse_fft_contrast_enhanced(x):
    x = torch.fft.ifft2(x, norm="ortho")
    x = x.abs()
    # Enhance the contrast
    x = x.cpu().numpy()
    p2, p98 = np.percentile(x, (2, 98))
    x_rescaled = exposure.rescale_intensity(x, in_range=(p2, p98))
    # And back to torch tensor
    return torch.from_numpy(x_rescaled)


def abs_contrast_enhanced(x):
    x = x.abs()
    # Enhance the contrast
    x = x.cpu().numpy()
    p2, p98 = np.percentile(x, (2, 98))
    x_rescaled = exposure.rescale_intensity(x, in_range=(p2, p98))
    # And back to torch tensor
    return torch.from_numpy(x_rescaled)


def to_db(x):
    img = 20 * torch.log10(torch.abs(x) + _EPS)
    return img


def to_db_complex_chen(x):
    img = 0.4 * torch.log10(torch.abs(x) + _NEPS) * torch.exp(1j * x.angle())
    return img


def to_db_complex(x):
    return 0.4 * torch.log10(x.abs() + _EPS) * torch.exp(1j * x.angle())


def to_db_complex_blured(x):
    blurrer = GaussianBlur((3, 3), (0.7, 0.7))
    return blurrer(
        0.4 * torch.log10(x.abs() + _EPS).unsqueeze(0)
    ).squeeze() * torch.exp(1j * x.angle())


def subsample(x, factor=1, random=True):
    """Sub-sample an image of shape [C, H, W]"""
    id = 0
    if random:
        id = torch.randint(0, factor - 1, (1,)).item()
    return x[:, id::factor, id::factor]


def to_complex(x):
    x = transforms.ToTensor()(x)
    x = transforms.Pad(2)(x)
    return x + x * 1j


def subsample_fft(x, factor=2):
    x = torch.fft.fft2(x, norm="ortho")
    N = x.shape[-1]
    s = 2 * factor
    x = torch.fft.fftshift(x)
    x = x[
        :,
        N // 2 - N // s + 1 : N // 2 + N // s + 1,
        N // 2 - N // s + 1 : N // 2 + N // s + 1,
    ]
    x = torch.fft.ifft2(x, norm="ortho")
    return x


def abs(x):
    return x.abs()


def angle(x):
    return x.angle()
