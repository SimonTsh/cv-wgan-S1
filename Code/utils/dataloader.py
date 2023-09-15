from pathlib import Path
import torch
import torchvision
import argparse
import glob
import numpy as np
import tqdm
from torchvision.utils import make_grid, save_image
from pdb import set_trace

try:
    from .transform import *
    from .dataset import SAR_DatasetFolder, SAR_DatasetRaw, SAR_DatasetNpy
except:
    from transform import *
    from dataset import SAR_DatasetFolder, SAR_DatasetRaw


def get_MNISTLike_dataloader(
    dataset_builder,
    batch_size: int,
    transform,
    length: int = -1,
    num_workers: int = 8,
    pin_memory: bool = True,
):
    if isinstance(transform, str):
        transform = eval(transform)

    root = Path.home() / "Datasets"

    dataset = dataset_builder(root=root, download=True, train=True, transform=transform)
    valid_dataset = dataset_builder(
        root=root, download=True, train=False, transform=transform
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    return train_dataloader, valid_dataloader


def get_EMNIST_dataloader(
    batch_size: int,
    transform,
    length: int = -1,
    num_workers: int = 8,
    pin_memory: bool = True,
):
    if isinstance(transform, str):
        transform = eval(transform)
    dataset_builder = torchvision.datasets.EMNIST

    root = Path.home() / "Datasets"
    dataset = dataset_builder(
        root=root, download=True, train=True, transform=transform, split="byclass"
    )
    valid_dataset = dataset_builder(
        root=root, download=True, train=False, transform=transform, split="byclass"
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    return train_dataloader, valid_dataloader


def get_FashionMNIST_dataloader(
    batch_size: int,
    transform,
    length: int = -1,
    num_workers: int = 8,
    pin_memory: bool = True,
):
    return get_MNISTLike_dataloader(
        torchvision.datasets.FashionMNIST,
        batch_size,
        transform,
        length,
        num_workers,
        pin_memory,
    )


def get_MNIST_dataloader(
    batch_size: int,
    transform,
    length: int = -1,
    num_workers: int = 8,
    pin_memory: bool = True,
):
    return get_MNISTLike_dataloader(
        torchvision.datasets.MNIST,
        batch_size,
        transform,
        length,
        num_workers,
        pin_memory,
    )


def get_SAR_dataloader(
    transform=None,
    train_datapath=None,
    valid_datapath=None,
    subsample_factor: int = 1,
    batch_size: int = 32,
    image_size: int = 64,
    length: int = -1,
    stride: int = 1,
    random_subset: bool = False,
    num_workers: int = 8,
    pin_memory: bool = True,
):
    if isinstance(transform, str):
        t = eval(transform)
    else:
        t = None

    # TODO : Make it cleaner
    if subsample_factor > 1:
        transform = lambda x: subsample_fft(t(x), factor=subsample_factor)
    else:
        transform = t

    def loader_complex_image(image_path):
        """Load image given its path."""
        image = np.load(image_path)
        return torch.from_numpy(image).unsqueeze(0).to(dtype=torch.complex64)

    if train_datapath is None:
        raise ValueError("train_datapath cannot be undefined")

    train_slc_files = glob.glob(f"{train_datapath}/*.slc")
    train_npy_files = glob.glob(f"{train_datapath}/*.npy")
    train_files = None
    if len(train_slc_files) != 0:
        train_SAR_dataset = SAR_DatasetRaw(
            raw_data_dir=train_datapath,
            image_size=image_size,
            subsample_factor=subsample_factor,
            transform=transform,
            stride=stride,
            random_subset=random_subset,
        )
        num_workers -= num_workers  # Set to zero otherwise can bug
        train_files = train_slc_files
    elif len(train_npy_files) != 0:
        train_SAR_dataset = SAR_DatasetNpy(
            raw_data_dir=train_datapath,
            image_size=image_size,
            subsample_factor=subsample_factor,
            transform=transform,
            stride=stride,
            random_subset=random_subset,
        )
        num_workers -= num_workers  # Set to zero otherwise can bug
        train_files = train_npy_files
    else:
        raise RuntimeError(
            f"We did not find either ann or npy files in {train_datapath}"
        )
    if length > 0:
        # Take a subset of the dataset
        train_indices = np.random.choice(len(train_SAR_dataset), length, replace=False)
        train_SAR_dataset = torch.utils.data.Subset(train_SAR_dataset, train_indices)

    if valid_datapath is None:
        valid_SAR_dataset = None
        valid_files = None
    else:
        valid_slc_files = glob.glob(f"{valid_datapath}/*.slc")
        valid_npy_files = glob.glob(f"{valid_datapath}/*.npy")
        valid_files = None
        if len(valid_slc_files) != 0:
            valid_SAR_dataset = SAR_DatasetRaw(
                raw_data_dir=valid_datapath,
                image_size=image_size,
                subsample_factor=subsample_factor,
                transform=transform,
                stride=stride,
                random_subset=random_subset,
            )
            num_workers -= num_workers  # Set to zero otherwise can bug
            valid_files = valid_slc_files
        elif len(valid_npy_files) != 0:
            valid_SAR_dataset = SAR_DatasetNpy(
                raw_data_dir=valid_datapath,
                image_size=image_size,
                subsample_factor=subsample_factor,
                transform=transform,
                stride=stride,
                random_subset=random_subset,
            )
            num_workers -= num_workers  # Set to zero otherwise can bug
            valid_files = valid_npy_files
        else:
            raise RuntimeError(
                f"We did not find either ann or npy files in {valid_datapath}"
            )
        if length > 0:
            # Take a subset of the dataset
            valid_indices = np.random.choice(
                len(valid_SAR_dataset), length, replace=False
            )
            valid_SAR_dataset = torch.utils.data.Subset(
                valid_SAR_dataset, valid_indices
            )

    print(
        f"We will be using : \n For training: {train_files} \n For validation : {valid_files}"
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_SAR_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    if valid_SAR_dataset is not None:
        valid_dataloader = torch.utils.data.DataLoader(
            valid_SAR_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    else:
        valid_dataloader = None
    return train_dataloader, valid_dataloader


def dataloader_info(dataloader):
    print("Number of samples:", len(dataloader.dataset))
    print("Number of batches:", len(dataloader))
    images, _ = next(iter(dataloader))
    print(images.shape)
    print("Data shape:", images.shape, " - Data type:", images.dtype)
    return images


if __name__ == "__main__":
    from logs import hsv_colorscale

    parser = argparse.ArgumentParser()
    parser.add_argument("--nrow", "-n", type=int, default=5)
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default="/mounts/Datasets3/2022-SONDRA/raw_data_2x8/",
    )
    parser.add_argument("--transform", "-t", type=str, default=None)
    parser.add_argument("--size", "-s", type=int, default=256)
    parser.add_argument("--suffix", "-sf", type=str, default="")
    parser.add_argument("--saving_path", "-sv", type=str, default="./logs/")
    args = parser.parse_args()

    transform = eval(args.transform) if args.transform != None else None

    if args.data_path == "MNIST":
        train_dataloader, valid_dataloader = get_MNIST_dataloader(
            batch_size=args.nrow * args.nrow,
            transform=transform,
            num_workers=8,
            pin_memory=True,
            train=True,
        )

    else:
        train_dataloader, valid_dataloader = get_SAR_dataloader(
            train_datapath=args.data_path + "/train/",
            valid_datapath=args.data_path + "/valid/",
            transform=transform,
            subsample_factor=1,
            image_size=args.size,
            batch_size=args.nrow * args.nrow,
            stride=20,
            num_workers=0,
            pin_memory=True,
        )

    dataloader = train_dataloader

    images = dataloader_info(dataloader)

    print(
        "Min module:",
        images.abs().min().item(),
        " - Max module:",
        images.abs().max().item(),
    )

    grid_abs = make_grid(images.abs(), nrow=args.nrow, scale_each=True)
    save_image(grid_abs, f"{args.saving_path}data_abs{args.suffix}.png")

    grid_angle = make_grid(images.angle(), nrow=args.nrow, scale_each=True)
    save_image(grid_angle, f"{args.saving_path}data_angle{args.suffix}.png")

    grid_hsv = make_grid(
        hsv_colorscale(images.squeeze()), nrow=args.nrow, scale_each=True
    )
    save_image(grid_hsv, f"{args.saving_path}data_hsv{args.suffix}.png")
