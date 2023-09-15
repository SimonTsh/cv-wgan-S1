import glob
import sys
import os
import re
import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import DatasetFolder
from torchvision.utils import make_grid, save_image

try:
    from .transform import *
except:
    from transform import *


class SAR_DatasetFolder(DatasetFolder):
    def __init__(self, data_path, loader, extensions, transform) -> None:
        self.root_data_path = os.path.dirname(os.path.normpath(data_path))
        self.data_dir = os.path.basename(os.path.normpath(data_path))
        super().__init__(
            root=self.root_data_path,
            loader=loader,
            extensions=extensions,
            transform=transform,
        )

    def find_classes(self, directory: str):
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(
            entry.name
            for entry in os.scandir(directory)
            if (entry.is_dir() and entry.name == self.data_dir)
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __len__(self) -> int:
        return len(self.samples)


class SAR_DatasetNpy_SingleFile(Dataset):
    def __init__(
        self,
        filepath: str,
        image_size: int,
        transform,
        subsample_factor: int,
        stride: int,
        random_subset: bool,
    ) -> None:
        super().__init__()

        self.im_size = image_size
        self.subsample = subsample_factor
        self.transform = transform
        self.stride = stride
        self.raw_image = self._load_sar_image(filepath)
        self.random_subset = random_subset

    @staticmethod
    def _load_sar_image(filepath: str):
        """Load the whole slc image from a directory with two files:
        - .slc data file
        - .ann annotation file

        Args:
            filepath (str): raw file directory
        """
        # .ann and .slc paths

        raw_image = torch.from_numpy(np.load(filepath))
        print(f"Loaded a SLC image of shape {raw_image.shape}")

        return raw_image

    def __len__(self) -> int:
        """The len of the dataset is the number of images
        of size <image_size> that one can tile on the original
        raw image.
        """
        H, W = self.raw_image.shape

        if self.random_subset:
            return (H // self.im_size) * (W // self.im_size)
        else:
            return ((H - self.im_size + 1) // self.stride) * (
                (W - self.im_size + 1) // self.stride
            )

    def __getitem__(self, idx):
        """Randomly draws an image in the large raw data slc image."""
        H, W = self.raw_image.shape

        # Top left corner of the cropped image
        # In the width, we can have  H - self.im_size + 1 indices
        if self.random_subset:
            # Top left corner of the cropped image
            h, w = (
                torch.randint(H - self.im_size - 1, (1,)).item(),
                torch.randint(W - self.im_size - 1, (1,)).item(),
            )
        else:
            row_size = (W - self.im_size + 1) // self.stride
            h = (idx // row_size) * self.stride
            w = (idx % row_size) * self.stride

        sample = self.raw_image[h : h + self.im_size, w : w + self.im_size].unsqueeze(0)

        if self.subsample > 1:
            sample = subsample_fft(sample, self.subsample)

        if self.transform:
            sample = self.transform(sample)

        return sample, idx


def SAR_DatasetNpy(
    raw_data_dir: str,
    image_size: int,
    transform,
    subsample_factor: int,
    stride: int,
    random_subset: bool,
):
    filepaths = glob.glob(f"{raw_data_dir}/*.npy")
    return ConcatDataset(
        [
            SAR_DatasetNpy_SingleFile(
                filepath,
                image_size,
                transform,
                subsample_factor,
                stride=stride,
                random_subset=random_subset,
            )
            for filepath in filepaths
        ]
    )


class SAR_DatasetRaw_SingleFile(Dataset):
    def __init__(
        self,
        ann_filepath: str,
        image_size: int,
        transform,
        subsample_factor: int,
        stride: int,
        random_subset: bool,
    ) -> None:
        super().__init__()

        self.im_size = image_size
        self.subsample = subsample_factor
        self.transform = transform
        self.stride = stride

        self.nrow = 0
        self.ncol = 0
        self.sample_shape = self.im_size // self.subsample

        self.raw_image = self._load_sar_image(ann_filepath)

        self.random_subset = random_subset

    @staticmethod
    def _load_sar_image(ann_filepath: str):
        """Load the whole slc image from a directory with two files:
        - .slc data file
        - .ann annotation file

        Args:
            filepath (str): raw data directory
        """

        slc_filepath = glob.glob(f"{ann_filepath[:-4]}*.slc")
        if len(slc_filepath) != 1:
            print(
                f"Cannot find a single SLC file associated with the annotation file {ann_filepath}; Found {len(slc_filepath)} match(es)"
            )
            sys.exit(-1)
        slc_filepath = slc_filepath[0]
        slc_basename = os.path.basename(slc_filepath)
        # Read meta data
        slc_shortname = None
        nrow = None
        ncol = None
        print(f"I will be looking for the slc shortname {slc_basename}")
        with open(ann_filepath, "r") as f:
            for line in f:  # Iterate on each line
                # If we find the name of the slc, we extract the
                # short name to then find the right Rows/Columns
                if slc_basename in line:
                    slc_shortname = line.split(" ")[0]
                    print(f"Found the shortname : {slc_shortname}")

                if slc_shortname is not None and slc_shortname in line:
                    if "Rows" in line:
                        nrow = int(re.split("\s+", line)[4])
                        print(f"Rows : {nrow}")
                    elif "Columns" in line:
                        ncol = int(re.split("\s+", line)[4])
                        print(f"Cols : {ncol}")

        if nrow is None or ncol is None:
            raise ValueError(
                f"I was not able to find for the Rows and Cols of the slc {slc_filepath} in annotation file {ann_filepath}"
            )
        raw_image = torch.from_numpy(
            np.fromfile(slc_filepath, dtype=np.complex64).reshape((nrow, ncol))
        )
        print(f"Loaded the SLC image {slc_filepath} of shape {raw_image.shape}")

        return raw_image

    def __len__(self) -> int:
        """The len of the dataset is the approximate number of images
        of size <image_size> that one can tile on the original
        raw image.
        """
        H, W = self.raw_image.shape
        print(
            f"The size is {((H - self.im_size + 1) // self.stride) * ((W - self.im_size + 1) // self.stride)}"
        )
        if self.random_subset:
            return (H // self.im_size) * (W // self.im_size)
        else:
            return ((H - self.im_size + 1) // self.stride) * (
                (W - self.im_size + 1) // self.stride
            )

    def __getitem__(self, idx):
        """Randomly draws an image in the large raw data slc image."""
        H, W = self.raw_image.shape

        # Top left corner of the cropped image
        if self.random_subset:
            # Top left corner of the cropped image
            h, w = (
                torch.randint(H - self.im_size - 1, (1,)).item(),
                torch.randint(W - self.im_size - 1, (1,)).item(),
            )
        else:
            # In the width, we can have  H - self.im_size + 1 indices
            row_size = (W - self.im_size + 1) // self.stride
            h = (idx // row_size) * self.stride
            w = (idx % row_size) * self.stride

        sample = self.raw_image[h : h + self.im_size, w : w + self.im_size].unsqueeze(0)

        if self.subsample > 1:
            sample = subsample_fft(sample, self.subsample)

        if self.transform:
            sample = self.transform(sample)

        return sample, idx


def SAR_DatasetRaw(
    raw_data_dir: str,
    image_size: int,
    transform,
    subsample_factor: int,
    stride: int,
    random_subset: bool,
):
    filepaths = glob.glob(f"{raw_data_dir}/*.ann")
    return ConcatDataset(
        [
            SAR_DatasetRaw_SingleFile(
                filepath,
                image_size,
                transform,
                subsample_factor,
                stride=stride,
                random_subset=random_subset,
            )
            for filepath in filepaths
        ]
    )


def test_dataset():
    DATA_DIR = "/mounts/Datasets3/2022-SONDRA/raw_data_2x8/train"
    IMAGE_SIZE = 128
    TRANSFORM = to_db_complex
    SUBSAMPLE_FACTOR = 2
    STRIDE = 2

    dataset = SAR_DatasetRaw(
        raw_data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        subsample_factor=SUBSAMPLE_FACTOR,
        transform=TRANSFORM,
        stride=STRIDE,
    )

    print("Len of the dataset:", len(dataset))
    img, _ = dataset[0]
    print("Image shape:", img.shape)


def test_full_read_raw():
    DATA_DIR = "/mounts/Datasets3/2022-SONDRA/raw_data_2x8/train"
    IMAGE_SIZE = 128
    TRANSFORM = to_db_complex
    SUBSAMPLE_FACTOR = 2
    BATCH_SIZE = 512
    NUM_WORKERS = 7
    STRIDE = 20

    dataset = SAR_DatasetRaw(
        raw_data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        subsample_factor=SUBSAMPLE_FACTOR,
        transform=TRANSFORM,
        stride=STRIDE,
    )

    print("Len of the dataset:", len(dataset))
    print(f"Trying to browse it all with a dataloader with batch size {BATCH_SIZE}")

    # Build a dataloader for parallel reading
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )
    for _ in tqdm.tqdm(loader):
        pass


def test_full_read_npy():
    DATA_DIR = "/mounts/Datasets3/2022-SONDRA/norm_data_2x8/train"
    IMAGE_SIZE = 128
    TRANSFORM = to_db_complex
    SUBSAMPLE_FACTOR = 2
    BATCH_SIZE = 512
    NUM_WORKERS = 7
    STRIDE = 20

    dataset = SAR_DatasetNpy(
        raw_data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        subsample_factor=SUBSAMPLE_FACTOR,
        transform=TRANSFORM,
        stride=STRIDE,
    )

    print("Len of the dataset:", len(dataset))
    print(f"Trying to browse it all with a dataloader with batch size {BATCH_SIZE}")

    # Build a dataloader for parallel reading
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )
    for _ in tqdm.tqdm(loader):
        pass


if __name__ == "__main__":
    print("Test read")
    test_dataset()
    print("Test full read raw")
    test_full_read_raw()
    print("Test full read NPY")
    test_full_read_npy()
