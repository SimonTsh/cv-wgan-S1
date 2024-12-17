# coding: utf-8

import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

from fid_compute_statistics import evaluate_fid
from config import load_config
from logs import find_run_path, find_config
from generator import *
import transform
import dataloader
from pathlib import Path
import numpy as np

import matplotlib

matplotlib.use("Agg")  # terminal backend
import matplotlib.pyplot as plt


def get_from_cfg(cfg_dict: dict, param=None):
    """
    Return item from config dictionary.
    dictionary has to have a key "NAME" that is the name
    of the instance to return and a key "PARAMETERS"
    which is a dict of the required parameters.
    """
    if not param:
        return eval(cfg_dict["NAME"])(**cfg_dict["PARAMETERS"])
    else:
        return eval(cfg_dict["NAME"])(param, **cfg_dict["PARAMETERS"])


def post_1chan_reshape(transform):
    def wrapped_transform(x):
        x = transform(x)
        # x is (1, H, W)
        # expand it to (3, H, W)
        return x.expand(3, *(x.shape[1:]))

    return wrapped_transform


def get_dataset(dataset_name: str, fold=str, parameters=None):
    """
    Parameters: used only for the SAR dataset (not very clean I admit)
    """
    print(parameters)
    if dataset_name in ["MNIST", "FashionMNIST"]:
        root = Path.home() / "Datasets"
        if fold == "train":
            fold = {"train": True}
        else:
            fold = {"train": False}
        dataset_builder = eval(f"torchvision.datasets.{dataset_name}")

        def transform(x):
            x = transforms.ToTensor()(x)
            x = transforms.Pad(2)(x)
            return x

        dataset = dataset_builder(root=root, download=True, transform=transform, **fold)
    elif dataset_name == "EMNIST":
        root = Path.home() / "Datasets"
        dataset_params = {"split": "byclass"}
        if fold == "train":
            dataset_params["train"] = True
        else:
            dataset_params["train"] = False
        dataset_builder = torchvision.datasets.EMNIST

        def transform(x):
            x = transforms.ToTensor()(x)
            x = transforms.Pad(2)(x)
            return x

        dataset = dataset_builder(
            root=root, download=True, transform=transform, **dataset_params
        )
    elif dataset_name == "SAR":
        train_dataloader, valid_dataloader = dataloader.get_SAR_dataloader(**parameters)
        if fold == "train":
            return train_dataloader
        elif fold == "test":
            return valid_dataloader
    else:
        raise RuntimeError(f"Unknown dataset {dataset_name}")
    return dataset


def compute_fid(
    dataset_name: str,
    fold: str,
    gen: nn.Module,
    postprocess_str,
    # mode="clean",
    z_dim=128,
    batch_size=128,
    run_test=True,#False,
    dataset_parameters=None,
):
    """
    gen: nn.Module gen(z) -> complex valued sample
    """

    device = next(gen.parameters()).device

    # Wrap, if necessary, the generator
    # For MNIST like, we apply the ifft to go in the image domain
    # For SAR, we do not do anything here. A common function
    #          is applied for both the dataset and generated sample
    def post_gen(z):
        x = gen(z)
        # x is complex valued
        if postprocess_str == "ifft":
            return transform.inverse_fft(x)
        elif postprocess_str == "None":
            return x
        else:
            raise RuntimeError(f"Unknown postprocessing {postprocess_str} for FID")

    def postprocess(x):
        if dataset_name in ["MNIST", "FashionMNIST", "EMNIST"]:
            return x
        elif dataset_name == "SAR":
            # Both data from the dataset and generated data
            # are complex, we take the magnitude of them
            return x.abs()

    # Get the reference dataset
    dataset = get_dataset(dataset_name, fold, dataset_parameters)

    # To test the shape, and bounds of the dataset and generator
    if run_test:
        test(dataset, post_gen, z_dim, device)

    # Compute the FID score in the image space
    score = evaluate_fid(
        dataset,
        post_gen,
        device=device,
        z_dim=z_dim,
        dtype=gen.dtype,
        postprocess=postprocess,
        batch_size=batch_size,
    )

    return score


def test(dataset, gen, z_dim, device):
    """
    Test function which take real samples, generated samples
    and postprocess them the exact same way to ensure, at least visually
    that we get samples from the same space
    """

    print("Run the FID test")
    N = 3 #10
    fig, axes = plt.subplots(nrows=2, ncols=N)#, figsize=(5, 1))

    def post_process_sample(X):
        return (X * 255).clip(0, 255).permute(1, 2, 0)#.numpy().astype(np.uint8)

    # Let us take real samples and save them as an image
    dataset = dataset.dataset # assumes data_loader is created from a dataset
    for iax, axi in enumerate(axes[0]):
        X, _ = dataset[iax]
        X = X.abs()
        print(
            f"Real sample : bounds=[{X.min()}, {X.max()}], shape ={X.shape}, dtype={X.dtype}"
        )
        X = post_process_sample(X)
        axi.imshow(X, cmap="gray")
        axi.set_axis_off()

    # Let us generate fake samples and save them as an image as well
    # print(x_gen.max(), x_gen.min(), x_gen.shape)
    with torch.no_grad():
        # input = dataset.datasets[0].raw_image.to(device) # noise.shape = N x z_dim = 2 x 512 for eg
        # row, col = input.shape
        # Xgen = gen((input[row // 2 - N // 2 : row // 2 + N // 2, \
        #             col // 2 - z_dim // 2 : col // 2 + z_dim // 2])).cpu() # transform.fft_shift_tensor
        noise = torch.randn(N, z_dim, dtype=torch.complex64).to(device)
        Xgen = gen(noise).cpu()

    for iax, axi in enumerate(axes[1]):
        X = Xgen[iax]

        # check generated patch statistics
        fig1, ax = plt.subplots(1,2)
        X_f = np.fft.fftshift(np.fft.fft2(X.squeeze())) # freq
        X = X.abs()
        hist, bin_edges = np.histogram(X.squeeze().flatten()) # hist
        print(f'hist sum = {hist.sum()}, {np.sum(hist * np.diff(bin_edges))}')
        ax[0].imshow(np.abs(X_f),cmap='gray')
        ax[0].set_title('fft output')
        ax[1].hist(X.squeeze().flatten(),bins='auto')
        ax[1].set_title('hist output')
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
        fig1.savefig("Code/logs/fid_analysis.png", dpi=300)
        plt.close()
        
        # save generated (abs) image
        print(
            f"Generated sample : bounds=[{X.min()}, {X.max()}], shape ={X.shape}, dtype={X.dtype}"
        )
        X = post_process_sample(X)
        axi.imshow(X, cmap="gray")
        axi.set_axis_off()

    plt.suptitle("Top: real; Bottom: generated")
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("Code/logs/fid_test.png", dpi=300)
    plt.close()
    print("fid_test.png saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", "-lm", type=str, default="SAR_WGAN_28")#, required=True)
    parser.add_argument("--postprocess", "-p", choices=["ifft", "None"], type=str, default="None")#, required=True)
    parser.add_argument("--test", "-t", action="store_true", default="store_true")
    parser.add_argument("--fold", "-f", choices=["train", "test"], default="test")#, required=True)
    args = parser.parse_args()

    run_path = find_run_path(args.load_model, toplogdir="Code/logs") #"./logs")
    config_path = find_config(run_path)

    if not config_path:
        print("Config path not found")
        exit(1)

    config = load_config(config_path)
    print("Run path found :", run_path)
    print("Config loaded :", config_path)
    print(
        f"The FID score will be computed between the generator and the dataset {config['DATASET']['NAME']}, fold {args.fold}"
    )

    cfg_gen = config["GENERATOR"]
    cfg_data = config["DATASET"]

    # Load model
    generator = get_from_cfg(cfg_gen)
    generator_state_path = os.path.join(run_path, cfg_gen["NAME"] + ".pt")
    generator.load_state_dict(torch.load(generator_state_path))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = "cuda"
    else:
        device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    generator = generator.to(device)

    score = compute_fid(
        dataset_name=config["DATASET"]["NAME"],
        fold=args.fold,
        gen=generator,
        postprocess_str=args.postprocess,
        z_dim=cfg_gen["PARAMETERS"]["latent_dim"],
        batch_size=config["DATASET"]["PARAMETERS"]["batch_size"],
        run_test=args.test,
        dataset_parameters=cfg_data["PARAMETERS"],
    )

    print(f"FID = {score}") # FID = Run1: 95.82718742133738; Run 2: 95.28351811571854
