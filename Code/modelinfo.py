# coding: utf-8

# Standard imports
import argparse


# External imports
from torchinfo import summary
import torch

# Local imports
from main import get_from_cfg
from utils.config import load_config
from utils.logs import find_run_path, find_config


def get_generator_info(config, device):
    # Generator
    cfg_gen = config["GENERATOR"]

    # Load model
    generator = get_from_cfg(cfg_gen)
    generator.eval()
    generator.to(device)

    zdim = generator.latent_dim
    print(f"Latent dimension: {zdim}")
    summary(generator, input_size=(1, zdim), device=device, dtypes=[torch.complex64])


def get_discriminator_info(config, device, image_size, input_channels):
    # Critic
    cfg_disc = config["DISCRIMINATOR"]

    cfg_disc["PARAMETERS"]["image_size"] = image_size
    cfg_disc["PARAMETERS"]["input_channels"] = input_channels

    # Load model
    discriminator = get_from_cfg(cfg_disc)
    discriminator.eval()
    discriminator.to(device)

    # input_channels = 1
    # num_conv_layers = cfg_dis["PARAMETERS"]["num_conv_layers"]
    # print(f"Latent dimension: {zdim}")
    summary(
        discriminator,
        input_size=(1, input_channels, image_size, image_size),
        device=device,
        dtypes=[torch.complex64],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", "-lm", type=str, default="", required=True)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--input_channels", type=int, default=1)
    args = parser.parse_args()

    run_path = find_run_path(args.load_model, toplogdir="./logs")
    config_path = find_config(run_path)

    device = "cpu"

    config = load_config(config_path)

    # Generator
    print("=====================================")
    print("======== GENERATOR  =================")
    get_generator_info(config, device)
    print("\n\n")
    # Critic
    print("=====================================")
    print("======== CRITIC  ====================")
    get_discriminator_info(config, device, args.image_size, args.input_channels)
