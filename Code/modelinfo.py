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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", "-lm", type=str, default="", required=True)
    args = parser.parse_args()

    run_path = find_run_path(args.load_model, toplogdir="./logs")
    config_path = find_config(run_path)

    device = "cpu"

    config = load_config(config_path)
    cfg_gen = config["GENERATOR"]

    # Load model
    generator = get_from_cfg(cfg_gen)
    generator.eval()
    generator.to(device)

    zdim = generator.latent_dim

    summary(generator, input_size=(1, zdim), device=device, dtypes=[torch.complex64])
