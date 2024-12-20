import argparse
import yaml
import torch
import torch.nn as nn
from torchsummary import summary

try:
    from utils.C_layers import (
        C_UpsampleConv2d,
        get_activation_function,
        get_convolution_block,
    )
    from utils.C_activation import *
except:
    from C_layers import (
        C_UpsampleConv2d,
        get_activation_function,
        get_convolution_block,
    )
    from C_activation import *


class CNN_Generator(nn.Module):
    def __init__(
        self,
        output_channels: int,
        latent_dim: int,
        num_conv_layers: int,
        channels_multiplicator: int = 0.25,
        step_upsample: int = 2,
        num_filter: int = 128,
        size_first: int = 8,
        kernel_size_conv: int = 3,
        stride_conv: int = 1,
        padding_conv: int = 1,
        batch_norm: bool = True,
        activation: str = "crelu",
        dtype=torch.complex64,
        name: str = "CNN_Generator",
    ) -> None:
        super(CNN_Generator, self).__init__()

        self.name = name
        self.latent_dim = latent_dim
        self.dtype = dtype

        self.size_first = size_first
        self.num_filter = num_filter
        self.input_layer = nn.Linear(
            latent_dim, num_filter * size_first * size_first, dtype=dtype
        )

        # Layers of the models
        self.conv_layers = []

        ### CONV BLOCK
        in_channels = num_filter
        out_channels = num_filter

        for l in range(num_conv_layers):

            if l % step_upsample == 0 and l != 0:
                in_channels = max(out_channels, 1)
                out_channels = max(int(in_channels * channels_multiplicator), 1)
                upsample = True

            else:
                in_channels = out_channels
                upsample = False

            conv_block = get_convolution_block(
                in_channels,
                out_channels,
                kernel_size_conv,
                stride_conv,
                padding_conv,
                0,  # No pooling
                0,
                0,
                batch_norm,  # BatchNorm
                activation,
                dtype,
            )

            self.conv_layers.append(conv_block)

            if upsample:
                upsample_conv = C_UpsampleConv2d(
                    out_channels,
                    out_channels,
                    kernel_size_conv,
                    stride_conv,
                    padding_conv,
                    dtype=dtype,
                )
                self.conv_layers.append(upsample_conv)
                self.conv_layers.append(get_activation_function(activation))

        self.out_layer = get_convolution_block(
            out_channels,
            output_channels,
            2 * kernel_size_conv - 1,
            stride_conv,
            2 * padding_conv,
            0,  # No pooling
            0,
            0,
            False,
            "none",
            dtype,
        )

        self.model = nn.Sequential(
            *self.conv_layers,
        )

    def forward(self, z):
        if z.dtype != torch.complex64: # and z.shape[-1] == 2:
            z = z.to(dtype=torch.complex64)
            # z = torch.view_as_complex(z)

        z = self.input_layer(z).view(
            -1, self.num_filter, self.size_first, self.size_first
        )
        z = self.model(z)
        z = self.out_layer(z)
        return z

    # def generate(self, batch_size: int = 1):
    #     device = next(self.parameters()).device
    #     noise = torch.randn(batch_size, self.latent_dim, dtype=self.dtype).to(device)
    #     return self(noise)

    # def from_noise(self, noise):
    #     device = next(self.parameters()).device
    #     return self(noise.to(device))


def get_generator_from_config(cfg_gen: dict):
    """
    Return a generator from a yaml configuration file (loaded as a dict).
    """
    generator = eval(cfg_gen["NAME"])(dtype=torch.complex64, **cfg_gen["PARAMETERS"])
    return generator


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-cfg", type=str, default="./config/config.yaml")
    parser.add_argument("--device", "-d", type=str, default="cpu")

    args = parser.parse_args()

    ### Load config
    with open(args.config, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.CFullLoader)

    ### Generator
    cfg_gen = cfg["GENERATOR"]
    generator = get_generator_from_config(cfg_gen)
    generator.to(args.device)

    ### Test
    # img = generator.generate(2)
    # print("Output shape", img.shape)

    summary(generator, (1, cfg_gen["PARAMETERS"]["latent_dim"], 2), device=args.device)
