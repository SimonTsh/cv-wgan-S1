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
        self.input_layer = get_convolution_block(
            size_first, size_first, dtype=dtype
        ) # * num_filter

        # Layers of the models
        self.conv_layers = []

        ### CONV BLOCK
        # in_channels = num_filter
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

        # z = self.input_layer(z).view(
        #     -1, self.size_first, self.size_first
        # ) # self.num_filter,

        if z.dim() == 3 and z.size(0) != self.num_filter:
            z = z.permute(1,2,0)
            z = z.unsqueeze(0)
        
        if z.dim() == 4 and z.size(1) != self.num_filter:
            z = z.permute(2,3,0,1)
        
        z = self.model(z)
        z = self.out_layer(z) # needed for 2048x256x256--> 1x256x256

        if z.dim() == 4:
            z = z.squeeze(0)

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
    parser.add_argument("--config", "-cfg", type=str, default="Code/logs/SAR_WGAN_28/config.yaml") #"./config/config.yaml")
    parser.add_argument("--device", "-d", type=str, default="cpu")
    parser.add_argument("--img_size", "-s", type=int, default=64) # 64 # 256

    args = parser.parse_args()    
    img_size = args.img_size

    ### Load config
    with open(args.config, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.CFullLoader)

    ### Generator
    cfg_gen = cfg["GENERATOR"]
    generator = get_generator_from_config(cfg_gen)
    generator.to(args.device)

    ### Test
    lr_img = torch.rand(1, img_size, img_size, dtype=torch.complex64).to(args.device)
    hr_img = generator(lr_img)
    print("Output shape", hr_img.shape)

    summary(generator, (1, img_size, img_size), device=args.device)
