import argparse
import yaml
import torch
import torch.nn as nn
from torchsummary import summary

try:
    from utils.C_layers import get_convolution_block, C_AvgPool2d
    from utils.C_activation import Mod
except:
    from C_layers import get_convolution_block, C_AvgPool2d
    from C_activation import Mod


class CNN_Discriminator(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_conv_layers: int,
        channels_multiplicator: int = 1,
        step_pooling: int = 1,
        max_channels: int = 128,
        num_filter: int = 128,
        kernel_size_conv: int = 3,
        stride_conv: int = 1,
        padding_conv: int = 1,
        batch_norm: bool = False,
        activation: str = "modReLU",
        image_size: int = 32,
        min_spatial_size: int = 8,
        dtype=torch.complex64,
        name: str = "CNN_Discriminator",
        **kwargs,
    ) -> None:
        super(CNN_Discriminator, self).__init__()

        self.name = name

        self.input_channels = input_channels
        self.image_size = image_size
        self.dtype = dtype

        # Layers of the models
        self.conv_layers = []
        self.fc_layers = []

        ### CONV BLOCK
        in_channels = input_channels
        out_channels = num_filter
        conv_size = image_size
        kernel_size_pool_l = stride_pool_l = padding_pool_l = 0  # No pooling
        stride_conv_l = stride_conv

        for l in range(num_conv_layers):
            if l != 0:
                in_channels = min(out_channels, max_channels)
                stride_conv_l = stride_conv

                if l % step_pooling == 0 and conv_size > min_spatial_size:
                    out_channels = min(
                        in_channels * channels_multiplicator, max_channels
                    )
                    stride_conv_l = 2
                    conv_size = conv_size // stride_conv_l

            conv_block = get_convolution_block(
                in_channels,
                out_channels,
                kernel_size_conv,
                stride_conv_l,
                padding_conv,
                kernel_size_pool_l,
                stride_pool_l,
                padding_pool_l,
                batch_norm,
                activation,
                dtype,
            )

            self.conv_layers.append(conv_block)

        ### FULLY CONNECTED BLOCK

        dummy_tensor = torch.rand(
            2, input_channels, image_size, image_size, dtype=dtype
        )
        dummy_conv_model = nn.Sequential(*self.conv_layers, *self.fc_layers)
        out = dummy_conv_model(dummy_tensor)
        conv_size = out.shape[-1]
        num_features = out.shape[1]

        self.fc_layers.append(C_AvgPool2d(conv_size, 1, 0))
        self.fc_layers.append(nn.Flatten())
        self.fc_layers.append(nn.Linear(num_features, 1, dtype=dtype))
        self.fc_layers.append(Mod())

        ### DEFINE MODEL
        self.model = nn.Sequential(*self.conv_layers, *self.fc_layers)

    def forward(self, z):
        if z.dtype != torch.complex64: # and z.shape[-1] == 2:
            z = z.to(dtype=torch.complex64)
            # z = torch.view_as_complex(z)

        return (self.model(z).real).to(
            dtype=self.dtype
        )  # Last activation in loss function


def get_discriminator_from_config(cfg_disc: dict, input_channel=1):
    """
    Return a discriminator from a yaml configuration file (loaded as a dict).
    """
    discriminator = eval(cfg_disc["NAME"])(
        dtype=torch.complex64, **cfg_disc["PARAMETERS"], input_channels=input_channel
    )
    return discriminator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-cfg", type=str, default="./config/config.yaml")
    parser.add_argument("--device", "-d", type=str, default="cpu")
    parser.add_argument("--img_size", "-s", type=int, default=64)

    args = parser.parse_args()

    img_size = args.img_size

    ### Load config
    with open(args.config, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.CFullLoader)

    ### Discriminator
    cfg_disc = cfg["DISCRIMINATOR"]
    cfg_disc["PARAMETERS"]["image_size"] = img_size
    cfg_disc["PARAMETERS"]["input_channels"] = 1
    discriminator = eval(cfg_disc["NAME"])(**cfg_disc["PARAMETERS"])

    # Test
    discriminator.to(args.device)
    x = torch.rand(2, 1, img_size, img_size, dtype=torch.complex64).to(args.device)
    y = discriminator(x)

    summary(discriminator, (1, img_size, img_size, 2), device=args.device)
