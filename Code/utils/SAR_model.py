import torch
import torch.nn as nn
import torch.nn.functional as F 

def init_weight(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            if m.bias.data is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, affine: bool = True):
        super().__init__()
        self.norm = nn.GroupNorm(num_channels, num_channels, affine=affine)

    def forward(self, x):
        return self.norm(x)

class SAR_Generator(nn.Module):
    def __init__(
        self, 
        latent_dim:int, 
        out_channels:int, 
        norm_layer:str,
        final_activation:str,
    ):

        super(SAR_Generator, self).__init__()

        self.name = "SAR_generator"
        
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.final_activation = eval(f"{final_activation}")

        norm_layer = eval(f"{norm_layer}")

        self.net = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 2048, 4, 1, 0, bias=False),
            norm_layer(2048),
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),
            norm_layer(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            norm_layer(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            norm_layer(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            norm_layer(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            norm_layer(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.out_channels, 4, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.net(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x
    
    def generate(self, batch_size):
        device = next(self.parameters()).device
        noise = torch.randn(batch_size, self.latent_dim, 1, 1).to(device)
        return self(noise)
    
    def from_noise(self, noise):
        device = next(self.parameters()).device
        return self(noise.to(device))


class SAR_Discriminator(nn.Module):
    def __init__(
        self,
        in_channels:int,
        norm_layer:str,
        final_activation:str,
    ):

        super(SAR_Discriminator, self).__init__()

        self.name = "SAR_discriminator"

        self.in_channels = in_channels
        self.final_activation = eval(f"{final_activation}")

        norm_layer = eval(f"{norm_layer}")

        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            norm_layer(128, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            norm_layer(256, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            norm_layer(512, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            norm_layer(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 2048, 4, 2, 1, bias=False),
            norm_layer(2048),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2048, 1, 4, 1, 0, bias=False)
        )
    
    def forward(self, x):
        x = self.net(x)
        return (
            x if self.final_activation is None else self.final_activation(x)
        )

if __name__ == "__main__":

    from torchsummary import summary

    latent_dim = 512
    out_channels = 3
    in_channels = 3
    batch_size = 128

    gen = SAR_Generator(latent_dim, out_channels, norm_layer="nn.BatchNorm2d", final_activation="None")
    summary(gen, input_size=(latent_dim, 1, 1), device="cpu")
    fake = gen.generate(batch_size)
    print(fake.shape)

    disc = SAR_Discriminator(in_channels, norm_layer="LayerNorm2d", final_activation="None")
    summary(disc, input_size=(3, 256, 256), device="cpu")
    output = disc(fake)
    print(output.shape)
