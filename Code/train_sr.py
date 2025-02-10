import os
import argparse
import yaml
import pickle

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.generator import get_generator_from_config
from utils.discriminator import get_discriminator_from_config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import grad

# Assuming you have the generator and discriminator models defined
class WGAN_GP:
    def __init__(self, generator, discriminator, device, g_loss, d_loss, lambda_gp=10, critic_iterations=5):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.lambda_gp = lambda_gp
        self.critic_iterations = critic_iterations
        self.g_loss = g_loss
        self.d_loss = d_loss

        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand((real_data.size(0), 1, 1), device=self.device)
        alpha = alpha.expand_as(real_data)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates.requires_grad = True

        d_interpolates = self.discriminator(interpolates)
        gradients = grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size(), device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.view(-1, gradients.size(-1)).norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, data_HR, data_LR):
        # Train Critic
        for i in range(self.critic_iterations):
            print(f'Critic iteration {i}...')
            self.optimizer_D.zero_grad()

            # Sample real and fake data
            real_data = data_HR.to(self.device) # 3x64x64; 1x256x256
            z = data_LR.to(self.device) # 3x64x64; 1x256x256
            fake_data = self.generator(z) # output size [1, 8192, 384]; should be 1x64x64 so can upsample x4
            print(fake_data.shape)
            # noise = torch.randn(batch_size, self.generator.latent_dim, dtype=torch.complex64).to(self.device)
            # fake_data = self.generator(noise)

            # Compute critic loss
            d_fake = self.discriminator(fake_data).mean() # input: 4x4x256x256 --> output as 1x256x256
            d_real = self.discriminator(real_data).mean() # input: 1x256x256
            gp = self.gradient_penalty(real_data, fake_data)
            self.d_loss = d_fake - d_real + self.lambda_gp * gp

            # Backpropagate and optimize
            # d_loss.backward()
            params = [p for p in self.discriminator.parameters() if p.requires_grad]
            grads = torch.autograd.grad(self.d_loss, params, create_graph=True, retain_graph=True)

            for param, grad in zip(params, grads):
                param.data = param.data.contiguous()
                
                param.grad = grad # manually set gradients for each parameter in order to retain graph
                if param.grad is not None:
                    param.grad.data = param.grad.data.contiguous()
            self.optimizer_D.step()

        # Train Generator
        self.optimizer_G.zero_grad()

        # # Sample fake data
        # noise = torch.randn(batch_size, self.generator.latent_dim, dtype=torch.complex64).to(self.device)
        # fake_data = self.generator(noise)

        # Compute generator loss
        self.g_loss = -self.discriminator(fake_data).mean()

        # Backpropagate and optimize
        self.g_loss.backward()
        self.optimizer_G.step()

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-cfg", type=str, default=f"{os.getcwd()}/Code/logs/SAR_WGAN_28/config.yaml") # /Code
    parser.add_argument("--batch_size", "-bs", type=int, default=32)
    parser.add_argument("--epochs", "-e", type=int, default=100)

    args = parser.parse_args()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") # for debugging

    ### Load config
    with open(args.config, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.CFullLoader)

    ### Generator and Discriminator
    cfg_gen = cfg["GENERATOR"]
    cfg_disc = cfg["DISCRIMINATOR"]
    cfg_epoch = args.epochs # cfg["TRAIN"]["EPOCH"]
    generator = get_generator_from_config(cfg_gen)
    discriminator = get_discriminator_from_config(cfg_disc)
    g_loss = 0
    d_loss = 0

    generator.to(device)
    discriminator.to(device)

    wgan_gp = WGAN_GP(generator, discriminator, device, g_loss, d_loss)

    ### Load input data
    working_dir = f'{os.getcwd()}/Code/data' # /Code
    file_name = 's1a-s3-slc-hh-20241108t213605-20241108t213629-056468-06ebd0-001' #'s1a-s6-slc-vh-20241125t214410-20241125t214439-056716-06f5c2-001' # 
    with open(f'{working_dir}/{file_name}.pickle', 'rb') as file:
        dataset = pickle.load(file)
    print(f'Dataset {file_name} loaded successfully...')

    # Analyse input array
    # dataset = np.array(dataset)
    data_HR = np.array(dataset[0])
    data_LR = np.array(dataset[1])
    num_samples = data_HR.shape[0]
    print(f'Dataset {file_name} has {num_samples} patches')
    del dataset

    # to display
    cut_init = (num_samples // 2) - 3
    cut_samples = 3 # to reduce memory usage for debug; still insuffient GPU memory
    # dataset_cut = dataset[:,cut_init:cut_init+cut_samples,:,:]
    data_HR_cut = data_HR[cut_init:cut_init+cut_samples,:,:]
    data_LR_cut = data_LR[cut_init:cut_init+cut_samples,:,:]

    tiny_e = 1
    fig, axes = plt.subplots(2, cut_samples, figsize=(15, 12))
    axes_flat = axes.flatten()
    for i, ax in enumerate(axes_flat):
        if i // cut_samples == 0:
            ax.imshow(10*np.log10(np.abs(data_HR_cut[i%cut_samples, :, :])+1), cmap='gray')
        else:
            ax.imshow(10*np.log10(np.abs(data_LR_cut[i%cut_samples, :, :])+1), cmap='gray')
        ax.axis('off')  # Turn off axis labels
    plt.tight_layout()
    plt.savefig(f'{working_dir}/{file_name}_dataExample.png', dpi=300)

    # Input data into dataloader
    data_HR = data_HR_cut
    data_LR = data_LR_cut
    data_loader = DataLoader([data_HR, data_LR], num_workers=0, batch_size=args.batch_size, shuffle=True)
    del data_HR, data_LR
    
    # Training loop
    for epoch in range(cfg_epoch): # range(args.epochs):
        print(f'Training epoch {epoch}...')
        train_bar = tqdm(data_loader)
        for batch_idx, (data_HR, data_LR) in enumerate(train_bar):
            # data_HR, data_LR = data_HR.unsqueeze(1), data_LR.unsqueeze(1)
            wgan_gp.train(data_HR, data_LR)
            # if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, G Loss: {wgan_gp.g_loss.item()}, D Loss: {wgan_gp.d_loss.item()}")


        # # Sample real and fake data
        # batch_size = data_HR.size(0)
        # real_data = data_HR.to(self.device) # 3x64x64; 1x256x256

        # # Train discriminator
        # self.optimizer_D.zero_grad()

        # # Real data
        # real_labels = torch.ones(batch_size).to(self.device)
        # real_data_4d = torch.stack((torch.real(real_data), torch.imag(real_data)), dim=0) # but this gives real value
        # output = discriminator(real_data_4d) # real_data
        # d_loss_real = self.criterion(output, real_labels)

        # # Fake data
        # # z = data_LR.to(self.device) # 3x64x64; 1x256x256
        # z = torch.randn(batch_size, 1, 2).to(self.device)
        # fake_data = self.generator(z) # output size [1, 8192, 384]; should be 1x64x64 so can upsample x4
        # fake_labels = torch.zeros(batch_size).to(self.device)
        # output = discriminator(fake_data.detach())
        # d_loss_fake = self.criterion(output, fake_labels)

        # self.d_loss = d_loss_real + d_loss_fake
        # self.d_loss.backward()
        # self.optimizer_D.step()

        # # Train Generator
        # self.optimizer_G.zero_grad()
        # output = discriminator(fake_data)
        # self.g_loss = self.criterion(output, real_labels)
        # self.g_loss.backward()
        # self.optimizer_G.step()