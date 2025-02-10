import os
import argparse
import yaml
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.generator import get_generator_from_config
from utils.discriminator import get_discriminator_from_config

import torch
from torch.utils.data import DataLoader

# Assuming you have the generator and discriminator models defined
class WGAN_GP:
    def __init__(self, generator, discriminator, device, latent_dim, img_size, g_loss=0, d_loss=0, lambda_gp=10, critic_iterations=5):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.lambda_gp = lambda_gp
        self.n_critic = critic_iterations
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.g_loss = g_loss
        self.d_loss = d_loss

        # Optimizers
        self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-5, betas=(0.9, 0.999)) # lr=3e-5, betas=(0.5, 0.9)
        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.9, 0.999)) # lr=3e-5, betas=(0.5, 0.9)

    def train(self, data_HR):
        real_data = data_HR.to(device)
        real_part = real_data.real.unsqueeze(1)
        imag_part = real_data.imag.unsqueeze(1)

        # Concatenate along the first dimension
        real_imgs = torch.cat([real_part, imag_part], dim=0).to(torch.complex64)
        # real_imgs = torch.rand(2, 1, self.img_size, self.img_size, dtype=torch.complex64)
        # real_imgs = torch.stack((torch.real(real_data), torch.imag(real_data)), dim=0).to(torch.complex64) # -1
        
        # Train Discriminator
        for _ in range(self.n_critic):
            self.optimizer_D.zero_grad()
            
            # Generate fake images
            z = torch.randn(real_imgs.size(0), self.latent_dim, 2, device=device)
            fake_imgs = generator(z)
            
            # Discriminator outputs
            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs)
            
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
            
            # WGAN-GP loss
            self.d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
            self.d_loss.backward()
            self.optimizer_D.step()
        
        # Train Generator
        self.optimizer_G.zero_grad()
        
        # Generate new fake images
        fake_imgs = generator(z)
        self.g_loss = -torch.mean(discriminator(fake_imgs))
        self.g_loss.backward()
        self.optimizer_G.step()


# Gradient Penalty
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-cfg", type=str, default=f"{os.getcwd()}/Code/logs/Sentinel-1/config.yaml") # /Code
    parser.add_argument("--batch_size", "-bs", type=int, default=32)
    parser.add_argument("--epochs", "-e", type=int, default=200) # 100
    parser.add_argument("--epoch_start", "-es", type=int, default=100) # 0
    parser.add_argument("--is_display", "-d", type=int, default=0)

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
    epoch_start = args.epoch_start
    generator = get_generator_from_config(cfg_gen)
    discriminator = get_discriminator_from_config(cfg_disc)
    latent_dim = cfg_gen["PARAMETERS"]["latent_dim"]
    image_size = cfg_disc["PARAMETERS"]["image_size"]

    generator.to(device)
    discriminator.to(device)

    if epoch_start != 0:
        print('Starting from epoch %d...' % (epoch_start + 1))
        generator.load_state_dict(torch.load('Code/data/epochs/generator_epoch%d.pth' % epoch_start))
        discriminator.load_state_dict(torch.load('Code/data/epochs/discriminator_epoch%d.pth' % epoch_start))

    wgan_gp = WGAN_GP(generator, discriminator, device, latent_dim, image_size)

    ### Load input data
    working_dir = f'{os.getcwd()}/Code/data' # /Code
    file_name = 's1a-s3-slc-hh-20241108t213605-20241108t213629-056468-06ebd0-001' #'s1a-s6-slc-vh-20241125t214410-20241125t214439-056716-06f5c2-001' # 
    with open(f'{working_dir}/{file_name}_{image_size}.pickle', 'rb') as file:
        dataset = pickle.load(file)

    # Analyse input array
    # dataset = np.array(dataset)
    data_HR = np.array(dataset[0])
    data_LR = np.array(dataset[1])
    num_samples = data_HR.shape[0]
    print(f'Dataset {file_name} loaded successfully with {num_samples} patches...')
    del dataset

    # to display
    if args.is_display:
        cut_init = (num_samples // 2) - 3
        cut_samples = 3 # 110 # 100 # to reduce memory usage for debug; still insuffient GPU memory
        # dataset_cut = dataset[:,cut_init:cut_init+cut_samples,:,:]
        data_HR_cut = data_HR[cut_init-(cut_samples//2):cut_init+(cut_samples//2),:,:]
        data_LR_cut = data_LR[cut_init-(cut_samples//2):cut_init+(cut_samples//2),:,:]
    
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
    # data_LR = data_LR_cut
    # data_HR = data_HR_cut # to reduce # of input data
    data_loader = DataLoader(data_HR, num_workers=8, batch_size=args.batch_size, shuffle=True) # 4
    results = {'d_loss': [], 'g_loss': []}
    del data_LR, data_HR

    # Training loop
    for epoch in range(epoch_start + 1, cfg_epoch + 1):      
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0}
        train_bar = tqdm(data_loader)

        for data_HR in train_bar:
            batch_size = data_HR.size(0)
            running_results['batch_sizes'] += batch_size

            wgan_gp.train(data_HR)

            running_results['g_loss'] += wgan_gp.g_loss.item() * batch_size
            running_results['d_loss'] += wgan_gp.d_loss.item() * batch_size
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f' % (
                epoch, cfg_epoch, running_results['d_loss'].real / running_results['batch_sizes'],
                running_results['g_loss'].real / running_results['batch_sizes']))
            
        print(f"Epoch {epoch+1}, G Loss: {wgan_gp.g_loss.item():.4f}, D Loss: {wgan_gp.d_loss.item():.4f}")
        
        # Save models and results
        torch.save(generator.state_dict(), f'{working_dir}/epochs/generator_epoch{epoch}.pth') # epoch+1
        torch.save(discriminator.state_dict(), f'{working_dir}/epochs/discriminator_epoch{epoch}.pth') # epoch+1
        results['d_loss'].append(running_results['d_loss'].real / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'].real / running_results['batch_sizes'])
        
        if epoch != 0:
            out_statistics_path = f'{working_dir}/statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss']},
                index= range(epoch_start + 1, epoch + 1))
            data_frame.to_csv(out_statistics_path + 'train_results.csv', index_label='Epoch')
