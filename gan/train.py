import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import random
from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
 
from utils import normalization, common

def resume_training(checkpoint_dir="../../model_checkpoints", epoch=0, gen=None, critic=None, optimizer_G=None, optimizer_C=None, device='cpu'):
    # Check if the latest checkpoint exists
    gen_checkpoint_path = os.path.join(checkpoint_dir, f"gen_epoch_{epoch}.pth")
    critic_checkpoint_path = os.path.join(checkpoint_dir, f"critic_epoch_{epoch}.pth")

    if os.path.exists(gen_checkpoint_path) and os.path.exists(critic_checkpoint_path):
        # Load model states
        gen.load_state_dict(torch.load(gen_checkpoint_path, map_location=device, weights_only=True))
        critic.load_state_dict(torch.load(critic_checkpoint_path, map_location=device, weights_only=True))
        
        # Optionally, load optimizer states if saved
        optimizer_G_path = os.path.join(checkpoint_dir, f"optimizer_G_epoch_{epoch}.pth")
        optimizer_C_path = os.path.join(checkpoint_dir, f"optimizer_C_epoch_{epoch}.pth")
        
        if os.path.exists(optimizer_G_path) and os.path.exists(optimizer_C_path):
            optimizer_G.load_state_dict(torch.load(optimizer_G_path, map_location=device, weights_only=True))
            optimizer_C.load_state_dict(torch.load(optimizer_C_path, map_location=device, weights_only=True))

        print("Checkpoint loaded. Resuming training...")
    else:
        print("No checkpoint found. Starting training from scratch.")
    
    return gen, critic, optimizer_G, optimizer_C

def train_gan(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
              gen=None, critic=None,
              num_epochs: int = 500, train_dataloader=None,
              optimizer_G=None, optimizer_C=None,
              clip_value: float = 0.01, N: int = 11,
              z_dim: int = 100,
              conditional: bool = True,
              model_save: bool = False,
              model_save_dir:str = "../model_parameter",
              start_epoch:int=0,
              checkpoint_interval: int = 20,  # 每隔多少个epoch保存检查点
              checkpoint_dir: str = "../model_checkpoints"):  # 检查点保存路径
    loss_mses = []
    loss_gs = []
    loss_cs = []

    gen=gen.train().to(device)
    critic=critic.train().to(device)
 
    c_loss = torch.tensor(0, dtype=torch.float32).to(device)
    mse_epochs = 100
    buffer_epochs = 10
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in range(num_epochs):
        for i, (refsrimage, lrimage, generated_lrimage) in enumerate(train_dataloader):
            batch_size = refsrimage.size(0)
 
            # LR Image
            ## Fake
            low_freq_generated_lrimage = common.frequency_filter(generated_lrimage,N)
            low_freq_generated_lrimage = low_freq_generated_lrimage[:,[2,5],:,:]
            c_fake = generated_lrimage
            z = torch.randn(batch_size, z_dim, refsrimage.size(-2), refsrimage.size(-1),device=device)
            z[:,:,0::2,1::2]=0
            z[:,:,1::2,0::2]=0
            fake_image = gen(z,c_fake)
            low_freq_fake_image = common.frequency_filter(fake_image,N)
            high_freq_fake_image = fake_image - low_freq_fake_image

            ## Real
            c_real = lrimage
            low_freq_refsrimage = common.frequency_filter(refsrimage,N)
            high_freq_refsrimage = refsrimage - low_freq_refsrimage

            # critic
            if start_epoch+epoch>mse_epochs:
                optimizer_C.zero_grad()
                real_validity = critic(high_freq_refsrimage,c_real)
                fake_validity = critic(high_freq_fake_image,c_fake)
                c_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                # critic training
                c_loss.backward(retain_graph=True)
                optimizer_C.step()
                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)
    
            # generator
            optimizer_G.zero_grad()
            
            if start_epoch+epoch <= mse_epochs:
                alpha = 0  
            elif start_epoch+epoch <= mse_epochs + buffer_epochs:
                alpha = (start_epoch+epoch - mse_epochs) / buffer_epochs  
            else:
                alpha = 1

            if start_epoch+epoch <= mse_epochs:
                mse_loss = nn.MSELoss()(low_freq_fake_image,low_freq_generated_lrimage)
                g_loss = mse_loss
            else:
                fake_validity = critic(high_freq_fake_image,c_fake)
                mse_loss = nn.MSELoss()(low_freq_fake_image,low_freq_generated_lrimage)
                g_loss = mse_loss + alpha * (-torch.mean(fake_validity))

            # generator training
            g_loss.backward()
            optimizer_G.step() 

            # loss record
            loss_mses.append(mse_loss.cpu().detach().numpy())
            loss_gs.append(g_loss.cpu().detach().numpy())
            loss_cs.append(c_loss.cpu().detach().numpy())

        

        if (start_epoch+epoch + 1) % 1 == 0:
            print(f"Epoch [{start_epoch+epoch+1}/{start_epoch+num_epochs}], D Loss: {c_loss.item():.4f}, G Loss: {g_loss.item():.4f}, rmse: {np.sqrt(mse_loss.item()):.4f}")

        # 保存检查点
        if (start_epoch+epoch + 1) % checkpoint_interval == 0:
            checkpoint_path_gen = os.path.join(checkpoint_dir, f"gen_epoch_{start_epoch+epoch+1}.pth")
            checkpoint_path_critic = os.path.join(checkpoint_dir, f"critic_epoch_{start_epoch+epoch+1}.pth")
            torch.save(gen.state_dict(), checkpoint_path_gen)
            torch.save(critic.state_dict(), checkpoint_path_critic)
            torch.save(optimizer_G.state_dict(), os.path.join(checkpoint_dir, f"optimizer_G_epoch_{start_epoch+epoch+1}.pth"))
            torch.save(optimizer_C.state_dict(), os.path.join(checkpoint_dir, f"optimizer_C_epoch_{start_epoch+epoch+1}.pth"))
            print(f"Checkpoint saved at epoch {start_epoch+epoch+1}")

    if model_save:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if conditional:
            torch.save(critic.state_dict(), f"{model_save_dir}/{current_time}_{N}_{z_dim}_c_critic.pth")
            torch.save(gen.state_dict(), f"{model_save_dir}/{current_time}_{N}_{z_dim}_c_generator.pth")
        else:
            torch.save(critic.state_dict(), f"{model_save_dir}/{current_time}_{N}_{z_dim}_critic.pth")
            torch.save(gen.state_dict(), f"{model_save_dir}/{current_time}_{N}_{z_dim}_generator.pth")
        print(current_time)

    loss_curve_plot(loss_gs,loss_cs)
    
    return loss_mses, loss_gs, loss_cs

def loss_curve_plot(loss_gs,loss_cs):
    steps = np.arange(0, len(loss_gs))

    def moving_average(data, window_size=10):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    smoothed_gen_loss = moving_average(loss_gs)
    smoothed_disc_loss = moving_average(loss_cs)

    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    ax=axes
    ax.plot(steps[:len(smoothed_gen_loss)], smoothed_gen_loss, label='Generator Loss', color='blue', linewidth=2)
    ax.plot(steps[:len(smoothed_disc_loss)], smoothed_disc_loss, label='Discriminator Loss', color='red', linewidth=2)

    ax.axhline(y=0,xmin=ax.get_xlim()[0],xmax=ax.get_xlim()[-1],color='black',alpha=0.4,lw=1)

    ax.set_title('Loss Curve', fontsize=16, pad=10)
    ax.set_xlabel('Steps (counts)', fontsize=12, labelpad=10)
    ax.set_ylabel('Loss', fontsize=12, labelpad=10)
    ax.legend(fontsize=10,ncol=2,loc='center', bbox_to_anchor=(0.5,0.1))

    plt.tight_layout()
    plt.show()