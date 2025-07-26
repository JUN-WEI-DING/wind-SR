import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from utils import prepare_data
from model import gan
from gan import train


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Data = prepare_data.Data(
        icon_path=args.icon_path,
        era5_path=args.era5_path
    )
    dataset = prepare_data.GAN_Dataset(device, *Data.get_traindata())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    gan_gen = gan.Generator(z_dim=args.z_dim, input_channel=args.input_channel,
                            output_channel=args.output_channel, conditional=args.conditional)
    gan_critic = gan.Critic(input_channel=args.input_channel,
                            output_channel=args.output_channel, conditional=args.conditional)

    optimizer_G = optim.RMSprop(gan_gen.parameters(), lr=args.lr)
    optimizer_C = optim.RMSprop(gan_critic.parameters(), lr=args.lr)

    gan_gen, gan_critic, optimizer_G, optimizer_C = train.resume_training(
        args.checkpoint_dir, args.start_epoch, gan_gen, gan_critic, optimizer_G, optimizer_C, device
    )

    loss_mses, loss_gs, loss_cs = train.train_gan(
        device=device,
        gen=gan_gen,
        critic=gan_critic,
        num_epochs=args.epochs,
        train_dataloader=dataloader,
        optimizer_G=optimizer_G,
        optimizer_C=optimizer_C,
        clip_value=args.clip_value,
        N=args.N,
        z_dim=args.z_dim,
        conditional=args.conditional,
        model_save=args.model_save,
        model_save_dir=args.model_save_dir,
        start_epoch=args.start_epoch,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir
    )

    return loss_mses, loss_gs, loss_cs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAN model for wind super-resolution")

    parser.add_argument("--z_dim", type=int, default=1, help="Latent dimension")
    parser.add_argument("--N", type=int, default=3, help="Number of critic iterations per generator iteration")
    parser.add_argument("--input_channel", type=int, default=8, help="Input channel dimension")
    parser.add_argument("--output_channel", type=int, default=2, help="Output channel dimension")
    parser.add_argument("--conditional", action="store_true", help="Use conditional GAN")

    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--clip_value", type=float, default=0.01, help="Clipping value for WGAN")
    parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch for resume training")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Epoch interval for saving checkpoints")

    parser.add_argument("--icon_path", type=str, default="/home/gary/Desktop/wind_sr_github/data/icon.zarr")
    parser.add_argument("--era5_path", type=str, default="/home/gary/Desktop/wind_sr_github/data/era5.zarr")
    parser.add_argument("--checkpoint_dir", type=str, default="/home/gary/Desktop/wind_sr_github/model_checkpoiints")
    parser.add_argument("--model_save_dir", type=str, default="/home/gary/Desktop/wind_sr_github/model_parameter")
    parser.add_argument("--model_save", action="store_true", help="Whether to save model parameters")

    args = parser.parse_args()
    train_model(args)
