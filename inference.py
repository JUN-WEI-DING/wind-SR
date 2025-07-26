import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader

from utils import prepare_data
from model import gan
from gan import evaluate


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Data = prepare_data.Data(icon_path=args.icon_path, era5_path=args.era5_path)
    srimage_normalizer = Data.get_normalizer()
    lrimage = Data.get_testdata()[0]

    gen = gan.Generator(z_dim=args.z_dim, input_channel=args.input_channel,
                        output_channel=args.output_channel, conditional=args.conditional)
    critic = gan.Critic(input_channel=args.input_channel, output_channel=args.output_channel,
                        conditional=args.conditional)

    gen.load_state_dict(torch.load(args.generator_path, weights_only=True))
    critic.load_state_dict(torch.load(args.critic_path, weights_only=True))

    gen.to(device).eval()
    critic.to(device).eval()

    # === 🚀 推論主流程 ===
    batch_size = args.batch_size
    n_sample = args.n_sample
    n_images = len(lrimage)
    result = torch.empty((n_images, args.output_channel, args.height, args.width), dtype=torch.float32, device="cpu")

    for i in range(0, n_images, batch_size):
        print(f"Processing batch {i // batch_size + 1} / {n_images // batch_size + 1}")
        batch = lrimage[i:i + batch_size].to(device)
        srimages, _ = evaluate.inference(
            device=device,
            gen=gen,
            critic=critic,
            z_dim=args.z_dim,
            N=args.N,
            lrimage=batch,
            n_sample=n_sample
        )
        srimages = srimages.view(n_sample, int(srimages.shape[0] / n_sample), args.output_channel, args.height, args.width)
        mean_srimages = srimages.mean(dim=0)
        result[i:i + batch_size] = mean_srimages.cpu()

    np.savez(args.output_path, gen_srimage=result.numpy(), allow_pickle=True)
    print(f"Saved result to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN-based SR inference")

    parser.add_argument("--icon_path", type=str, default="/home/gary/Desktop/wind_sr_github/data/icon.zarr")
    parser.add_argument("--era5_path", type=str, default="/home/gary/Desktop/wind_sr_github/data/era5.zarr")
    parser.add_argument("--generator_path", type=str, required=True, help="Path to generator .pth")
    parser.add_argument("--critic_path", type=str, required=True, help="Path to critic .pth")
    parser.add_argument("--output_path", type=str, default="/home/gary/Desktop/wind_sr_github/data/srimage.npz")

    parser.add_argument("--z_dim", type=int, default=1)
    parser.add_argument("--input_channel", type=int, default=8)
    parser.add_argument("--output_channel", type=int, default=2)
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--N", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--height", type=int, default=41)
    parser.add_argument("--width", type=int, default=41)

    args = parser.parse_args()
    inference(args)
