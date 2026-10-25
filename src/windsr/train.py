"""WGAN-GP training: content MSE on LP3 against ERA5, conditional critic on the 3x3 high-pass of ICON.

An epoch is one shuffled pass over the fit hours in batches of 64. During warm-up every batch is one generator
step on the content loss alone. After it, every batch is one critic step, and every n_critic-th batch is also one
generator step on its own random batch of hours. Every generator step updates the weight average, which is the
generator that is scored. Its half-life is `ema_half_life` training hours (samples), as in StyleGAN. Training runs
50 epochs and saves epoch 50; nothing in training depends on the total number of epochs.
"""
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .grid import crop
from .model.inputs import HEIGHTS, Fields, Norm, fit_fields
from .model.network import BoxLowPass, Critic, Generator
from .periods import FIT


@dataclass(frozen=True)
class TrainConfig:
    w: float = 1.0
    epochs: int = 50
    save_at: tuple[int, ...] = (50,)
    seed: int = 0
    batch: int = 64
    lr: float = 1e-4
    betas: tuple[float, float] = (0.0, 0.9)
    n_critic: int = 5
    gp_weight: float = 10.0
    warmup: int = 2
    box: int = 3
    lambda_max: float = 1e4
    ema_half_life: float = 10_000


def content_loss(lp: BoxLowPass, y: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """MSE(LP3(G), LP3(ERA5)) over the 41x41 window and both heights."""
    return (lp(y - base) ** 2).mean()


def highpass(lp: BoxLowPass, x: torch.Tensor) -> torch.Tensor:
    """HP3(x) = x - LP3(x), the field the critic judges."""
    return x - lp(x)


def gradient_penalty(critic: Critic, real: torch.Tensor, fake: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    eps = torch.rand(real.shape[0], 1, 1, 1, device=real.device)
    mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
    (g,) = torch.autograd.grad(critic(mix, cond).sum(), mix, create_graph=True)
    return ((g.flatten(1).norm(dim=1) - 1) ** 2).mean()


def train(cfg: TrainConfig, data: Fields, norm: Norm, out: Path, device: str = "cuda") -> None:
    out.mkdir(parents=True, exist_ok=True)
    norm.save(out / "norm.json")
    json.dump(asdict(cfg), open(out / "config.json", "w"), indent=1)
    torch.manual_seed(cfg.seed)
    order = torch.Generator().manual_seed(cfg.seed)
    x, base, target = (torch.tensor(a, device=device) for a in (data.x, data.base, data.target))
    g, d, lp = Generator().to(device), Critic().to(device), BoxLowPass(cfg.box)
    g_ema = copy.deepcopy(g).requires_grad_(False)
    decay = 0.5 ** (cfg.batch / cfg.ema_half_life)
    opt_g = torch.optim.Adam(g.parameters(), cfg.lr, betas=cfg.betas)
    opt_d = torch.optim.Adam(d.parameters(), cfg.lr, betas=cfg.betas)
    n_batches = len(x) // cfg.batch
    log = open(out / "log.jsonl", "a")

    for epoch in range(1, cfg.epochs + 1):
        t0, stats = time.time(), {}
        add = lambda k, v: stats.setdefault(k, []).append(v.detach())
        perm = torch.randperm(len(x), generator=order)[: n_batches * cfg.batch].view(n_batches, cfg.batch).to(device)
        for i, idx in enumerate(perm):
            xb, bb, tb = x[idx], base[idx], target[idx]
            cond = crop(xb)
            if epoch <= cfg.warmup:
                loss = content_loss(lp, g(xb, bb), bb)
                opt_g.zero_grad(set_to_none=True)
                loss.backward()
                opt_g.step()
                _update_average(g_ema, g, decay)
                add("content", loss)
                continue

            with torch.no_grad():
                fake = g(xb, bb)
            real, fake = highpass(lp, tb), highpass(lp, fake)
            d_real, d_fake = d(real, cond).mean(), d(fake, cond).mean()
            gp = gradient_penalty(d, real, fake, cond)
            loss_d = d_fake - d_real + cfg.gp_weight * gp
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
            add("wasserstein", d_real - d_fake)
            add("gp", gp)

            if i % cfg.n_critic == cfg.n_critic - 1:
                d.requires_grad_(False)
                ig = torch.randint(len(x), (cfg.batch,), generator=order).to(device)
                xg, bg = x[ig], base[ig]
                y = g(xg, bg)
                content, adv = content_loss(lp, y, bg), -d(highpass(lp, y), crop(xg)).mean()
                (gc,) = torch.autograd.grad(content, g.head.weight, retain_graph=True)
                (ga,) = torch.autograd.grad(adv, g.head.weight, retain_graph=True)
                ratio = gc.norm() / (ga.norm() + 1e-12)
                lam = (cfg.w * ratio).clamp(0, cfg.lambda_max).detach()
                opt_g.zero_grad(set_to_none=True)
                (content + lam * adv).backward()
                opt_g.step()
                _update_average(g_ema, g, decay)
                d.requires_grad_(True)
                add("content", content)
                add("adv", adv)
                add("grad_ratio", ratio)
                add("lambda", lam)
                hp = highpass(lp, y - bg)
                for j, h in enumerate(HEIGHTS):
                    add(f"hp_rms_{h}", hp[:, j].pow(2).mean().sqrt())

        row = {"epoch": epoch, "phase": "warmup" if epoch <= cfg.warmup else "gan", "seconds": round(time.time() - t0, 2),
               **{k: torch.stack(v).float().mean().item() for k, v in stats.items()}}
        if not all(np.isfinite(v) for v in row.values() if isinstance(v, float)):
            raise FloatingPointError(f"non-finite training statistics at epoch {epoch}: {row}")
        log.write(json.dumps(row) + "\n")
        log.flush()
        if epoch in cfg.save_at:
            torch.save({"generator": g.state_dict(), "generator_ema": g_ema.state_dict(), "critic": d.state_dict(),
                        "epoch": epoch, "config": asdict(cfg)}, out / f"epoch{epoch:03d}.pt")


@torch.no_grad()
def _update_average(avg: Generator, g: Generator, decay: float) -> None:
    for a, p in zip(avg.parameters(), g.parameters()):
        a.lerp_(p, 1 - decay)


def train_run(data: Path, run: Path) -> None:
    if (run / "log.jsonl").exists():
        raise SystemExit(f"{run} already has a training log; use a new directory")
    norm, fields = fit_fields(data, FIT)
    train(TrainConfig(), fields, norm, run)
