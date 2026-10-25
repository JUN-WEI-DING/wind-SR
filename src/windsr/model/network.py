"""The generator, the critic and the 3x3 box scale split.

Data shapes: the generator maps (batch, 16, 57, 57) inputs and (batch, 2, 41, 41) ERA5 speed to (batch, 2, 41, 41)
speed at the HEIGHTS; the critic scores a (batch, 2, 41, 41) field given the inputs on the output window,
(batch, 16, 41, 41), with one value per sample.
"""
import torch
import torch.nn.functional as F
from torch import nn

from ..grid import crop
from .inputs import CHANNELS, HEIGHTS


class BoxLowPass(nn.Module):
    """The size x size mean of each field on the last two axes after whole-sample reflect padding (LP3 at size 3).
    Not a projection."""

    def __init__(self, size: int = 3):
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.size // 2
        flat = F.pad(x.reshape(-1, 1, *x.shape[-2:]), (p, p, p, p), mode="reflect")
        return F.avg_pool2d(flat, self.size, stride=1).reshape(x.shape)


class Block(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.LeakyReLU(0.2),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch))

    def forward(self, x):
        return x + self.body(x)


class Generator(nn.Module):
    """57x57 input CHANNELS -> 41x41 speed at the HEIGHTS, each ERA5 speed at that height plus a correction
    (global residual per channel). Deterministic."""

    def __init__(self, in_ch: int = len(CHANNELS), ch: int = 64, blocks: int = 8):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(ch) for _ in range(blocks)])
        self.head = nn.Conv2d(ch, len(HEIGHTS), 3, padding=1)

    def correction(self, x: torch.Tensor) -> torch.Tensor:
        return crop(self.head(self.blocks(self.stem(x))))

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        return base + self.correction(x)


class Critic(nn.Module):
    """Scores a (2, 41, 41) speed field given the input CHANNELS on the same window; receptive field 40 cells."""

    def __init__(self, cond_ch: int = len(CHANNELS)):
        super().__init__()
        act = lambda: nn.LeakyReLU(0.2)
        self.net = nn.Sequential(
            nn.Conv2d(len(HEIGHTS) + cond_ch, 64, 3, padding=1), act(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), act(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), act(),
            nn.Conv2d(256, 256, 4, stride=2, padding=1), act(),
            nn.Conv2d(256, 1, 3, padding=1))

    def forward(self, field: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([field, cond], 1)).mean((1, 2, 3))
