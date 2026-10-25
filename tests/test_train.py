import json
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from windsr.grid import crop
from windsr.model.inputs import CHANNELS, Fields, Norm
from windsr.model.network import BoxLowPass, Critic, Generator
from windsr.model.predict import load_generator, predict
from windsr.train import TrainConfig, _update_average, content_loss, highpass, train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _toy(n=128):
    r = np.random.default_rng(0)
    x = r.standard_normal((n, len(CHANNELS), 57, 57)).astype(np.float32)
    base = r.standard_normal((n, 2, 41, 41)).astype(np.float32)
    return Fields(np.arange(n), x, base, (base + 0.3 * r.standard_normal((n, 2, 41, 41))).astype(np.float32))


def test_content_loss_is_the_low_pass_mse_over_both_heights():
    lp, base = BoxLowPass(), torch.zeros(1, 2, 41, 41, dtype=torch.float64)
    y = base.clone()
    y[:, 1] += 0.6
    assert torch.isclose(content_loss(lp, y, base), torch.tensor(0.36 / 2, dtype=torch.float64)), "a 100 m offset counts"
    i = np.arange(41)
    wave = np.cos(2 * np.pi * (i[:, None] + i[None]) / 3) + np.cos(2 * np.pi * i[:, None] / 3)
    y = base + torch.tensor(wave)
    assert lp(y - base)[..., 1:-1, 1:-1].abs().max() < 1e-12, "a 3-cell wave is left to the critic away from the edge"
    assert content_loss(lp, y, base) < 0.05 * (y ** 2).mean(), "only the edge cells, through reflect padding, keep some"


def test_the_critic_sees_only_the_high_pass():
    lp = BoxLowPass()
    icon, g = torch.randn(3, 2, 41, 41, dtype=torch.float64), torch.randn(3, 2, 41, 41, dtype=torch.float64)
    shift = torch.tensor([1.5, -0.8], dtype=torch.float64)[:, None, None]
    for field in (icon, g):
        assert torch.allclose(highpass(lp, field + shift), highpass(lp, field)), "a constant per height is invisible"


def test_the_release_trains_50_epochs_and_saves_epoch_50():
    assert asdict(TrainConfig()) == {"w": 1.0, "epochs": 50, "save_at": (50,), "seed": 0, "batch": 64, "lr": 1e-4,
                                     "betas": (0.0, 0.9), "n_critic": 5, "gp_weight": 10.0, "warmup": 2, "box": 3,
                                     "lambda_max": 1e4, "ema_half_life": 10_000}


def test_short_run_logs_every_epoch_saves_and_predicts(tmp_path):
    data, norm = _toy(320), Norm((0.0,) * 6, (1.0,) * 6, (5.0, 7.0), (2.0, 3.0))
    cfg = TrainConfig(epochs=3, save_at=(2, 3), warmup=1, batch=32)
    train(cfg, data, norm, tmp_path, DEVICE)
    assert json.load(open(tmp_path / "config.json"))["w"] == 1
    rows = [json.loads(l) for l in open(tmp_path / "log.jsonl")]
    assert [r["phase"] for r in rows] == ["warmup", "gan", "gan"]
    assert {"wasserstein", "gp", "content", "adv", "grad_ratio", "lambda", "hp_rms_10m", "hp_rms_100m"} <= set(rows[-1])
    assert 0 <= rows[-1]["lambda"] <= cfg.lambda_max
    assert Norm.load(tmp_path / "norm.json") == norm
    g = load_generator(tmp_path / "epoch003.pt", DEVICE)
    pred = predict(g, data, norm, batch=100, device=DEVICE)
    assert pred.shape == (320, 2, 41, 41) and np.isfinite(pred).all()
    assert not np.array_equal(predict(load_generator(tmp_path / "epoch002.pt", DEVICE), data, norm, device=DEVICE), pred)
    ckpt = torch.load(tmp_path / "epoch003.pt", map_location=DEVICE)
    assert set(ckpt) == {"generator", "generator_ema", "critic", "epoch", "config"} and ckpt["epoch"] == 3
    raw = Generator().to(DEVICE)
    raw.load_state_dict(ckpt["generator"])
    assert any(not torch.equal(a, b) for a, b in zip(raw.parameters(), g.parameters())), "the average is its own copy"
    d = Critic().to(DEVICE)
    d.load_state_dict(ckpt["critic"])
    x, t = torch.tensor(data.x[:4], device=DEVICE), torch.tensor(data.target[:4], device=DEVICE)
    with torch.no_grad():
        assert torch.isfinite(d(t, crop(x))).all()


def test_predict_returns_each_height_in_its_own_m_per_s_scale(tmp_path):
    data, norm = _toy(8), Norm((0.0,) * 6, (1.0,) * 6, (5.0, 7.0), (2.0, 3.0))
    train(TrainConfig(epochs=1, save_at=(1,), warmup=1, batch=8), data, norm, tmp_path, DEVICE)
    g = load_generator(tmp_path / "epoch001.pt", DEVICE)
    torch.nn.init.zeros_(g.head.weight)
    torch.nn.init.zeros_(g.head.bias)
    pred = predict(g, data, norm, batch=3, device=DEVICE)
    assert np.allclose(pred[:, 0], data.base[:, 0] * 2 + 5, atol=1e-5)
    assert np.allclose(pred[:, 1], data.base[:, 1] * 3 + 7, atol=1e-5)


def test_warmup_drives_the_large_scale_correction_toward_zero(tmp_path):
    data, norm = _toy(256), Norm((0.0,) * 6, (1.0,) * 6, (0.0, 0.0), (1.0, 1.0))
    train(TrainConfig(epochs=6, save_at=(6,), warmup=6, batch=32), data, norm, tmp_path, DEVICE)
    rows = [json.loads(l) for l in open(tmp_path / "log.jsonl")]
    assert rows[-1]["content"] < 0.5 * rows[0]["content"]


def test_generator_average_follows_the_weights_with_its_decay():
    a, g = Generator(), Generator()
    before = [p.clone() for p in a.parameters()]
    _update_average(a, g, 0.9)
    for p0, pa, pg in zip(before, a.parameters(), g.parameters()):
        assert torch.allclose(pa, 0.9 * p0 + 0.1 * pg)
