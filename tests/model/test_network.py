import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import uniform_filter

torch = pytest.importorskip("torch")

from windsr.grid import FINE_LAT, FINE_LON, NATIVE_LAT, NATIVE_LON
from windsr.model.inputs import upsample_matrix
from windsr.model.network import BoxLowPass, Critic, Generator

rng = np.random.default_rng(0)


def test_box_lowpass_is_the_3x3_mean_with_whole_sample_reflect_padding_per_field():
    x = rng.standard_normal((4, 2, 41, 41))
    want = uniform_filter(x, size=(1, 1, 3, 3), mode="mirror")
    assert np.abs(BoxLowPass()(torch.tensor(x)).numpy() - want).max() < 1e-12, "scipy mirror is torch reflect"
    assert np.abs(BoxLowPass()(torch.tensor(x[:, 1])).numpy() - want[:, 1]).max() < 1e-12, "any leading axes"
    c = torch.full((2, 2, 41, 41), 0.7, dtype=torch.float64)
    assert torch.allclose(BoxLowPass()(c), c), "a constant passes unchanged, edges included"


@pytest.mark.parametrize("wavelength", [8, 4, 3])
def test_box_lowpass_response_is_the_3_tap_mean_in_the_interior(wavelength):
    i = np.arange(41)
    wave = np.cos(2 * np.pi * i / wavelength + 0.3)
    x = torch.tensor(np.broadcast_to(wave[:, None], (41, 41)).copy())
    gain = (1 + 2 * np.cos(2 * np.pi / wavelength)) / 3
    got = BoxLowPass()(x).numpy()[1:-1]
    assert np.abs(got - gain * wave[1:-1, None]).max() < 1e-12, f"gain {gain:.4f} away from the edge rows"


def test_upsample_matrix_is_bilinear_interpolation():
    f = rng.standard_normal((2, 29, 29))
    da = xr.DataArray(f, dims=("t", "latitude", "longitude"), coords={"latitude": NATIVE_LAT, "longitude": NATIVE_LON})
    want = da.interp(latitude=FINE_LAT, longitude=FINE_LON, method="linear").values
    a = upsample_matrix()
    assert np.abs(np.einsum("ij,tjk,lk->til", a, f, a) - want).max() < 1e-12


def _receptive_field(f, shape, out_index):
    x = torch.zeros(shape, requires_grad=True)
    f(x)[out_index].sum().backward()
    rows = torch.nonzero(x.grad.abs().sum((0, 1, 3)))
    return int(rows.max() - rows.min() + 1)


def test_generator_outputs_both_heights_with_a_residual_per_height():
    torch.manual_seed(0)
    g = Generator()
    with torch.no_grad():
        for p in g.parameters():
            p.add_(0.05)
    x, base = torch.randn(2, 16, 57, 57), torch.randn(2, 2, 41, 41)
    assert g(x, base).shape == (2, 2, 41, 41)
    torch.nn.init.zeros_(g.head.weight)
    torch.nn.init.zeros_(g.head.bias)
    assert torch.equal(g(x, base), base), "a zero correction returns ERA5 itself at both heights"
    with torch.no_grad():
        g.head.bias[1] = 0.5
    y = g(x, base)
    assert torch.equal(y[:, 0], base[:, 0]) and torch.allclose(y[:, 1], base[:, 1] + 0.5), "channel j corrects height j"


def test_generator_receptive_field_is_37_cells():
    conv_only = Generator()
    for m in conv_only.modules():
        for name, child in m.named_children():
            if isinstance(child, torch.nn.GroupNorm):
                setattr(m, name, torch.nn.Identity())
    for height in (0, 1):
        rf = _receptive_field(conv_only.correction, (1, 16, 57, 57), (0, height, 20, 20))
        assert rf == 37, "convolutional receptive field"


def test_critic_sees_the_whole_window():
    torch.manual_seed(0)
    d = Critic()
    assert d(torch.randn(3, 2, 41, 41), torch.randn(3, 16, 41, 41)).shape == (3,)
    assert d.net(torch.zeros(1, 18, 41, 41)).shape[-2:] == (5, 5)
    rf = _receptive_field(lambda x: d.net(x), (1, 18, 41, 41), (0, 0, 2, 2))
    assert rf >= 37


def test_generator_is_deterministic():
    torch.manual_seed(0)
    g = Generator()
    x, base = torch.randn(2, 16, 57, 57), torch.randn(2, 2, 41, 41)
    with torch.no_grad():
        assert torch.equal(g(x, base), g(x, base)), "two calls on the same input give the same field"
