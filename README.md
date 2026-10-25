# WindSR

Code for Ding, J.-W. & Hsieh, I-Y. L., *Super-Resolution Wind Mapping with Deep Learning for Scalable Renewable
Energy Planning*, Communications Earth & Environment 7, 51 (2026),
[doi:10.1038/s43247-025-03072-9](https://doi.org/10.1038/s43247-025-03072-9).

WindSR turns hourly ERA5 wind (0.25 deg) into a 0.125 deg wind map of Taiwan at 10 m and 100 m. ERA5 is its only
input. In training, a content loss holds the map's large scales to ERA5 and a critic judges its small scales
against the ICON model.

## Run

You need Linux, [uv](https://docs.astral.sh/uv/), and the dataset of Zenodo record
[doi:10.5281/zenodo.17647354](https://doi.org/10.5281/zenodo.17647354) unpacked into a directory `DATA`
([DATA.md](DATA.md)). Training and prediction also need an NVIDIA GPU.

```sh
git clone https://github.com/JUN-WEI-DING/wind-SR.git && cd wind-SR
uv sync --frozen --extra train
uv run --frozen windsr run DATA OUT
```

`windsr run` checks DATA against its SHA256SUMS, trains the network and predicts into the new directory
OUT/model, and evaluates into OUT/evaluation. The same steps one at a time:

```sh
uv run --frozen windsr verify-data DATA
uv run --frozen windsr train DATA RUN
uv run --frozen windsr predict DATA RUN
uv run --frozen windsr evaluate DATA OUT --model windsr=RUN/test.zarr,RUN/train.zarr
```

`windsr evaluate` scores any prediction store with `wind_speed_10m` on the 0.125 deg grid. `uv sync --frozen` alone,
without torch, is enough to verify DATA and evaluate prediction stores. Tests run with `uv run --frozen pytest`; set
`WINDSR_DATA=DATA` to include those that read the data. To build DATA from the raw sources, see [DATA.md](DATA.md).

## Citation and license

Cite the paper and the dataset. The code is under the MIT License ([LICENSE](LICENSE)). The dataset is under
CC BY 4.0; [DATA.md](DATA.md) lists its sources and the credit each requires.
