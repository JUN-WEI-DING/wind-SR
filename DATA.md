# WindSR data

The data directory that WindSR trains on and evaluates against (Ding & Hsieh, Commun. Earth Environ. 7, 51, 2026,
doi:10.1038/s43247-025-03072-9). `windsr build-data` writes it from the raw sources below.

License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Every file is modified from its sources as
described below. Credit the sources as listed under Credits.

## Download

The Zenodo record holds one zip per store, one for `taipower/`, and `SHA256SUMS`. Download them into one directory,
then unzip them there; that directory is DATA:

```sh
for f in *.zip; do unzip -q "$f"; done
sha256sum -c SHA256SUMS
```

## Files

| Path | Contents | Grid | Hours (UTC) |
|---|---|---|---|
| `era5_native.zarr` | ERA5 u and v at 10 m and 100 m, the network input | 29x29, 0.25 deg, 20.5-27.5 N, 117.5-124.5 E | 2022-12-01 to 2025-08-29 |
| `era5.zarr` | ERA5 on the 0.125 deg grid, with speed and direction | 41x41, 0.125 deg, 21.375-26.375 N, 118.125-123.125 E | same |
| `icon.zarr` | ICON 10 m and 100 m wind speed, the training target | same 41x41 | same |
| `era5_land.zarr` | ERA5-Land 10 m wind speed, NaN over the sea | 52x52, 0.1 deg | 2023-01-01 to 2025-07-01 |
| `stations.zarr` | CWA hourly 10 m wind speed at 345 stations | | 2022-12-01 to 2025-08-29 |
| `taipower/` | Taipower 10-minute net generation of six wind farms, for Fig. 4 | | 2022-11 to 2025-08, local months |
| `SHA256SUMS` | SHA-256 of every file | | |

The stores are Zarr format 2 with consolidated metadata, chunks of 720 hours, latitude ascending, wind speed in m/s.

## Sources and processing

- **`era5_native.zarr`**: `u10`, `v10`, `u100`, `v100`. ERA5 hourly single levels (Hersbach et al. 2020,
  doi:10.24381/cds.adbb2d47), cut to the 29x29 box. Values unchanged.
- **`era5.zarr`**: the same four components interpolated bilinearly to the 0.125 deg grid,
  `wind_speed_10m` and `wind_speed_100m` from hypot(u, v) interpolated the same way, and `wind_dir_10m` and
  `wind_dir_100m` (degrees, from north, the direction the wind blows from) from the interpolated u and v.
- **`icon.zarr`**: `wind_speed_10m`, `wind_speed_100m`. ICON global forecasts of the Deutscher Wetterdienst from the
  Open-Meteo historical forecast API, model `icon_global`, as in the first published dataset (Zenodo
  doi:10.5281/zenodo.17647355, `hr`). `wind_speed_100m` is the log-height interpolation of 80 m and 120 m. From
  2025-06-02T18 the published field is one column east; the build moves it back.
- **`era5_land.zarr`**: `wind_speed`. ERA5-Land hourly (doi:10.24381/cds.e2161bac), as in the first published
  dataset (`bm`), with its sea fill of 0 set back to NaN using the land mask of NSF NCAR GDEX d633008. The
  evaluation counts a sea cell as 0 m/s.
- **`stations.zarr`**: CWA CODIS (https://codis.cwa.gov.tw/StationData), column `WindSpeed.Mean`, at the 345
  stations of the first published dataset (`gs`). Moved from local time (UTC+8) to UTC.
  - `wind_speed_raw`: negative missing codes are NaN; calm hours are kept.
  - `stuck_mask`: hours in a run of 12 or more hours at one value above 0, or 48 or more hours at 0.
  - `wind_speed`: `wind_speed_raw` with the stuck hours set to NaN.
  - Coordinates: `latitude`, `longitude`, `elevation` and `group` from `gs`, and the official CODIS coordinates.
- **`taipower/`**: Taiwan Power Company unit generation, cut to the six farms of `src/windsr/data/taipower.py`.
  - `partitions/<YYYY-MM>/data.parquet` by local month: `timestamp` (Asia/Taipei, every 10 minutes), `unit_name`,
    `net_generation_mw`, `capacity_mw`, and the quality flags `qa_exclude` and `eligible_reported_item_analysis`.
  - `metadata/observation_quality_flags.csv`: runs of frozen or zero output that the evaluation excludes.
  - `release.json`: the identifier of the release it was cut from.

## Limitations

- ERA5-Land has no value over the sea and covers a shorter period.
- ICON's easternmost column (123.125 E) is NaN from 2025-06-02T18, and its northernmost row on 2025-08-29. No
  station uses either.
- Neighbouring ICON cells often carry identical series: 1,681 cells hold 1,480 distinct series.
- The stuck mask does not catch slow drift or level shifts.
- A fresh CODIS download may differ from the station files used here.
- The Taipower quality flags come from the processed collection the extract was cut from, not from Taipower.

## Rebuild

`windsr build-data` takes each raw source as an argument:

```sh
uv run --frozen windsr build-data DATA --era5 ERA5_LIST --zenodo ZENODO --era5-land ERA5_LAND --codis CODIS \
  --taipower TAIPOWER
cd DATA && sha256sum -c SHA256SUMS
```

- `ERA5_LIST`: a text file listing, one path per line, a Zarr store per UTC day from 2022-12-01 to 2025-08-29
  (1,003 days) in the variable layout of ARCO-ERA5 (`gs://gcp-public-data-arco-era5`):
  `10m_u_component_of_wind`, `10m_v_component_of_wind`, `100m_u_component_of_wind`,
  `100m_v_component_of_wind`, covering at least 20.5-27.5 N and 117.5-124.5 E.
- `ZENODO`: the unpacked first published dataset, doi:10.5281/zenodo.17647355 (`lr.zarr`, `hr.zarr`, `bm.zarr`,
  `gs.zarr`).
- `ERA5_LAND`: the 182 GDEX d633008 files `e5land.oper.fc.sfc.instan.u10_165.*.nc` from 2022-12-26 to 2025-07-06,
  each downloaded as
  `https://tds.gdex.ucar.edu/thredds/ncss/grid/files/d633008/e5land.oper.fc.sfc.instan/<YYYYMM>/<file>?north=26.55&south=21.15&west=117.95&east=123.35&horizStride=1&temporal=all&accept=netcdf&var=u10`.
- `CODIS`: a directory with `data/stations/<ID>.parquet` (hourly CODIS reports indexed by local `DataTime`, with
  `WindSpeed.Mean`), `metadata/official/station_list.json` (the response of
  `https://codis.cwa.gov.tw/api/station_list`) and `metadata/station_profiles.csv` (`station_id`,
  `latitude_wgs84`, `longitude_wgs84`).
- `TAIPOWER`: a release of Taipower unit generation with `release.json`, `partitions/<YYYY-MM>/data.parquet`
  (the columns above) and `metadata/observation_quality_flags.csv` (`unit_name`, `rule_id`,
  `run_first_timestamp`, `run_last_timestamp`), built from data.gov.tw datasets 37331 and 8931.

With the package versions of `uv.lock`, the build writes the same bytes as the published directory, so
`sha256sum -c` against the published SHA256SUMS passes. With other versions, compare values instead:
`uv run --frozen python -m windsr.data.compare DATA PUBLISHED_DATA`.

## Credits

- ERA5: Copernicus Climate Change Service (2018), ERA5 hourly data on single levels from 1940 to present,
  doi:10.24381/cds.adbb2d47; Hersbach, H. et al. (2020), Q. J. R. Meteorol. Soc. 146, 1999-2049. Read from
  ARCO-ERA5: Carver, R. W. & Merose, A. (2023), ARCO-ERA5: an analysis-ready cloud-optimized reanalysis
  dataset, AMS Annual Meeting, 4A.1.
- ERA5-Land: Copernicus Climate Change Service (2019), ERA5-Land hourly data from 1950 to present,
  doi:10.24381/cds.e2161bac; Muñoz-Sabater, J. et al. (2021), Earth Syst. Sci. Data 13, 4349-4383. Land mask from
  the NSF NCAR GDEX subset, doi:10.5065/SNVE-T250.
- Contains modified Copernicus Climate Change Service information (2025). Neither the European Commission nor
  ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.
- ICON: Datenbasis Deutscher Wetterdienst, bearbeitet. Via Open-Meteo: Zippenfenig, P. (2023), Open-Meteo.com
  Weather API, doi:10.5281/zenodo.7970649, CC BY 4.0.
- Stations: 中央氣象署 (Central Weather Administration), 氣候觀測資料查詢服務 CODiS, Open Government Data License
  version 1.0 (https://data.gov.tw/license).
- Wind farms: 台灣電力股份有限公司 (Taiwan Power Company), 台灣電力公司各機組過去發電量 and
  台灣電力公司各機組發電量即時資訊, Open Government Data License version 1.0 (https://data.gov.tw/license).
