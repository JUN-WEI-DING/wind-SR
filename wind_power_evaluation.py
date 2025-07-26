import pandas as pd
import numpy as np
import torch
import xarray as xr
from pathlib import Path
from utils import prepare_data


class WindFarmEvaluator:
    def __init__(self):
        self.wind_farm_info_df = pd.DataFrame({
            "ID": ["海洋竹南", "離岸一期", "鹿威彰濱", "彰工", "王功", "雲麥"],
            "Engname": ["Formosa 1 Offshore", "Offshore Phase 1", "Lu-Wei Changbin", "Changgong", "Wanggong", "Yun-Mai"],
            "Turbine Model": ["SWT-6.0-154", "HTW5.2-127", "Enercon E70", "Vestas V80", "Enercon E70", "Vestas V80"],
            "Hub height": [90, 90, 64, 67, 75, 67],
            "latitude": [24.7, 23.96, 24.09, 24.14, 23.99, 23.81],
            "longitude": [120.82, 120.27, 120.38, 120.42, 120.33, 120.26]
        })
        self.power_curves = self._load_power_curves()

    def _load_power_curves(self):
        # 所有風機型號的風速-功率對照表（略，與你提供相同，建議外部 JSON 管理）
        return {
            "Vestas V80": {1: 0.0, 2: 0.0, 3: 0.0, 3.5: 35, 4: 70, 4.5: 117.0, 5: 165.0,
                     5.5: 225.0, 6.0: 285.0, 6.5: 372, 7.0: 459, 7.5: 580, 
                     8.0: 701, 8.5: 832, 9: 964.0, 9.5: 1127, 10: 1289.0, 
                     10.5: 1428, 11.0: 1567, 11.5: 1678, 12.0: 1788, 12.5: 1865, 
                     13: 1941, 13.5: 1966, 14.0: 1990, 14.5: 2000, 15.0: 2000.0, 
                     16: 2000.0, 17: 2000.0, 18: 2000.0, 19: 2000.0, 20: 2000.0, 
                     21: 2000.0, 22: 2000.0, 23: 2000.0, 24: 2000.0, 25: 2000.0, 
                     26: 2000.0, 27: 2000.0, 28: 2000.0, 29: 2000.0, 30: 2000.0},
            "GE 1.5se": {1: 0.0, 2: 0.0, 3: 0.0, 3.5: 19, 4: 35, 4.5: 55.0, 5: 75.0,
                     5.5: 114.0, 6.0: 153.0, 6.5: 213, 7.0: 274, 7.5: 363, 
                     8.0: 453, 8.5: 583, 9: 714.0, 9.5: 862, 10: 1011.0, 
                     10.5: 1163, 11.0: 1315, 11.5: 1450, 12.0: 1489, 12.5: 1495, 
                     13: 1500, 13.5: 1500, 14.0: 1500, 14.5: 1500, 15.0: 1500.0, 
                     16: 1500.0, 17: 1500.0, 18: 1500.0, 19: 1500.0, 20: 1500.0, 
                     21: 1500.0, 22: 1500.0, 23: 1500.0, 24: 1500.0, 25: 1500.0, 
                     26: 1500.0, 27: 1500.0, 28: 1500.0, 29: 1500.0, 30: 1500.0},
            "Enercon E70": {1: 0.0, 2: 0.0, 2.5: 10.0, 3: 18.0, 3.5: 37, 4: 56, 4.5: 91.0, 5: 127.0,
                     5.5: 183.0, 6.0: 240.0, 6.5: 320, 7.0: 400, 7.5: 513, 
                     8.0: 626, 8.5: 759, 9: 892.0, 9.5: 1058, 10: 1223.0, 
                     10.5: 1407, 11.0: 1590, 11.5: 1745, 12.0: 1900, 12.5: 1990, 
                     13: 2080, 13.5: 2155, 14.0: 2230, 14.5: 2265, 15.0: 2300.0, 
                     16: 2310.0, 17: 2310.0, 18: 2310.0, 19: 2310.0, 20: 2310.0, 
                     21: 2310.0, 22: 2310.0, 23: 2310.0, 24: 2310.0, 25: 2310.0, 
                     26: 2310.0, 27: 2310.0, 28: 2310.0, 29: 2310.0, 30: 2310.0},
            "HTW5.2-127": {1: 0.0, 2: 0.0, 2.5: 0.0, 3: 0.0, 3.5: 0, 4: 0, 4.5: 150.0, 5: 300.0,
                     5.5: 440.0, 6.0: 599.0, 6.5: 830, 7.0: 1073, 7.5: 1400, 
                     8.0: 1735, 8.5: 2130, 9: 2524.0, 9.5: 2885, 10: 3218.0, 
                     10.5: 3610, 11.0: 3975, 11.5: 4330, 12.0: 4621, 12.5: 4850, 
                     13: 5020, 13.5: 5140, 14.0: 5200, 14.5: 5200, 15.0: 5200.0, 
                     16: 5200.0, 17: 5200.0, 18: 5200.0, 19: 5200.0, 20: 5200.0, 
                     21: 5200.0, 22: 5200.0, 23: 5200.0, 24: 5200.0, 25: 5200.0, 
                     26: 5200.0, 27: 5200.0, 28: 5200.0, 29: 5200.0, 30: 5200.0},
            "SWT-6.0-154": {1: 0.0, 2: 0.0, 2.5: 0.0, 3: 0.0, 3.5: 100, 4: 220, 4.5: 320.0, 5: 440.0,
                     5.5: 575.0, 6.0: 721.0, 6.5: 945, 7.0: 1173, 7.5: 1485, 
                     8.0: 1796, 8.5: 2157, 9: 2517.0, 9.5: 2940, 10: 3360.0, 
                     10.5: 3930, 11.0: 4485, 11.5: 5160, 12.0: 5792, 12.5: 5960, 
                     13: 6000, 13.5: 6000, 14.0: 6000, 14.5: 6000, 15.0: 6000.0, 
                     16: 6000.0, 17: 6000.0, 18: 6000.0, 19: 6000.0, 20: 6000.0, 
                     21: 6000.0, 22: 6000.0, 23: 6000.0, 24: 6000.0, 25: 6000.0,}
        }

    def assign_icon_grid_indices(self, icon_ds_path):
        ds = xr.open_dataset(icon_ds_path)
        lat_arr, lon_arr = ds['lat'].values, ds['lon'].values
        self.wind_farm_info_df['lat_index'] = self.wind_farm_info_df['latitude'].apply(lambda lat: np.abs(lat_arr - lat).argmin())
        self.wind_farm_info_df['lon_index'] = self.wind_farm_info_df['longitude'].apply(lambda lon: np.abs(lon_arr - lon).argmin())

    def load_wind_farm_power_data(self, csv_dir="../../data/csv/power"):
        power_df = pd.DataFrame()
        for station in self.wind_farm_info_df["ID"]:
            all_years_df = []
            for year in ['2022', '2023']:
                fpath = Path(csv_dir) / f"{station}_{year}.csv"
                if not fpath.exists():
                    continue
                df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                df = df[df["status"] == "online"].apply(pd.to_numeric, errors="coerce")
                df['group'] = (df['used'] != df['used'].shift()).cumsum()
                df.loc[df.groupby('group')['used'].transform('count') >= 5, 'used'] = np.nan
                df = df.drop(columns='group')
                for cap in df['capacity'].dropna().unique():
                    max_val = df[df['capacity'] == cap]['used'].max()
                    df.loc[df['capacity'] == cap, 'used'] /= max_val
                all_years_df.append(df['used'].rename(station))
            if all_years_df:
                power_df = pd.concat([power_df, pd.concat(all_years_df)], axis=1)
        power_df[power_df < 0] = 0
        power_df = power_df.sort_index().asfreq("10min")
        self.power_df = power_df[power_df.index.minute == 0]
        return power_df

    def wind_speed_at_hub_height(self, ws10: pd.Series, ws100: pd.Series, height: float) -> pd.Series:
        """Log interpolation from 10m and 100m to hub height."""
        return ws10 + (np.log(height / 10) / np.log(100 / 10)) * (ws100 - ws10)

    def interpolate_power(self, speed: float, model: str) -> float:
        curve = self.power_curves.get(model, {})
        if speed < 1:
            return 0.0
        if speed > 25:
            return max(curve.values(), default=0.0)
        speeds = sorted(curve.keys())
        for i in range(len(speeds) - 1):
            if speeds[i] <= speed <= speeds[i+1]:
                s1, s2 = speeds[i], speeds[i+1]
                p1, p2 = curve[s1], curve[s2]
                return p1 + (p2 - p1) * (speed - s1) / (s2 - s1)
        return curve.get(speed, 0.0)

    def apply_loss_model(self, ws: float, rated_speed: float, loss_base: float = 0.1) -> float:
        if ws < rated_speed - 0.5:
            return loss_base
        elif ws <= rated_speed + 2:
            return loss_base * (1 - (ws - (rated_speed - 0.5)) / 2.5)
        return 0

    def estimate_actual_power(self, ws_series: pd.Series, model: str, loss_base: float = 0.1) -> pd.Series:
        rated_speed = max(self.power_curves[model].keys())
        return ws_series.apply(lambda ws: self.interpolate_power(ws * (1 - self.apply_loss_model(ws, rated_speed, loss_base)), model))

    def calculate_metrics(self, df, pred_col, obs_col):
        valid = df[[pred_col, obs_col]].dropna()
        mse = ((valid[pred_col] - valid[obs_col]) ** 2).mean()
        mae = (valid[pred_col] - valid[obs_col]).abs().mean()
        return {'rmse': np.sqrt(mse), 'mae': mae, 'n': len(valid)}

    def prepare_all_datasets(self):
        data_loader = prepare_data.Data(
            icon_path="/home/gary/Desktop/wind_sr_github/data/icon.zarr",
            era5_path="/home/gary/Desktop/wind_sr_github/data/era5.zarr"
        )
        self.timeindex = data_loader.get_testindex()
        self.era5, self.icon = data_loader.get_testdata()
        self.gen_srimage = torch.tensor(np.load('srimage.npz', allow_pickle=True)['gen_srimage'])
        self.stations_ds = xr.open_dataset('/home/gary/Desktop/wind_sr_github/data/groundstation.zarr')
        normalizer = data_loader.get_normalizer()
        self.era5 = normalizer.inverse_normalize(self.era5)
        self.icon = normalizer.inverse_normalize(self.icon, [0, 1])
        self.gen_srimage = normalizer.inverse_normalize(self.gen_srimage, [0, 1])
        self.gen_srimage = np.clip(self.gen_srimage, 0, None)

    def evaluate_estimation_performance(self, wind_speed_source="srimage", loss_base=0.1):
        result = {}
        for i, row in self.wind_farm_info_df.iterrows():
            ID = row["ID"]
            model = row["Turbine Model"]
            hub_height = row["Hub height"]
            lat_index = row["lat_index"]
            lon_index = row["lon_index"]

            # 取風速資料
            era5_ws10 = pd.Series(self.era5[:, 0, lat_index, lon_index], index=self.timeindex)
            era5_ws100 = pd.Series(self.era5[:, 1, lat_index, lon_index], index=self.timeindex)
            icon_ws10 = pd.Series(self.icon[:, 0, lat_index, lon_index], index=self.timeindex)
            icon_ws100 = pd.Series(self.icon[:, 1, lat_index, lon_index], index=self.timeindex)
            sr_ws10 = pd.Series(self.gen_srimage[:, 0, lat_index, lon_index], index=self.timeindex)
            sr_ws100 = pd.Series(self.gen_srimage[:, 1, lat_index, lon_index], index=self.timeindex)

            # 插值至 hub 高度
            if wind_speed_source == "era5":
                ws = self.wind_speed_at_hub_height(era5_ws10, era5_ws100, hub_height)
            elif wind_speed_source == "icon":
                ws = self.wind_speed_at_hub_height(icon_ws10, icon_ws100, hub_height)
            else:
                ws = self.wind_speed_at_hub_height(sr_ws10, sr_ws100, hub_height)

            # 功率估計
            est_power = self.estimate_actual_power(ws, model=model, loss_base=loss_base)
            est_power.name = "estimated"
            try:
                obs_power = self.power_df[ID].rename("observed")
                df = pd.concat([est_power, obs_power], axis=1).dropna()
                metrics = self.calculate_metrics(df, "estimated", "observed")
                result[ID] = metrics
            except Exception as e:
                print(f"Skip {ID} due to missing or malformed observed power data.")
        return pd.DataFrame(result).T

if __name__ == "__main__":
    evaluator = WindFarmEvaluator()
    evaluator.assign_icon_grid_indices("/home/gary/Desktop/wind_sr_github/data/icon.zarr")
    evaluator.prepare_all_datasets()
    evaluator.load_wind_farm_power_data()

    df_sr = evaluator.evaluate_estimation_performance(wind_speed_source="srimage")
    df_icon = evaluator.evaluate_estimation_performance(wind_speed_source="icon")
    df_era5 = evaluator.evaluate_estimation_performance(wind_speed_source="era5")
