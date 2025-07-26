import pandas as pd
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np 

from utils import normalization,common

class Data:
    def __init__(self, icon_path, era5_path, train_range=("2024", "2025"), test_range=("2022", "2024")):
        self.TRAIN_DATE_RANGE = train_range
        self.TEST_DATE_RANGE = test_range

        # === 讀取資料 ===
        icon_ds = xr.open_dataset(icon_path)
        era5_ds = xr.open_dataset(era5_path)
        icon_ds['date'] = icon_ds['date'] 
        era5_ds['date'] = era5_ds['date'] 

        # === ICON 插值計算 100m 風速 ===
        h1, h2, h = 80, 120, 100
        log_h1, log_h2, log_h = np.log(h1), np.log(h2), np.log(h)
        ws_h1, ws_h2 = icon_ds['wind_speed_80m'], icon_ds['wind_speed_120m']
        icon_ds['wind_speed_100m'] = ws_h1 + ((log_h - log_h1) / (log_h2 - log_h1)) * (ws_h2 - ws_h1)
        icon_ds = icon_ds[['wind_speed_10m', 'wind_speed_100m']]

        # === ERA5 計算風速與風向 ===
        era5_ds['ws10'] = np.sqrt(era5_ds['u10']**2 + era5_ds['v10']**2)
        era5_ds['ws100'] = np.sqrt(era5_ds['u100']**2 + era5_ds['v100']**2)
        era5_ds['wd10'] = (np.arctan2(-era5_ds['u10'], -era5_ds['v10']) * 180 / np.pi) % 360
        era5_ds['wd100'] = (np.arctan2(-era5_ds['u100'], -era5_ds['v100']) * 180 / np.pi) % 360
        era5_ds = era5_ds[['ws10','ws100','u10','v10','wd10','u100','v100','wd100']]

        # === 切分訓練與測試資料 ===
        train_era5 = era5_ds.sel(date=slice(*self.TRAIN_DATE_RANGE))
        test_era5 = era5_ds.sel(date=slice(*self.TEST_DATE_RANGE))
        train_icon = icon_ds.sel(date=slice(*self.TRAIN_DATE_RANGE))
        test_icon = icon_ds.sel(date=slice(*self.TEST_DATE_RANGE))
        
        # === 轉換為 torch tensor ===
        self.raw_train_era5 = torch.tensor(train_era5.to_array().values, dtype=torch.float32).permute(1, 0, 2, 3)
        self.raw_test_era5 = torch.tensor(test_era5.to_array().values, dtype=torch.float32).permute(1, 0, 2, 3)
        self.raw_train_icon = torch.tensor(train_icon.to_array().values, dtype=torch.float32).permute(1, 0, 2, 3)
        self.raw_test_icon = torch.tensor(test_icon.to_array().values, dtype=torch.float32).permute(1, 0, 2, 3)
        self.test_time_index = test_era5['date'].values

        # === 對 ERA5 上採樣到 ICON 尺寸 ===
        self.size = (self.raw_train_icon.size(-2), self.raw_train_icon.size(-1))
        self.upsample_train_era5 = common.resize_image(self.raw_train_era5, self.size)
        self.upsample_test_era5 = common.resize_image(self.raw_test_era5, self.size)

        # === 正規化 ===
        self.srimage_normalizer = normalization.Normalizer(self.upsample_train_era5)
        self.train_icon = self.srimage_normalizer.normalize(self.raw_train_icon,[0,1])
        self.test_icon = self.srimage_normalizer.normalize(self.raw_test_icon,[0,1])
        self.upsample_train_era5 = self.srimage_normalizer.normalize(self.upsample_train_era5)
        self.upsample_test_era5 = self.srimage_normalizer.normalize(self.upsample_test_era5)

    # === API ===
    def get_traindata(self):
        return self.upsample_train_era5, self.train_icon

    def get_testdata(self):
        return self.upsample_test_era5, self.test_icon

    def get_normalizer(self):
        return self.srimage_normalizer

    def get_raw_data(self):
        return self.raw_train_era5, self.raw_test_era5, self.raw_train_icon
    
    def get_testindex(self):
        return self.test_time_index
    
class GAN_Dataset(Dataset):
    def __init__(self, device, labels, images):
        self.images = images.to(device)
        self.labels = labels.to(device)
        shuffled_indices = torch.randperm(self.labels.size(0))  
        self.generated_label = self.labels[shuffled_indices].to(device)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.generated_label[idx]
