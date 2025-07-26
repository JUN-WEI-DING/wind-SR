import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from collections import defaultdict
from scipy.spatial import Delaunay
import torch
import xarray as xr
import argparse
from utils import prepare_data


class GroundStationEvaluator:
    def __init__(self):
        self.station_df = None
        self.stations_info_df = None
        self.era5 = None
        self.icon = None
        self.gen_srimage = None
        self.stations_ds = None
        self.timeindex = None

    def establish_clustering(self, k=5):
        self.station_df = pd.read_csv('/home/gary/Desktop/wind_sr_github/data/CWA_station_group_info.csv').iloc[:, :-2]
        lon_list = np.arange(118.125, 123.25, 0.125)
        lat_list = np.arange(21.375, 26.5, 0.125)
        grid_centers = [(lon, lat) for lat in lat_list for lon in lon_list]
        grid_centers_arr = np.array(grid_centers)
        diffs = self.station_df[['longitude', 'latitude']].values[:, None, :] - grid_centers_arr[None, :, :]
        dists = np.sqrt((diffs**2).sum(axis=2))
        self.station_df['grid_index'] = dists.argmin(axis=1)

        density_map = self.station_df['grid_index'].value_counts()
        cluster_center_grids = density_map.head(k).index.tolist()
        cluster_assignments = {}
        cluster_members = defaultdict(list)

        for i, grid_idx in enumerate(cluster_center_grids):
            center_stations = self.station_df[self.station_df['grid_index'] == grid_idx]
            if not center_stations.empty:
                center_id = center_stations.iloc[0]['ID']
                cluster_assignments[center_id] = i
                cluster_members[i].append(center_id)

        coords = self.station_df[['longitude', 'latitude']].values
        distance_matrix = pairwise_distances(coords)
        id_to_index = {id_: i for i, id_ in enumerate(self.station_df['ID'])}
        unassigned = set(self.station_df['ID']) - set(cluster_assignments.keys())

        for sid in unassigned:
            i = id_to_index[sid]
            best_cluster, best_distance = None, float('inf')
            for cid, members in cluster_members.items():
                indices = [id_to_index[mid] for mid in members]
                avg_distance = distance_matrix[i, indices].mean()
                if avg_distance < best_distance:
                    best_distance = avg_distance
                    best_cluster = cid
            cluster_assignments[sid] = best_cluster
            cluster_members[best_cluster].append(sid)

        self.station_df['cluster'] = self.station_df['ID'].map(cluster_assignments)

    def establish_sampling_probabilities(self):
        points = self.station_df[['longitude', 'latitude']].values
        tri = Delaunay(points)
        neighbors = defaultdict(set)
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                neighbors[a].add(b)
                neighbors[b].add(a)

        d_a_list = []
        for i, neighbor_ids in neighbors.items():
            dists = [np.linalg.norm(points[i] - points[j]) for j in neighbor_ids]
            d_a = np.mean(dists) if dists else 0
            d_a_list.append((i, d_a))

        d_a_df = pd.DataFrame(d_a_list, columns=["index", "d_a"]).set_index("index")
        self.station_df['d_a'] = d_a_df['d_a']
        self.station_df['d_a_prime'] = self.station_df['d_a'] / self.station_df['d_a'].sum()
        self.station_df['sampling_prob'] = self.station_df['d_a_prime'] / self.station_df['d_a_prime'].sum()

    def assign_icon_grid_indices(self):
        icon_ds = xr.open_dataset("/home/gary/Desktop/wind_sr_github/data/icon.zarr")
        lat_array = icon_ds['lat'].values
        lon_array = icon_ds['lon'].values
        self.stations_info_df = self.station_df.copy()
        self.stations_info_df['lat_index'] = self.stations_info_df['latitude'].apply(lambda lat: np.abs(lat_array - lat).argmin())
        self.stations_info_df['lon_index'] = self.stations_info_df['longitude'].apply(lambda lon: np.abs(lon_array - lon).argmin())

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

    @staticmethod
    def calculate_rmse(df, col1, col2):
        diff = df[col1] - df[col2]
        return np.sqrt((diff ** 2).mean())

    def evaluate_cluster_performance(self, cluster_id):
        cluster_df = self.stations_info_df[self.stations_info_df['cluster'] == cluster_id]
        total_df = pd.DataFrame()
        for ID, lat_index, lon_index in zip(cluster_df["ID"], cluster_df["lat_index"], cluster_df["lon_index"]):
            era5_df = pd.DataFrame(self.era5[:, 0, lat_index, lon_index], index=self.timeindex, columns=["era5"])
            icon_df = pd.DataFrame(self.icon[:, 0, lat_index, lon_index], index=self.timeindex, columns=["icon"])
            srimage_df = pd.DataFrame(self.gen_srimage[:, 0, lat_index, lon_index], index=self.timeindex, columns=["srimage"])
            ground_station_df = self.stations_ds.sel(station_id=ID)['WindSpeed.Mean'].to_dataframe(name='ground_station').drop(columns=["station_id"], errors='ignore')
            ground_station_df.index.name = None
            ws_df = pd.concat([era5_df, icon_df, srimage_df, ground_station_df], axis=1).dropna()
            total_df = pd.concat([total_df, ws_df], axis=0)
        for col in ['era5', 'icon', 'srimage']:
            total_df[col] = self.calculate_rmse(total_df, col, 'ground_station')
        return total_df.iloc[0, :]

    def evaluate_sampling_strategy(self):
        results = {}
        for n in range(1, self.stations_info_df.shape[0]):
            collected = []
            while True:
                sampled_df = self.stations_info_df.sample(n=n, weights="sampling_prob", replace=False, random_state=None)
                total_df = pd.DataFrame()
                for ID, lat_index, lon_index in zip(sampled_df["ID"], sampled_df["lat_index"], sampled_df["lon_index"]):
                    era5_df = pd.DataFrame(self.era5[:, 0, lat_index, lon_index], index=self.timeindex, columns=["era5"])
                    icon_df = pd.DataFrame(self.icon[:, 0, lat_index, lon_index], index=self.timeindex, columns=["icon"])
                    srimage_df = pd.DataFrame(self.gen_srimage[:, 0, lat_index, lon_index], index=self.timeindex, columns=["srimage"])
                    ground_station_df = self.stations_ds.sel(station_id=ID)['WindSpeed.Mean'].to_dataframe(name='ground_station').drop(columns=["station_id"], errors='ignore')
                    ground_station_df.index.name = None
                    ws_df = pd.concat([era5_df, icon_df, srimage_df, ground_station_df], axis=1).dropna()
                    total_df = pd.concat([total_df, ws_df], axis=0)
                for col in ['era5', 'icon', 'srimage']:
                    total_df[col] = self.calculate_rmse(total_df, col, 'ground_station')
                collected.append(total_df.iloc[0, :-1].values)
                if len(collected) >= 5:
                    recent = np.array(collected[-5:])
                    if np.std(recent, axis=0).mean() < 0.01:
                        results[n] = np.mean(recent, axis=0)
                        break
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate wind station model performance")
    parser.add_argument('--mode', choices=['sampling', 'cluster'], required=True, help="Evaluation mode")
    parser.add_argument('--cluster_id', type=int, help="Specify cluster id if mode=cluster")
    args = parser.parse_args()

    evaluator = GroundStationEvaluator()
    evaluator.establish_clustering()
    evaluator.establish_sampling_probabilities()
    evaluator.assign_icon_grid_indices()
    evaluator.prepare_all_datasets()

    if args.mode == "sampling":
        result = evaluator.evaluate_sampling_strategy()
        print("Sampling evaluation result:")
        for n, metrics in result.items():
            print(f"Sample size {n}: era5={metrics[0]:.4f}, icon={metrics[1]:.4f}, srimage={metrics[2]:.4f}")
    elif args.mode == "cluster":
        if args.cluster_id is None:
            raise ValueError("Please provide --cluster_id when mode is 'cluster'")
        result = evaluator.evaluate_cluster_performance(args.cluster_id)
        print(f"Cluster {args.cluster_id} evaluation result:")
        print(result)
