import numpy as np
import torch

class Normalizer:
    def __init__(self, raw_data=None, load_path=None):
        self.channel_min = None
        self.channel_max = None
        self.raw_data = raw_data
        self.is_tensor = True

        if raw_data is not None and load_path is None:
            self.fit(raw_data)
            self.normalized_data = self.normalize(raw_data)
        elif load_path is not None and raw_data is None:
            self.load(load_path)
        else:
            raise ValueError("Either raw_data or load_path must be provided, but not both.")

    def fit(self, arr):
        """
        Computes the minimum and maximum values for each channel, 
        used later for normalize() and inverse_normalize().

        Args:
            arr (torch.Tensor): A 4D tensor with shape [batch_size, channels, height, width].
        
        Raises:
            TypeError: If the input is not a torch.Tensor.
        """
        if isinstance(arr, torch.Tensor):
            num_channels = arr.shape[1]
            self.channel_min = []
            self.channel_max = []
            for i in range(num_channels):
                self.channel_min.append(arr[:, i, :, :].min().item())
                self.channel_max.append(arr[:, i, :, :].max().item())
            self.channel_min = torch.tensor(self.channel_min, dtype=arr.dtype, device=arr.device) + 1e-10
            self.channel_max = torch.tensor(self.channel_max, dtype=arr.dtype, device=arr.device)
        else:
            raise TypeError("Input must be a torch.Tensor")

    def normalize(self, arr, channel_map=None):
        """
        Normalize the input data to the [0, 1] range.
        Optionally, use a channel_map to apply normalization based on specific original channels.

        Args:
            arr (torch.Tensor): Input data to be normalized, with shape [batch_size, channels, height, width].
            channel_map (list[int], optional): A list that maps each input channel to a corresponding original channel
                                            for min-max normalization. If None, channels are mapped directly (i → i).

        Returns:
            torch.Tensor: A tensor with per-channel min-max normalization applied.

        Raises:
            TypeError: If the input is not a torch.Tensor.
        """

        if not isinstance(arr, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        arr = arr.clone()
        if channel_map is None:
            channel_map = list(range(arr.shape[1]))  # Default to direct mapping

        for i, original_channel in enumerate(channel_map):
            arr[:, i, :, :] = (arr[:, i, :, :] - self.channel_min[original_channel]) / (
                self.channel_max[original_channel] - self.channel_min[original_channel]
            )
        return arr

    def inverse_normalize(self, arr, channel_map=None):
        """
        Restore normalized data back to its original value range.
        Optionally, use a channel_map to specify which original channel's min/max to apply.

        Args:
            arr (torch.Tensor): Input data to be inverse-normalized, with shape [batch_size, channels, height, width].
            channel_map (list[int], optional): A list that maps each input channel to a corresponding original channel.
                                            If None, channels are mapped directly (i → i).

        Returns:
            torch.Tensor: A tensor with values restored to the original scale.

        Raises:
            TypeError: If the input is not a torch.Tensor.
        """

        if not isinstance(arr, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        arr = arr.clone()
        if channel_map is None:
            channel_map = list(range(arr.shape[1])) 

        for i, original_channel in enumerate(channel_map):
            arr[:, i, :, :] = arr[:, i, :, :] * (self.channel_max[original_channel] - self.channel_min[original_channel]) + self.channel_min[original_channel]
        return arr


    def save(self, file_path):
        """
        Save the per-channel minimum and maximum values to a file.

        Args:
            file_path (str): Path to the output .npz file.
        """
        save_data = {
            'channel_min': self.channel_min.cpu().numpy(),
            'channel_max': self.channel_max.cpu().numpy()
        }
        np.savez(file_path, **save_data)
        print(f"Normalization parameters saved to {file_path}")

    def load(self, file_path):
        """
        Load the per-channel minimum and maximum values from a file.

        Args:
            file_path (str): Path to the .npz file containing 'channel_min' and 'channel_max'.
        """
        data = np.load(file_path)
        channel_min = data['channel_min']
        channel_max = data['channel_max']

        self.channel_min = torch.tensor(channel_min, dtype=torch.float32)
        self.channel_max = torch.tensor(channel_max, dtype=torch.float32)
        print(f"Normalization parameters loaded from {file_path}")
