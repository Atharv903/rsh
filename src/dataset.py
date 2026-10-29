"""
PyTorch Dataset and DataLoader module for IMU GPS-denied navigation.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class IMUNavigationDataset(Dataset):
    """
    Dataset wrapping windowed 6-axis IMU sequences and 2D velocity targets.
    
    Channels: [accel_x, accel_y, accel_z, gyro_yaw, gyro_pitch, gyro_roll]
    Target: [v_east, v_north] in m/s
    """
    def __init__(
        self,
        npz_path: str,
        filter_valid_only: bool = True,
        target_mode: str = "velocity",
        window_duration_s: float = 10.0,
    ):
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)
        self.X_raw = data["X"].astype(np.float32)  # (N, seq_len, 6)
        # Support both 'Y' and 'y'
        if "y" in data:
            raw_y = data["y"].astype(np.float32)
        elif "Y" in data:
            raw_y = data["Y"].astype(np.float32)
        else:
            raise KeyError("Neither 'y' nor 'Y' found in npz archive!")

        # Target conversion: convert 10s displacement to window-average velocity (m/s)
        if target_mode == "velocity":
            self.y_raw = raw_y / float(window_duration_s)
        else:
            self.y_raw = raw_y

        # Parse metadata
        if "valid_mask" in data:
            self.valid_mask = data["valid_mask"].astype(bool)
        elif "meta" in data:
            meta = data["meta"]
            self.valid_mask = np.array([m.get("valid_ratio", 1.0) >= 0.99 for m in meta], dtype=bool)
        else:
            self.valid_mask = np.ones(len(self.X_raw), dtype=bool)

        if "meta" in data:
            meta = data["meta"]
            self.drive_id = np.array([m.get("drive_id", "") for m in meta])
            self.t_start = np.array([m.get("start_time_s", 0.0) for m in meta], dtype=np.float64)
            self.t_end = np.array([m.get("end_time_s", 0.0) for m in meta], dtype=np.float64)
        else:
            self.drive_id = data.get("drive_id", np.array([""] * len(self.X_raw)))
            self.t_start = data.get("t_start", np.zeros(len(self.X_raw), dtype=np.float64))
            self.t_end = data.get("t_end", np.zeros(len(self.X_raw), dtype=np.float64))

        if filter_valid_only:
            valid_indices = np.where(self.valid_mask)[0]
            self.X = self.X_raw[valid_indices]
            self.y = self.y_raw[valid_indices]
            self.t_start = self.t_start[valid_indices]
            self.t_end = self.t_end[valid_indices]
            self.drive_id = self.drive_id[valid_indices]
            self.indices = valid_indices
        else:
            self.X = self.X_raw
            self.y = self.y_raw
            self.indices = np.arange(len(self.valid_mask))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


def create_dataloaders(
    train_npz: str,
    val_npz: str,
    test_npz: str,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for train, val, and test splits.
    Train split is shuffled; val and test are sequential.
    """
    train_ds = IMUNavigationDataset(train_npz, filter_valid_only=True)
    val_ds = IMUNavigationDataset(val_npz, filter_valid_only=True)
    test_ds = IMUNavigationDataset(test_npz, filter_valid_only=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
