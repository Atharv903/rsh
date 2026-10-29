"""
Signal Preprocessing and Normalization Module.

Provides configurable, modular signal conditioning:
- Missing-value handling
- Outlier clipping
- Optional low-pass filtering (Butterworth zero-phase)
- Optional gravity removal / attitude correction
- Leakage-free sensor normalization (fit strictly on training data only)
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import signal


class SensorNormalizer:
    """Sensor normalizer with strict leakage prevention.
    
    Fits scaling parameters strictly on training split data only,
    and applies saved parameters to validation and test splits.
    """
    
    def __init__(self, method: str = "standard", channels: Optional[List[str]] = None):
        """
        Args:
            method: 'standard' (z-score), 'robust' (median & IQR), 'minmax', or 'none'.
            channels: List of sensor columns to normalize.
        """
        self.method = method
        self.channels = channels or ["accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll"]
        self.params: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> "SensorNormalizer":
        """Compute normalization statistics from training dataframe ONLY."""
        self.params = {}
        for ch in self.channels:
            if ch not in df.columns:
                continue
            vals = df[ch].values.astype(float)
            vals = vals[~np.isnan(vals)]
            
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            if std_val < 1e-8:
                std_val = 1.0  # Avoid zero division
                
            median_val = float(np.median(vals))
            q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
            iqr_val = q75 - q25
            if iqr_val < 1e-8:
                iqr_val = 1.0
                
            min_val = float(np.min(vals))
            max_val = float(np.max(vals))
            span = max_val - min_val
            if span < 1e-8:
                span = 1.0
                
            self.params[ch] = {
                "mean": mean_val,
                "std": std_val,
                "median": median_val,
                "iqr": iqr_val,
                "min": min_val,
                "max": max_val,
                "span": span,
            }
            
        self.is_fitted = True
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted normalization to a dataframe (train, val, or test)."""
        if not self.is_fitted and self.method != "none":
            raise RuntimeError("SensorNormalizer must be fitted on training data before transform.")
            
        df_out = df.copy()
        if self.method == "none":
            return df_out
            
        for ch in self.channels:
            if ch not in df_out.columns or ch not in self.params:
                continue
            p = self.params[ch]
            vals = df_out[ch].values.astype(float)
            
            if self.method == "standard":
                df_out[ch] = (vals - p["mean"]) / p["std"]
            elif self.method == "robust":
                df_out[ch] = (vals - p["median"]) / p["iqr"]
            elif self.method == "minmax":
                df_out[ch] = (vals - p["min"]) / p["span"]
            else:
                raise ValueError(f"Unknown normalization method: {self.method}")
                
        return df_out
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df and return transformed df."""
        return self.fit(df).transform(df)
        
    def save(self, json_path: str) -> None:
        """Save fitted parameters to JSON."""
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        data = {
            "method": self.method,
            "channels": self.channels,
            "is_fitted": self.is_fitted,
            "params": self.params,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load(cls, json_path: str) -> "SensorNormalizer":
        """Load fitted parameters from JSON."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        norm = cls(method=data["method"], channels=data["channels"])
        norm.params = data["params"]
        norm.is_fitted = data["is_fitted"]
        return norm


def handle_missing_values(
    df: pd.DataFrame,
    channels: List[str],
    max_gap_samples: int = 5,
    strategy: str = "interpolate_small",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Handle missing (NaN or Inf) sensor observations.
    
    Args:
        df: DataFrame containing sensor columns.
        channels: List of sensor columns.
        max_gap_samples: Maximum consecutive missing samples to interpolate.
        strategy: 'interpolate_small', 'forward_fill', or 'drop'.
        
    Returns:
        (cleaned_df, report)
    """
    df_out = df.copy()
    missing_report = {}
    
    for ch in channels:
        if ch not in df_out.columns:
            continue
        vals = df_out[ch].replace([np.inf, -np.inf], np.nan)
        missing_count = int(vals.isna().sum())
        
        if missing_count > 0:
            if strategy == "interpolate_small":
                # Linearly interpolate gaps up to max_gap_samples
                vals = vals.interpolate(method="linear", limit=max_gap_samples, limit_direction="both")
                # Forward/backward fill any remaining boundary edge
                vals = vals.ffill().bfill()
            elif strategy == "forward_fill":
                vals = vals.ffill().bfill()
            elif strategy == "drop":
                pass
            df_out[ch] = vals
            
        missing_report[ch] = {
            "initial_nan_count": missing_count,
            "residual_nan_count": int(df_out[ch].isna().sum()),
        }
        
    return df_out, missing_report


def clip_outliers(
    df: pd.DataFrame,
    accel_channels: List[str] = ["accel_x", "accel_y", "accel_z"],
    gyro_channels: List[str] = ["gyro_yaw", "gyro_pitch", "gyro_roll"],
    accel_limit_mps2: float = 30.0,
    gyro_limit_rads: float = 10.0,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clip physical outliers beyond extreme automotive dynamics limits."""
    df_out = df.copy()
    clipped_counts = {}
    
    for ch in accel_channels:
        if ch in df_out.columns:
            vals = df_out[ch].values
            mask = np.abs(vals) > accel_limit_mps2
            clipped_counts[ch] = int(np.sum(mask))
            df_out[ch] = np.clip(vals, -accel_limit_mps2, accel_limit_mps2)
            
    for ch in gyro_channels:
        if ch in df_out.columns:
            vals = df_out[ch].values
            mask = np.abs(vals) > gyro_limit_rads
            clipped_counts[ch] = int(np.sum(mask))
            df_out[ch] = np.clip(vals, -gyro_limit_rads, gyro_limit_rads)
            
    return df_out, clipped_counts


def apply_butterworth_lowpass(
    df: pd.DataFrame,
    channels: List[str],
    cutoff_hz: float = 3.5,
    sampling_rate_hz: float = 10.0,
    order: int = 4,
) -> pd.DataFrame:
    """Apply zero-phase Butterworth low-pass filter to attenuate high-frequency acoustic/engine noise.
    
    Args:
        df: DataFrame containing sensor columns.
        channels: Channels to filter.
        cutoff_hz: Filter cutoff frequency (< sampling_rate_hz / 2).
        sampling_rate_hz: Sampling rate in Hz.
        order: Filter order.
        
    Returns:
        Filtered DataFrame.
    """
    nyquist = 0.5 * sampling_rate_hz
    if cutoff_hz >= nyquist:
        raise ValueError(f"Cutoff frequency ({cutoff_hz} Hz) must be less than Nyquist ({nyquist} Hz).")
        
    normal_cutoff = cutoff_hz / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    
    df_out = df.copy()
    for ch in channels:
        if ch in df_out.columns:
            vals = df_out[ch].values
            # filtfilt performs forward-backward zero-phase filtering (no phase distortion)
            df_out[ch] = signal.filtfilt(b, a, vals)
            
    return df_out


def apply_gravity_removal(
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    method: str = "device_estimate",
) -> pd.DataFrame:
    """Optional gravity subtraction from accelerometer channels.
    
    Args:
        df: Aligned dataframe with accel_x, accel_y, accel_z.
        raw_df: Raw dataframe containing GRAVITY X, Y, Z if available.
        method: 'device_estimate' or 'none'.
    """
    df_out = df.copy()
    if method == "device_estimate":
        if all(c in raw_df.columns for c in ["GRAVITY X (m/s²)", "GRAVITY Y (m/s²)", "GRAVITY Z (m/s²)"]):
            df_out["accel_x"] = df_out["accel_x"] - raw_df["GRAVITY X (m/s²)"].values
            df_out["accel_y"] = df_out["accel_y"] - raw_df["GRAVITY Y (m/s²)"].values
            df_out["accel_z"] = df_out["accel_z"] - raw_df["GRAVITY Z (m/s²)"].values
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unsupported gravity removal method: {method}")
        
    return df_out
