"""
Module 1: Raw Signal Ingestion Layer
Captures reality without interpretation.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
import time
import math


@dataclass
class RawSignalTensor:
    """Opaque tensor bundle - no semantic meaning"""
    data: torch.Tensor  # Shape: [sequence_len, feature_dim]
    timestamps: torch.Tensor  # Shape: [sequence_len]
    
    def to(self, device):
        return RawSignalTensor(
            data=self.data.to(device),
            timestamps=self.timestamps.to(device)
        )


class RawSignalIngestion:
    """
    Pure signal capture with zero interpretation.
    All outputs are normalized tensors without named semantics.
    """
    
    def __init__(self, config):
        self.config = config
        self.buffer_size = config.raw_signals.context_window_samples
        
        # Circular buffers (no interpretation)
        self._init_buffers()
        
        # Normalization statistics (learned online)
        self._init_normalizers()
    
    def _init_buffers(self):
        """Initialize raw signal buffers"""
        cfg = self.config.raw_signals
        
        # Calculate total feature dimensionality
        self.feature_dim = (
            cfg.gps_dims +
            cfg.wifi_hash_dims +
            cfg.bluetooth_rssi_dims +
            cfg.accel_dims +
            cfg.gyro_dims +
            cfg.audio_level_dims +
            cfg.screen_dims +
            cfg.time_dims
        )
        
        # Circular buffer
        self.buffer = np.zeros((self.buffer_size, self.feature_dim), dtype=np.float32)
        self.timestamps = np.zeros(self.buffer_size, dtype=np.float64)
        self.buffer_idx = 0
        self.buffer_full = False
    
    def _init_normalizers(self):
        """Initialize online normalization statistics"""
        self.running_mean = np.zeros(self.feature_dim, dtype=np.float32)
        self.running_var = np.ones(self.feature_dim, dtype=np.float32)
        self.n_samples = 0
    
    def ingest(self, raw_data: Dict) -> None:
        """
        Ingest raw sensor data without interpretation.
        
        Args:
            raw_data: Dict with keys corresponding to sensor streams
                     (values are raw floats/ints, no preprocessing)
        """
        # Encode raw signals into feature vector
        features = self._encode_raw_signals(raw_data)
        
        # Update circular buffer
        self.buffer[self.buffer_idx] = features
        self.timestamps[self.buffer_idx] = time.time()
        
        # Update normalizers online
        self._update_normalizers(features)
        
        # Advance buffer
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        if self.buffer_idx == 0:
            self.buffer_full = True
    
    def _encode_raw_signals(self, raw_data: Dict) -> np.ndarray:
        """
        Encode raw sensor readings into flat feature vector.
        NO semantic interpretation - pure signal encoding.
        """
        features = []
        cfg = self.config.raw_signals
        
        # GPS coordinates (raw floats)
        gps = raw_data.get('gps', [0.0, 0.0])
        features.extend(gps[:cfg.gps_dims])
        
        # WiFi BSSID hashes (hash to fixed-dim embedding indices)
        wifi_bssids = raw_data.get('wifi_bssids', [])
        wifi_features = self._hash_to_features(wifi_bssids, cfg.wifi_hash_dims)
        features.extend(wifi_features)
        
        # Bluetooth RSSI time series (recent window)
        bt_rssi = raw_data.get('bluetooth_rssi', [])
        bt_features = self._timeseries_to_features(bt_rssi, cfg.bluetooth_rssi_dims)
        features.extend(bt_features)
        
        # Accelerometer (3-axis)
        accel = raw_data.get('accelerometer', [0.0, 0.0, 0.0])
        features.extend(accel[:cfg.accel_dims])
        
        # Gyroscope (3-axis)
        gyro = raw_data.get('gyroscope', [0.0, 0.0, 0.0])
        features.extend(gyro[:cfg.gyro_dims])
        
        # Ambient audio level (SPL)
        audio_level = raw_data.get('audio_level', [0.0])
        features.extend(audio_level[:cfg.audio_level_dims])
        
        # Screen interaction features (derived, not interpreted)
        screen_data = raw_data.get('screen_interaction', {})
        screen_features = self._screen_to_features(screen_data, cfg.screen_dims)
        features.extend(screen_features)
        
        # Time encoding (cyclic)
        time_features = self._time_to_cyclic(time.time(), cfg.time_dims)
        features.extend(time_features)
        
        return np.array(features, dtype=np.float32)
    
    def _hash_to_features(self, items: list, dim: int) -> list:
        """Hash items to fixed-dim feature vector (no lookup table)"""
        if not items:
            return [0.0] * dim
        
        # Simple hash aggregation (no semantic meaning)
        hashes = [hash(str(item)) % (2**31) for item in items]
        features = []
        for i in range(dim):
            # Deterministic but opaque transformation
            val = sum(h * (i + 1) for h in hashes) % (2**31)
            features.append(val / (2**31))  # Normalize to [0, 1]
        
        return features
    
    def _timeseries_to_features(self, series: list, dim: int) -> list:
        """Convert time series to fixed-dim features (statistical moments)"""
        if not series:
            return [0.0] * dim
        
        arr = np.array(series, dtype=np.float32)
        
        # Statistical features (no interpretation)
        features = []
        for i in range(dim):
            if i == 0:
                features.append(float(np.mean(arr)))
            elif i == 1:
                features.append(float(np.std(arr)))
            elif i == 2:
                features.append(float(np.min(arr)))
            elif i == 3:
                features.append(float(np.max(arr)))
            else:
                # Higher moments or percentiles
                percentile = (i - 3) * (100 / (dim - 3))
                features.append(float(np.percentile(arr, percentile)))
        
        return features
    
    def _screen_to_features(self, screen_data: dict, dim: int) -> list:
        """Encode screen interaction data to features"""
        features = []
        
        # Opaque transformations of screen state
        timestamp_delta = screen_data.get('last_interaction_delta', 0.0)
        interaction_count = screen_data.get('interaction_count', 0)
        is_locked = screen_data.get('is_locked', False)
        foreground_entropy = screen_data.get('foreground_entropy', 0.0)
        
        if dim >= 1:
            features.append(timestamp_delta)
        if dim >= 2:
            features.append(float(interaction_count))
        if dim >= 3:
            features.append(float(is_locked))
        if dim >= 4:
            features.append(foreground_entropy)
        
        # Pad if needed
        while len(features) < dim:
            features.append(0.0)
        
        return features[:dim]
    
    def _time_to_cyclic(self, timestamp: float, dim: int) -> list:
        """Encode time as cyclic features (no semantic labels)"""
        features = []
        
        # Multiple periodic encodings at different scales
        for i in range(dim // 2):
            # Different periods: daily, weekly, monthly, etc.
            period = 86400 * (2 ** i)  # Exponential periods
            phase = (timestamp % period) / period * 2 * math.pi
            features.append(math.sin(phase))
            features.append(math.cos(phase))
        
        # Pad if odd dim
        if dim % 2 == 1:
            features.append(timestamp % 1.0)
        
        return features[:dim]
    
    def _update_normalizers(self, features: np.ndarray):
        """Update running normalization statistics"""
        self.n_samples += 1
        
        # Welford's online algorithm for mean/variance
        delta = features - self.running_mean
        self.running_mean += delta / self.n_samples
        delta2 = features - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.n_samples
    
    def get_context_window(self) -> RawSignalTensor:
        """
        Return current context window as normalized tensor.
        Output is opaque - no semantic meaning.
        """
        # Determine valid range
        if self.buffer_full:
            # Full circular buffer
            data = np.concatenate([
                self.buffer[self.buffer_idx:],
                self.buffer[:self.buffer_idx]
            ])
            ts = np.concatenate([
                self.timestamps[self.buffer_idx:],
                self.timestamps[:self.buffer_idx]
            ])
        else:
            # Partial buffer
            data = self.buffer[:self.buffer_idx]
            ts = self.timestamps[:self.buffer_idx]
        
        # Normalize using running statistics
        if self.n_samples > 1:
            std = np.sqrt(self.running_var)
            std = np.where(std < 1e-6, 1.0, std)  # Avoid division by zero
            data = (data - self.running_mean) / std
        
        # Convert to tensors
        data_tensor = torch.from_numpy(data).float()
        ts_tensor = torch.from_numpy(ts).float()
        
        return RawSignalTensor(data=data_tensor, timestamps=ts_tensor)
