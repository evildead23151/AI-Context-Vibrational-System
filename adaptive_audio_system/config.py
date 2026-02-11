"""
System Configuration
All values are architectural constants, not domain-specific parameters.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class LatentDimensions:
    """Latent space dimensionalities - learned, not designed"""
    context_embedding: int = 512
    control_tensor: int = 256
    audio_conditioning: int = 768
    behavioral_state: int = 128


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # ACE-Step LM planner size
    lm_size: Literal["0.6b", "1.7b", "4b"] = "1.7b"
    
    # ACE-Step DiT variant
    dit_variant: Literal["turbo", "sft", "base"] = "turbo"
    
    # Context encoder architecture
    context_encoder_type: Literal["transformer", "s4", "temporal_conv"] = "transformer"
    context_encoder_layers: int = 6
    context_encoder_heads: int = 8
    
    # User response model type
    urm_type: Literal["online_rl", "bayesian"] = "online_rl"


@dataclass
class RawSignalConfig:
    """Raw sensor signal configuration"""
    # Temporal sampling
    signal_sampling_hz: float = 1.0  # Once per second
    
    # Buffer sizes (architectural)
    context_window_samples: int = 3600  # 1 hour at 1Hz
    
    # Tensor shapes
    gps_dims: int = 2  # lat, lon
    wifi_hash_dims: int = 16  # learned embedding lookup
    bluetooth_rssi_dims: int = 8
    accel_dims: int = 3
    gyro_dims: int = 3
    audio_level_dims: int = 1
    screen_dims: int = 4  # interaction features
    time_dims: int = 4  # cyclic encodings


@dataclass
class AdaptationConfig:
    """Continuous adaptation loop configuration"""
    # Update intervals (control theory)
    update_interval_seconds: int = 300  # 5 minutes
    
    # Stability constraints
    max_control_delta: float = 0.1  # L2 norm constraint
    reward_drop_abort_threshold: float = -0.5
    
    # Learning rates (optimizer config, not domain parameters)
    planner_lr: float = 1e-5
    context_lr: float = 1e-4
    urm_lr: float = 1e-4


@dataclass
class AudioGenerationConfig:
    """Audio generator configuration"""
    # Generation interval
    generation_chunk_seconds: int = 60  # Generate 1 min at a time
    
    # Buffer overlap for continuity
    overlap_seconds: int = 5
    
    # Diffusion steps (architectural)
    dit_steps: int = 8  # Turbo default


@dataclass
class SystemConfig:
    """Complete system configuration"""
    latent_dims: LatentDimensions = LatentDimensions()
    model: ModelConfig = ModelConfig()
    raw_signals: RawSignalConfig = RawSignalConfig()
    adaptation: AdaptationConfig = AdaptationConfig()
    audio_gen: AudioGenerationConfig = AudioGenerationConfig()
    
    # System paths
    acestep_repo_path: str = "./ACE-Step-1.5"
    weights_cache_path: str = "./weights_cache"
    user_state_path: str = "./user_state"
    
    # Device
    device: str = "cuda"
    dtype: str = "float32"


# Global instance
DEFAULT_CONFIG = SystemConfig()
