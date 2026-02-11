"""
Adaptive Audio Regulation System
Main system integrator and entry point.
"""

import torch
import logging
from pathlib import Path

from .config import SystemConfig
from .modules.raw_signal_ingestion import RawSignalIngestion
from .modules.latent_context_model import LatentContextModel, LatentContextTrainer
from .modules.user_response_model import UserResponseModel
from .modules.latent_audio_planner import LatentAudioPlanner
from .modules.audio_generator import AudioGenerator
from .modules.adaptation_loop import ContinuousAdaptationLoop


class AdaptiveAudioSystem:
    """
    Production-grade adaptive audio regulation system.
    
    Zero human-interpretable parameters.
    All control via learned latent spaces.
    Continuous adaptation from implicit behavioral feedback.
    """
    
    def __init__(
        self,
        lm_model_size: str = "1.7b",
        dit_variant: str = "turbo",
        device: str = "cuda",
        config: SystemConfig = None
    ):
        """
        Initialize the adaptive audio system.
        
        Args:
            lm_model_size: ACE-Step LM size ("0.6b", "1.7b", "4b")
            dit_variant: ACE-Step DiT variant ("turbo", "sft", "base")
            device: Torch device ("cuda", "cpu", "mps")
            config: Optional custom configuration
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AdaptiveAudioSystem')
        
        # Configuration
        if config is None:
            config = SystemConfig()
        config.model.lm_size = lm_model_size
        config.model.dit_variant = dit_variant
        config.device = device
        
        self.config = config
        self.device = torch.device(device)
        
        self.logger.info(f"Initializing system on {device}")
        
        # Initialize modules
        self._init_modules()
        
        # Load pretrained models
        self._load_pretrained_models()
        
        # Initialize adaptation loop
        self.adaptation_loop = ContinuousAdaptationLoop(
            raw_signal_ingestion=self.raw_signals,
            latent_context_model=self.lcm,
            user_response_model=self.urm,
            latent_audio_planner=self.planner,
            audio_generator=self.generator,
            config=self.config,
            device=self.device
        )
        
        self.logger.info("System initialized successfully")
    
    def _init_modules(self):
        """Initialize all system modules"""
        self.logger.info("Initializing modules...")
        
        # Module 1: Raw Signal Ingestion
        self.raw_signals = RawSignalIngestion(self.config)
        
        # Module 2: Latent Context Model
        self.lcm = LatentContextModel(self.config)
        self.lcm_trainer = LatentContextTrainer(
            self.lcm,
            self.config,
            self.device
        )
        
        # Module 3: User Response Model
        self.urm = UserResponseModel(self.config)
        
        # Module 4: Latent Audio Planner
        self.planner = LatentAudioPlanner(self.config)
        
        # Module 5: Audio Generator
        self.generator = AudioGenerator(self.config)
        
        self.logger.info("Modules initialized")
    
    def _load_pretrained_models(self):
        """Load ACE-Step pretrained models"""
        self.logger.info("Loading ACE-Step models...")
        
        # Create cache directory
        weights_cache = Path(self.config.weights_cache_path)
        weights_cache.mkdir(parents=True, exist_ok=True)
        
        # Load ACE-Step LM (planner)
        self.planner.load_ace_lm(str(weights_cache))
        
        # Load ACE-Step DiT (generator)
        self.generator.load_ace_dit(str(weights_cache))
        
        self.logger.info("ACE-Step models loaded")
    
    def ingest_sensor_data(self, sensor_data: dict):
        """
        Ingest raw sensor data.
        
        Args:
            sensor_data: Dict with sensor readings (GPS, WiFi, accel, etc.)
        """
        self.raw_signals.ingest(sensor_data)
    
    def record_behavior(self, signal_type: str, value: float):
        """
        Record implicit behavioral signal.
        
        Args:
            signal_type: Signal identifier (opaque)
            value: Signal value (opaque)
        """
        if hasattr(self, 'adaptation_loop'):
            self.adaptation_loop.record_behavior(signal_type, value)
    
    def start(self):
        """Start the continuous adaptation loop"""
        self.logger.info("Starting adaptive audio system...")
        self.adaptation_loop.start()
        self.logger.info("System running")
    
    def stop(self):
        """Stop the system"""
        self.logger.info("Stopping system...")
        self.adaptation_loop.stop()
        self.logger.info("System stopped")
    
    def get_system_state(self):
        """Get current system state"""
        return self.adaptation_loop.get_state()
    
    def save_state(self, path: str):
        """Save system state (models, control tensors, etc.)"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model states
        torch.save({
            'lcm': self.lcm.state_dict(),
            'urm': self.urm.state_dict(),
            'planner_adapters': {
                k: v for k, v in self.planner.state_dict().items()
                if 'adapter' in k or 'projector' in k
            },
            'generator_adapter': {
                k: v for k, v in self.generator.state_dict().items()
                if 'control_to_conditioning' in k
            },
            'config': self.config
        }, save_path / 'system_state.pt')
        
        # Save raw signal normalizers
        import pickle
        with open(save_path / 'signal_normalizers.pkl', 'wb') as f:
            pickle.dump({
                'running_mean': self.raw_signals.running_mean,
                'running_var': self.raw_signals.running_var,
                'n_samples': self.raw_signals.n_samples
            }, f)
        
        self.logger.info(f"State saved to {path}")
    
    def load_state(self, path: str):
        """Load system state"""
        save_path = Path(path)
        
        # Load model states
        checkpoint = torch.load(save_path / 'system_state.pt', map_location=self.device)
        
        self.lcm.load_state_dict(checkpoint['lcm'])
        self.urm.load_state_dict(checkpoint['urm'])
        
        # Load adapters only
        planner_state = self.planner.state_dict()
        planner_state.update(checkpoint['planner_adapters'])
        self.planner.load_state_dict(planner_state)
        
        generator_state = self.generator.state_dict()
        generator_state.update(checkpoint['generator_adapter'])
        self.generator.load_state_dict(generator_state)
        
        # Load signal normalizers
        import pickle
        with open(save_path / 'signal_normalizers.pkl', 'rb') as f:
            normalizers = pickle.load(f)
            self.raw_signals.running_mean = normalizers['running_mean']
            self.raw_signals.running_var = normalizers['running_var']
            self.raw_signals.n_samples = normalizers['n_samples']
        
        self.logger.info(f"State loaded from {path}")


def create_system(
    lm_model_size: str = "1.7b",
    dit_variant: str = "turbo",
    device: str = "cuda"
) -> AdaptiveAudioSystem:
    """
    Factory function to create adaptive audio system.
    
    Args:
        lm_model_size: ACE-Step LM size
        dit_variant: ACE-Step DiT variant
        device: Torch device
    
    Returns:
        Initialized AdaptiveAudioSystem
    """
    return AdaptiveAudioSystem(
        lm_model_size=lm_model_size,
        dit_variant=dit_variant,
        device=device
    )
