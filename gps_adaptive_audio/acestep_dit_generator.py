"""
ACE-Step DiT Generator - REAL INTEGRATION
Replaces sine wave fallback with actual ACE-Step Diffusion Transformer
"""

import torch
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger('ACEStepDiTGenerator')


class ACEStepDiTGenerator:
    """
    REAL ACE-Step DiT integration for GPS-adaptive audio system
    
    Uses ACE-Step Diffusion Transformer to generate music from control tensors
    """
    
    def __init__(self, device="cpu", variant="turbo"):
        self.device = torch.device(device)
        self.variant = variant
        self.sample_rate = 44100
        
        logger.info(f"Initializing ACE-Step DiT Generator (variant: {variant}, device: {device})")
        
        try:
            # Import ACE-Step modules
            self._import_acestep()
            
            # Load model
            self._load_model()
            
            logger.info(f"✅ ACE-Step DiT loaded successfully")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize ACE-Step DiT: {e}")
            logger.error("Ensure ACE-Step repository is cloned and dependencies installed")
            raise
    
    def _import_acestep(self):
        """Import ACE-Step modules"""
        try:
            # Add ACE-Step to path if needed
            import sys
            acestep_path = Path(__file__).parent.parent.parent / "ACE-Step-1.5"
            if acestep_path.exists():
                sys.path.insert(0, str(acestep_path))
            
            # Import DiT components
            from datasets import infer
            from datasets.configs.config import Config
            
            self.infer_module = infer
            self.Config = Config
            
            logger.info("ACE-Step modules imported")
            
        except ImportError as e:
            logger.error(f"Cannot import ACE-Step: {e}")
            logger.error("Please clone ACE-Step-1.5 repository")
            logger.error("Run: git clone https://github.com/ace-step/ACE-Step-1.5")
            raise
    
    def _load_model(self):
        """Load ACE-Step DiT model"""
        # Configure model
        config = self.Config()
        config.model_name = f"acestep-v15-{self.variant}"
        config.device = str(self.device)
        
        # Load DiT
        logger.info(f"Loading DiT model: {config.model_name}")
        self.dit = self.infer_module.load_dit_model(config)
        
        # Get model metadata
        self.max_duration = getattr(config, 'max_duration', 600)  # 10 minutes max
        self.sample_rate = getattr(config, 'sample_rate', 44100)
        
        logger.info(f"DiT loaded. Max duration: {self.max_duration}s")
    
    def generate(self, control_tensor: torch.Tensor, duration_seconds=30):
        """
        Generate music from control tensor
        
        Args:
            control_tensor: Latent control tensor from planner
            duration_seconds: Duration of audio to generate
        
        Returns:
            tuple: (audio_waveform, sample_rate)
        """
        try:
            logger.info(f"Generating {duration_seconds}s audio from control tensor {control_tensor.shape}")
            
            # Convert control tensor to caption/conditioning
            # In a full system, this would use the LM to interpret control
            # For now, we use a simple mapping
            caption = self._control_to_caption(control_tensor)
            
            logger.info(f"Generated caption: {caption}")
            
            # Generate audio using DiT
            audio_data = self.dit.generate(
                prompt=caption,
                duration=duration_seconds,
                num_inference_steps=8 if self.variant == "turbo" else 50,
                guidance_scale=3.5,
                sample_rate=self.sample_rate
            )
            
            # Extract waveform
            if isinstance(audio_data, dict):
                audio_waveform = audio_data['audio']
            else:
                audio_waveform = audio_data
            
            # Ensure numpy array
            if isinstance(audio_waveform, torch.Tensor):
                audio_waveform = audio_waveform.cpu().numpy()
            
            # Normalize
            audio_waveform = audio_waveform.flatten()
            audio_waveform = audio_waveform / (np.max(np.abs(audio_waveform)) + 1e-8) * 0.8
            
            logger.info(f"Generated audio: {len(audio_waveform)/self.sample_rate:.2f}s")
            
            return audio_waveform.astype(np.float32), self.sample_rate
            
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise
    
    def _control_to_caption(self, control_tensor: torch.Tensor) -> str:
        """
        Convert control tensor to music caption/prompt
        
        This is a simplified mapping. In production, you would:
        1. Use ACE-Step LM to interpret control tensor into semantic codes
        2. Generate detailed caption with metadata
        
        For now, we extract basic features from control tensor values
        """
        control_np = control_tensor.cpu().numpy()
        
        # Extract musical characteristics from control tensor
        # These mappings are non-semantic but musically relevant
        
        # Feature 1-10: Instrument/timbre selection (from control values)
        instruments = ["piano", "guitar", "strings", "synth", "drums", 
                      "bass", "horns", "woodwinds", "choir", "percussion"]
        dominant_instrument = instruments[int(abs(control_np[0]) * len(instruments)) % len(instruments)]
        
        # Feature 11-20: Genre/style (from control values)
        styles = ["ambient", "electronic", "classical", "jazz", "rock",
                 "folk", "cinematic", "experimental", "lo-fi", "orchestral"]
        dominant_style = styles[int(abs(control_np[1]) * len(styles)) % len(styles)]
        
        # Feature 21-30: Mood/energy (from control values)
        moods = ["calm", "energetic", "melancholic", "uplifting", "mysterious",
                "dramatic", "peaceful", "intense", "playful", "contemplative"]
        dominant_mood = moods[int(abs(control_np[2]) * len(moods)) % len(moods)]
        
        # Feature 31-40: Tempo/rhythm characteristics
        tempo_value = 60 + int(abs(control_np[3]) * 120)  # 60-180 BPM range
        
        # Build caption
        caption = f"{dominant_mood} {dominant_style} music featuring {dominant_instrument}, {tempo_value} BPM"
        
        return caption


def test_dit_generator():
    """Test function for ACE-Step DiT generator"""
    print("=" * 60)
    print("ACE-Step DiT Generator - Test")
    print("=" * 60)
    
    try:
        # Initialize
        print("\n1. Initializing generator...")
        generator = ACEStepDiTGenerator(device="cpu", variant="turbo")
        print("✅ Generator initialized")
        
        # Create test control tensor
        print("\n2. Creating test control tensor...")
        control_tensor = torch.randn(64)
        print(f"Control tensor shape: {control_tensor.shape}")
        
        # Generate audio
        print("\n3. Generating audio (this may take 10-30 seconds)...")
        audio, sr = generator.generate(control_tensor, duration_seconds=10)
        print(f"✅ Generated audio: {len(audio)/sr:.2f}s at {sr}Hz")
        
        # Save test audio
        import soundfile as sf
        output_path = "test_acestep_output.wav"
        sf.write(output_path, audio, sr)
        print(f"✅ Saved to: {output_path}")
        
        print("\n" + "=" * 60)
        print("TEST PASSED - ACE-Step DiT is working!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_dit_generator()
