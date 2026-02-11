# Quick Start Guide

## Installation

### 1. Clone ACE-Step Repository

```bash
cd "c:\Users\Gitesh malik\Documents\Backup\OneDrive\Desktop\Folders\Projects\Anti-Gravity\AI\1"
git clone https://github.com/ace-step/ACE-Step-1.5
```

### 2. Install Dependencies

```bash
# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install system dependencies
pip install -r requirements.txt

# Install ACE-Step dependencies
cd ACE-Step-1.5
pip install -r requirements.txt
cd ..
```

### 3. Download ACE-Step Models

The system will auto-download models on first run, but you can pre-download:

```bash
# Using huggingface-cli
pip install huggingface-hub

# Download LM (1.7B recommended)
huggingface-cli download ACE-Step/acestep-5Hz-lm-1.7b --local-dir ./weights_cache/lm-1.7b

# Download DiT (turbo recommended)
huggingface-cli download ACE-Step/Ace-Step1.5 --local-dir ./weights_cache/dit-turbo
```

---

## Usage

### Basic Example

```python
from adaptive_audio_system import create_system

# Create system
system = create_system(
    lm_model_size="1.7b",  # or "0.6b", "4b"
    dit_variant="turbo",    # or "sft", "base"
    device="cuda"           # or "cpu"
)

# Ingest sensor data
sensor_data = {
    'gps': [40.7128, -74.0060],  # NYC coordinates
    'wifi_bssids': ['aa:bb:cc:dd:ee:ff', '11:22:33:44:55:66'],
    'bluetooth_rssi': [-60, -70, -65],
    'accelerometer': [0.1, 0.2, 9.8],
    'gyroscope': [0.01, 0.02, 0.01],
    'audio_level': [45.0],
    'screen_interaction': {
        'last_interaction_delta': 30.0,
        'interaction_count': 10,
        'is_locked': False,
        'foreground_entropy': 1.5
    }
}

system.ingest_sensor_data(sensor_data)

# Start adaptive audio generation
system.start()

# Record behavioral signals
system.record_behavior('session_active', 1.0)
system.record_behavior('engagement', 0.9)

# Let it run...
import time
time.sleep(300)  # 5 minutes

# Stop
system.stop()
```

---

## Simulated Demo (No Real Sensors)

```bash
# Run example with simulated data
python example_usage.py
```

This will:
1. Create the system
2. Simulate sensor data
3. Simulate user behavior
4. Demonstrate continuous adaptation
5. Save/load state

---

## Verification

```bash
# Verify system compliance
python verify_system.py
```

Expected output:
```
✅ ALL CHECKS PASSED - SYSTEM VALID
```

---

## Configuration

Edit `adaptive_audio_system/config.py` to customize:

```python
from adaptive_audio_system.config import SystemConfig

config = SystemConfig()

# Adjust latent dimensions
config.latent_dims.context_embedding = 512
config.latent_dims.control_tensor = 256

# Change update frequency
config.adaptation.update_interval_seconds = 300  # 5 minutes

# Select models
config.model.lm_size = "1.7b"
config.model.dit_variant = "turbo"

# Create system with custom config
system = create_system(device="cuda")
system.config = config
```

---

## Production Deployment

### Step 1: Pre-train Latent Context Model

```python
from adaptive_audio_system.modules.latent_context_model import LatentContextTrainer
from adaptive_audio_system import create_system

system = create_system()
trainer = LatentContextTrainer(system.lcm, system.config, device="cuda")

# Collect sensor data over days/weeks
for epoch in range(100):
    for batch in sensor_data_loader:  # Your data
        loss = trainer.train_step(batch)
        print(f"Epoch {epoch}, Loss: {loss}")

# Save
system.save_state("./pretrained_state")
```

### Step 2: Deploy to Users

```python
# On user device
system = create_system()
system.load_state("./pretrained_state")
system.start()

# System adapts online from user behavior
```

### Step 3: Monitor

```python
# Get system state
state = system.get_system_state()
print(f"Running: {state['running']}")
print(f"Context: {state['z_context'].shape}")
print(f"Control: {state['u_current'].shape}")
```

---

## Troubleshooting

### Issue: Out of Memory

**Solution**: Use smaller models
```python
system = create_system(
    lm_model_size="0.6b",  # Smaller LM
    dit_variant="turbo",
    device="cpu"           # Use CPU
)
```

### Issue: Slow Generation

**Solution**: Use GPU and turbo variant
```python
system = create_system(
    dit_variant="turbo",  # Fastest (8 steps)
    device="cuda"
)
```

### Issue: ACE-Step Models Not Found

**Solution**: Manually download
```bash
cd weights_cache
git lfs install
git clone https://huggingface.co/ACE-Step/acestep-5Hz-lm-1.7b lm-1.7b
git clone https://huggingface.co/ACE-Step/Ace-Step1.5 dit-turbo
```

---

## API Reference

### AdaptiveAudioSystem

```python
system = create_system(lm_model_size, dit_variant, device)

# Methods
system.ingest_sensor_data(sensor_data: dict)
system.record_behavior(signal_type: str, value: float)
system.start()
system.stop()
system.get_system_state() -> dict
system.save_state(path: str)
system.load_state(path: str)
```

### Sensor Data Format

```python
sensor_data = {
    'gps': [lat: float, lon: float],
    'wifi_bssids': [str, ...],
    'bluetooth_rssi': [float, ...],
    'accelerometer': [x: float, y: float, z: float],
    'gyroscope': [x: float, y: float, z: float],
    'audio_level': [float],
    'screen_interaction': {
        'last_interaction_delta': float,
        'interaction_count': int,
        'is_locked': bool,
        'foreground_entropy': float
    }
}
```

### Behavioral Signals

```python
# Signal types (opaque identifiers)
'session_active'      # 1.0 if active, 0.0 if inactive
'volume_delta'        # Change in volume (-1.0 to 1.0)
'interruption'        # 1.0 for each interruption
'earphone_removed'    # 1.0 if removed
'lock_duration'       # Seconds locked
'skip'                # 1.0 for each skip
'engagement'          # 0.0 to 1.0 engagement level
```

---

## Next Steps

1. **Collect Real Data**: Replace simulated sensors with actual device APIs
2. **Integrate Playback**: Connect audio output to system audio/speakers
3. **Deploy at Scale**: Set up cloud inference for multiple users
4. **Monitor Performance**: Track reward trends, user engagement
5. **Iterate**: Collect feedback, retrain models, improve

---

## Support

- **Documentation**: See `ARCHITECTURE.md` for technical details
- **Contract**: See `SYSTEM_CONTRACT.md` for compliance rules
- **Examples**: See `example_usage.py` for code samples
- **Verification**: Run `verify_system.py` to check compliance

---

## License

MIT License - see LICENSE file for details
