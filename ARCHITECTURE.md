# Technical Architecture Documentation

## System Overview

The Adaptive Audio Regulation System is a production-grade closed-loop control system that generates and adapts audio using **pure latent-space control** with zero human-interpretable parameters.

### Core Innovation

Traditional music generation systems use human-designed parameters (tempo, key, genre, mood). This system **eliminates all semantic representations** and instead:

1. **Learns** context patterns from raw sensor data (GPS, WiFi, motion, etc.)
2. **Infers** user response from implicit behavior (session duration, volume changes, skips)
3. **Plans** in learned latent control space (opaque tensors with no named dimensions)
4. **Generates** audio via ACE-Step models conditioned solely on latent controls
5. **Adapts** continuously via online reinforcement learning

### Control Flow

```
┌─────────────────┐
│  Raw Sensors    │ GPS, WiFi, Accel, Gyro, Audio, Screen
│  (Multi-modal)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Raw Signal      │ Normalize, buffer, no interpretation
│ Ingestion       │ Output: [T, D] normalized tensor
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Latent Context  │ Self-supervised sequence encoder
│ Model (LCM)     │ Transformer/S4, predictive training
└────────┬────────┘ Output: Z_context ∈ ℝ⁵¹²
         │
         │         ┌─────────────────┐
         │         │ User Response   │ Online RL, implicit signals
         │         │ Model (URM)     │ Session, volume, skips, engagement
         │         └────────┬────────┘
         │                  │ Reward signal
         │                  │
         ▼                  ▼
┌─────────────────────────────┐
│ Latent Audio Planner        │ ACE-Step LM (0.6B/1.7B/4B)
│ (ACE-Step LM)               │ Input: Z_context, U_prev, reward
│                             │ Output: U_next ∈ ℝ²⁵⁶ (opaque)
└────────┬────────────────────┘
         │ Latent control tensor U_t
         │
         ▼
┌─────────────────┐
│ Audio Generator │ ACE-Step DiT (turbo/sft/base)
│ (ACE-Step DiT)  │ Conditioned ONLY on U_t
└────────┬────────┘ Output: Audio waveform
         │
         ▼
┌─────────────────┐
│  Audio Output   │ Continuous stream, cross-fade overlap
│  (Playback)     │
└────────┬────────┘
         │
         │ Implicit behavioral feedback
         │
         └──────────────────┐
                            │
                  ┌─────────▼────────┐
                  │ Adaptation Loop  │ minutes-scale updates
                  │ (Control Theory) │ Stability constraints
                  └──────────────────┘
```

## Module Details

### Module 1: Raw Signal Ingestion

**Purpose**: Capture reality without interpretation

**Inputs**:
- GPS coordinates (float32 × 2)
- WiFi BSSID hashes (uint64, embedded to ℝ¹⁶)
- Bluetooth RSSI time series (float32 × 8, statistical features)
- Accelerometer (float32 × 3)
- Gyroscope (float32 × 3)
- Ambient audio level (float32)
- Screen interaction features (float32 × 4)
- Time encoding (cyclic, float32 × 4)

**Processing**:
- Hash-based embeddings for categorical signals
- Statistical aggregation for time series
- Cyclic encoding for temporal signals
- Online normalization (Welford's algorithm)

**Output**: 
- `RawSignalTensor` with shape `[sequence_len, feature_dim]`
- Feature dim = 2 + 16 + 8 + 3 + 3 + 1 + 4 + 4 = 41

**Key Property**: Zero semantic interpretation - pure signal encoding

---

### Module 2: Latent Context Model (LCM)

**Purpose**: Learn recurrent patterns of existence without naming them

**Architecture**:
- **Input Projection**: Linear(41 → 512)
- **Positional Encoding**: Learned, shape [4096, 512]
- **Temporal Encoder**: 
  - Option A: Transformer (6 layers, 8 heads, 512 dim)
  - Option B: Temporal Conv (6 layers, dilated)
- **Latent Projection**: Linear(512 → 512) + GELU + Linear(512 → 512)
- **Output**: Z_context ∈ ℝ⁵¹² (opaque embedding)

**Training**: Self-supervised
- **Objective 1**: Predict future inputs from past (autoregressive)
- **Objective 2**: Contrastive learning (optional, SimCLR-style)
- **Loss**: MSE on predicted future states

**Parameters**: ~15M (Transformer) or ~8M (Conv)

**Key Property**: Z_context has no semantic meaning, never decoded to labels

---

### Module 3: User Response Model (URM)

**Purpose**: Learn how nervous system reacts to sound under contexts

**Implicit Behavioral Signals** (NO explicit feedback):
1. `session_active` - Binary continuation signal
2. `volume_delta` - Volume adjustment magnitude
3. `interruption` - Count of interruptions
4. `earphone_removed` - Binary removal event
5. `lock_duration` - Device lock time
6. `return_probability` - Inferred from return intervals
7. `skip` - Count of skip events
8. `engagement` - Time since last interaction

**Architecture**:
- **Behavior Encoder**: Linear(8 → 128) + GELU + Linear(128 → 128)
- **Reward Net**: Linear(512 + 256 + 128 → 256) + GELU + Linear(256 → 1)
- **Value Net**: Linear(512 + 128 → 128) + GELU + Linear(128 → 1)

**Training**: Online RL
- **Trajectory Buffer**: Recent (context, control, behavior) tuples
- **Proxy Reward**: Learned associations from implicit signals
  - Session continuation → positive
  - Volume increase → positive
  - Skips/interruptions → negative
- **Update**: Policy gradient with advantage estimation
- **Frequency**: Every 5+ trajectory steps

**Parameters**: ~500K

**Key Property**: Learns reward from behavior, not labels

---

### Module 4: Latent Audio Planner

**Purpose**: Navigate latent control space to maximize URM reward

**Architecture**:

**ACE-Step LM** (frozen):
- Size options: 0.6B, 1.7B, 4B parameters
- Hidden dim: 768, 1024, or 2048 (model-dependent)
- Source: `ACE-Step/acestep-5Hz-lm-{size}`

**Adapter Layers** (trainable):
- **Context Adapter**: Linear(512 → hidden_dim) + GELU + LayerNorm
- **Control Adapter**: Linear(256 → hidden_dim) + GELU + LayerNorm
- **Control Projector**: Linear(hidden_dim → 512) + GELU + Linear(512 → 256) + Tanh

**Forward Pass**:
```python
# Inputs
z_context in R^512       # From LCM
u_prev in R^256          # Previous control
reward in R^1            # From URM (optional)

# Adapt to LM space
context_token = context_adapter(z_context)  # R^hidden_dim
control_token = control_adapter(u_prev)     # R^hidden_dim

# Condition on reward
if reward:
    context_token += reward * 0.1

# LM forward (frozen)
sequence = stack([context_token, control_token])  # [B, 2, hidden_dim]
lm_output = ACE_LM(inputs_embeds=sequence)        # [B, 2, hidden_dim]
lm_hidden = lm_output[:, -1, :]                    # [B, hidden_dim]

# Project to control tensor (trainable)
u_next = control_projector(lm_hidden)  # R^256
```

**Training**: Policy gradient
- **Objective**: Maximize URM reward at next control
- **Loss**: -reward(z_context, u_next)
- **Optimizer**: AdamW, lr=1e-5
- **Constraints**: 
  - Max L2 delta = 0.1 (stability)
  - Abort if reward drop > -0.5

**Parameters**: 
- Frozen LM: 0.6B / 1.7B / 4B
- Trainable adapters: ~2M

**Key Property**: U_next has NO named dimensions - pure latent control

---

### Module 5: Audio Generator

**Purpose**: Generate audio from latent control tensors

**Architecture**:

**ACE-Step DiT** (frozen):
- Variant options: turbo (8 steps), sft (50 steps), base (50 steps)
- Hidden dim: 768
- Source: `ACE-Step/acestep-v15-{variant}`
- Supports continuous generation with latent state carryover

**Conditioning Adapter** (trainable):
- **Control-to-Cond**: Linear(256 → 768) + GELU + LayerNorm + Linear(768 → 768) + GELU + LayerNorm

**Forward Pass**:
```python
# Inputs
u_t in R^256           # Latent control from planner
prev_latent (optional) # Previous audio latent (for continuity)

# Adapt control to DiT conditioning
conditioning = control_to_cond(u_t)  # R^768

# Generate audio via diffusion (frozen)
audio, next_latent = ACE_DiT.generate(
    conditioning=conditioning,
    num_samples=sample_rate * duration,
    num_steps=8,  # for turbo
    prev_latent=prev_latent
)
```

**Generation**:
- **Sample Rate**: 44.1kHz
- **Chunk Duration**: 60 seconds
- **Overlap**: 5 seconds (cross-faded)
- **Continuity**: Uses previous latent state

**Parameters**:
- Frozen DiT: ~1B
- Trainable adapter: ~1M

**Key Property**: Conditioned ONLY by latent U_t, no semantic parameters

---

### Module 6: Continuous Adaptation Loop

**Purpose**: Closed-loop control with online learning

**Control Theory**:
- **Update Interval**: 300 seconds (5 minutes)
- **Control Law**: Policy gradient on URM reward
- **Stability Constraints**:
  - Max control delta: ||U_next - U_prev||₂ ≤ 0.1
  - Reward drop abort: Δreward < -0.5

**Loop Cycle** (every 60 seconds):
1. **Update Context**: Encode recent raw signals → Z_context
2. **Generate Audio**: U_current → audio chunk (60s)
3. **Collect Behavior**: Drain behavior queue → implicit signals
4. **Update URM**: Record trajectory, update if ≥5 steps
5. **Check Timer**: If 300s elapsed → update control

**Control Update** (every 5 minutes):
1. Get current reward: r_current = URM(Z_context, U_current, behavior)
2. Propose new control: U_proposed = Planner(Z_context, U_current, r_current)
3. Check delta: Δ = ||U_proposed - U_current||₂
4. If Δ > 0.1 → clamp to max delta
5. Estimate new reward: r_proposed = URM(Z_context, U_proposed, behavior)
6. If r_proposed - r_current < -0.5 → abort update
7. Otherwise: Backprop -r_proposed, update planner adapters
8. Commit: U_current ← U_proposed

**Threading**:
- Main loop runs in separate daemon thread
- Behavior signals queued from external threads
- Thread-safe state access

**Optimizers**:
- Planner: AdamW, lr=1e-5
- URM: AdamW, lr=1e-4

---

## Training Procedures

### Phase 1: Self-Supervised Pre-Training (LCM)

**Data**: Raw sensor logs (no labels)
**Duration**: Days to weeks
**Procedure**:
1. Collect continuous sensor data streams
2. Chunk into sequences (1 hour windows)
3. Train LCM to predict future from past
4. Monitor: Prediction MSE (should decrease)
5. Save best checkpoint

**Expected**: MSE reduction by 2-3x

---

### Phase 2: Adapter Fine-Tuning (Planner + Generator)

**Data**: Paired (control, audio) samples (if available) or random initialization
**Duration**: Hours to days
**Procedure**:

**Option A: Supervised** (if have labeled data)
1. Feed known control tensors through generator
2. Compute perceptual loss vs target audio
3. Update adapter weights

**Option B: Self-Supervised** (no labels)
1. Generate audio with random controls
2. Compute diversity + quality losses (spectral, temporal)
3. Update adapter weights

**Expected**: Coherent audio generation

---

### Phase 3: Online RL (System Deployment)

**Data**: User interactions (implicit signals)
**Duration**: Continuous (weeks to months)
**Procedure**:
1. Deploy system to users
2. Collect trajectory: (Z_context, U_t, behavior)
3. Update URM every 5 steps
4. Update Planner every 5 minutes
5. Monitor: Reward trend (should increase)

**Expected**: Per-user personalization, reward increase over time

---

## Deployment Architecture

### On-Device Deployment

**Hardware Requirements**:
- CPU: 4+ cores
- RAM: 8+ GB
- Storage: 10+ GB (for models)
- GPU: Optional (for faster inference)
  - Min: 4GB VRAM (RTX 2060)
  - Recommended: 8GB VRAM (RTX 3070)

**Software Stack**:
- Python 3.11+
- PyTorch 2.0+
- ACE-Step models (downloaded locally)

**Directory Structure**:
```
~/.adaptive_audio/
├── weights_cache/        # ACE-Step models
│   ├── lm-1.7b/
│   └── dit-turbo/
├── user_state/           # Per-user state
│   ├── system_state.pt   # Model checkpoints
│   └── signal_normalizers.pkl
└── logs/                 # System logs
```

---

### Cloud Deployment

**Architecture**:
```
User Device (Edge)          Cloud (Backend)
├── Sensor Ingestion  ────▶ ├── Model Serving
├── Local LCM               │   └── ACE-Step models (GPU)
└── Playback           ◀──── ├── Inference API
                             └── State Sync
```

**API Endpoints**:
- `POST /encode_context` - Encode raw signals → Z_context
- `POST /plan_control` - Z_context + U_prev → U_next
- `POST /generate_audio` - U_t → audio chunk
- `POST /update_models` - Sync adapted weights

**Scaling**:
- GPU instances for ACE-Step inference
- Redis for state caching
- PostgreSQL for trajectory logging

---

## Performance Characteristics

### Latency

**Per-module** (on RTX 3090):
- Raw Signal Ingestion: <1ms
- LCM Encoding: ~10ms (Transformer) / ~5ms (Conv)
- URM Forward: ~2ms
- Planner Forward: ~50ms (1.7B LM)
- Generator: ~8s (turbo, 60s audio) / ~30s (sft)

**End-to-end** (60s audio chunk):
- Cold start: ~10s
- Warm (with cache): ~8s

**Update overhead**:
- URM update: ~20ms
- Planner update: ~100ms

---

### Memory

**Model Sizes**:
- LCM: 60MB (15M params)
- URM: 2MB (0.5M params)
- Planner adapters: 8MB (2M params)
- Generator adapter: 4MB (1M params)
- ACE-Step LM (frozen): 700MB / 4GB / 8GB (0.6B / 1.7B / 4B)
- ACE-Step DiT (frozen): 3GB

**Total** (1.7B LM + turbo):
- Models: ~8 GB
- Runtime buffers: ~2 GB
- **Total**: ~10 GB

**Min Config** (0.6B LM + turbo, CPU):
- ~5 GB total

---

### Throughput

**Batch Generation**: Up to 8 parallel streams (GPU-dependent)

**User Capacity**:
- Single GPU: 10-20 concurrent users
- CPU-only: 2-3 concurrent users

---

## Monitoring & Observability

### Key Metrics

**System Health**:
- Loop uptime
- Update frequency (should match config)
- Control delta magnitude (should be < 0.1)
- Reward trend (should increase over time)

**Model Performance**:
- LCM prediction loss (lower = better context)
- URM reward estimation error
- Planner policy gradient
- Generator audio quality (perceptual metrics)

**User Engagement** (implicit):
- Session duration
- Volume stability
- Skip rate
- Return frequency

### Logging

**Levels**:
- INFO: Normal operations (updates, state changes)
- WARNING: Constraint violations (delta clamp, reward abort)
- ERROR: System failures (model load, inference errors)

**Artifacts**:
- Control trajectory plots
- Reward curves
- Audio spectrograms (for debugging)

---

## Future Enhancements

1. **Multi-modal Context**: Vision (camera), text (calendar)
2. **Hierarchical Planning**: Long-term (hours) + short-term (minutes)
3. **Multi-task Learning**: Speech, ambient, sound effects
4. **Federated Learning**: Cross-user knowledge transfer (privacy-preserving)
5. **Compression**: Quantization, pruning for mobile deployment

---

## References

**ACE-Step**:
- Paper: https://arxiv.org/abs/2602.00744
- Code: https://github.com/ace-step/ACE-Step-1.5
- Models: https://huggingface.co/ACE-Step

**RL**:
- Policy Gradient: Sutton & Barto, "Reinforcement Learning"
- Implicit Feedback: "Learning from Implicit Feedback" (various)

**Self-Supervised Learning**:
- Contrastive: SimCLR, MoCo
- Predictive: BERT, GPT (autoregressive)
