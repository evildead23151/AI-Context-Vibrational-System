"""
Module 4: Latent Audio Planner
Uses ACE-Step Language Model to output latent control tensors (NOT music parameters).

CRITICAL: The planner outputs U_t ∈ ℝᵐ with NO named dimensions.
The meaning is learned jointly with the generator via reward gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


class LatentAudioPlanner(nn.Module):
    """
    ACE-Step LM-based planner that outputs latent control tensors.
    
    The planner learns to navigate latent control space via URM rewards.
    It does NOT reason about music - only about control tensor movements.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.context_dim = config.latent_dims.context_embedding
        self.control_dim = config.latent_dims.control_tensor
        
        # ACE-Step LM model path
        self.lm_size = config.model.lm_size
        self.model_name = f"ACE-Step/acestep-5Hz-lm-{self.lm_size}"
        
        # Load ACE-Step LM (will be initialized externally)
        self.ace_lm = None
        self.ace_hidden_dim = None
        
        # Adapter layers (context → LM input space)
        # These will be initialized after loading the LM
        self.context_adapter = None
        self.control_adapter = None
        
        # Control history buffer
        self.control_history = []
        self.max_history = 10
        
        # Latent control projection (LM hidden → control tensor)
        # Initialized after loading LM
        self.control_projector = None
        
    def load_ace_lm(self, weights_cache_path):
        """
        Load ACE-Step Language Model from cache or download.
        
        Args:
            weights_cache_path: Path to cache ACE-Step weights
        """
        print(f"Loading ACE-Step LM: {self.model_name}")
        
        # Check if running locally or need to download
        cache_dir = os.path.join(weights_cache_path, f"lm-{self.lm_size}")
        
        try:
            # Try loading from cache
            self.ace_lm = AutoModelForCausalLM.from_pretrained(
                cache_dir,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            print(f"Loaded from cache: {cache_dir}")
        except:
            # Download from HuggingFace
            print(f"Downloading from HuggingFace: {self.model_name}")
            self.ace_lm = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
        
        # Freeze LM weights (we only train adapters)
        for param in self.ace_lm.parameters():
            param.requires_grad = False
        
        # Get hidden dimension
        self.ace_hidden_dim = self.ace_lm.config.hidden_size
        
        # Initialize adapter layers
        self.context_adapter = nn.Sequential(
            nn.Linear(self.context_dim, self.ace_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.ace_hidden_dim)
        )
        
        self.control_adapter = nn.Sequential(
            nn.Linear(self.control_dim, self.ace_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.ace_hidden_dim)
        )
        
        # Control projector (LM output → latent control tensor)
        self.control_projector = nn.Sequential(
            nn.Linear(self.ace_hidden_dim, self.control_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.control_dim * 2, self.control_dim),
            nn.Tanh()  # Bound control tensor to [-1, 1]
        )
        
        print(f"ACE-Step LM loaded. Hidden dim: {self.ace_hidden_dim}")
    
    def forward(self, z_context, u_prev, reward_signal=None):
        """
        Generate next latent control tensor from context and previous control.
        
        Args:
            z_context: [B, context_dim] - latent context embedding
            u_prev: [B, control_dim] - previous control tensor
            reward_signal: [B, 1] - optional reward from URM (for RL)
        
        Returns:
            u_next: [B, control_dim] - next latent control tensor
            lm_hidden: [B, ace_hidden_dim] - LM hidden state (for gradient flow)
        """
        if self.ace_lm is None:
            raise RuntimeError("ACE-Step LM not loaded. Call load_ace_lm() first.")
        
        batch_size = z_context.shape[0]
        device = z_context.device
        
        # Adapt context to LM input space
        context_token = self.context_adapter(z_context)  # [B, hidden_dim]
        
        # Adapt previous control to LM input space
        control_token = self.control_adapter(u_prev)  # [B, hidden_dim]
        
        # Optional: inject reward signal as conditioning
        if reward_signal is not None:
            # Simple reward embedding (learned)
            reward_embedding = reward_signal * 0.1  # Scale factor
            context_token = context_token + reward_embedding
        
        # Construct input sequence for LM
        # Format: [context_token, control_token, <predict_next>]
        input_sequence = torch.stack([context_token, control_token], dim=1)  # [B, 2, hidden_dim]
        
        # Pass through ACE-Step LM
        with torch.no_grad():
            lm_outputs = self.ace_lm.forward_features(input_sequence)
        
        # Extract last hidden state
        lm_hidden = lm_outputs[:, -1, :]  # [B, hidden_dim]
        
        # Project to latent control tensor (trainable)
        u_next = self.control_projector(lm_hidden)  # [B, control_dim]
        
        return u_next, lm_hidden
    
    def forward_features(self, input_sequence):
        """
        Compatibility method for ACE-Step LM.
        Passes input sequence through LM and returns hidden states.
        """
        # Use the LM's internal forward
        outputs = self.ace_lm(
            inputs_embeds=input_sequence,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Return hidden states
        return outputs.hidden_states[-1]
    
    def update_control(self, z_context, u_prev, reward_signal):
        """
        Update control tensor based on reward gradient.
        This is the RL update step.
        
        Args:
            z_context: Current latent context
            u_prev: Previous control tensor
            reward_signal: Reward from URM
        
        Returns:
            u_next: Updated control tensor
        """
        # Forward pass with gradient tracking
        u_next, lm_hidden = self.forward(z_context, u_prev, reward_signal)
        
        # Store in history
        self.control_history.append(u_next.detach().clone())
        if len(self.control_history) > self.max_history:
            self.control_history.pop(0)
        
        return u_next
    
    def compute_control_delta(self, u_next, u_prev):
        """
        Compute L2 norm of control change.
        Used for stability constraints.
        """
        delta = torch.norm(u_next - u_prev, p=2, dim=-1)
        return delta
    
    def clamp_control_delta(self, u_next, u_prev, max_delta):
        """
        Clamp control change to maximum delta (stability).
        
        Args:
            u_next: Proposed next control
            u_prev: Previous control
            max_delta: Maximum allowed L2 change
        
        Returns:
            u_clamped: Clamped control tensor
        """
        delta = u_next - u_prev
        delta_norm = torch.norm(delta, p=2, dim=-1, keepdim=True)
        
        # Clamp if exceeds max
        scale = torch.clamp(max_delta / (delta_norm + 1e-8), max=1.0)
        delta_clamped = delta * scale
        
        u_clamped = u_prev + delta_clamped
        return u_clamped
    
    def get_control_trajectory(self):
        """Return recent control history (for visualization/debugging)"""
        if not self.control_history:
            return None
        return torch.stack(self.control_history)


class LatentPlannerTrainer:
    """
    Trainer for the latent audio planner.
    Uses policy gradient from URM rewards.
    """
    
    def __init__(self, planner, urm, config, device):
        self.planner = planner.to(device)
        self.urm = urm.to(device)
        self.config = config
        self.device = device
        
        # Only optimize adapter and projector (LM is frozen)
        trainable_params = [
            p for p in planner.parameters() if p.requires_grad
        ]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.adaptation.planner_lr,
            weight_decay=0.01
        )
    
    def update_step(self, z_context, u_prev, behavior_signals):
        """
        Single RL update step.
        
        Args:
            z_context: Latent context
            u_prev: Previous control tensor
            behavior_signals: Implicit behavioral feedback
        
        Returns:
            loss: Scalar loss
            u_next: New control tensor
        """
        # Generate new control tensor
        u_next, lm_hidden = self.planner.forward(z_context, u_prev)
        
        # Get reward estimate from URM
        reward = self.urm.forward(z_context, u_next, behavior_signals)
        
        # Policy gradient: maximize reward
        loss = -reward.mean()
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.planner.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item(), u_next.detach()
