"""
Module 3: User Response Model (URM)
Learns how the specific nervous system reacts to sound under latent contexts.
Uses ONLY implicit behavioral signals - no explicit feedback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import numpy as np


class ImplicitBehavior:
    """Container for implicit behavioral signals (no semantic labels)"""
    
    def __init__(self):
        self.signals = {}
    
    def record(self, signal_type: str, value: float):
        """Record an implicit signal (opaque, no interpretation)"""
        if signal_type not in self.signals:
            self.signals[signal_type] = []
        self.signals[signal_type].append(value)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to fixed-size feature vector"""
        features = []
        
        # Session continuation (binary)
        features.append(self.signals.get('session_active', [0.0])[-1])
        
        # Volume adjustment delta (continuous)
        vol_deltas = self.signals.get('volume_delta', [0.0])
        features.append(np.mean(vol_deltas) if vol_deltas else 0.0)
        
        # Interruption events (count)
        interruptions = self.signals.get('interruption', [])
        features.append(float(len(interruptions)))
        
        # Earphone removal (binary)
        features.append(self.signals.get('earphone_removed', [0.0])[-1])
        
        # Device lock duration (continuous)
        lock_duration = self.signals.get('lock_duration', [0.0])
        features.append(np.mean(lock_duration) if lock_duration else 0.0)
        
        # Return probability (inferred from gaps)
        return_times = self.signals.get('return_time', [])
        features.append(1.0 / (np.mean(return_times) + 1.0) if return_times else 0.5)
        
        # Playback skip events
        skips = self.signals.get('skip', [])
        features.append(float(len(skips)))
        
        # Engagement proxy (time since last interaction)
        engagement = self.signals.get('engagement', [1.0])
        features.append(engagement[-1])
        
        return torch.tensor(features, dtype=torch.float32)


class UserResponseModel(nn.Module):
    """
    Online RL model that learns reward from implicit behavior.
    No explicit reward signal - all inference from behavioral patterns.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        context_dim = config.latent_dims.context_embedding
        control_dim = config.latent_dims.control_tensor
        behavior_dim = config.latent_dims.behavioral_state
        
        # Behavior feature dimension (from ImplicitBehavior.to_tensor)
        self.behavior_input_dim = 8
        
        # Behavior encoder
        self.behavior_encoder = nn.Sequential(
            nn.Linear(self.behavior_input_dim, behavior_dim),
            nn.GELU(),
            nn.Linear(behavior_dim, behavior_dim)
        )
        
        # Reward prediction network
        # Inputs: z_context, control_tensor, behavioral_state
        self.reward_net = nn.Sequential(
            nn.Linear(context_dim + control_dim + behavior_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)  # Scalar reward estimate
        )
        
        # Value network (for advantage estimation)
        self.value_net = nn.Sequential(
            nn.Linear(context_dim + behavior_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        # Policy gradient buffer
        self.trajectory_buffer = []
        
    def forward(self, z_context, control_tensor, behavior_signals):
        """
        Predict reward from context, control, and implicit behavior.
        
        Args:
            z_context: [B, context_dim] latent context
            control_tensor: [B, control_dim] latent audio control
            behavior_signals: ImplicitBehavior or tensor [B, behavior_input_dim]
        
        Returns:
            reward_estimate: [B, 1] scalar reward
        """
        # Encode behavior if needed
        if isinstance(behavior_signals, ImplicitBehavior):
            behavior_tensor = behavior_signals.to_tensor().unsqueeze(0)
        else:
            behavior_tensor = behavior_signals
        
        behavior_tensor = behavior_tensor.to(z_context.device)
        
        # Encode behavior
        behavior_state = self.behavior_encoder(behavior_tensor)  # [B, behavior_dim]
        
        # Concatenate all inputs
        combined = torch.cat([z_context, control_tensor, behavior_state], dim=-1)
        
        # Predict reward
        reward = self.reward_net(combined)  # [B, 1]
        
        return reward
    
    def estimate_value(self, z_context, behavior_signals):
        """
        Estimate state value (for advantage computation).
        
        Args:
            z_context: [B, context_dim]
            behavior_signals: ImplicitBehavior or tensor
        
        Returns:
            value: [B, 1]
        """
        if isinstance(behavior_signals, ImplicitBehavior):
            behavior_tensor = behavior_signals.to_tensor().unsqueeze(0)
        else:
            behavior_tensor = behavior_signals
        
        behavior_tensor = behavior_tensor.to(z_context.device)
        behavior_state = self.behavior_encoder(behavior_tensor)
        
        combined = torch.cat([z_context, behavior_state], dim=-1)
        value = self.value_net(combined)
        
        return value
    
    def record_trajectory(self, z_context, control_tensor, behavior_signals):
        """Record trajectory step for policy gradient update"""
        self.trajectory_buffer.append({
            'z_context': z_context.detach(),
            'control_tensor': control_tensor.detach(),
            'behavior': behavior_signals
        })
    
    def compute_returns(self, gamma=0.99):
        """
        Compute discounted returns from trajectory buffer.
        Uses implicit behavioral signals as proxy rewards.
        """
        if not self.trajectory_buffer:
            return None
        
        returns = []
        running_return = 0.0
        
        # Convert behavioral signals to proxy rewards
        for step in reversed(self.trajectory_buffer):
            behavior = step['behavior']
            
            # Implicit reward heuristic (NO SEMANTIC LABELS)
            # These are learned associations, not designed rules
            proxy_reward = 0.0
            
            # Tensor-based reward computation (opaque)
            if isinstance(behavior, ImplicitBehavior):
                b_tensor = behavior.to_tensor()
            else:
                b_tensor = behavior
            
            # Session continuation is positive signal
            proxy_reward += b_tensor[0].item() * 1.0
            
            # Volume adjustments (up = positive, down = negative)
            proxy_reward += b_tensor[1].item() * 0.5
            
            # Interruptions are negative
            proxy_reward -= b_tensor[2].item() * 0.3
            
            # Earphone removal is negative
            proxy_reward -= b_tensor[3].item() * 1.0
            
            # Long lock duration is negative
            proxy_reward -= b_tensor[4].item() * 0.2
            
            # Return probability is positive
            proxy_reward += b_tensor[5].item() * 0.5
            
            # Skips are negative
            proxy_reward -= b_tensor[6].item() * 0.3
            
            # Engagement is positive
            proxy_reward += b_tensor[7].item() * 0.3
            
            # Discounted return
            running_return = proxy_reward + gamma * running_return
            returns.insert(0, running_return)
        
        return torch.tensor(returns, dtype=torch.float32)
    
    def update_from_trajectory(self, optimizer):
        """
        Policy gradient update using collected trajectory.
        No explicit labels - learns from implicit behavioral associations.
        """
        if len(self.trajectory_buffer) < 2:
            return None
        
        # Compute returns
        returns = self.compute_returns()
        
        if returns is None:
            return None
        
        # Extract trajectory components
        contexts = torch.stack([s['z_context'] for s in self.trajectory_buffer])
        controls = torch.stack([s['control_tensor'] for s in self.trajectory_buffer])
        behaviors = torch.stack([
            s['behavior'].to_tensor() if isinstance(s['behavior'], ImplicitBehavior)
            else s['behavior']
            for s in self.trajectory_buffer
        ])
        
        device = contexts.device
        returns = returns.to(device)
        behaviors = behaviors.to(device)
        
        # Forward pass
        predicted_rewards = self.forward(contexts, controls, behaviors).squeeze(-1)
        predicted_values = self.estimate_value(contexts, behaviors).squeeze(-1)
        
        # Compute advantages
        advantages = returns - predicted_values.detach()
        
        # Policy gradient loss (reward prediction)
        policy_loss = -torch.mean(predicted_rewards * advantages)
        
        # Value function loss
        value_loss = F.mse_loss(predicted_values, returns)
        
        # Combined loss
        loss = policy_loss + 0.5 * value_loss
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        
        # Clear buffer
        self.trajectory_buffer.clear()
        
        return loss.item()


class BayesianUserResponseModel(nn.Module):
    """
    Alternative: Bayesian preference inference model.
    Learns probabilistic reward model from implicit comparisons.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        context_dim = config.latent_dims.context_embedding
        control_dim = config.latent_dims.control_tensor
        behavior_dim = config.latent_dims.behavioral_state
        
        # Behavior encoder
        self.behavior_encoder = nn.Sequential(
            nn.Linear(8, behavior_dim),
            nn.GELU(),
            nn.Linear(behavior_dim, behavior_dim)
        )
        
        # Preference model (outputs mean and log_std)
        self.preference_net = nn.Sequential(
            nn.Linear(context_dim + control_dim + behavior_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU()
        )
        
        self.mean_head = nn.Linear(128, 1)
        self.log_std_head = nn.Linear(128, 1)
    
    def forward(self, z_context, control_tensor, behavior_signals):
        """
        Predict preference distribution.
        
        Returns:
            mean: [B, 1]
            std: [B, 1]
        """
        if isinstance(behavior_signals, ImplicitBehavior):
            behavior_tensor = behavior_signals.to_tensor().unsqueeze(0)
        else:
            behavior_tensor = behavior_signals
        
        behavior_tensor = behavior_tensor.to(z_context.device)
        behavior_state = self.behavior_encoder(behavior_tensor)
        
        combined = torch.cat([z_context, control_tensor, behavior_state], dim=-1)
        features = self.preference_net(combined)
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample_reward(self, z_context, control_tensor, behavior_signals):
        """Sample from reward distribution"""
        mean, std = self.forward(z_context, control_tensor, behavior_signals)
        reward = torch.normal(mean, std)
        return reward
