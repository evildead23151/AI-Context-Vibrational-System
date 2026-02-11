"""
Module 2: Latent Context Model (LCM)
Self-supervised sequence encoder that learns recurrent patterns without naming them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class LatentContextModel(nn.Module):
    """
    Self-supervised temporal encoder.
    Learns to predict future from past without semantic labels.
    
    Output: Opaque latent embedding Z_context ∈ ℝⁿ
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_dim = config.raw_signals.gps_dims + \
                        config.raw_signals.wifi_hash_dims + \
                        config.raw_signals.bluetooth_rssi_dims + \
                        config.raw_signals.accel_dims + \
                        config.raw_signals.gyro_dims + \
                        config.raw_signals.audio_level_dims + \
                        config.raw_signals.screen_dims + \
                        config.raw_signals.time_dims
        
        self.embed_dim = config.latent_dims.context_embedding
        
        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.embed_dim)
        
        # Positional encoding (learned)
        self.positional_enc = nn.Parameter(
            torch.randn(1, 4096, self.embed_dim) * 0.02
        )
        
        # Temporal encoder
        if config.model.context_encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=config.model.context_encoder_heads,
                dim_feedforward=self.embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.model.context_encoder_layers
            )
        elif config.model.context_encoder_type == "temporal_conv":
            self.encoder = TemporalConvEncoder(
                input_dim=self.embed_dim,
                hidden_dim=self.embed_dim,
                num_layers=config.model.context_encoder_layers
            )
        else:
            raise ValueError(f"Unknown encoder type: {config.model.context_encoder_type}")
        
        # Latent projection (to fixed-size context embedding)
        self.latent_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Self-supervised prediction head (for training)
        self.prediction_head = nn.Linear(self.embed_dim, self.input_dim)
        
        # Contrastive projection (for self-supervised learning)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 128)
        )
    
    def forward(self, raw_signal_tensor, return_prediction=False):
        """
        Encode raw signal sequence to opaque latent context.
        
        Args:
            raw_signal_tensor: RawSignalTensor with .data [B, T, D]
            return_prediction: If True, also return future prediction (for training)
        
        Returns:
            z_context: [B, embed_dim] - opaque latent embedding
            (optional) prediction: [B, T, input_dim] - for self-supervised loss
        """
        x = raw_signal_tensor.data
        
        # Handle single sequence (add batch dim)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        B, T, D = x.shape
        
        # Input projection
        x = self.input_proj(x)  # [B, T, embed_dim]
        
        # Add positional encoding
        x = x + self.positional_enc[:, :T, :]
        
        # Temporal encoding
        if isinstance(self.encoder, nn.TransformerEncoder):
            # Transformer encoder
            encoded = self.encoder(x)  # [B, T, embed_dim]
        else:
            # Temporal conv
            encoded = self.encoder(x)  # [B, T, embed_dim]
        
        # Pool over time (mean pooling for permutation invariance)
        pooled = encoded.mean(dim=1)  # [B, embed_dim]
        
        # Project to latent context
        z_context = self.latent_proj(pooled)  # [B, embed_dim]
        
        if return_prediction:
            # Self-supervised prediction of future input
            prediction = self.prediction_head(encoded)  # [B, T, input_dim]
            return z_context, prediction
        
        return z_context
    
    def contrastive_forward(self, raw_signal_tensor):
        """
        Forward pass for contrastive learning.
        Used during self-supervised pre-training.
        """
        z_context = self.forward(raw_signal_tensor)
        z_contrast = self.contrastive_proj(z_context)
        return F.normalize(z_contrast, dim=-1)
    
    def compute_self_supervised_loss(self, raw_signal_tensor, future_signal_tensor=None):
        """
        Compute self-supervised loss (predictive or contrastive).
        
        Args:
            raw_signal_tensor: Current context window
            future_signal_tensor: Future window (for predictive loss)
        
        Returns:
            loss: Scalar loss for backprop
        """
        # Predictive loss: predict future input from past
        z_context, prediction = self.forward(raw_signal_tensor, return_prediction=True)
        
        if future_signal_tensor is not None:
            # Predict future inputs
            target = future_signal_tensor.data
            if target.dim() == 2:
                target = target.unsqueeze(0)
            
            # MSE on predictions
            pred_loss = F.mse_loss(prediction, target)
        else:
            # Autoregressive: predict next step from previous
            target = raw_signal_tensor.data[:, 1:, :]  # Shift by 1
            pred = prediction[:, :-1, :]
            pred_loss = F.mse_loss(pred, target)
        
        # Optional: Add contrastive loss for better representations
        # (SimCLR-style with augmented views)
        
        return pred_loss


class TemporalConvEncoder(nn.Module):
    """
    Temporal convolutional encoder (alternative to transformer).
    Uses dilated convolutions for long-range dependencies.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                nn.Conv1d(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation
                )
            )
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, D]
        Returns:
            out: [B, T, D]
        """
        # Conv1d expects [B, D, T]
        x = x.transpose(1, 2)
        
        for layer in self.layers:
            if isinstance(layer, nn.Conv1d):
                x = layer(x)
            elif isinstance(layer, nn.GELU):
                x = layer(x)
            elif isinstance(layer, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = layer(x)
                x = x.transpose(1, 2)
        
        # Back to [B, T, D]
        x = x.transpose(1, 2)
        return x


class LatentContextTrainer:
    """
    Trainer for self-supervised pre-training of LCM.
    No labels required - learns from temporal structure.
    """
    
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.adaptation.context_lr,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10000
        )
    
    def train_step(self, raw_signal_batch):
        """
        Single training step on raw signal batch.
        No external labels needed.
        """
        self.model.train()
        
        # Move to device
        raw_signal_batch = raw_signal_batch.to(self.device)
        
        # Compute self-supervised loss
        loss = self.model.compute_self_supervised_loss(raw_signal_batch)
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    @torch.no_grad()
    def encode(self, raw_signal_tensor):
        """Encode raw signals to latent context (inference mode)"""
        self.model.eval()
        raw_signal_tensor = raw_signal_tensor.to(self.device)
        z_context = self.model(raw_signal_tensor)
        return z_context
