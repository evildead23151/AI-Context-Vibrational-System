"""
Module 6: Continuous Adaptation Loop
Closed-loop control system with online learning from implicit feedback.
"""

import torch
import time
import threading
from queue import Queue
from typing import Optional
import logging


class ContinuousAdaptationLoop:
    """
    Main control loop that ties everything together.
    
    Flow:
    Raw Signals → LCM → URM → Planner → Generator → Audio Output
                                ↑                       ↓
                                └─── Behavior Feedback ─┘
    """
    
    def __init__(
        self,
        raw_signal_ingestion,
        latent_context_model,
        user_response_model,
        latent_audio_planner,
        audio_generator,
        config,
        device
    ):
        self.raw_signals = raw_signal_ingestion
        self.lcm = latent_context_model
        self.urm = user_response_model
        self.planner = latent_audio_planner
        self.generator = audio_generator
        self.config = config
        self.device = device
        
        # Control state
        self.u_current = None  # Current latent control tensor
        self.z_context = None  # Current latent context
        self.behavior_queue = Queue()
        
        # Adaptation parameters
        self.update_interval = config.adaptation.update_interval_seconds
        self.max_control_delta = config.adaptation.max_control_delta
        self.reward_abort_threshold = config.adaptation.reward_drop_abort_threshold
        
        # Optimizers
        self.planner_optimizer = torch.optim.AdamW(
            [p for p in planner.parameters() if p.requires_grad],
            lr=config.adaptation.planner_lr
        )
        
        self.urm_optimizer = torch.optim.AdamW(
            urm.parameters(),
            lr=config.adaptation.urm_lr
        )
        
        # Loop control
        self.running = False
        self.loop_thread = None
        
        # Logging
        self.logger = logging.getLogger('AdaptationLoop')
        
    def initialize(self):
        """Initialize control state"""
        # Initialize control tensor (random at start)
        self.u_current = torch.randn(
            1, self.config.latent_dims.control_tensor,
            device=self.device
        ) * 0.1
        
        # Get initial context
        raw_signal_tensor = self.raw_signals.get_context_window()
        raw_signal_tensor = raw_signal_tensor.to(self.device)
        
        with torch.no_grad():
            self.z_context = self.lcm(raw_signal_tensor)
        
        self.logger.info("Control loop initialized")
    
    def start(self):
        """Start the continuous adaptation loop"""
        if self.running:
            self.logger.warning("Loop already running")
            return
        
        self.initialize()
        self.running = True
        
        # Start loop thread
        self.loop_thread = threading.Thread(target=self._loop, daemon=True)
        self.loop_thread.start()
        
        self.logger.info("Adaptation loop started")
    
    def stop(self):
        """Stop the loop"""
        self.running = False
        if self.loop_thread:
            self.loop_thread.join(timeout=10)
        
        self.logger.info("Adaptation loop stopped")
    
    def _loop(self):
        """Main control loop (runs in separate thread)"""
        last_update_time = time.time()
        
        while self.running:
            try:
                # 1. Update context from raw signals
                self._update_context()
                
                # 2. Generate audio with current control
                audio_chunk = self._generate_audio()
                
                # 3. Collect implicit behavioral feedback
                behavior_signals = self._collect_behavior()
                
                # 4. Update URM from trajectory
                self._update_urm(behavior_signals)
                
                # 5. Check if update interval elapsed
                current_time = time.time()
                if current_time - last_update_time >= self.update_interval:
                    # 6. Update control tensor via planner
                    self._update_control(behavior_signals)
                    last_update_time = current_time
                
                # 7. Sleep until next audio chunk needed
                sleep_time = self.config.audio_gen.generation_chunk_seconds * 0.9
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(1.0)  # Avoid tight error loop
    
    def _update_context(self):
        """Update latent context from recent raw signals"""
        raw_signal_tensor = self.raw_signals.get_context_window()
        raw_signal_tensor = raw_signal_tensor.to(self.device)
        
        with torch.no_grad():
            self.z_context = self.lcm(raw_signal_tensor)
    
    def _generate_audio(self):
        """Generate audio chunk using current control tensor"""
        with torch.no_grad():
            audio_chunk = self.generator.generate_continuous(self.u_current)
        
        # Audio output would go to playback system here
        # (Not implemented - would use PyAudio, sounddevice, etc.)
        
        return audio_chunk
    
    def _collect_behavior(self):
        """Collect implicit behavioral signals from queue"""
        from .user_response_model import ImplicitBehavior
        
        behavior = ImplicitBehavior()
        
        # Drain queue
        while not self.behavior_queue.empty():
            signal_type, value = self.behavior_queue.get()
            behavior.record(signal_type, value)
        
        # Record default signals if queue was empty
        if not behavior.signals:
            behavior.record('session_active', 1.0)
            behavior.record('engagement', 1.0)
        
        return behavior
    
    def _update_urm(self, behavior_signals):
        """Update User Response Model from trajectory"""
        # Record trajectory step
        self.urm.record_trajectory(
            self.z_context.detach(),
            self.u_current.detach(),
            behavior_signals
        )
        
        # Update if enough trajectory collected
        if len(self.urm.trajectory_buffer) >= 5:
            loss = self.urm.update_from_trajectory(self.urm_optimizer)
            if loss is not None:
                self.logger.info(f"URM updated. Loss: {loss:.4f}")
    
    def _update_control(self, behavior_signals):
        """Update control tensor via planner + URM reward"""
        # Get current reward estimate
        with torch.no_grad():
            reward_current = self.urm.forward(
                self.z_context,
                self.u_current,
                behavior_signals
            )
        
        # Propose new control via planner
        u_proposed, _ = self.planner.forward(
            self.z_context,
            self.u_current,
            reward_signal=reward_current
        )
        
        # Check control delta (stability constraint)
        delta = self.planner.compute_control_delta(u_proposed, self.u_current)
        
        if delta.item() > self.max_control_delta:
            # Clamp delta
            u_proposed = self.planner.clamp_control_delta(
                u_proposed,
                self.u_current,
                self.max_control_delta
            )
            self.logger.warning(f"Control delta clamped: {delta.item():.4f}")
        
        # Estimate reward for proposed control
        with torch.no_grad():
            reward_proposed = self.urm.forward(
                self.z_context,
                u_proposed,
                behavior_signals
            )
        
        # Check reward drop (abort if too negative)
        reward_change = reward_proposed - reward_current
        
        if reward_change.item() < self.reward_abort_threshold:
            self.logger.warning(
                f"Control update aborted. Reward drop: {reward_change.item():.4f}"
            )
            return
        
        # Update planner via policy gradient
        # (Maximize reward at new control)
        policy_loss = -reward_proposed.mean()
        
        self.planner_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.planner.parameters(), 1.0)
        self.planner_optimizer.step()
        
        # Commit new control
        self.u_current = u_proposed.detach()
        
        self.logger.info(
            f"Control updated. Reward: {reward_current.item():.4f} → "
            f"{reward_proposed.item():.4f}, Delta: {delta.item():.4f}"
        )
    
    def record_behavior(self, signal_type: str, value: float):
        """
        Record implicit behavioral signal.
        Called from external systems (playback, sensors, etc.)
        
        Args:
            signal_type: Opaque signal identifier
            value: Scalar value (no semantic meaning)
        """
        self.behavior_queue.put((signal_type, value))
    
    def get_state(self):
        """Get current system state (for monitoring)"""
        return {
            'z_context': self.z_context.detach().cpu() if self.z_context is not None else None,
            'u_current': self.u_current.detach().cpu() if self.u_current is not None else None,
            'running': self.running,
            'trajectory_length': len(self.urm.trajectory_buffer)
        }
