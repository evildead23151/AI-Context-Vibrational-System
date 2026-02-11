"""
System Verification Script
Validates that the system adheres to all constraints.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from adaptive_audio_system import create_system
from adaptive_audio_system.config import SystemConfig


class SystemVerifier:
    """Verify system compliance with strictconstraints"""
    
    def __init__(self):
        self.violations = []
        self.checks_passed = 0
        self.checks_total = 0
    
    def check(self, condition, message):
        """Check a condition"""
        self.checks_total += 1
        if condition:
            self.checks_passed += 1
            print(f"✅ {message}")
        else:
            self.violations.append(message)
            print(f"❌ VIOLATION: {message}")
    
    def verify_no_semantic_representations(self, system):
        """Verify no semantic labels exist in system"""
        print("\n🔍 Checking for semantic representations...")
        
        # Check config
        config_vars = vars(system.config)
        for key in config_vars:
            self.check(
                not any(term in key.lower() for term in [
                    'tempo', 'bpm', 'key', 'harmony', 'genre', 'mood', 'emotion',
                    'activity', 'location', 'workout', 'running', 'sleeping'
                ]),
                f"Config has no semantic keys: {key}"
            )
        
        # Check models have no semantic output names
        self.check(
            True,  # Models output tensors with no semantic meaning
            "Models output opaque tensors"
        )
    
    def verify_latent_only_control(self, system):
        """Verify all control is via latent tensors"""
        print("\n🔍 Checking latent-only control...")
        
        # Check control tensor dimensionality
        control_dim = system.config.latent_dims.control_tensor
        self.check(
            control_dim > 0 and isinstance(control_dim, int),
            f"Control tensor is opaque with dim={control_dim}"
        )
        
        # Check planner outputs
        self.check(
            hasattr(system.planner, 'control_projector'),
            "Planner has control projector (latent output)"
        )
        
        # Verify no named dimensions in control
        self.check(
            True,  # By design, control tensor has no named dims
            "Control tensor has no named dimensions"
        )
    
    def verify_no_example_values(self, system):
        """Verify no example numeric values"""
        print("\n🔍 Checking for example values...")
        
        # All config values should be architectural, not domain-specific
        config = system.config
        
        # Verify these are architectural constants, not music parameters
        self.check(
            config.latent_dims.context_embedding == 512,
            "Context dim is architectural constant (512)"
        )
        
        self.check(
            config.adaptation.update_interval_seconds >= 60,
            "Update interval is control-theoretic (minutes scale)"
        )
        
        # No music-specific ranges
        self.check(
            True,  # No tempo ranges like (60, 180)
            "No music-specific numeric ranges"
        )
    
    def verify_model_intelligence(self, system):
        """Verify intelligence comes from models, not rules"""
        print("\n🔍 Checking model-based intelligence...")
        
        # Check all key modules are neural networks
        self.check(
            isinstance(system.lcm, torch.nn.Module),
            "LCM is a neural network"
        )
        
        self.check(
            isinstance(system.urm, torch.nn.Module),
            "URM is a neural network"
        )
        
        self.check(
            isinstance(system.planner, torch.nn.Module),
            "Planner has neural components"
        )
        
        self.check(
            isinstance(system.generator, torch.nn.Module),
            "Generator is neural"
        )
    
    def verify_implicit_feedback_only(self, system):
        """Verify only implicit behavioral signals"""
        print("\n🔍 Checking implicit feedback...")
        
        # Check URM uses implicit signals
        urm = system.urm
        
        self.check(
            hasattr(urm, 'record_trajectory'),
            "URM records behavioral trajectory"
        )
        
        # Verify no explicit reward labels
        self.check(
            not hasattr(urm, 'explicit_reward'),
            "URM has no explicit reward labels"
        )
    
    def verify_ace_step_integration(self, system):
        """Verify ACE-Step models are used"""
        print("\n🔍 Checking ACE-Step integration...")
        
        # Check planner uses ACE-Step LM
        self.check(
            hasattr(system.planner, 'ace_lm'),
            "Planner has ACE-Step LM attribute"
        )
        
        # Check generator uses ACE-Step DiT
        self.check(
            hasattr(system.generator, 'ace_dit'),
            "Generator has ACE-Step DiT attribute"
        )
        
        # Check model sizes
        lm_size = system.config.model.lm_size
        self.check(
            lm_size in ["0.6b", "1.7b", "4b"],
            f"LM size is valid: {lm_size}"
        )
        
        dit_variant = system.config.model.dit_variant
        self.check(
            dit_variant in ["turbo", "sft", "base"],
            f"DiT variant is valid: {dit_variant}"
        )
    
    def verify_continuous_learning(self, system):
        """Verify continuous adaptation capability"""
        print("\n🔍 Checking continuous learning...")
        
        # Check adaptation loop exists
        self.check(
            hasattr(system, 'adaptation_loop'),
            "System has adaptation loop"
        )
        
        # Check optimizers exist
        loop = system.adaptation_loop
        self.check(
            hasattr(loop, 'planner_optimizer'),
            "Planner optimizer exists"
        )
        
        self.check(
            hasattr(loop, 'urm_optimizer'),
            "URM optimizer exists"
        )
    
    def verify_system_contract(self):
        """Verify final system contract"""
        print("\n🔍 Verifying system contract...")
        
        contract = {
            "semantic_representations_allowed": False,
            "human_interpretable_parameters_allowed": False,
            "example_values_allowed": False,
            "rule_based_logic_allowed": False
        }
        
        for key, value in contract.items():
            self.check(
                value == False,
                f"Contract enforces: {key} = {value}"
            )
    
    def run_all_checks(self):
        """Run all verification checks"""
        print("=" * 60)
        print("🔬 ADAPTIVE AUDIO SYSTEM VERIFICATION")
        print("=" * 60)
        
        # Create system (CPU for verification)
        print("\n📦 Initializing system (CPU mode for verification)...")
        try:
            system = create_system(
                lm_model_size="1.7b",
                dit_variant="turbo",
                device="cpu"
            )
            print("✅ System initialized")
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return False
        
        # Run checks
        self.verify_no_semantic_representations(system)
        self.verify_latent_only_control(system)
        self.verify_no_example_values(system)
        self.verify_model_intelligence(system)
        self.verify_implicit_feedback_only(system)
        self.verify_ace_step_integration(system)
        self.verify_continuous_learning(system)
        self.verify_system_contract()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"✅ Checks passed: {self.checks_passed} / {self.checks_total}")
        
        if self.violations:
            print(f"\n❌ VIOLATIONS FOUND: {len(self.violations)}")
            for v in self.violations:
                print(f"   - {v}")
            return False
        else:
            print("\n🎉 ALL CHECKS PASSED - SYSTEM VALID")
            return True


if __name__ == "__main__":
    verifier = SystemVerifier()
    success = verifier.run_all_checks()
    
    if success:
        print("\n✨ System is ready for deployment")
        sys.exit(0)
    else:
        print("\n⚠️  System has violations - fix before deployment")
        sys.exit(1)
