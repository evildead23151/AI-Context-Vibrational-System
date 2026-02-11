"""
Example Usage of Adaptive Audio System
Demonstrates how to use the system with simulated sensors and behavior.
"""

import torch
import time
import numpy as np
from adaptive_audio_system import create_system


def simulate_sensor_data():
    """Generate simulated sensor data (for testing)"""
    return {
        'gps': [np.random.uniform(-90, 90), np.random.uniform(-180, 180)],
        'wifi_bssids': [f"wifi_{i}" for i in range(np.random.randint(3, 10))],
        'bluetooth_rssi': list(np.random.uniform(-80, -30, 10)),
        'accelerometer': list(np.random.normal(0, 1, 3)),
        'gyroscope': list(np.random.normal(0, 0.5, 3)),
        'audio_level': [np.random.uniform(30, 70)],
        'screen_interaction': {
            'last_interaction_delta': np.random.uniform(0, 300),
            'interaction_count': np.random.randint(0, 50),
            'is_locked': np.random.choice([True, False]),
            'foreground_entropy': np.random.uniform(0, 2)
        }
    }


def simulate_user_behavior(system, duration_seconds=10):
    """
    Simulate user behavior (implicit signals).
    In production, these would come from actual playback/interaction events.
    """
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        # Simulate various behavioral signals
        
        # Session active
        system.record_behavior('session_active', 1.0)
        
        # Random volume adjustment
        if np.random.random() < 0.1:
            vol_delta = np.random.uniform(-0.2, 0.2)
            system.record_behavior('volume_delta', vol_delta)
        
        # Random skip event
        if np.random.random() < 0.05:
            system.record_behavior('skip', 1.0)
        
        # Engagement signal
        engagement = np.random.uniform(0.5, 1.0)
        system.record_behavior('engagement', engagement)
        
        time.sleep(1.0)


def example_basic_usage():
    """Example 1: Basic system usage"""
    print("=" * 60)
    print("Example 1: Basic System Usage")
    print("=" * 60)
    
    # Create system
    print("\n1. Creating system...")
    system = create_system(
        lm_model_size="1.7b",
        dit_variant="turbo",
        device="cpu"  # Use CPU for demo
    )
    print("✅ System created")
    
    # Ingest some sensor data
    print("\n2. Ingesting sensor data...")
    for i in range(10):
        sensor_data = simulate_sensor_data()
        system.ingest_sensor_data(sensor_data)
        time.sleep(0.1)
    print("✅ Sensor data ingested")
    
    # Start system
    print("\n3. Starting adaptive audio system...")
    system.start()
    print("✅ System started")
    
    # Simulate operation
    print("\n4. Running for 30 seconds (simulated)...")
    simulate_user_behavior(system, duration_seconds=30)
    
    # Check state
    print("\n5. Checking system state...")
    state = system.get_system_state()
    print(f"   - Running: {state['running']}")
    print(f"   - Trajectory length: {state['trajectory_length']}")
    if state['z_context'] is not None:
        print(f"   - Context embedding shape: {state['z_context'].shape}")
    if state['u_current'] is not None:
        print(f"   - Control tensor shape: {state['u_current'].shape}")
    
    # Stop system
    print("\n6. Stopping system...")
    system.stop()
    print("✅ System stopped")


def example_state_persistence():
    """Example 2: Save and load system state"""
    print("\n" + "=" * 60)
    print("Example 2: State Persistence")
    print("=" * 60)
    
    # Create and run system
    print("\n1. Creating system and collecting data...")
    system = create_system(device="cpu")
    
    # Ingest data
    for i in range(20):
        system.ingest_sensor_data(simulate_sensor_data())
    
    # Save state
    print("\n2. Saving system state...")
    system.save_state("./saved_state")
    print("✅ State saved to ./saved_state")
    
    # Create new system
    print("\n3. Creating new system...")
    system2 = create_system(device="cpu")
    
    # Load state
    print("\n4. Loading saved state...")
    system2.load_state("./saved_state")
    print("✅ State loaded")
    
    print("\n✅ State persistence verified")


def example_continuous_adaptation():
    """Example 3: Demonstrate continuous adaptation"""
    print("\n" + "=" * 60)
    print("Example 3: Continuous Adaptation")
    print("=" * 60)
    
    print("\n1. Creating system...")
    system = create_system(device="cpu")
    
    print("\n2. Pre-training context model (simulated)...")
    # In production, this would be long-term self-supervised training
    for i in range(5):
        sensor_data = simulate_sensor_data()
        system.ingest_sensor_data(sensor_data)
    print("✅ Context model has learned patterns")
    
    print("\n3. Starting adaptive loop...")
    system.start()
    
    print("\n4. Running adaptive system for 60 seconds...")
    print("   (System learns from implicit behavioral feedback)")
    
    # Simulate different behavioral patterns
    for phase in range(3):
        print(f"\n   Phase {phase + 1}/3:")
        
        if phase == 0:
            print("   - Simulating positive engagement...")
            for _ in range(10):
                system.record_behavior('session_active', 1.0)
                system.record_behavior('engagement', 0.9)
                time.sleep(1)
        
        elif phase == 1:
            print("   - Simulating mixed engagement...")
            for _ in range(10):
                system.record_behavior('session_active', 1.0)
                system.record_behavior('engagement', 0.5)
                if np.random.random() < 0.3:
                    system.record_behavior('skip', 1.0)
                time.sleep(1)
        
        else:
            print("   - Simulating high engagement...")
            for _ in range(10):
                system.record_behavior('session_active', 1.0)
                system.record_behavior('engagement', 1.0)
                system.record_behavior('volume_delta', 0.1)
                time.sleep(1)
    
    print("\n5. Checking adapted state...")
    state = system.get_system_state()
    print(f"   - Trajectory collected: {state['trajectory_length']} steps")
    print("   ✅ System has adapted to behavioral patterns")
    
    print("\n6. Stopping system...")
    system.stop()
    print("✅ Adaptation demo complete")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("🎵 ADAPTIVE AUDIO SYSTEM - USAGE EXAMPLES")
    print("=" * 60)
    
    try:
        # Example 1
        example_basic_usage()
        
        # Example 2
        example_state_persistence()
        
        # Example 3
        example_continuous_adaptation()
        
        print("\n" + "=" * 60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
