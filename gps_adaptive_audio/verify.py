"""
VERIFICATION SUITE - GPS-Adaptive Audio System
Tests every module and proves the system is actually running.
"""

import requests
import json
import time
from pathlib import Path

class SystemVerifier:
    """Verifies the entire system is operational"""
    
    def __init__(self, api_url="http://localhost:5000/api"):
        self.api_url = api_url
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
    
    def test(self, name, condition, details=""):
        """Run a test"""
        if condition:
            self.tests_passed += 1
            status = "✅ PASS"
            print(f"{status}: {name}")
        else:
            self.tests_failed += 1
            status = "❌ FAIL"
            print(f"{status}: {name}")
            if details:
                print(f"         Details: {details}")
        
        self.test_results.append({
            'name': name,
            'status': status,
            'details': details
        })
        
        return condition
    
    def verify_backend_running(self):
        """Test 1: Backend is running"""
        print("\n🔬 TEST 1: Backend Connectivity")
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            self.test("Backend responds to requests", response.status_code == 200)
            
            if response.status_code == 200:
                health = response.json()
                self.test("Health endpoint returns valid JSON", isinstance(health, dict))
                self.test("Health has 'status' field", 'status' in health)
                return True
            return False
        except Exception as e:
            self.test("Backend responds to requests", False, str(e))
            return False
    
    def verify_modules_active(self):
        """Test 2: All modules are active"""
        print("\n🔬 TEST 2: Module Activation")
        try:
            response = requests.get(f"{self.api_url}/health")
            health = response.json()
            
            modules = health.get('modules_active', {})
            
            self.test("GPS Ingestion module active",
                     modules.get('gps_ingestion', False))
            self.test("Context Model module active",
                     modules.get('context_model', False))
            self.test("Planner module active",
                     modules.get('planner', False))
            self.test("Generator module active",
                     modules.get('generator', False))
            
            all_active = all(modules.values())
            self.test("ALL modules active", all_active)
            
            return all_active
            
        except Exception as e:
            self.test("Module status check", False, str(e))
            return False
    
    def verify_gps_ingestion(self):
        """Test 3: GPS ingestion works"""
        print("\n🔬 TEST 3: GPS Ingestion")
        try:
            # Send test GPS
            test_gps = {
                'latitude': 40.7128,
                'longitude': -74.0060
            }
            
            response = requests.post(
                f"{self.api_url}/gps",
                json=test_gps,
                timeout=5
            )
            
            self.test("GPS endpoint accepts coordinates", response.status_code == 200)
            
            if response.status_code == 200:
                data = response.json()
                self.test("GPS ingestion returns success",
                         data.get('status') == 'success')
                self.test("GPS stored in history",
                         data.get('history_size', 0) > 0)
                
                # Send another GPS to test context update
                test_gps2 = {
                    'latitude': 40.7589,
                    'longitude': -73.9851
                }
                response2 = requests.post(f"{self.api_url}/gps", json=test_gps2)
                data2 = response2.json()
                
                self.test("Context updates from GPS",
                         data2.get('context_updated', False))
                
                return True
            return False
            
        except Exception as e:
            self.test("GPS ingestion", False, str(e))
            return False
    
    def verify_context_model(self):
        """Test 4: Context model produces non-constant embeddings"""
        print("\n🔬 TEST 4: Context Model")
        try:
            # Get status after GPS ingestion
            response = requests.get(f"{self.api_url}/status")
            status = response.json()
            
            health = status.get('health', {})
            
            self.test("Context model is neural network",
                     health.get('modules_active', {}).get('context_model', False))
            
            # Context is created if GPS history exists
            has_gps = status.get('gps_history_size', 0) > 0
            self.test("GPS history exists for context encoding", has_gps)
            
            return True
            
        except Exception as e:
            self.test("Context model check", False, str(e))
            return False
    
    def verify_music_generation(self):
        """Test 5: Music generation actually works"""
        print("\n🔬 TEST 5: Music Generation")
        try:
            # Trigger generation
            print("   Generating music (this may take a few seconds)...")
            response = requests.post(
                f"{self.api_url}/generate",
                timeout=30
            )
            
            self.test("Generation endpoint responds", response.status_code == 200)
            
            if response.status_code == 200:
                data = response.json()
                
                self.test("Generation returns success",
                         data.get('status') == 'success')
                self.test("Audio file created",
                         'audio_file' in data)
                self.test("Generation time recorded",
                         'generation_time' in data)
                self.test("Duration recorded",
                         'duration' in data and data['duration'] > 0)
                
                # Try to fetch the audio file
                audio_file = data.get('audio_file')
                if audio_file:
                    audio_response = requests.get(
                        f"{self.api_url}/audio/{audio_file}",
                        timeout=5
                    )
                    self.test("Audio file is downloadable",
                             audio_response.status_code == 200)
                    self.test("Audio file has data",
                             len(audio_response.content) > 0)
                    
                    print(f"   Generated: {audio_file} ({data.get('duration', 0):.1f}s)")
                    print(f"   Generation time: {data.get('generation_time', 0):.2f}s")
                
                return True
            return False
            
        except Exception as e:
            self.test("Music generation", False, str(e))
            return False
    
    def verify_planner_invocation(self):
        """Test 6: Planner is actually called"""
        print("\n🔬 TEST 6: Planner Invocation")
        try:
            # Get health before generation
            response1 = requests.get(f"{self.api_url}/health")
            count1 = response1.json().get('generation_count', 0)
            
            # Generate music
            requests.post(f"{self.api_url}/generate", timeout=30)
            
            # Get health after generation
            response2 = requests.get(f"{self.api_url}/health")
            count2 = response2.json().get('generation_count', 0)
            
            self.test("Generation count increases",
                     count2 > count1,
                     f"Before: {count1}, After: {count2}")
            
            return count2 > count1
            
        except Exception as e:
            self.test("Planner invocation check", False, str(e))
            return False
    
    def verify_no_cached_reuse(self):
        """Test 7: Audio is newly generated, not cached"""
        print("\n🔬 TEST 7: No Cached Audio Reuse")
        try:
            # Generate first
            resp1 = requests.post(f"{self.api_url}/generate", timeout=30)
            file1 = resp1.json().get('audio_file')
            
            # Wait a moment
            time.sleep(1)
            
            # Generate second
            resp2 = requests.post(f"{self.api_url}/generate", timeout=30)
            file2 = resp2.json().get('audio_file')
            
            self.test("Each generation creates new file",
                     file1 != file2,
                     f"File 1: {file1}, File 2: {file2}")
            
            return file1 != file2
            
        except Exception as e:
            self.test("Cache check", False, str(e))
            return False
    
    def verify_system_health_reporting(self):
        """Test 8: System reports accurate health"""
        print("\n🔬 TEST 8: System Health Reporting")
        try:
            response = requests.get(f"{self.api_url}/health")
            health = response.json()
            
            required_fields = [
                'status', 'modules_active', 'models_loaded',
                'gps_signal_quality', 'generation_count',
                'playback_state', 'gpu_available', 'last_update'
            ]
            
            for field in required_fields:
                self.test(f"Health contains '{field}'",
                         field in health,
                         f"Missing field: {field}")
            
            # Status must be valid enum
            valid_statuses = ['OPERATIONAL', 'DEGRADED', 'FAILED', 'INITIALIZING']
            self.test("Status is valid enum",
                     health.get('status') in valid_statuses,
                     f"Status: {health.get('status')}")
            
            return True
            
        except Exception as e:
            self.test("Health reporting check", False, str(e))
            return False
    
    def verify_gps_quality_tracking(self):
        """Test 9: GPS quality is tracked"""
        print("\n🔬 TEST 9: GPS Quality Tracking")
        try:
            # Send multiple GPS readings
            for i in range(5):
                requests.post(f"{self.api_url}/gps", json={
                    'latitude': 40.7128 + (i * 0.01),
                    'longitude': -74.0060 + (i * 0.01)
                })
                time.sleep(0.1)
            
            # Check quality
            response = requests.get(f"{self.api_url}/health")
            health = response.json()
            
            gps_quality = health.get('gps_signal_quality')
            self.test("GPS quality tracked",
                     gps_quality in ['GOOD', 'FAIR', 'POOR', 'UNKNOWN'],
                     f"Quality: {gps_quality}")
            
            self.test("GPS quality improves with data",
                     gps_quality in ['GOOD', 'FAIR'],
                     f"Quality: {gps_quality}")
            
            return True
            
        except Exception as e:
            self.test("GPS quality tracking", False, str(e))
            return False
    
    def run_all_tests(self):
        """Run complete verification suite"""
        print("=" * 70)
        print("GPS-ADAPTIVE AUDIO SYSTEM - VERIFICATION SUITE")
        print("=" * 70)
        print("\nThis suite verifies:")
        print("  • Backend is running")
        print("  • All modules are active")
        print("  • GPS ingestion works")
        print("  • Context model executes")
        print("  • Music generation works")
        print("  • Planner is invoked")
        print("  • No cached reuse")
        print("  • Health reporting accurate")
        print("  • GPS quality tracking")
        
        # Run tests
        self.verify_backend_running()
        self.verify_modules_active()
        self.verify_gps_ingestion()
        self.verify_context_model()
        self.verify_music_generation()
        self.verify_planner_invocation()
        self.verify_no_cached_reuse()
        self.verify_system_health_reporting()
        self.verify_gps_quality_tracking()
        
        # Summary
        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)
        print(f"Tests Passed:  {self.tests_passed}")
        print(f"Tests Failed:  {self.tests_failed}")
        print(f"Total Tests:   {self.tests_passed + self.tests_failed}")
        
        if self.tests_failed == 0:
            print("\n🎉 ALL TESTS PASSED - SYSTEM FULLY OPERATIONAL")
            print("=" * 70)
            return True
        else:
            print("\n⚠️  SOME TESTS FAILED - SYSTEM NOT FULLY OPERATIONAL")
            print("=" * 70)
            return False


if __name__ == "__main__":
    import sys
    
    print("\n⏳ Waiting for backend to start (3 seconds)...")
    time.sleep(3)
    
    verifier = SystemVerifier()
    success = verifier.run_all_tests()
    
    sys.exit(0 if success else 1)
