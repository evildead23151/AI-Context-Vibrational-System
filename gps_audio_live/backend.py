"""
GPS Audio System - LIVE LOCAL BACKEND
Generates WAV audio from GPS coordinates
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import wave
import struct
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Storage
audio_dir = Path("./audio_output")
audio_dir.mkdir(exist_ok=True)

latest_audio_file = None
gps_log = []


def generate_audio_from_gps(lat, lon, duration=3):
    """
    Generate WAV audio deterministically from GPS
    
    Mapping:
    - Latitude (-90 to 90) → Frequency (200 to 800 Hz)
    - Longitude (-180 to 180) → Amplitude (0.2 to 0.8)
    """
    # Map latitude to frequency (200-800 Hz)
    freq = 200 + ((lat + 90) / 180) * 600
    
    # Map longitude to amplitude (0.2-0.8)
    amp = 0.2 + ((lon + 180) / 360) * 0.6
    
    # Generate sine wave
    sample_rate = 44100
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples)
    
    # Create audio: sine wave + harmonics
    audio = amp * (
        np.sin(2 * np.pi * freq * t) +
        0.5 * np.sin(4 * np.pi * freq * t)
    )
    
    # Normalize
    audio = audio * 32767 / np.max(np.abs(audio))
    audio = audio.astype(np.int16)
    
    # Save WAV file
    timestamp = int(datetime.now().timestamp())
    filename = f"gps_audio_{timestamp}.wav"
    filepath = audio_dir / filename
    
    with wave.open(str(filepath), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    
    print(f"✅ AUDIO GENERATED: {filename}")
    print(f"   GPS: ({lat:.4f}, {lon:.4f})")
    print(f"   Frequency: {freq:.1f} Hz")
    print(f"   Amplitude: {amp:.2f}")
    print(f"   File: {filepath}")
    
    return str(filepath), filename


@app.route('/api/gps', methods=['POST'])
def receive_gps():
    """Receive GPS and generate audio"""
    data = request.json
    lat = data.get('latitude')
    lon = data.get('longitude')
    
    if lat is None or lon is None:
        return jsonify({'error': 'Missing GPS data'}), 400
    
    print(f"\n📍 GPS RECEIVED: lat={lat:.6f}, lon={lon:.6f}")
    
    # Log GPS
    gps_log.append({
        'lat': lat,
        'lon': lon,
        'timestamp': datetime.now().isoformat()
    })
    
    # Generate audio
    try:
        global latest_audio_file
        filepath, filename = generate_audio_from_gps(lat, lon)
        latest_audio_file = filename
        
        return jsonify({
            'status': 'success',
            'audio_file': filename,
            'gps': {'lat': lat, 'lon': lon}
        })
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/audio/<filename>', methods=['GET'])
def get_audio(filename):
    """Serve audio file"""
    filepath = audio_dir / filename
    if filepath.exists():
        print(f"🔊 SERVING AUDIO: {filename}")
        return send_file(filepath, mimetype='audio/wav')
    return jsonify({'error': 'File not found'}), 404


@app.route('/api/audio/latest', methods=['GET'])
def get_latest_audio():
    """Get latest generated audio"""
    if latest_audio_file:
        return get_audio(latest_audio_file)
    return jsonify({'error': 'No audio generated yet'}), 404


@app.route('/api/status', methods=['GET'])
def get_status():
    """System status"""
    return jsonify({
        'status': 'LIVE',
        'gps_count': len(gps_log),
        'latest_audio': latest_audio_file
    })


if __name__ == '__main__':
    print("=" * 60)
    print("GPS AUDIO SYSTEM - LOCAL BACKEND")
    print("=" * 60)
    print("Starting server on http://localhost:5000")
    print("Waiting for GPS data from frontend...")
    print("=" * 60)
    
    app.run(host='localhost', port=5000, debug=False)
