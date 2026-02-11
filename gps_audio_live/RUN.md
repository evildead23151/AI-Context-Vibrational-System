# GPS AUDIO SYSTEM - RUN INSTRUCTIONS

## START THE SYSTEM

### 1. Install dependencies
```
pip install flask flask-cors numpy
```

### 2. Start backend
```
cd gps_audio_live
python backend.py
```

### 3. Open frontend
```
Open frontend.html in Chrome or Firefox
```

### 4. Grant GPS permission
Click "START SYSTEM" → Allow GPS

### 5. Generate audio
Click "GENERATE AUDIO FROM GPS" → Audio plays automatically

## VERIFY IT WORKS

- GPS coordinates should display
- Console shows "AUDIO GENERATED"
- WAV file created in audio_output/
- Audio plays through speakers
- **Change location (or spoof GPS) → Sound changes**

## WHAT HAPPENS

1. Browser captures GPS (lat, lon)
2. Sends to http://localhost:5000/api/gps
3. Backend generates WAV (lat→frequency, lon→amplitude)
4. Browser fetches and plays WAV
5. **Different GPS = Different sound**
