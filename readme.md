# EyEbot - Voice-Controlled Object Tracking Robot

A smart IoT robot car that uses voice commands and computer vision to detect and track objects in real-time. The system combines YOLOv5/v8 object detection, speech recognition, and ESP8266-based motor control.

---

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## ✨ Features

- **Voice-Activated Object Detection**: Speak the name of any COCO dataset object to track
- **Real-Time Object Tracking**: Uses YOLOv5/YOLOv8 for accurate detection
- **Autonomous Navigation**: Car automatically aligns and moves toward detected objects
- **Adaptive Search**: Rotates to search for objects when lost from view
- **Multiple Operation Modes**:
  - Full robot control (`iotFinal.py`)
  - Phone camera testing (`iotPhone.py`)
  - System camera testing (`voice_object_finder.py`)
- **WiFi-Controlled Robot**: ESP8266-based wireless motor control
- **Smart Stabilization**: Prevents jittery movements with deadband logic

---

## 🏗️ System Architecture
```
┌─────────────┐      WiFi/HTTP     ┌──────────────┐
│   Python    │ ◄─────────────────► │   ESP8266    │
│   Script    │    Commands         │   NodeMCU    │
└─────────────┘                     └──────────────┘
      │                                     │
      │ Camera Stream                       │ GPIO Control
      ▼                                     ▼
┌─────────────┐                     ┌──────────────┐
│  IP Camera  │                     │  L298N Motor │
│  (Phone)    │                     │   Driver     │
└─────────────┘                     └──────────────┘
                                            │
                                            ▼
                                    ┌──────────────┐
                                    │ DC Motors x4 │
                                    └──────────────┘
```

---

## 🔧 Hardware Requirements

### Robot Components
- **ESP8266 NodeMCU** (WiFi microcontroller)
- **L298N Motor Driver** (H-bridge for DC motors)
- **4x DC Motors** (with wheels)
- **Power Supply**: 7-12V battery pack
- **Buzzer** (optional, for audio feedback)
- **LED** (optional, for visual indicators)

### Wiring Connections
```
ESP8266 NodeMCU → L298N Motor Driver
─────────────────────────────────────
D5 (IN1) → Motor FR/BR Control 1
D6 (IN2) → Motor FR/BR Control 2
D7 (IN3) → Motor FL/BL Control 1
D8 (IN4) → Motor FL/BL Control 2
D0       → Buzzer (optional)
D1       → LED (optional)
D2       → WiFi Status LED
```

### Computer/Laptop
- **Camera Phone**: Any Android/iOS device with IP Webcam app
- **Microphone**: For voice commands
- **GPU**: CUDA-compatible GPU recommended (optional)

---

## 💻 Software Requirements

### Python Dependencies
```bash
pip install opencv-python
pip install torch torchvision
pip install SpeechRecognition
pip install requests
pip install numpy
pip install ultralytics  # for YOLOv8 (voice_object_finder.py)
```

### Additional Tools
- **IP Webcam App**: Install on smartphone ([Android](https://play.google.com/store/apps/details?id=com.pas.webcam))
- **Arduino IDE**: For uploading firmware to ESP8266
- **ESP8266 Board Package**: Install via Arduino IDE Board Manager

---

## 📥 Installation

### 1. Set Up Arduino Environment

1. Install Arduino IDE
2. Add ESP8266 board:
   - File → Preferences → Additional Board Manager URLs
   - Add: `http://arduino.esp8266.com/stable/package_esp8266com_index.json`
3. Install ESP8266 board package:
   - Tools → Board → Boards Manager → Search "ESP8266" → Install

### 2. Upload Robot Firmware

1. Open `EyEbot_robot_script/EyEbot_robot_script.ino` in Arduino IDE
2. Configure WiFi credentials:
```cpp
   String sta_ssid = "YourWiFiName";
   String sta_password = "YourPassword";
```
3. Select Board: **NodeMCU 1.0 (ESP-12E Module)**
4. Upload the sketch to ESP8266
5. Note the IP address from Serial Monitor

### 3. Set Up Python Environment
```bash
# Clone or download the repository
cd eyebot-project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure IP Addresses

**In Python scripts** (`iotFinal.py`):
```python
CAR_IP = "172.28.188.69"  # Replace with your ESP8266 IP
cap = cv2.VideoCapture("http://192.0.0.4:8080/video")  # Phone IP camera URL
```

**In IP Webcam app**:
- Start server on phone
- Note the IP address shown (e.g., `192.168.1.100:8080`)

---

## 🚀 Usage

### Mode 1: Full Robot Control (`iotFinal.py`)

**Complete object tracking with robot movement**
```bash
python iotFinal.py
```

**Voice Commands:**
- "Find person" - Track and follow a person
- "Find bottle" - Track and follow a bottle
- "Find cup" / "Find mouse" / "Find phone" - Track any COCO object
- "Stop" - Stop tracking and halt the robot

**Behavior:**
- Detects target object using YOLOv5
- Aligns robot to center the object
- Moves forward when aligned
- Rotates to search if object is lost
- Automatically stops and re-centers

---

### Mode 2: Phone Camera Testing (`iotPhone.py`)

**Object detection without robot control (for testing)**
```bash
python iotPhone.py
```

**Features:**
- Same voice commands as Mode 1
- Displays bounding boxes on detected objects
- No robot movement commands sent
- Useful for debugging detection accuracy

---

### Mode 3: System Camera Testing (`voice_object_finder.py`)

**Uses laptop/desktop webcam for local testing**
```bash
python voice_object_finder.py
```

**Features:**
- Uses YOLOv8 instead of YOLOv5
- Displays zoomed view of detected object
- No network/robot connection needed
- Perfect for algorithm development

---

## 📂 File Descriptions

| File | Purpose |
|------|---------|
| `iotFinal.py` | Main script for full robot operation with object tracking |
| `iotPhone.py` | Detection-only script using phone camera (no robot control) |
| `voice_object_finder.py` | Local testing script using system webcam and YOLOv8 |
| `EyEbot_robot_script.ino` | Arduino firmware for ESP8266 motor control |

---

## ⚙️ Configuration

### Tunable Parameters (in `iotFinal.py`)
```python
# Detection sensitivity
CONF_THRESHOLD = 0.3  # Lower = more detections, higher = more accuracy

# Alignment tuning
align_thresh = 80     # Pixels of acceptable deviation from center
margin = 50           # Deadband to ignore tiny movements

# Search behavior
SEARCH_ROTATE_SECS = 1           # Rotation duration when searching
MAX_RECHECK_SECONDS = 5.0        # Time to look for object after rotation
MAX_MISS_FRAMES = 5              # Frames before triggering search

# Motor speeds (in Arduino)
SPEED = 70            # Base motor speed (0-1023)
```

### Supported Objects (COCO Dataset)

The robot can track any of the 80 COCO objects including:
- **People**: person
- **Vehicles**: car, bicycle, motorcycle, bus, truck
- **Animals**: dog, cat, bird, horse, cow
- **Household**: bottle, cup, fork, knife, spoon, bowl, chair, couch, bed
- **Electronics**: laptop, mouse, keyboard, cell phone, tv, remote

Full list: [COCO Dataset Classes](https://github.com/ultralytics/yolov5/blob/master/data/coco.yaml)

---

## 🐛 Troubleshooting

### Camera Connection Issues

**Problem**: `Failed to open camera stream`

**Solutions:**
1. Verify phone and computer are on same WiFi network
2. Check IP camera URL format: `http://192.168.x.x:8080/video`
3. Test URL in browser first
4. Disable phone screen lock/power saving

### Robot Not Responding

**Problem**: Voice commands detected but robot doesn't move

**Solutions:**
1. Check ESP8266 IP address in `CAR_IP` variable
2. Verify ESP8266 is connected to WiFi (check Serial Monitor)
3. Test manual commands:
```bash
   curl "http://172.28.188.69/?State=F"  # Forward
   curl "http://172.28.188.69/?State=S"  # Stop
```
4. Check motor driver connections and power supply

### Poor Detection Accuracy

**Problem**: Objects not detected reliably

**Solutions:**
1. Increase lighting conditions
2. Lower `CONF_THRESHOLD` (e.g., `0.2`)
3. Use `yolov5s` or larger model instead of `yolov5n`
4. Ensure object is clearly visible and not occluded
5. Move camera closer to target

### Voice Recognition Issues

**Problem**: Commands not recognized

**Solutions:**
1. Check microphone permissions
2. Speak clearly and closer to microphone
3. Reduce background noise
4. Use exact object names from COCO dataset
5. Test with: "Find person" (most reliable)

### Robot Movement Unstable

**Problem**: Jittery or erratic movement

**Solutions:**
1. Increase `margin` deadband (e.g., `margin = 100`)
2. Increase `align_thresh` for less strict alignment
3. Adjust `MICRO_TURN_SECS` for smoother turns
4. Check motor power supply voltage
5. Calibrate motor speeds in Arduino code

---

## 🔒 Safety Notes

- Always test in open space away from obstacles
- Keep emergency stop ready (say "stop" or press Ctrl+C)
- Monitor battery levels to prevent brownouts
- Don't point camera at bright lights (affects detection)
- Ensure stable WiFi connection before operation

---

## 🛠️ Advanced Customization

### Change Detection Model

Replace YOLOv5 with custom trained model:
```python
model = torch.hub.load("path/to/your/model", "custom", path="best.pt")
```

### Add New Motor Patterns

In Arduino `.ino` file:
```cpp
// Add new command case
else if (command == "C") CirclePattern();

// Implement function
void CirclePattern() {
  // Your custom movement code
}
```

### Modify Detection Frequency
```python
# Run detection every N frames (lower = faster, higher = efficient)
if frame_counter % 3 != 0:  # Change 3 to desired value
```

---

## 📊 Performance Optimization

- **GPU Acceleration**: Use CUDA-enabled GPU for 5-10x faster detection
- **Model Selection**: 
  - `yolov5n`: Fastest, lower accuracy
  - `yolov5s`: Balanced (recommended)
  - `yolov5m/l/x`: Higher accuracy, slower
- **Frame Skipping**: Process every 2-3 frames to reduce CPU load
- **Resolution**: Lower camera resolution (640x480) for faster processing

---

## 📜 License

This project is provided as-is for educational purposes.

---

## 🙏 Acknowledgments

- **YOLOv5**: [Ultralytics](https://github.com/ultralytics/yolov5)
- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **SpeechRecognition**: [anthony-zhang](https://github.com/Uberi/speech_recognition)
- **ESP8266 Core**: [ESP8266 Community](https://github.com/esp8266/Arduino)

---

## 📧 Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review code comments for parameter explanations
3. Test individual components separately
4. Verify all hardware connections

---

**Happy Building! 🤖**