# lstm-gesture-recognition
Real-time gesture recognition for robotic arm control using LSTM and MediaPipe. 91.2% accuracy, 61.4ms latency.
Real-Time Gesture Recognition for Robotic Arm Control
## Overview
LSTM-based gesture recognition system achieving 91.2% accuracy for controlling an Igus Robolink robotic arm in real-time using only a laptop webcam.

## Key Features
- **91.2% Classification Accuracy** across 6 gesture classes (Greet, Move Left/Right, Grasp, Release, Idle)
- **61.4 ms Average Latency** (well below 100ms perception threshold)
- **Hardware Integration** - Direct control of Igus robotic arm via CRI protocol
- **Consumer-Grade Hardware** - Runs on CPU without GPU acceleration
- **Open-Source Tools** - MediaPipe, TensorFlow/Keras, OpenCV

## Performance Metrics
| Gesture | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Release | 1.000 | 1.000 | 1.000 |
| Idle | 1.000 | 0.933 | 0.966 |
| Move Left | 1.000 | 0.750 | 0.867 |
| Move Right | 1.000 | 0.750 | 0.857 |
| Greet User | 0.810 | 1.000 | 0.895 |
| Grasp | 0.700 | 1.000 | 0.824 |

## Technology Stack
- **Deep Learning Framework**: TensorFlow/Keras
- **Computer Vision**: OpenCV, MediaPipe (21-point hand landmark detection)
- **Model Architecture**: LSTM (128 → 64 units with dropout, ReLU + Softmax)
- **Real-Time Inference**: 30-frame temporal sequences at 30 fps
- **Robot Integration**: TCP/IP socket communication via CRI protocol

## Dataset
- **Videos**: 97 raw gesture videos (30 fps, 640×480 resolution)
- **Preprocessed Sequences**: 133 valid 30-frame landmark sequences
- **Classes**: 6 hand gestures
- **Training/Test Split**: 80/20 (stratified)

## Model Architecture
```
Input (30×63 sequence)
    ↓
LSTM Layer 1 (128 units, return_sequences=True)
Dropout (0.2)
    ↓
LSTM Layer 2 (64 units)
Dropout (0.2)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Output Layer (6 units, Softmax → gesture classes)
```

## Results
- **Training Accuracy**: Converged to ~90%+ by epoch 6
- **Test Accuracy**: 91.2% macro average
- **Latency**: 50-80ms (mean 61.4ms) ✓ Real-time capable
- **Usability Rating**: 4.4/5 gesture intuitiveness, 4.2/5 responsiveness

## Installation
```bash
pip install tensorflow>=2.0 opencv-python mediapipe numpy
```

## Usage
```python
from gesture_recognition import GestureRecognizer
from robot_control import RobotController

# Initialize gesture recognizer
recognizer = GestureRecognizer(model_path='trained-model.h5')

# Initialize robot connection
robot = RobotController(ip='192.168.1.100', port=1234)

# Run real-time recognition
recognizer.run_realtime(robot_controller=robot)
```

## Files
- `lstm_model_training.ipynb` - Complete training pipeline
- `gesture_recognition.py` - Real-time inference
- `robot_control.py` - Igus robot integration via CRI
- `results/` - Confusion matrix, training curves, metrics

## References
- Master's Thesis: "Motion and Gesture Recognition in Human-Robot Interaction Integrating OpenCV and Machine Learning using Igus Robotic Arm" (Sept 2025, SRH Berlin)
- MediaPipe Documentation: https://mediapipe.dev/
- Igus CRI Protocol: https://www.igus.eu/info/robotics-software-irc

## Author
Varaha Venkat Karri  
varaha.venkat@icloud.com | Berlin, Germany
