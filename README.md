# Gesture Control for an Igus Robot Arm (LSTM + MediaPipe)

Real-time hand-gesture recognition that drives an **Igus Robolink** robot arm, using only a standard laptop webcam. Built for my M.Eng. thesis at SRH Berlin (2025).

**91.2% overall accuracy** on 6 gestures · **~61 ms average command latency** · runs on a laptop CPU, no GPU or depth camera

## How it works

```
Laptop webcam (640×480, 30 fps)
   ↓
MediaPipe Hands — 21 landmarks × (x, y, z) = 63 features per frame
   ↓
30-frame sliding window → input of shape (30, 63)
   ↓
LSTM classifier → one of 6 gestures (confidence threshold 0.6)
   ↓
CRI command string → Igus Robolink arm over TCP/IP
```

Gestures: **Greet User, Move Right, Move Left, Grasp, Release, Idle**

### Why MediaPipe?
I compared it with OpenPose and BlazePose. OpenPose is accurate but too slow on a normal CPU for real-time control. BlazePose is built for full-body pose, not fine hand movement. MediaPipe Hands gives 21 precise 3D hand landmarks and runs in real time on a CPU.

### Why an LSTM?
A gesture is a movement, not a single pose. The LSTM classifies the motion across 30 frames, which separates gestures that look similar in any single frame.

## Model

```
Input (30 × 63)
LSTM 128 units (return_sequences=True) → Dropout 0.2
LSTM 64 units → Dropout 0.2
Dense 64 (ReLU)
Dense 6 (Softmax)
```
Adam (lr = 1e-3), categorical cross-entropy, batch size 32, 20 epochs, 80/20 stratified split.

## Results

| Gesture | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Greet User | 0.810 | 1.000 | 0.895 | 17 |
| Move Right | 1.000 | 0.750 | 0.857 | 16 |
| Move Left | 1.000 | 0.750 | 0.867 | 17 |
| Grasp | 0.700 | 1.000 | 0.824 | 14 |
| Release | 1.000 | 1.000 | 1.000 | 20 |
| Idle | 1.000 | 0.933 | 0.966 | 30 |
| **Overall accuracy** | | | **0.912** | 114 |
| Macro average | 0.918 | 0.908 | 0.901 | 114 |

**Main weakness:** Move Right was sometimes read as Greet User, and Move Left as Grasp (about 25% of cases each), because their hand paths overlap. Better gesture design or more data would help most here.

**Latency:** mean 61.4 ms (range 49–80 ms), measured from gesture recognition to command execution. That's below the ~100 ms point where people start to notice delay.

> Note: the code as published here includes a fixed `time.sleep(0.1)` pause in `send_cri_command()`, inside the section the latency timer measures. The latency figures above are the values reported in the thesis; a fresh measurement with this exact code will include that pause.

**Usability (pilot, 5 participants, 1–5 scale):** intuitiveness 4.4, responsiveness 4.2, ease of use 4.0.

## Run it

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python gesture_control.py
```

The menu offers three options:
1. **Record** new gesture videos with the webcam (press `s` to start, `q` to stop), then train
2. **Train and evaluate** on an existing `captured_videos/` folder with `labels.json`
3. **Run real-time recognition** and robot control

Without a robot connected, option 3 runs in simulation mode: recognition works, no commands are sent. For real hardware, set `ROBOT_IP` in the config section (the default CRI port is 3920).

Training writes the model to `gesture_recognition_model.keras` and the figures (confusion matrices, training curves, per-gesture metrics) to `thesis_figures/`.

## Limitations

This is a proof of concept. The dataset is small: 97 videos from 3 people, recorded under controlled indoor lighting against a plain background. The sliding-window sequences come from overlapping frames of the same videos, so the test score probably overstates performance on new users. Next steps: more participants, harder lighting, a video-level train/test split, and testing a GRU or Transformer encoder.

## Repository

| File | Contents |
|---|---|
| `gesture_control.py` | Full pipeline: data capture, preprocessing, training, evaluation, real-time control |
| `requirements.txt` | Python dependencies |
| `3117839@MASTER THESIS.pdf` | Full thesis |

The recorded videos are not included.

## Author

**Varaha Venkat Karri** · M.Eng. Industry 4.0, SRH Berlin
[LinkedIn](https://www.linkedin.com/in/varaha-venkat) · varaha.venkat@icloud.com
