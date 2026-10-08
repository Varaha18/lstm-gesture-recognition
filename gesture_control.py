"""
Real-time gesture recognition for Igus Robolink robot arm control.

M.Eng. thesis code (SRH Berlin, 2025): "Motion and Gesture Recognition in
Human-Robot Interaction Integrating OpenCV and Machine Learning using Igus
Robotic Arm" - Varaha Venkat Karri.

Pipeline: webcam -> MediaPipe Hands (21 landmarks x 3 = 63 features/frame)
-> 30-frame sequence -> LSTM classifier -> CRI command over TCP/IP.

Run:  python gesture_control.py   and pick 1 / 2 / 3 from the menu.
"""

import os
import json
import time
import uuid
import socket
import threading
from collections import deque
from typing import Dict, List, Tuple

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# -----------------------------
# 1) CONFIGURATION
# -----------------------------
# Paths
DATASET_PATH = "captured_videos"
LABELS_PATH = "labels.json"
FIG_DIR = "thesis_figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Model
MODEL_PATH = "gesture_recognition_model.keras"  # use modern Keras format
SEQUENCE_LENGTH = 30   # frames per sample
IMG_WIDTH, IMG_HEIGHT = 640, 480

# Classes (keep in sync with labels.json values)
GESTURES = ["Greet User", "Move Right", "Move Left", "Grasp", "Release", "Idle"]
NUM_CLASSES = len(GESTURES)
CONFIDENCE_THRESHOLD = 0.6

# Training params (explicit for thesis)
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VAL_SIZE = 0.2
RANDOM_STATE = 42

# Robot / CRI
ROBOT_IP = "127.0.0.1"  # change to robot IP for real hardware
ROBOT_PORT = 3920
COMMAND_COUNTER = 1

CRI_COMMANDS = {
    'Greet User': "CRISTART {} CMD Move Joint 0.0 -80.0 90.0 90.0 0.0 0.0 0.0 0.0 0.0 50.0 CRIEND",
    'Move Right': "CRISTART {} CMD Move Joint 90.0 45.0 45.0 90.0 0.0 0.0 0.0 0.0 0.0 50.0 CRIEND",
    'Move Left':  "CRISTART {} CMD Move Joint 90.0 -45.0 -45.0 -90.0 0.0 0.0 0.0 0.0 0.0 50.0 CRIEND",
    'Grasp':      "CRISTART {} CMD Move Joint 0.0 80.0 90.0 0.0 0.0 0.0 0.0 0.0 0.0 50.0 CRIEND",
    'Release':    "CRISTART {} CMD Move Joint 0.0 -80.0 90.0 90.0 0.0 0.0 0.0 0.0 0.0 50.0 CRIEND",
    'Idle':       "CRISTART {} CMD Move Stop CRIEND"
}
CRI_ALIVEJOG = "CRISTART {} ALIVEJOG 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 CRIEND"
alive_thread_running = False


# -----------------------------
# 2) ROBOT UTILS
# -----------------------------
def initialize_robot_connection() -> socket.socket | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((ROBOT_IP, ROBOT_PORT))
        print(f"[Robot] Connected: {ROBOT_IP}:{ROBOT_PORT}")
        return s
    except socket.error as e:
        print(f"[Robot] Connection failed: {e}")
        return None


def send_cri_command(sock: socket.socket, command: str) -> bool:
    global COMMAND_COUNTER
    try:
        msg = command.format(COMMAND_COUNTER)
        sock.sendall(msg.encode("utf-8"))
        COMMAND_COUNTER = 1 if COMMAND_COUNTER >= 9999 else COMMAND_COUNTER + 1
        time.sleep(0.1)
        return True
    except socket.error as e:
        print(f"[Robot] Send error: {e}")
        return False


def send_alive_jog_periodically(sock: socket.socket):
    global alive_thread_running, COMMAND_COUNTER
    print("[Robot] AliveJog thread started")
    while alive_thread_running:
        try:
            msg = CRI_ALIVEJOG.format(COMMAND_COUNTER)
            sock.sendall(msg.encode("utf-8"))
            COMMAND_COUNTER = 1 if COMMAND_COUNTER >= 9999 else COMMAND_COUNTER + 1
            time.sleep(0.5)
        except Exception as e:
            print(f"[Robot] AliveJog thread exiting: {e}")
            break


# -----------------------------
# 3) DATA COLLECTION
# -----------------------------
def _load_labels() -> Dict[str, str]:
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_labels(labels: Dict[str, str]):
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)


def capture_videos():
    """
    Interactive capture of videos for each gesture.
    Press 's' to start recording; press 'q' to stop current recording or exit.
    """
    os.makedirs(DATASET_PATH, exist_ok=True)
    labels_map = _load_labels()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)
    if not cap.isOpened():
        print("[Capture] Webcam not available.")
        return

    for gesture in GESTURES:
        print(f"\nPress 's' to start recording for '{gesture}'. Press 'q' to stop.")
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.putText(frame, f"Ready: {gesture}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Recording Gestures", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                _save_labels(labels_map)
                return

        fname = f"video_{str(uuid.uuid4())}.mp4"
        path = os.path.join(DATASET_PATH, fname)
        labels_map[fname] = gesture

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, 20.0, (IMG_WIDTH, IMG_HEIGHT))
        print(f"[Capture] Recording '{gesture}'... Press 'q' to stop.")

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            out.write(frame)
            show = frame.copy()
            cv2.putText(show, f"RECORDING: {gesture}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Recording Gestures", show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        print(f"[Capture] Saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    _save_labels(labels_map)
    print(f"[Capture] Labels saved -> {LABELS_PATH}")


# -----------------------------
# 4) DATA LOADING & PREPROCESSING
# -----------------------------
def extract_sequences_from_videos() -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads videos in DATASET_PATH using MediaPipe Hands.
    Produces sequences of length SEQUENCE_LENGTH with 21*3=63 features per frame.
    Skips videos with inconsistent landmarks.
    """
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Label file missing: {LABELS_PATH}")

    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        raise FileNotFoundError(f"No videos in: {DATASET_PATH}")

    with open(LABELS_PATH, "r") as f:
        labels_map = json.load(f)

    mp_hands = mp.solutions.hands
    sequences, labels = [], []
    valid_count, skipped = 0, 0

    with mp_hands.Hands(static_image_mode=False, max_num_hands=3) as hands:
        for video_file in tqdm(os.listdir(DATASET_PATH), desc="[Preprocess]"):
            if not video_file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue
            gesture = labels_map.get(video_file)
            if gesture not in GESTURES:
                skipped += 1
                continue

            path = os.path.join(DATASET_PATH, video_file)
            cap = cv2.VideoCapture(path)
            frame_lms = []
            ok_video = True

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(frame_rgb)

                lms = []
                if res.multi_hand_landmarks:
                    # Use the first detected hand for fixed-size input (63 features)
                    first = res.multi_hand_landmarks[0]
                    for lm in first.landmark:
                        lms.extend([lm.x, lm.y, lm.z])

                if len(lms) == 63:
                    frame_lms.append(lms)
                else:
                    ok_video = False
                    break

            cap.release()
            if not ok_video or len(frame_lms) < SEQUENCE_LENGTH:
                skipped += 1
                continue

            # Sliding window extraction
            for i in range(len(frame_lms) - SEQUENCE_LENGTH + 1):
                seq = frame_lms[i:i + SEQUENCE_LENGTH]
                sequences.append(np.array(seq))
                labels.append(gesture)
                valid_count += 1

    if valid_count == 0:
        raise RuntimeError("No valid sequences extracted. Check recordings/labels.")

    # Encode labels to one-hot
    label_to_int = {g: i for i, g in enumerate(GESTURES)}
    y_int = np.array([label_to_int[g] for g in labels], dtype=np.int32)
    X = np.array(sequences, dtype=np.float32)
    y = to_categorical(y_int, num_classes=NUM_CLASSES)

    print(f"[Preprocess] Sequences: {X.shape}, Labels: {y.shape}. Skipped videos: {skipped}")
    return X, y


# -----------------------------
# 5) MODEL
# -----------------------------
def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = Sequential([
        LSTM(128, return_sequences=True, activation="relu", input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_model_architecture(model: tf.keras.Model, out_path: str):
    try:
        tf.keras.utils.plot_model(model, to_file=out_path, show_shapes=True, dpi=200)
        print(f"[Saved] {out_path}")
    except Exception as e:
        print("[Info] Could not render model diagram (install graphviz + pydot).", e)


# -----------------------------
# 6) TRAINING + EVALUATION
# -----------------------------
def plot_training_curves(history: tf.keras.callbacks.History, out_path: str):
    plt.figure(figsize=(10, 4))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Model Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] {out_path}")


def save_classification_assets(y_true_int: np.ndarray,
                               y_pred_int: np.ndarray,
                               out_report_txt: str,
                               out_cm_png: str,
                               out_cm_norm_png: str,
                               out_bars_png: str):
    # Classification report
    report = classification_report(y_true_int, y_pred_int, target_names=GESTURES, digits=3)
    with open(out_report_txt, "w") as f:
        f.write(report)
    print(f"[Saved] {out_report_txt}")

    # Confusion Matrix
    cm = confusion_matrix(y_true_int, y_pred_int, labels=np.arange(NUM_CLASSES))

    # Raw CM
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix (Counts)")
    plt.colorbar()
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, GESTURES, rotation=45, ha="right")
    plt.yticks(tick_marks, GESTURES)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_cm_png)
    plt.close()
    print(f"[Saved] {out_cm_png}")

    # Normalized CM
    with np.errstate(all='ignore'):
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm_norm, interpolation='nearest')
    plt.title("Confusion Matrix (Proportions)")
    plt.colorbar()
    plt.xticks(tick_marks, GESTURES, rotation=45, ha="right")
    plt.yticks(tick_marks, GESTURES)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_cm_norm_png)
    plt.close()
    print(f"[Saved] {out_cm_norm_png}")

    # Per-class Precision/Recall/F1 bars
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_int, y_pred_int, labels=np.arange(NUM_CLASSES), zero_division=0
    )
    x = np.arange(NUM_CLASSES)
    width = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, prec, width, label='Precision')
    plt.bar(x,         rec,  width, label='Recall')
    plt.bar(x + width, f1,   width, label='F1-score')
    plt.xticks(x, GESTURES, rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1-score per Gesture")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_bars_png)
    plt.close()
    print(f"[Saved] {out_bars_png}")


def train_and_evaluate():
    print("[Train] Loading & preprocessing dataset ...")
    X, y = extract_sequences_from_videos()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=RANDOM_STATE,
        stratify=np.argmax(y, axis=1)
    )

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()

    # Optional: model diagram
    plot_model_architecture(model, os.path.join(FIG_DIR, "model_architecture.png"))

    print(f"[Train] Starting training: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Save model
    model.save(MODEL_PATH)
    print(f"[Saved] {MODEL_PATH}")

    # Training curves
    plot_training_curves(history, os.path.join(FIG_DIR, "training_curves.png"))

    # Evaluate
    y_test_int = np.argmax(y_test, axis=1)
    y_pred_int = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # Save reports & figures
    save_classification_assets(
        y_true_int=y_test_int,
        y_pred_int=y_pred_int,
        out_report_txt=os.path.join(FIG_DIR, "classification_report.txt"),
        out_cm_png=os.path.join(FIG_DIR, "confusion_matrix.png"),
        out_cm_norm_png=os.path.join(FIG_DIR, "confusion_matrix_percent.png"),
        out_bars_png=os.path.join(FIG_DIR, "gesture_metrics_bar.png")
    )

    # Print summary to console
    report_text = classification_report(y_test_int, y_pred_int, target_names=GESTURES, digits=3)
    print("\n--- Classification Report ---\n" + report_text)


# -----------------------------
# 7) REAL-TIME RECOGNITION & CONTROL
# -----------------------------
def run_realtime_control():
    if not os.path.exists(MODEL_PATH):
        print(f"[Runtime] Model '{MODEL_PATH}' not found. Train first.")
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[Runtime] Model loaded.")
    except Exception as e:
        print(f"[Runtime] Model load error: {e}")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=3,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)
    if not cap.isOpened():
        print("[Runtime] Webcam not available.")
        return

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    current_gesture = "No Gesture"
    last_command_sent = None

    # Robot
    robot_socket = initialize_robot_connection()
    if robot_socket:
        global alive_thread_running
        alive_thread_running = True
        alive_thread = threading.Thread(target=send_alive_jog_periodically, args=(robot_socket,), daemon=True)
        alive_thread.start()
    else:
        print("[Runtime] Robot not connected - simulation mode (no commands sent).")

    print("\n[Runtime] Starting. Press 'q' to quit.")
    latency_ms_list = []
    gesture_to_idx = {i: g for i, g in enumerate(GESTURES)}

    confidence_display = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        landmarks = []
        if results.multi_hand_landmarks:
            first = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, first, mp_hands.HAND_CONNECTIONS)
            for lm in first.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == 63:
            sequence.append(landmarks)
        else:
            # keep temporal length stable; append zeros if no hand to avoid stalling
            sequence.append([0.0] * 63)

        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(list(sequence), axis=0)
            probs = model.predict(input_data, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            confidence_display = confidence

            if confidence >= CONFIDENCE_THRESHOLD:
                predicted = gesture_to_idx[pred_idx]
                if predicted != current_gesture:
                    current_gesture = predicted
                    print(f"[Runtime] Detected: {current_gesture} (conf {confidence:.2f})")

                    if robot_socket and current_gesture in CRI_COMMANDS and current_gesture != last_command_sent:
                        cmd = CRI_COMMANDS[current_gesture]
                        t0 = time.time()
                        ok = send_cri_command(robot_socket, cmd)
                        t1 = time.time()
                        if ok:
                            lat = (t1 - t0) * 1000.0
                            latency_ms_list.append(lat)
                            print(f"[Runtime] Command latency: {lat:.2f} ms")
                            last_command_sent = current_gesture
            else:
                current_gesture = "No Gesture"

        # HUD
        cv2.rectangle(frame, (0, 0), (320, 70), (245, 117, 16), -1)
        cv2.putText(frame, 'DETECTED GESTURE', (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.putText(frame, current_gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if len(sequence) == SEQUENCE_LENGTH:
            cv2.putText(frame, f"Conf: {confidence_display:.2f}", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if robot_socket:
        alive_thread_running = False
        robot_socket.close()

    # Latency summary
    if latency_ms_list:
        avg = sum(latency_ms_list) / len(latency_ms_list)
        print("\n--- Performance Summary ---")
        print(f"Total commands sent: {len(latency_ms_list)}")
        print(f"Average end-to-end latency: {avg:.2f} ms")
    else:
        print("\n[Runtime] No commands sent; no latency measured.")


# -----------------------------
# 8) MAIN MENU
# -----------------------------
def main():
    print("Welcome to the Thesis Project Application.")
    print("Please choose an option:")
    print("1. Record new gesture videos (and re-train model)")
    print("2. Train/Evaluate on existing dataset (no recording)")
    print("3. Run real-time gesture recognition and robot control")
    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == '1':
        capture_videos()
        train_and_evaluate()
    elif choice == '2':
        train_and_evaluate()
    elif choice == '3':
        run_realtime_control()
    else:
        print("Invalid choice. Please restart and choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
