import cv2
import torch
import threading
import queue
import requests
import time
import speech_recognition as sr
import difflib
import re
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------
# Config
# -------------------------
CAR_IP = "172.28.188.69"
CONF_THRESHOLD = 0.3
FRAME_QUEUE_MAX = 3
PROCESSED_QUEUE_MAX = 3
SEARCH_ROTATE_SECS = 1
MICRO_TURN_SECS = 1
MAX_RECHECK_SECONDS = 5.0  # max time to wait after rotation while trying to re-detect
frames_since_seen = 0
MAX_MISS_FRAMES = 5

# -------------------------
# Globals
# -------------------------
target_object = None
frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAX)
processed_frame_queue = queue.Queue(maxsize=PROCESSED_QUEUE_MAX)
lock = threading.Lock()
last_state = None

# -------------------------
# Helpers
# -------------------------
def clear_queue(q):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        return
    
last_command_time = 0
command_lock = threading.Lock()

def send_command_http(cmd, timeout=0.4):
    """Send command to car safely with cooldown and no overlaps."""
    global last_state, last_command_time

    def _send():
        global last_state, last_command_time
        with command_lock:
            now = time.time()
            # Cooldown: 150 ms to avoid command spamming
            if now - last_command_time < 0.15:
                return

            if cmd != last_state:
                url = f"http://{CAR_IP}/?State={cmd}"
                print(f"🚗 Sending command → {url}")
                try:
                    requests.get(url, timeout=timeout)
                    last_state = cmd
                    last_command_time = now
                except requests.exceptions.RequestException as e:
                    print(f"⚠️ Request error: {e}")

    threading.Thread(target=_send, daemon=True).start()

def car_stop(): send_command_http("S")
def car_forward(): send_command_http("F")
def car_left(): send_command_http("L")
def car_right(): send_command_http("R")
def car_rotate(): send_command_http("R")  # rotate mapping for your firmware

# -------------------------
# Load model
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "yolov5n" if DEVICE == "cpu" else "yolov5s"
print(f"[INIT] device={DEVICE}, model={MODEL_NAME}")

model = torch.hub.load("ultralytics/yolov5", MODEL_NAME, verbose=False)
model.conf = CONF_THRESHOLD
if DEVICE == "cuda":
    model.to(DEVICE)

yolo_labels = [name.lower() for name in model.names.values()]

# -------------------------
# Speech recognition
# -------------------------
recognizer = sr.Recognizer()
mic = sr.Microphone()

def extract_object_name(text):
    text = text.lower()
    if "stop" in text:
        return "STOP"
    for label in yolo_labels:
        if label in text:
            return label
    words = re.findall(r"\w+", text)
    for w in words:
        best = difflib.get_close_matches(w, yolo_labels, n=1, cutoff=0.6)
        if best:
            return best[0]
    return None

def callback(recognizer_obj, audio):
    global target_object
    try:
        command = recognizer_obj.recognize_google(audio).lower()
    except Exception:
        return
    if "stop" in command:
        with lock:
            target_object = None
        car_stop()
        print(f"🗣️ {command} → 🛑 Stop tracking.")
        return
    obj = extract_object_name(command)
    if obj:
        with lock:
            target_object = obj
        print(f"🗣️ {command} → 🎯 Target set: {obj}")
    else:
        print(f"🗣️ {command} → ⚠️ No valid object found.")

# -------------------------
# Detection helpers
# -------------------------
def run_model_on_rgb(rgb_img):
    """Run the model on an RGB numpy image and return numpy detections or None."""
    results = model(rgb_img)
    xyxy_np = None
    try:
        if hasattr(results, "xyxy") and len(results.xyxy) > 0:
            xyxy_np = results.xyxy[0].cpu().numpy()
    except Exception:
        try:
            df = results.pandas().xyxy[0]
            if not df.empty:
                xyxy_np = df[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']].values
        except Exception:
            xyxy_np = None
    return xyxy_np

def detection_matches_object(xyxy_np, obj_name):
    """Return first matching detection (x1,y1,x2,y2,conf,cls) or None."""
    if xyxy_np is None:
        return None
    for det in xyxy_np:
        if len(det) < 6:
            continue
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]
        if label.lower() == obj_name.lower() and float(conf) > CONF_THRESHOLD:
            return (int(x1), int(y1), int(x2), int(y2), float(conf), int(cls))
    return None

# -------------------------
# Detection thread
# -------------------------
def detection_loop():
    global target_object, frames_since_seen
    frame_counter = 0

    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        frame_counter += 1

        # Only run heavy detection on every 3rd frame
        if frame_counter % 3 != 0:
            try:
                if not processed_frame_queue.full():
                    processed_frame_queue.put_nowait(frame)
            except queue.Full:
                pass
            continue

        with lock:
            obj = target_object
        if not obj:
            try:
                if not processed_frame_queue.full():
                    processed_frame_queue.put_nowait(frame)
            except queue.Full:
                pass
            continue

        # convert to RGB and run model
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        xyxy_np = run_model_on_rgb(rgb)
        match = detection_matches_object(xyxy_np, obj)

        if match is not None:
            global frames_since_seen
            frames_since_seen = 0  # reset counter when detected
            # continue

            x1, y1, x2, y2, conf, cls = match
            obj_center_x = (x1 + x2) // 2
            frame_center_x = frame.shape[1] // 2
            deviation = obj_center_x - frame_center_x

            # draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{model.names[cls]} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            align_thresh = 80
            margin = 50 # deadband margin to ignore small deviations
            is_stabilizing = False

            # Alignment logic with tolerance
            if abs(deviation) <= align_thresh:
                print(f"✅ {obj} well aligned (deviation={deviation}) → moving forward")
                car_forward()
                time.sleep(0.5)
                car_stop()
                is_stabilizing = True

            elif abs(deviation) <= align_thresh + margin:
                print(f"⚖️ {obj} slightly off-center (deviation={deviation}) → ignoring small deviation")
                # Don’t move, just keep watching (to avoid unnecessary micro-turns)

            elif deviation > 0:
                print(f"↩️ {obj} right of center (deviation={deviation}) → micro-turn right")
                car_right()
                time.sleep(MICRO_TURN_SECS)
                car_stop()
                print("🔎 Scanning after right micro-align...")
                is_stabilizing = True

            else:
                print(f"↪️ {obj} left of center (deviation={deviation}) → micro-turn left")
                car_left()
                time.sleep(MICRO_TURN_SECS)
                car_stop()
                print("🔎 Scanning after left micro-align...")
                is_stabilizing = True

            # Stabilization scan delay
            if is_stabilizing:
                time.sleep(1.2)
                clear_queue(frame_queue)  # discard stale frames before new detection
                clear_queue(processed_frame_queue)
                print("⏸️ Waiting for stable camera view...")

            try:
                if not processed_frame_queue.full():
                    processed_frame_queue.put_nowait(frame)
            except queue.Full:
                pass
            continue

        
 
        
        


        # --- not found: only trigger search if object missing for several frames ---
        frames_since_seen += 1
        if frames_since_seen < MAX_MISS_FRAMES:
            try:
                if not processed_frame_queue.full():
                    processed_frame_queue.put_nowait(frame)
            except queue.Full:
                pass
            continue

        print(f"🔍 {obj} not found for {frames_since_seen} frames — rotating to search...")
        frames_since_seen = 0

        clear_queue(frame_queue)
        clear_queue(processed_frame_queue)

        car_rotate()
        time.sleep(SEARCH_ROTATE_SECS)
        car_stop()

        # dynamic re-check loop
        recheck_start = time.time()
        found_again = False
        while time.time() - recheck_start < MAX_RECHECK_SECONDS:
            try:
                new_frame = frame_queue.get(timeout=0.6)
            except queue.Empty:
                continue

            rgb_new = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
            xyxy_new = run_model_on_rgb(rgb_new)
            match_new = detection_matches_object(xyxy_new, obj)
            if match_new is not None:
                x1, y1, x2, y2, conf, cls = match_new
                cv2.rectangle(new_frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(new_frame, f"{model.names[cls]} {conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                print("✅ Object re-detected during recheck.")
                car_stop()  # 🚨 Immediately stop rotation
                found_again = True
                break
                try:
                    if not processed_frame_queue.full():
                        processed_frame_queue.put_nowait(new_frame)
                except queue.Full:
                    pass

                print("✅ Object re-detected during recheck.")
                found_again = True
                break
            else:
                try:
                    if not processed_frame_queue.full():
                        processed_frame_queue.put_nowait(new_frame)
                except queue.Full:
                    pass
                continue

        if not found_again:
            print(f"⏳ Re-check timed out ({MAX_RECHECK_SECONDS}s) — will rotate again if still not found.")

# -------------------------
# Main (camera + UI)
# -------------------------
if __name__ == "__main__":
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    stop_listening = recognizer.listen_in_background(mic, callback, phrase_time_limit=4)

    threading.Thread(target=detection_loop, daemon=True).start()

    cap = cv2.VideoCapture("http://192.0.0.4:8080/video")
    if not cap.isOpened():
        print("⚠️ Failed to open camera stream. Check URL.")
    skip_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.12)
                continue

            frame = cv2.resize(frame, (640, 480))
            skip_counter += 1

            if skip_counter % 2 == 0:
                try:
                    if not frame_queue.full():
                        frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass

            try:
                if not processed_frame_queue.empty():
                    display_frame = processed_frame_queue.get_nowait()
                else:
                    display_frame = frame
            except queue.Empty:
                display_frame = frame

            cv2.imshow("Object Tracker", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_listening(wait_for_stop=False)
        cap.release()
        cv2.destroyAllWindows()
