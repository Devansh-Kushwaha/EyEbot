import cv2
import difflib
import speech_recognition as sr
from ultralytics import YOLO
import threading

# Load YOLO model
model = YOLO("yolov8n.pt")
yolo_labels = [name.lower() for name in model.names.values()]

# Global target object (shared between threads)
target_object = None
lock = threading.Lock()

# Speech recognizer
recognizer = sr.Recognizer()
mic = sr.Microphone()

def extract_object_name(command):
    """
    Try to extract a YOLO object name from speech text.
    First check direct match, then fuzzy match.
    """
    command = command.lower()

    # Direct substring match
    for obj in yolo_labels:
        if obj in command:
            return obj

    # Otherwise fuzzy match
    words = command.split()
    for w in words:
        best = difflib.get_close_matches(w, yolo_labels, n=1, cutoff=0.7)
        if best:
            return best[0]
    return None

def listen_loop():
    global target_object
    while True:
        with mic as source:
            print("🎤 Say the object you want to find...")
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
                command = recognizer.recognize_google(audio).lower()
                print(f"🗣️ You said: {command}")

                obj = extract_object_name(command)
                if obj:
                    with lock:
                        target_object = obj
                    print(f"✅ Target object set to: {obj}")
                else:
                    print("⚠️ No valid object found in command")
            except Exception as e:
                print(f"⚠️ {e}")

# Start microphone thread
threading.Thread(target=listen_loop, daemon=True).start()

# Capture video
cap = cv2.VideoCapture(0)

zoom_window_open = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    found_object = False  # Track if target is visible this frame

    results = model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            with lock:
                if target_object and label == target_object:
                    found_object = True
                    print(f"🎯 Found {target_object}!")
                    
                    # Add margin around object
                    margin = 30
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)

                    # Draw rectangle & label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Zoom view (keep aspect ratio, no stretching)
                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size > 0:
                        h, w = cropped.shape[:2]
                        scale = min(frame.shape[1] / w, frame.shape[0] / h)
                        new_w, new_h = int(w * scale), int(h * scale)
                        zoomed = cv2.resize(cropped, (new_w, new_h))

                        # Center the zoomed image on black background
                        zoom_canvas = frame.copy()
                        zoom_canvas[:] = 0
                        y_offset = (zoom_canvas.shape[0] - new_h) // 2
                        x_offset = (zoom_canvas.shape[1] - new_w) // 2
                        zoom_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = zoomed

                        cv2.imshow("Zoomed View", zoom_canvas)
                        zoom_window_open = True

    # If object not found, close zoom window
    if not found_object and zoom_window_open:
        cv2.destroyWindow("Zoomed View")
        zoom_window_open = False

    cv2.imshow("Object Finder", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
