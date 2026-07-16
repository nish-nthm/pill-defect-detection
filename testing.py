from ultralytics import YOLO
import cv2
import serial
import time
from collections import Counter

MODEL_PATH = "runs/detect/pill_defect_finetuned/weights/best.pt"
CONF_THRESHOLD = 0.25
CAMERA_INDEX = 1
IMG_SIZE = 960

DEFECT_CLASS_NAMES = {"Crack", "stain"}
DEFECT_LIMIT = 0

PILL_GAP_FRAMES = 5           # frames of no detection in ROI -> current pill has exited
NO_DETECTION_TIMEOUT = 30     # frames of no detection in ROI -> batch has ended

ROI = (250, 100, 1000, 600)

ARDUINO_PORT = "COM6"
BAUD_RATE = 9600

model = YOLO(MODEL_PATH)
print("Classes:", model.names)

try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("Arduino connected.")
except Exception as e:
    arduino = None
    print(f"Warning: Arduino not connected ({e}). Running without servo signaling.")


def send_sort():
    if arduino is not None:
        arduino.write(b'1')
    print(">> SORT signal sent (batch rejected)")


def send_pass():
    if arduino is not None:
        arduino.write(b'0')
    print(">> PASS signal sent (batch clean)")


cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"Error: could not open camera index {CAMERA_INDEX}")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(f"Camera resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print("Starting live detection. Press 'q' to quit.")

x1, y1, x2, y2 = ROI

# Current pill being watched (while it's inside the ROI)
current_pill_votes = Counter()
pill_active = False
pill_empty_streak = 0     # consecutive frames with nothing in ROI, resets when a detection appears

# Batch tracking
batch_defect_count = 0
batch_pill_count = 0
batch_class_counts = Counter()
batch_empty_streak = 0    # separate, longer streak for ending the whole batch


def finalize_pill():
    """Called when the current pill leaves the ROI. Locks in its majority-voted class."""
    global current_pill_votes, pill_active, batch_pill_count, batch_defect_count, batch_class_counts

    if not current_pill_votes:
        pill_active = False
        return

    final_class = current_pill_votes.most_common(1)[0][0]
    is_defect = final_class in DEFECT_CLASS_NAMES

    batch_pill_count += 1
    batch_class_counts[final_class] += 1
    if is_defect:
        batch_defect_count += 1

    print(f"Pill finalized: {final_class} ({'DEFECT' if is_defect else 'OK'}) "
          f"| Batch count: {batch_pill_count}")

    current_pill_votes = Counter()
    pill_active = False


def finalize_batch():
    global batch_defect_count, batch_pill_count, batch_class_counts

    if batch_pill_count == 0:
        return

    if batch_defect_count > DEFECT_LIMIT:
        send_sort()
    else:
        send_pass()

    print(f"Batch result: {batch_pill_count} pill(s), {batch_defect_count} defect(s) -> "
          f"{'SORT' if batch_defect_count > DEFECT_LIMIT else 'PASS'}\n")

    batch_pill_count = 0
    batch_defect_count = 0
    batch_class_counts = Counter()


while True:
    ret, raw_frame = cap.read()
    if not ret:
        print("Error: failed to grab frame")
        break

    results = model.predict(source=raw_frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
    result = results[0]
    frame = result.plot()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    boxes = result.boxes
    detections_in_roi = []

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            inside_roi = (x1 <= cx <= x2) and (y1 <= cy <= y2)

            if inside_roi:
                detections_in_roi.append(cls_name)

    if detections_in_roi:
        # Something is in the ROI right now -> accumulate votes for the current pill
        pill_active = True
        pill_empty_streak = 0
        batch_empty_streak = 0
        for cls_name in detections_in_roi:
            current_pill_votes[cls_name] += 1
    else:
        pill_empty_streak += 1
        batch_empty_streak += 1

        # Pill has been gone long enough -> it has fully exited, lock in its result
        if pill_active and pill_empty_streak >= PILL_GAP_FRAMES:
            finalize_pill()

        # No pill activity for a longer stretch -> whole batch has ended
        if batch_empty_streak >= NO_DETECTION_TIMEOUT and batch_pill_count > 0:
            finalize_batch()
            batch_empty_streak = 0

    y = 30
    cv2.putText(frame, f"Crack: {batch_class_counts.get('Crack', 0)}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    y += 30
    cv2.putText(frame, f"Stain: {batch_class_counts.get('stain', 0)}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    y += 30
    cv2.putText(frame, f"Non-defect: {batch_class_counts.get('non defect', 0)}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y += 30
    cv2.putText(frame, f"Total: {batch_pill_count}  Defects: {batch_defect_count}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Pill Defect Detection - Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino is not None:
    arduino.close()