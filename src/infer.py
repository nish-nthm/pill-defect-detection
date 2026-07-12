from ultralytics import YOLO
import cv2
from collections import Counter

MODEL_PATH = "runs/detect/pill_defect_v1_nano-2/weights/best.pt"
CONF_THRESHOLD = 0.25
CAMERA_INDEX = 1
IMG_SIZE = 960  # match your training imgsz

model = YOLO(MODEL_PATH)
print("Classes:", model.names)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"Error: could not open camera index {CAMERA_INDEX}")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {actual_w}x{actual_h}")

print("Starting live detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: failed to grab frame")
        break

    results = model.predict(source=frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
    result = results[0]
    frame = result.plot(conf=False)

    counts = Counter()
    if result.boxes is not None:
        for cls_id in result.boxes.cls:
            counts[model.names[int(cls_id)]] += 1

    y = 30
    cv2.putText(frame, f"Crack: {counts.get('Crack', 0)}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    y += 30
    cv2.putText(frame, f"Stain: {counts.get('stain', 0)}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    y += 30
    cv2.putText(frame, f"Non-defect: {counts.get('non defect', 0)}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Pill Defect Detection - Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()