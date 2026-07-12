from ultralytics import YOLO
import cv2
import serial
import time
from collections import Counter, deque

MODEL_PATH = "runs/detect/pill_defect_v1_nano-2/weights/best.pt"
CONF_THRESHOLD = 0.20
CAMERA_INDEX = 1
IMG_SIZE = 960

DEFECT_CLASS_NAMES = {"Crack", "stain"}
VOTE_WINDOW = 7
VOTE_THRESHOLD = 0.5

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


def send_servo_on():
    if arduino is not None:
        arduino.write(b'1')
    print(">> Servo ON (defect confirmed)")


def send_servo_off():
    if arduino is not None:
        arduino.write(b'0')


cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"Error: could not open camera index {CAMERA_INDEX}")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {actual_w}x{actual_h}")

print("Starting live detection. Press 'q' to quit.")

vote_buffer = deque(maxlen=VOTE_WINDOW)
servo_state = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: failed to grab frame")
        break

    results = model.predict(source=frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
    result = results[0]
    frame = result.plot()

    counts = Counter()
    frame_has_defect = 0
    if result.boxes is not None:
        for cls_id in result.boxes.cls:
            cls_name = model.names[int(cls_id)]
            counts[cls_name] += 1
            if cls_name in DEFECT_CLASS_NAMES:
                frame_has_defect = 1

    vote_buffer.append(frame_has_defect)
    defect_ratio = sum(vote_buffer) / len(vote_buffer)
    should_turn_on = defect_ratio >= VOTE_THRESHOLD

    if should_turn_on and not servo_state:
        servo_state = True
        send_servo_on()
    elif not should_turn_on and servo_state:
        servo_state = False
        send_servo_off()

    y = 30
    cv2.putText(frame, f"Crack: {counts.get('Crack', 0)}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    y += 30
    cv2.putText(frame, f"Stain: {counts.get('stain', 0)}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    y += 30
    cv2.putText(frame, f"Non-defect: {counts.get('non defect', 0)}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y += 40
    cv2.putText(frame, f"Vote: {sum(vote_buffer)}/{len(vote_buffer)}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    y += 30
    servo_text = "SERVO: ON" if servo_state else "SERVO: OFF"
    servo_color = (0, 0, 255) if servo_state else (0, 255, 0)
    cv2.putText(frame, servo_text, (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, servo_color, 2)

    cv2.imshow("Pill Defect Detection - Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if arduino is not None:
    arduino.close()
cv2.destroyAllWindows()