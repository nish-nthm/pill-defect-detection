import cv2

print("Scanning for available cameras...\n")

for backend_name, backend in [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]:
    print(f"--- Backend: {backend_name} ---")
    for i in range(5):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  Index {i}: OPENED and frame read OK, shape={frame.shape}")
            else:
                print(f"  Index {i}: opened but failed to read frame")
        else:
            print(f"  Index {i}: failed to open")
        cap.release()
    print()