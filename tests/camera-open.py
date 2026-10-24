import cv2

for idx in [0, 1, 2, 3, 4, 5, 6, 7]:
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Index {idx} SUCCESS -> Resolution: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"Index {idx} failed to read frame")
        cap.release()
    else:
        print(f"Index {idx} failed to open")
