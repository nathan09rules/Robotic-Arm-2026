import cv2
import numpy as np
import time
import urllib.request

# ===== YOUR BOARD =====
board = cv2.aruco.CharucoBoard(
    (7, 5),
    25,
    18,
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
)

detector = cv2.aruco.CharucoDetector(board)

corners_all = []
ids_all = []

last_capture = time.time() - 2
frame_count = 0

cap = cv2.VideoCapture(4)

while len(corners_all) < 20:

    success, frame = cap.read()
    
    if not success:
        continue

    # Capture every 2 seconds
    if time.time() - last_capture >= 2:

        corners, ids, _, _ = detector.detectBoard(frame)

        if ids is not None and len(ids) >= 6:

            corners_all.append(corners)
            ids_all.append(ids)

            frame_count += 1
            print(f"Captured {frame_count}/20")

        else:
            print("Board not detected")

        last_capture = time.time()


# ===== CHECK =====

if len(corners_all) < 10:
    raise RuntimeError(
        f"Only captured {len(corners_all)} usable frames."
    )


# ===== CALIBRATE =====

obj_points = []
img_points = []

board_corners = board.getChessboardCorners()

for corners, ids in zip(corners_all, ids_all):

    ids = ids.flatten()

    # Known 3D locations of ChArUco corners
    obj_points.append(board_corners[ids])

    # Detected 2D image locations
    img_points.append(corners.reshape(-1, 2))


h, w = frame.shape[:2]

error, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points,
    img_points,
    (w, h),
    None,
    None
)


# ===== SAVE =====

np.savez(
    "calibration.npz",
    camera_matrix=K,
    distortion=dist
)

print("\nDone!")
print("Reprojection error:", error)
print("Camera matrix:")
print(K)
print("Distortion:")
print(dist)
print("\nSaved calibration.npz")
