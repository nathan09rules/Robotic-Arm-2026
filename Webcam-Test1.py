
import cv2
import time
import threading
from pathlib import Path

import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# CUSTOM CODE
from Utils.Webcam import *
from Utils.BlenderArm import finger

# Load stereo calibration
calibration = load_calibration(CALIBRATION_FILE)
print("Loaded stereo calibration:", CALIBRATION_FILE)

model = Path(MODEL_PATH)
if not model.exists():
    raise FileNotFoundError(f"MediaPipe model not found: {model}")

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_tracking_confidence=0.8
)

cam1 = cv2.VideoCapture(2)
cam2 = cv2.VideoCapture(4)

landmarker_cam1 = mp.tasks.vision.HandLandmarker.create_from_options(options)
landmarker_cam2 = mp.tasks.vision.HandLandmarker.create_from_options(options)

if not cam1.isOpened() or not cam2.isOpened():
    cam1.release()
    cam2.release()
    landmarker_cam1.close()
    landmarker_cam2.close()
    raise RuntimeError("Could not open both cameras")

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Triangulated 3D Hand")

scatter = ax.scatter([], [], [], color="red", s=20)
lines = [
    ax.plot([], [], [], color="green", linewidth=2)[0]
    for _ in HAND_CONNECTIONS
]
artists = [scatter] + lines

start = time.perf_counter()
last1 = -1
last2 = -1

# Starting target: MediaPipe landmark 8 (index-finger tip)
start_position = None
target = None

try:
    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()

        if not ret1 or frame1 is None or not ret2 or frame2 is None:
            continue

        now_ms = int((time.perf_counter() - start) * 1000)
        ts1 = max(now_ms, last1 + 1)
        ts2 = max(now_ms, last2 + 1)
        last1, last2 = ts1, ts2

        result1 = process_and_display(
            frame1, landmarker_cam1, ts1, "Camera 1"
        )
        result2 = process_and_display(
            frame2, landmarker_cam2, ts2, "Camera 2"
        )

        if not result1.hand_landmarks or not result2.hand_landmarks:
            plt.pause(0.001)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        Postion = Plot_Hand(
            calibration, result1, result2, ax, artists
        )

        if Postion is not None and len(Postion) > 8:
            # Capture landmark 8 once as the initial 3D target
            if start_position is None:
                start_position = np.array(
                    Postion[8], dtype=float
                ).copy()

                target = start_position.copy()

                print(
                    "\nStarting position set from landmark 8:",
                    start_position
                )

            Bone = [
                ["THUMB", 4, 13, 0.06, 0, 2],
                ["INDEX", 8, 5, 0.06, 0, 2],
                ["MIDDLE", 12, 9, 0.06, 0, 2],
                ["RING", 16, 13, 0.06, 0, 2],
                ["PINKY", 20, 17, 0.06, 0, 2],
            ]

            SPEACIAL_STATES = [
                ["INDEX-THUMB", 4, 8, 0.02, 0, -1],
            ]

            output = ""
            Input_Joint, Input_state = "", 0

            for Joint in Bone:
                distance = np.linalg.norm(
                    Postion[Joint[1]] - Postion[Joint[2]]
                )

                if distance < Joint[3]:
                    finger(Joint[0], Joint[4])
                    state = "CURLED"
                else:
                    finger(Joint[0], Joint[5])
                    state = "EXTENDED"

                output += (
                    f"\r{Joint[0]:<8}: "
                    f"{distance:.3f} [{state}]           \n"
                )

            for Joint in SPEACIAL_STATES:
                distance = np.linalg.norm(
                    Postion[Joint[1]] - Postion[Joint[2]]
                )

                if distance < Joint[3]:
                    finger(Joint[0], Joint[4])
                    state = "TRUE"
                else:
                    state = "FALSE"

                output += (
                    f"\r{Joint[0]:<12}: "
                    f"{distance:.3f} [{state}]           \n"
                )

            if Input_Joint:
                finger(Input_Joint, Input_state)

            output += (
                f"START TARGET: {target}"
                if target is not None
                else "Waiting for starting target..."
            )

            print(
                output
                + f"\033[{len(Bone) + len(SPEACIAL_STATES) + 1}A",
                end="",
                flush=True
            )

        plt.pause(0.001)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cam1.release()
    cam2.release()
    landmarker_cam1.close()
    landmarker_cam2.close()
    cv2.destroyAllWindows()
    plt.close("all")
    print()

