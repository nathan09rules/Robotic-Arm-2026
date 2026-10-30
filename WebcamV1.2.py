import cv2
import time
import threading
from pathlib import Path
import asyncio

import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

#CUSTOM CODE
from Utils.Webcam import *
from Utils.BlenderArm import finger, HandState
from Utils.Arm import move_to_target

arm_loop = asyncio.new_event_loop()

def run_arm_loop():
    asyncio.set_event_loop(arm_loop)
    arm_loop.run_forever()

threading.Thread(target=run_arm_loop, daemon=True).start()

X_LIMIT = (-0.25, 0.25)
Y_LIMIT = (-0.25, 0.25)
Z_LIMIT = ( 0.0, 0.5)

calibration = load_calibration(CALIBRATION_FILE)
print("Loaded stereo calibration:", CALIBRATION_FILE)

model = Path(MODEL_PATH)
if not model.exists():
    raise FileNotFoundError(f"MediaPipe model not found: {model}")

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_tracking_confidence = 0.8
)

CAM1_SOURCE = "/dev/v4l/by-path/pci-0000:04:00.4-usb-0:2:1.0-video-index0"
CAM2_SOURCE = "/dev/v4l/by-path/pci-0000:04:00.3-usb-0:2:1.0-video-index0"

cam1 = cv2.VideoCapture(CAM1_SOURCE)
cam2 = cv2.VideoCapture(CAM2_SOURCE)

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

ax.set_xlim(X_LIMIT)
ax.set_ylim(Y_LIMIT)
ax.set_zlim(Z_LIMIT)

scatter = ax.scatter([], [], [], color="red", s=20)
lines = [ax.plot([], [], [], color="green", linewidth=2)[0]
         for _ in HAND_CONNECTIONS]
artists = [scatter] + lines

start = time.perf_counter()
last1 = -1
last2 = -1


STATE = {
    "THUMB": 0,
    "INDEX": 0,
    "MIDDLE": 0,
    "RING": 0,
    "PINKY": 0,
}

active_traking = False
try:
    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        if not ret1 or frame1 is None or not ret2 or frame2 is None:
            continue

        now_ms = int((time.perf_counter() - start) * 1000)
        ts1 = max(now_ms, last1 + 1)
        ts2 = max(now_ms, last2 + 1)

        result1 = process_and_display(frame1, landmarker_cam1, ts1, "Camera 1")
        result2 = process_and_display(frame2, landmarker_cam2, ts2, "Camera 2")
        if result1.hand_landmarks == [] or result2.hand_landmarks == []:
            continue

        Postion = Plot_Hand(calibration, result1, result2, ax, artists)
        
        ax.set_xlim(X_LIMIT)
        ax.set_ylim(Y_LIMIT)
        ax.set_zlim(Z_LIMIT)

        if Postion:
            Bone = [
                ["THUMB",3, 5, 0.06, 0, 2],
                ["INDEX", 8, 5, 0.06, 0, 2],
                ["MIDDLE", 12, 9, 0.06, 0, 2],
                ["RING", 16, 13, 0.06, 0, 2],
                ["PINKY", 20, 17, 0.06, 0, 2],
            ]

            SPEACIAL_STATES = [
                ["INDEX-THUMB", 4, 8, 0.02, 0, -1],
            ]
                
            output = ""
            Pause_Time = 0
            Input_Joint, Input_state = "",0
            def dist(bone1, bone2):
                return np.linalg.norm(Postion[bone1] - Postion[bone2])

            for Joint in Bone:
                distance = np.linalg.norm(Postion[Joint[1]] - Postion[Joint[2]])
                if distance < Joint[3]:
                    finger(Joint[0], Joint[4])
                    state = "CURLED"
                else:
                    finger(Joint[0], Joint[5])
                    state = "EXTENDED"

            for Joint in SPEACIAL_STATES:
                distance = np.linalg.norm(Postion[Joint[1]] - Postion[Joint[2]])
                if distance < Joint[3]:
                    finger(Joint[0], Joint[4])
                    state = "TRUE"
                else:
                    state = "FALSE"

            if Input_Joint:
                finger(Input_Joint, Input_state)

            async def check_hand():
                STATE = HandState()

                Conditions_Code = [
                    ["track", ["THUMB", 0], ["INDEX", 0], ["MIDDLE", 0],
                     ["RING", 0], ["PINKY", 0]]
                ]

                for state in Conditions_Code:
                    passed = all(
                        STATE[bone[0]] == bone[1]
                        for bone in state[1:]
                    )

                    target = Postion[0]

                    #await move_to_target(target, traking=passed)


            asyncio.run(check_hand())
              
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
