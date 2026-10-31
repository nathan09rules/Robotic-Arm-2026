

import cv2
import time
import threading
import asyncio
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

from Utils.Webcam import *
from Utils.BlenderArm import finger, HandState
from Utils.Arm import move_to_target
from Utils.Boundry import BoundaryTracker


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

CAM1_SOURCE = "/dev/v4l/by-path/pci-0000:04:00.4-usb-0:2:1.0-video-index0"
CAM2_SOURCE = "/dev/v4l/by-path/pci-0000:04:00.3-usb-0:2:1.0-video-index0"

X_LIMIT = (-0.25, 0.25)
Y_LIMIT = (-0.25, 0.25)
Z_LIMIT = (0.0, 0.5)

BOUNDARY_CALIBRATION_TIME = 10
MIN_MARKER_SAMPLES = 5


# --------------------------------------------------
# ARM EVENT LOOP
# --------------------------------------------------

arm_loop = asyncio.new_event_loop()


def run_arm_loop():
    asyncio.set_event_loop(arm_loop)
    arm_loop.run_forever()


arm_thread = threading.Thread(
    target=run_arm_loop,
    daemon=True
)
arm_thread.start()


# --------------------------------------------------
# LOAD CALIBRATION
# --------------------------------------------------

calibration = load_calibration(CALIBRATION_FILE)


# --------------------------------------------------
# CAMERAS
# --------------------------------------------------

cam1 = cv2.VideoCapture(CAM1_SOURCE)
cam2 = cv2.VideoCapture(CAM2_SOURCE)

if not cam1.isOpened() or not cam2.isOpened():
    cam1.release()
    cam2.release()
    raise RuntimeError("Could not open both cameras")


# --------------------------------------------------
# MEDIAPIPE
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options1 = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

options2 = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

landmarker_cam1 = HandLandmarker.create_from_options(options1)
landmarker_cam2 = HandLandmarker.create_from_options(options2)


# --------------------------------------------------
# ONE SHARED 3D PLOT
# --------------------------------------------------

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

hand_scatter = ax.scatter(
    [], [], [], color="red", s=20
)

hand_lines = [
    ax.plot([], [], [], color="green", linewidth=2)[0]
    for _ in HAND_CONNECTIONS
]

hand_artists = [hand_scatter] + hand_lines

ax.set_xlim(X_LIMIT)
ax.set_ylim(Y_LIMIT)
ax.set_zlim(Z_LIMIT)

ax.set_xlabel("Floor X (m)")
ax.set_ylabel("Floor Y (m)")
ax.set_zlabel("Height Z (m)")

# Boundary uses this exact same axis.
boundary = BoundaryTracker(ax)


# --------------------------------------------------
# CALIBRATE BOUNDARY FOR 10 SECONDS
# --------------------------------------------------

print("Keep all six boundary markers visible...")
print("Calibrating boundary for 10 seconds.")

boundary_samples = {
    marker_id: []
    for marker_id in boundary.ids
}

start_time = time.monotonic()
last_second = -1

while time.monotonic() - start_time < BOUNDARY_CALIBRATION_TIME:

    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 or not ret2:
        continue

    # Detect marker centres independently in both cameras.
    detected1 = boundary.detect(frame1)
    detected2 = boundary.detect(frame2)

    # Only triangulate markers visible in BOTH cameras.
    for marker_id in boundary.ids:

        if (
            marker_id not in detected1
            or marker_id not in detected2
        ):
            continue

        point = boundary.triangulate(
            detected1[marker_id],
            detected2[marker_id],
            calibration
        )

        if point is not None:
            boundary_samples[marker_id].append(point)

    elapsed = int(time.monotonic() - start_time)

    if elapsed != last_second:
        last_second = elapsed
        print(
            f"Calibrating... "
            f"{BOUNDARY_CALIBRATION_TIME - elapsed} seconds remaining"
        )

    cv2.imshow("Boundary Calibration - Camera 1", frame1)
    cv2.imshow("Boundary Calibration - Camera 2", frame2)
    cv2.waitKey(1)


# --------------------------------------------------
# CALCULATE MEDIAN MARKER POSITIONS
# --------------------------------------------------

boundary_points = {}

print("\nBoundary calibration results:")

for marker_id in boundary.ids:

    samples = boundary_samples[marker_id]

    print(f"Marker {marker_id}: {len(samples)} samples")

    if len(samples) >= MIN_MARKER_SAMPLES:

        boundary_points[marker_id] = np.median(
            np.asarray(samples),
            axis=0
        )


missing = [
    marker_id for marker_id in boundary.ids
    if marker_id not in boundary_points
]

if missing:
    cam1.release()
    cam2.release()
    landmarker_cam1.close()
    landmarker_cam2.close()
    cv2.destroyAllWindows()
    plt.close(fig)
    arm_loop.call_soon_threadsafe(arm_loop.stop)

    raise RuntimeError(
        f"Boundary calibration failed. "
        f"Markers without enough samples: {missing}. "
        f"Ensure all six markers are visible in both cameras and restart."
    )


# --------------------------------------------------
# DRAW BOUNDARY ONCE
# --------------------------------------------------

boundary.set_boundary(boundary_points)

print("\nBoundary calibrated successfully.")
print("Boundary is now fixed.")

cv2.destroyWindow("Boundary Calibration - Camera 1")
cv2.destroyWindow("Boundary Calibration - Camera 2")


# --------------------------------------------------
# HAND CLASSIFICATION
# --------------------------------------------------

Bone = [
    ("THUMB", 3, 5, 0.06, 0, 2),
    ("INDEX", 8, 5, 0.06, 0, 2),
    ("MIDDLE", 12, 9, 0.06, 0, 2),
    ("RING", 16, 13, 0.06, 0, 2),
    ("PINKY", 20, 17, 0.06, 0, 2)
]

SPEACIAL_STATES = [
    ("INDEX-THUMB", 4, 8, 0.02, 0, -1)
]

Conditions_Code = [
    ("THUMB", 0),
    ("INDEX", 0),
    ("MIDDLE", 0),
    ("RING", 0),
    ("PINKY", 0)
]


async def check_hand():

    STATE = HandState()

    passed = all(
        STATE[name] == value
        for name, value in Conditions_Code
    )

    return passed


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

timestamp1 = 0
timestamp2 = 0

try:

    while True:

        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()

        if not ret1 or not ret2:
            continue

        timestamp = time.monotonic_ns() // 1_000_000

        timestamp1 = max(timestamp, timestamp1 + 1)
        timestamp2 = max(timestamp, timestamp2 + 1)

        # HAND TRACKING
        result1 = process_and_display(
            frame1,
            landmarker_cam1,
            timestamp1,
            "Camera 1"
        )

        result2 = process_and_display(
            frame2,
            landmarker_cam2,
            timestamp2,
            "Camera 2"
        )

        # HAND TRIANGULATION
        Postion = None

        if result1.hand_landmarks and result2.hand_landmarks:

            Postion = Plot_Hand(
                calibration,
                result1,
                result2,
                ax,
                hand_artists
            )

        else:

            hand_scatter._offsets3d = ([], [], [])

            for line in hand_lines:
                line.set_data([], [])
                line.set_3d_properties([])

        # Only classify when all required points are valid.
        required_points = {
            3, 5, 8, 12, 9, 16, 13, 20, 17, 4
        }

        hand_valid = (
            Postion is not None
            and all(
                i in Postion
                and np.all(np.isfinite(Postion[i]))
                for i in required_points
            )
        )

        if hand_valid:

            for name, joint, reference, threshold, curled, extended in Bone:

                distance = np.linalg.norm(
                    Postion[joint] - Postion[reference]
                )

                finger(
                    name,
                    curled if distance < threshold else extended
                )

            for name, joint, reference, threshold, true_state, false_state in SPEACIAL_STATES:

                distance = np.linalg.norm(
                    Postion[joint] - Postion[reference]
                )

                finger(
                    name,
                    true_state if distance < threshold else false_state
                )

            asyncio.run(check_hand())

        # Restore fixed viewing limits because Plot_Hand changes them.
        ax.set_xlim(X_LIMIT)
        ax.set_ylim(Y_LIMIT)
        ax.set_zlim(Z_LIMIT)

        plt.pause(0.001)


except KeyboardInterrupt:
    print("\nStopping...")


finally:

    cam1.release()
    cam2.release()

    landmarker_cam1.close()
    landmarker_cam2.close()

    cv2.destroyAllWindows()
    plt.close(fig)

    arm_loop.call_soon_threadsafe(arm_loop.stop)

