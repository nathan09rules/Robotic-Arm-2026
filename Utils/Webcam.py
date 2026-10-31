import cv2
import time
import threading
from pathlib import Path

import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

CALIBRATION_FILE = "Utils/webcam-calibration.npz"
FLOOR_FILE = "Utils/floor-calibration.npz"
MODEL_PATH = "Utils/hand_landmarker.task"


def load_calibration(filename=CALIBRATION_FILE):
    path = Path(filename)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parent.parent / filename

    floor_path = Path(FLOOR_FILE)
    if not floor_path.is_absolute() and not floor_path.exists():
        floor_path = Path(__file__).resolve().parent.parent / FLOOR_FILE

    if not path.exists():
        raise FileNotFoundError(f"Calibration not found: {path}")
    if not floor_path.exists():
        raise FileNotFoundError(f"Floor calibration not found: {floor_path}")

    with np.load(path) as data:
        position = {
            "K1": data["K1"].copy(),
            "dist1": data["dist1"].copy(),
            "K2": data["K2"].copy(),
            "dist2": data["dist2"].copy(),
            "R": data["R"].copy(),
            "T": data["T"].reshape(3).copy(),
            "size1": tuple(data["image_size1"].astype(int)),
            "size2": tuple(data["image_size2"].astype(int)),
        }

    with np.load(floor_path) as floor:
        position["R_floor"] = floor["R_cam1_to_floor"].copy()
        position["t_floor"] = floor["t_cam1_to_floor"].reshape(3).copy()

    return position


def process_and_display(frame, landmarker, timestamp_ms, window_name):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w = frame.shape[:2]
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            for p in pts:
                cv2.circle(frame, p, 4, (0, 0, 255), -1)

    cv2.imshow(window_name, frame)
    cv2.waitKey(1)

    return result


_prev_3d_points = {}


def Plot_Hand(position_cam, result_cam1, result_cam2, ax, artists):

    def clear_plot():
        artists[0]._offsets3d = ([], [], [])
        for line in artists[1:]:
            line.set_data([], [])
            line.set_3d_properties([])

    def get_rays(result, size, K, dist):
        if result is None or not result.hand_landmarks:
            return None

        hand = result.hand_landmarks[0]
        if len(hand) != 21:
            return None

        width, height = size
        pixels = np.array(
            [[lm.x * width, lm.y * height] for lm in hand],
            dtype=np.float64
        ).reshape(-1, 1, 2)

        xy = cv2.undistortPoints(pixels, K, dist).reshape(-1, 2)
        rays = np.column_stack((xy, np.ones(21)))
        rays /= np.maximum(
            np.linalg.norm(rays, axis=1, keepdims=True), 1e-12
        )
        return rays

    rays1 = get_rays(
        result_cam1, position_cam["size1"],
        position_cam["K1"], position_cam["dist1"]
    )
    rays2 = get_rays(
        result_cam2, position_cam["size2"],
        position_cam["K2"], position_cam["dist2"]
    )

    if rays1 is None or rays2 is None:
        clear_plot()
        return None

    # Camera 2 expressed in Camera 1 coordinates.
    R = position_cam["R"]
    T = position_cam["T"]

    C1 = np.zeros(3)
    C2 = -R.T @ T
    rays2 = (R.T @ rays2.T).T

    # Camera 1 coordinates -> shared floor coordinates.
    R_floor = position_cam["R_floor"]
    t_floor = position_cam["t_floor"]

    points = {}

    for i, (u, v) in enumerate(zip(rays1, rays2)):
        w0 = C1 - C2

        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        d = np.dot(u, w0)
        e = np.dot(v, w0)

        denom = a * c - b * b

        if denom <= 1e-8 * a * c:
            points[i] = np.full(3, np.nan)
            continue

        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom

        if s < 0 or t < 0:
            points[i] = np.full(3, np.nan)
            continue

        q1 = C1 + s * u
        q2 = C2 + t * v

        # Triangulated point in Camera 1 coordinates, millimetres.
        point_cam1 = (q1 + q2) * 0.5

        # Transform into the floor coordinate system.
        point_floor = R_floor @ point_cam1 + t_floor
        curr = point_floor / 1000.0

        # Smooth each landmark independently.
        if (
            i in _prev_3d_points
            and np.all(np.isfinite(curr))
            and np.all(np.isfinite(_prev_3d_points[i]))
        ):
            prev = _prev_3d_points[i]
            distance = np.linalg.norm(curr - prev)
            alpha = 0.1 + 0.8 * min(1.0, distance / 0.05)
            curr = alpha * curr + (1 - alpha) * prev

        if np.all(np.isfinite(curr)):
            _prev_3d_points[i] = curr

        points[i] = curr

    valid = [
        k for k, p in points.items()
        if np.all(np.isfinite(p))
    ]

    if not valid:
        clear_plot()
        return points

    xyz = np.array([points[k] for k in valid])

    artists[0]._offsets3d = (
        xyz[:, 0], xyz[:, 1], xyz[:, 2]
    )

    for line, (a, b) in zip(artists[1:], HAND_CONNECTIONS):
        if (
            a in points and b in points
            and np.all(np.isfinite(points[a]))
            and np.all(np.isfinite(points[b]))
        ):
            p1, p2 = points[a], points[b]

            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            line.set_3d_properties([p1[2], p2[2]])
        else:
            line.set_data([], [])
            line.set_3d_properties([])

    centre = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    radius = max(np.ptp(xyz, axis=0).max() * 0.6, 0.05)

    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)

    ax.set_xlabel("Floor X (m)")
    ax.set_ylabel("Floor Y (m)")
    ax.set_zlabel("Height Z (m)")

    return points
