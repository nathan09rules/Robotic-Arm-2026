import cv2
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CALIBRATION_FILE = BASE_DIR / "Utils" / "webcam-calibration.npz"
FLOOR_FILE = BASE_DIR / "Utils" / "floor-calibration.npz"
BOUNDARY_FILE = BASE_DIR / "Utils" / "boundry.npz"

CAM1_SOURCE = "/dev/v4l/by-path/pci-0000:04:00.4-usb-0:2:1.0-video-index0"
CAM2_SOURCE = "/dev/v4l/by-path/pci-0000:04:00.3-usb-0:2:1.0-video-index0"

MARKER_IDS = list(range(6))
CALIBRATION_TIME = 10
MIN_MARKER_SAMPLES = 5


def load_calibration(filename=CALIBRATION_FILE):
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(f"Calibration not found: {path}")
    if not FLOOR_FILE.exists():
        raise FileNotFoundError(f"Floor calibration not found: {FLOOR_FILE}")

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

    with np.load(FLOOR_FILE) as floor:
        position["R_floor"] = floor["R_cam1_to_floor"].copy()
        position["t_floor"] = floor["t_cam1_to_floor"].reshape(3).copy()

    return position


def create_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    else:
        detector = None

    return dictionary, parameters, detector


def detect_markers(frame, dictionary, parameters, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if detector is not None:
        corners, detected_ids, _ = detector.detectMarkers(gray)
    else:
        corners, detected_ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=parameters
        )

    detected = {}

    if detected_ids is None:
        return detected

    for marker_corners, marker_id in zip(corners, detected_ids.flatten()):
        marker_id = int(marker_id)

        if marker_id in MARKER_IDS:
            detected[marker_id] = marker_corners[0].mean(axis=0)

    return detected


def triangulate(p1, p2, calibration):
    K1, K2 = calibration["K1"], calibration["K2"]
    dist1, dist2 = calibration["dist1"], calibration["dist2"]
    R, T = calibration["R"], calibration["T"].reshape(3)

    ray1 = cv2.undistortPoints(
        np.asarray(p1, dtype=np.float64).reshape(1, 1, 2),
        K1, dist1
    ).reshape(2)

    ray2 = cv2.undistortPoints(
        np.asarray(p2, dtype=np.float64).reshape(1, 1, 2),
        K2, dist2
    ).reshape(2)

    u = np.array([ray1[0], ray1[1], 1.0])
    v = np.array([ray2[0], ray2[1], 1.0])

    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    C1 = np.zeros(3)
    C2 = -R.T @ T

    v = R.T @ v
    v /= np.linalg.norm(v)

    w0 = C1 - C2
    a, b, c = np.dot(u, u), np.dot(u, v), np.dot(v, v)
    d, e = np.dot(u, w0), np.dot(v, w0)

    denominator = a * c - b * b

    if abs(denominator) < 1e-10:
        return None

    s = (b * e - c * d) / denominator
    t = (a * e - b * d) / denominator

    if s < 0 or t < 0:
        return None

    q1 = C1 + s * u
    q2 = C2 + t * v
    point_cam1 = (q1 + q2) / 2

    point_floor = (
        calibration["R_floor"] @ point_cam1
        + calibration["t_floor"].reshape(3)
    )

    point = point_floor / 1000.0
    point[2] *= -1
    point[1] *= -1

    if not np.all(np.isfinite(point)):
        return None

    return point


def main():
    calibration = load_calibration()
    if calibration is None:
        return

    cap1 = cv2.VideoCapture(CAM1_SOURCE)
    cap2 = cv2.VideoCapture(CAM2_SOURCE)

    if not cap1.isOpened() or not cap2.isOpened():
        return

    dictionary, parameters, detector = create_detector()
    marker_samples = {mid: [] for mid in MARKER_IDS}
    start_time = time.time()

    while time.time() - start_time < CALIBRATION_TIME:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            continue

        det1 = detect_markers(frame1, dictionary, parameters, detector)
        det2 = detect_markers(frame2, dictionary, parameters, detector)

        common_ids = set(det1.keys()).intersection(set(det2.keys()))
        for mid in common_ids:
            pt = triangulate(det1[mid], det2[mid], calibration)
            if pt is not None:
                marker_samples[mid].append(pt)

        cv2.imshow("Cam 1", frame1)
        cv2.imshow("Cam 2", frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

    valid_ids = []
    valid_points = []

    for mid, samples in marker_samples.items():
        if len(samples) >= MIN_MARKER_SAMPLES:
            valid_ids.append(mid)
            valid_points.append(np.mean(samples, axis=0))

    if valid_ids:
        np.savez(
            BOUNDARY_FILE,
            ids=np.array(valid_ids, dtype=int),
            points=np.array(valid_points, dtype=float)
        )


if __name__ == "__main__":
    main()
