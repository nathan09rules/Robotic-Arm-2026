import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Line3DCollection

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]
_EDGE_A = np.array([a for a, _ in HAND_CONNECTIONS])
_EDGE_B = np.array([b for _, b in HAND_CONNECTIONS])

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


# ----------------------------------------------------------------------
# Camera side: inference and 2D overlay are separate so that inference can
# run in worker threads while all OpenCV windows stay in the main thread.
# ----------------------------------------------------------------------

def detect(frame, landmarker, timestamp_ms):
    """Run MediaPipe on one BGR frame. Safe to call from a worker thread."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def draw_overlay(frame, result):
    """Draw the 2D hand skeleton onto the frame in place."""
    if result is None or not result.hand_landmarks:
        return frame
    h, w = frame.shape[:2]
    for hand in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)
    return frame


def process_and_display(frame, landmarker, timestamp_ms, window_name,
                        show=True, wait=True):
    """Backwards-compatible wrapper (same behaviour as the original).
    show=False skips the OpenCV window; wait=False skips cv2.waitKey."""
    result = detect(frame, landmarker, timestamp_ms)
    if show:
        cv2.imshow(window_name, draw_overlay(frame, result))
        if wait:
            cv2.waitKey(1)
    return result


# ----------------------------------------------------------------------
# 3D side
# ----------------------------------------------------------------------

def create_hand_artists(ax):
    """Faster artist set: 1 scatter + 1 line collection (instead of
    1 scatter + 21 Line3D). Plot_Hand / clear_hand detect it automatically."""
    scat = ax.scatter([], [], [], c="r", s=20)
    bones = Line3DCollection([], colors="g", linewidths=2)
    ax.add_collection3d(bones)
    return [scat, bones]


def clear_hand(artists):
    """Hide the hand in the 3D plot (works with either artist layout)."""
    artists[0]._offsets3d = ([], [], [])
    if isinstance(artists[1], Line3DCollection):
        artists[1].set_segments([])
    else:
        for line in artists[1:]:
            line.set_data([], [])
            line.set_3d_properties([])


_prev_points = np.full((21, 3), np.nan)
_cache = {}


def _prepare(position_cam):
    """Everything that depends only on the calibration, computed once."""
    key = id(position_cam)
    c = _cache.get(key)
    if c is None:
        R = position_cam["R"]
        T = position_cam["T"]
        flip = np.array([1.0, -1.0, -1.0])
        # floor transform + mm->m + axis flips folded into one affine map
        A = (flip[:, None] * position_cam["R_floor"]) / 1000.0
        c = {
            "R": R,
            "C2": -R.T @ T,
            "A_T": np.ascontiguousarray(A.T),
            "b": flip * position_cam["t_floor"] / 1000.0,
            "scale1": np.asarray(position_cam["size1"], dtype=np.float64),
            "scale2": np.asarray(position_cam["size2"], dtype=np.float64),
        }
        _cache.clear()
        _cache[key] = c
    return c


def _get_rays(result, scale, K, dist):
    if result is None or not result.hand_landmarks:
        return None
    hand = result.hand_landmarks[0]
    if len(hand) != 21:
        return None

    pix = np.array([(lm.x, lm.y) for lm in hand], dtype=np.float64) * scale
    xy = cv2.undistortPoints(pix.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    rays = np.empty((21, 3))
    rays[:, :2] = xy
    rays[:, 2] = 1.0
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)  # norm >= 1, no guard needed
    return rays


def _triangulate(u, v, C2):
    """Midpoint of closest approach between rays C1=0 + s*u and C2 + t*v,
    for all landmarks at once. u, v are unit vectors (n,3). Invalid -> NaN."""
    w0 = -C2
    b = np.einsum("ij,ij->i", u, v)
    d = u @ w0
    e = v @ w0
    denom = 1.0 - b * b                 # a = c = 1 for unit rays
    ok = denom > 1e-8
    denom = np.where(ok, denom, 1.0)
    s = (b * e - d) / denom
    t = (e - b * d) / denom
    ok &= (s >= 0) & (t >= 0)

    pts = 0.5 * (s[:, None] * u + C2 + t[:, None] * v)
    pts[~ok] = np.nan
    return pts


def Plot_Hand(position_cam, result_cam1, result_cam2, ax, artists):
    global _prev_points
    fast_lines = isinstance(artists[1], Line3DCollection)

    c = _prepare(position_cam)

    rays1 = _get_rays(result_cam1, c["scale1"],
                      position_cam["K1"], position_cam["dist1"])
    rays2 = _get_rays(result_cam2, c["scale2"],
                      position_cam["K2"], position_cam["dist2"])
    if rays1 is None or rays2 is None:
        clear_hand(artists)
        return None

    rays2 = rays2 @ c["R"]                    # == (R.T @ rays2.T).T

    curr = _triangulate(rays1, rays2, c["C2"]) @ c["A_T"] + c["b"]
    
    # Swap X and Y axes
    curr = curr[:, [1, 0, 2]]

    # Per-landmark adaptive smoothing, vectorised.
    fin = np.isfinite(curr).all(axis=1)
    both = fin & np.isfinite(_prev_points).all(axis=1)
    if both.any():
        dist = np.linalg.norm(curr[both] - _prev_points[both], axis=1)
        alpha = (0.1 + 0.8 * np.minimum(1.0, dist / 0.05))[:, None]
        curr[both] = alpha * curr[both] + (1 - alpha) * _prev_points[both]
    _prev_points[fin] = curr[fin]

    valid = np.isfinite(curr).all(axis=1)
    if not valid.any():
        clear_hand(artists)
        return {i: curr[i] for i in range(21)}

    xyz = curr[valid]
    artists[0]._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])

    seg_ok = valid[_EDGE_A] & valid[_EDGE_B]
    if fast_lines:
        segs = np.stack((curr[_EDGE_A], curr[_EDGE_B]), axis=1)[seg_ok]
        artists[1].set_segments(segs)
    else:
        for line, ok, a, b in zip(artists[1:], seg_ok, _EDGE_A, _EDGE_B):
            if ok:
                line.set_data_3d([curr[a, 0], curr[b, 0]],
                                 [curr[a, 1], curr[b, 1]],
                                 [curr[a, 2], curr[b, 2]])
            else:
                line.set_data([], [])
                line.set_3d_properties([])

    return {i: curr[i] for i in range(21)}
