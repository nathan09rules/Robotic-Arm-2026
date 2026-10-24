import cv2
import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

def load_projection_matrices(npz_path):
    data = np.load("webcam-calibration.npz")
    K = data["camera_matrix"]
    dist = data["distortion"]
    R2 = data["R2"]
    T2 = data["T2"]

    # Compute 3x4 Projection Matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R2, T2))
    
    return K, dist, P1, P2

def process_and_display(frame, landmarker, timestamp_ms, window_name):
    import mediapipe as mp
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w, _ = frame.shape
    pts_2d = None

    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        raw_coords = np.array(
            [[lm.x, lm.y] for lm in result.hand_landmarks[0]],
            dtype=np.float32
        )
        pts_2d = raw_coords * [w, h]
        pts_int = pts_2d.astype(np.int32)

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts_int[start_idx]), tuple(pts_int[end_idx]), (0, 255, 0), 2)
        for pt in pts_int:
            cv2.circle(frame, tuple(pt), 4, (0, 0, 255), -1)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    cv2.imshow(window_name, frame)

    return pts_2d

def triangulate_2d_to_3d(pts1_2d, pts2_2d, K, dist, P1, P2):
    # Undistort 2D pixel coordinates using camera calibration
    pts1_undist = cv2.undistortPoints(
        pts1_2d.reshape(-1, 1, 2), K, dist, P=K
    ).reshape(-1, 2)

    pts2_undist = cv2.undistortPoints(
        pts2_2d.reshape(-1, 1, 2), K, dist, P=K
    ).reshape(-1, 2)

    # Solve DLT system for ray intersection
    pts4D = cv2.triangulatePoints(P1, P2, pts1_undist.T, pts2_undist.T)

    # Convert homogeneous (4, N) to 3D Cartesian (N, 3)
    return (pts4D[:3] / pts4D[3]).T    
