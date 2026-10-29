

import time
from pathlib import Path

import cv2
import numpy as np

CAM1_SOURCE = 2
CAM2_SOURCE = 4

CAMERA_FILE = "Utils/webcam-calibration.npz"
OUTPUT_FILE = "Utils/floor-calibration.npz"

BOARD_SIZE = (7, 5)
SQUARE_LENGTH = 25
MARKER_LENGTH = 18
MIN_CORNERS = 6

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)

board = cv2.aruco.CharucoBoard(
    BOARD_SIZE,
    SQUARE_LENGTH,
    MARKER_LENGTH,
    dictionary
)

detector = cv2.aruco.CharucoDetector(board)
board_corners = board.getChessboardCorners()


def detect(frame):
    corners, ids, _, _ = detector.detectBoard(frame)

    if ids is None or len(ids) < MIN_CORNERS:
        return None, None

    ids = ids.flatten()
    points = corners.reshape(-1, 2)

    return ids, points


def get_board_pose(frame, K, dist):
    ids, points = detect(frame)

    if ids is None:
        return None

    object_points = board_corners[ids].astype(np.float32)
    image_points = points.astype(np.float32)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)

    return R, tvec.reshape(3), len(ids)


def main():
    calibration_path = Path(CAMERA_FILE)

    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Camera calibration not found: {CAMERA_FILE}"
        )

    with np.load(calibration_path) as data:
        K1 = data["K1"].copy()
        dist1 = data["dist1"].copy()
        K2 = data["K2"].copy()
        dist2 = data["dist2"].copy()
        stereo_R = data["R"].copy()
        stereo_T = data["T"].copy()

    cap1 = cv2.VideoCapture(CAM1_SOURCE)
    cap2 = cv2.VideoCapture(CAM2_SOURCE)

    if not cap1.isOpened() or not cap2.isOpened():
        cap1.release()
        cap2.release()
        raise RuntimeError("Could not open both cameras.")

    try:
        print("FLOOR CALIBRATION")
        print("Place the ChArUco board flat where the floor should be.")
        print("Keep it stationary and visible in BOTH cameras.")
        print("Press Q to cancel.")

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                continue

            pose1 = get_board_pose(frame1, K1, dist1)
            pose2 = get_board_pose(frame2, K2, dist2)

            for frame, pose, name in (
                (frame1, pose1, "Camera 1"),
                (frame2, pose2, "Camera 2")
            ):
                if pose is not None:
                    cv2.putText(
                        frame,
                        f"{name}: Board detected ({pose[2]} corners)",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )
                else:
                    cv2.putText(
                        frame,
                        f"{name}: Board not detected",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2
                    )

            cv2.imshow("Floor - Camera 1", frame1)
            cv2.imshow("Floor - Camera 2", frame2)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Cancelled.")
                return

            if pose1 is None or pose2 is None:
                continue

            # Board pose in each camera's coordinate system.


# Board pose in each camera's coordinate system.
            R1, t1, _ = pose1
            R2, t2, _ = pose2

# Camera 1 -> floor coordinates
            R_cam1_to_floor = R1.T
            t_cam1_to_floor = -R1.T @ t1

# Camera 2 -> floor coordinates
            R_cam2_to_floor = R2.T
            t_cam2_to_floor = -R2.T @ t2

            output_path = Path(OUTPUT_FILE)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez(
                output_path,

                # Camera 1 -> floor
                R_cam1_to_floor=R_cam1_to_floor,
                t_cam1_to_floor=t_cam1_to_floor,

                # Camera 2 -> floor
                R_cam2_to_floor=R_cam2_to_floor,
                t_cam2_to_floor=t_cam2_to_floor,

                # Original board poses
                board_R_cam1=R1,
                board_t_cam1=t1,
                board_R_cam2=R2,
                board_t_cam2=t2,

                # Stereo calibration
                stereo_R=stereo_R,
                stereo_T=stereo_T,

                square_length=np.array(SQUARE_LENGTH)
            )

            print(f"Saved: {output_path}")
            print("Both cameras aligned to the floor.")
            print("Floor plane: Z = 0")
            print("Units: millimetres")

            break

    finally:
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

