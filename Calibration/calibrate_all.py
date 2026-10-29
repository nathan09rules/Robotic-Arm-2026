import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

CAM1_SOURCE = 6
CAM2_SOURCE = 4

OUTPUT_FILE = "Utils/webcam-calibration.npz"

NUM_VIEWS = 20
CAPTURE_INTERVAL = 2.0
MIN_CORNERS = 6

BOARD_SIZE = (7, 5)
SQUARE_LENGTH = 25       # mm
MARKER_LENGTH = 18       # mm

DICTIONARY = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)

board = cv2.aruco.CharucoBoard(
    BOARD_SIZE,
    SQUARE_LENGTH,
    MARKER_LENGTH,
    DICTIONARY
)

detector = cv2.aruco.CharucoDetector(board)
BOARD_CORNERS = board.getChessboardCorners()


# ============================================================
# DETECT CHARUCO CORNERS
# ============================================================

def detect_corner_map(frame):
    """
    Return {corner_id: (x, y)} for detected ChArUco corners.
    Returns None if too few corners are detected.
    """

    corners, ids, _, _ = detector.detectBoard(frame)

    if ids is None or len(ids) < MIN_CORNERS:
        return None

    ids = ids.reshape(-1)
    points = corners.reshape(-1, 2)

    corner_map = {}

    for corner_id, point in zip(ids, points):
        corner_map[int(corner_id)] = point.astype(np.float32)

    if len(corner_map) < MIN_CORNERS:
        return None

    return corner_map


# ============================================================
# CALIBRATION
# ============================================================

def main():
    cap1 = cv2.VideoCapture(2)
    cap2 = cv2.VideoCapture(4)

    # Intrinsic calibration observations, independently for each camera.
    object_points1 = []
    image_points1 = []

    object_points2 = []
    image_points2 = []

    # Matched observations for stereo calibration.
    stereo_object_points = []
    stereo_image_points1 = []
    stereo_image_points2 = []

    image_size1 = None
    image_size2 = None

    last_capture = -CAPTURE_INTERVAL

    try:


        print("=" * 60)
        print("TWO-CAMERA CHARUCO CALIBRATION")
        print("=" * 60)
        print(f"Camera 1: {CAM1_SOURCE}")
        print(f"Camera 2: {CAM2_SOURCE}")
        print(f"Required successful views: {NUM_VIEWS}")
        print()
        print("Move and tilt the board between captures.")
        print("Both cameras must see the board simultaneously.")
        print("Press Q to cancel.")
        print()

        while len(stereo_object_points) < NUM_VIEWS:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or frame1 is None:
                time.sleep(0.01)
                continue

            if not ret2 or frame2 is None:
                time.sleep(0.01)
                continue

            current_size1 = (
                frame1.shape[1],
                frame1.shape[0]
            )

            current_size2 = (
                frame2.shape[1],
                frame2.shape[0]
            )

            if image_size1 is None:
                image_size1 = current_size1
                image_size2 = current_size2

            if current_size1 != image_size1:
                raise RuntimeError(
                    "Camera 1 resolution changed during calibration."
                )

            if current_size2 != image_size2:
                raise RuntimeError(
                    "Camera 2 resolution changed during calibration."
                )

            # Display previews.
            preview1 = frame1.copy()
            preview2 = frame2.copy()

            detected1 = detect_corner_map(frame1)
            detected2 = detect_corner_map(frame2)

            if detected1 is not None:
                cv2.putText(
                    preview1,
                    f"Corners: {len(detected1)}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    preview1,
                    "Board not detected",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            if detected2 is not None:
                cv2.putText(
                    preview2,
                    f"Corners: {len(detected2)}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    preview2,
                    "Board not detected",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            progress = (
                f"Successful paired views: "
                f"{len(stereo_object_points)}/{NUM_VIEWS}"
            )

            cv2.putText(
                preview1,
                progress,
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            cv2.imshow("Calibration - Camera 1", preview1)
            cv2.imshow("Calibration - Camera 2", preview2)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Calibration cancelled.")
                return

            now = time.monotonic()

            if now - last_capture < CAPTURE_INTERVAL:
                continue

            last_capture = now

            # Both cameras must detect the board.
            if detected1 is None or detected2 is None:
                print(
                    "Waiting: board must be visible in both cameras."
                )
                continue

            # Match IDs so image points correspond to the same
            # physical ChArUco corners in both cameras.
            common_ids = sorted(
                set(detected1.keys()) & set(detected2.keys())
            )

            if len(common_ids) < MIN_CORNERS:
                print(
                    "Waiting: only "
                    f"{len(common_ids)} shared corners detected."
                )
                continue

            # ----- Camera 1 intrinsic observations -----

            ids1 = sorted(detected1.keys())

            obj1 = BOARD_CORNERS[ids1].astype(np.float32)
            img1 = np.array(
                [detected1[i] for i in ids1],
                dtype=np.float32
            )

            object_points1.append(
                obj1.reshape(-1, 1, 3)
            )

            image_points1.append(
                img1.reshape(-1, 1, 2)
            )

            # ----- Camera 2 intrinsic observations -----

            ids2 = sorted(detected2.keys())

            obj2 = BOARD_CORNERS[ids2].astype(np.float32)
            img2 = np.array(
                [detected2[i] for i in ids2],
                dtype=np.float32
            )

            object_points2.append(
                obj2.reshape(-1, 1, 3)
            )

            image_points2.append(
                img2.reshape(-1, 1, 2)
            )

            # ----- Matched stereo observations -----

            obj_stereo = BOARD_CORNERS[common_ids].astype(
                np.float32
            )

            img_stereo1 = np.array(
                [detected1[i] for i in common_ids],
                dtype=np.float32
            )

            img_stereo2 = np.array(
                [detected2[i] for i in common_ids],
                dtype=np.float32
            )

            stereo_object_points.append(
                obj_stereo.reshape(-1, 1, 3)
            )

            stereo_image_points1.append(
                img_stereo1.reshape(-1, 1, 2)
            )

            stereo_image_points2.append(
                img_stereo2.reshape(-1, 1, 2)
            )

            print(
                f"Captured {len(stereo_object_points)}/{NUM_VIEWS} "
                f"successful views "
                f"({len(common_ids)} shared corners)."
            )

        print()
        print("All 20 successful views collected.")
        print("Calculating camera intrinsics...")

        # ----------------------------------------------------
        # CALIBRATE CAMERA 1
        # ----------------------------------------------------

        error1, K1, dist1, _, _ = cv2.calibrateCamera(
            object_points1,
            image_points1,
            image_size1,
            None,
            None
        )

        # ----------------------------------------------------
        # CALIBRATE CAMERA 2
        # ----------------------------------------------------

        error2, K2, dist2, _, _ = cv2.calibrateCamera(
            object_points2,
            image_points2,
            image_size2,
            None,
            None
        )

        print("Calculating relative stereo geometry...")

        # ----------------------------------------------------
        # STEREO CALIBRATION
        #
        # X_cam2 = R @ X_cam1 + T
        #
        # Translation T will use millimetres because the board
        # dimensions were supplied in millimetres.
        # ----------------------------------------------------

        criteria = (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-6
        )

        stereo_error, K1, dist1, K2, dist2, R, T, E, F = (
            cv2.stereoCalibrate(
                stereo_object_points,
                stereo_image_points1,
                stereo_image_points2,
                K1,
                dist1,
                K2,
                dist2,
                image_size1,
                criteria=criteria,
                flags=cv2.CALIB_FIX_INTRINSIC
            )
        )

        # ----------------------------------------------------
        # SAVE ALL CALIBRATION DATA TO ONE FILE
        # ----------------------------------------------------

        output_path = Path(OUTPUT_FILE)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.savez(
            str(output_path),

            # Camera 1 intrinsics.
            K1=K1,
            dist1=dist1,

            # Camera 2 intrinsics.
            K2=K2,
            dist2=dist2,

            # Compatibility with the original single-camera file.
            camera_matrix=K1,
            distortion=dist1,

            # Camera 1 -> Camera 2 transformation.
            R=R,
            T=T,

            # Additional stereo calibration matrices.
            E=E,
            F=F,

            # Resolution of each calibrated camera.
            image_size1=np.array(image_size1, dtype=np.int32),
            image_size2=np.array(image_size2, dtype=np.int32),

            # Calibration errors for inspection.
            intrinsic_error1=np.array(error1),
            intrinsic_error2=np.array(error2),
            stereo_error=np.array(stereo_error)
        )

        print()
        print("=" * 60)
        print("CALIBRATION COMPLETE")
        print("=" * 60)
        print(f"Saved file: {output_path}")
        print(f"Camera 1 reprojection error: {error1:.5f}")
        print(f"Camera 2 reprojection error: {error2:.5f}")
        print(f"Stereo calibration error:   {stereo_error:.5f}")
        print()
        print("Camera 1 intrinsic matrix:")
        print(K1)
        print()
        print("Camera 2 intrinsic matrix:")
        print(K2)
        print()
        print("Relative rotation:")
        print(R)
        print()
        print("Relative translation (mm):")
        print(T)

    finally:
        if cap1 is not None:
            cap1.release()

        if cap2 is not None:
            cap2.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
