

import cv2
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class BoundaryTracker:

    def __init__(self, ax):

        self.ax = ax
        self.ids = [0, 1, 2, 3, 4, 5]

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        if hasattr(cv2.aruco, "DetectorParameters"):
            self.parameters = cv2.aruco.DetectorParameters()
        else:
            self.parameters = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(
                self.dictionary,
                self.parameters
            )
        else:
            self.detector = None

        # All artists use the axis supplied by main.py.
        self.marker_scatter = ax.scatter(
            [], [], [], color="orange", s=55
        )

        self.lines = [
            ax.plot([], [], [], color="orange", linewidth=2)[0]
            for _ in self.ids
        ]

        self.labels = [
            ax.text(0, 0, 0, str(i), color="orange")
            for i in self.ids
        ]

        self.surface = Poly3DCollection(
            [],
            alpha=0.15,
            facecolor="orange",
            edgecolor="orange"
        )

        ax.add_collection3d(self.surface)

        self.calibrated_points = None

    # Detect marker centres in one camera frame.
    def detect(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.detector is not None:
            corners, detected_ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, detected_ids, _ = cv2.aruco.detectMarkers(
                gray,
                self.dictionary,
                parameters=self.parameters
            )

        detected = {}

        if detected_ids is None:
            return detected

        for marker_corners, marker_id in zip(
            corners, detected_ids.flatten()
        ):

            marker_id = int(marker_id)

            if marker_id not in self.ids:
                continue

            detected[marker_id] = marker_corners[0].mean(axis=0)

        return detected

    # Triangulate one marker from the two camera observations.
    def triangulate(self, p1, p2, calibration):

        K1 = calibration["K1"]
        K2 = calibration["K2"]

        dist1 = calibration["dist1"]
        dist2 = calibration["dist2"]

        R = calibration["R"]
        T = calibration["T"].reshape(3)

        ray1 = cv2.undistortPoints(
            np.asarray(p1, dtype=np.float64).reshape(1, 1, 2),
            K1,
            dist1
        ).reshape(2)

        ray2 = cv2.undistortPoints(
            np.asarray(p2, dtype=np.float64).reshape(1, 1, 2),
            K2,
            dist2
        ).reshape(2)

        u = np.array([ray1[0], ray1[1], 1.0])
        v = np.array([ray2[0], ray2[1], 1.0])

        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)

        # Camera centres and rays in camera 1 coordinates.
        C1 = np.zeros(3)
        C2 = -R.T @ T

        v = R.T @ v
        v /= np.linalg.norm(v)

        w0 = C1 - C2

        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        d = np.dot(u, w0)
        e = np.dot(v, w0)

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

        # Same floor transformation as Plot_Hand().
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

    # Draw the final calibrated boundary once.
    def set_boundary(self, points):

        missing = [
            marker_id for marker_id in self.ids
            if marker_id not in points
        ]

        if missing:
            raise ValueError(
                f"Missing boundary markers: {missing}"
            )

        self.calibrated_points = {
            marker_id: np.asarray(points[marker_id], dtype=float).copy()
            for marker_id in self.ids
        }

        xyz = np.array([
            self.calibrated_points[i]
            for i in self.ids
        ])

        self.marker_scatter._offsets3d = (
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2]
        )

        for i, marker_id in enumerate(self.ids):

            p1 = self.calibrated_points[marker_id]

            next_id = self.ids[(i + 1) % len(self.ids)]
            p2 = self.calibrated_points[next_id]

            self.labels[i].set_position_3d(p1)
            self.labels[i].set_visible(True)

            self.lines[i].set_data(
                [p1[0], p2[0]],
                [p1[1], p2[1]]
            )

            self.lines[i].set_3d_properties(
                [p1[2], p2[2]]
            )

        self.surface.set_verts([
            [self.calibrated_points[i] for i in self.ids]
        ])

