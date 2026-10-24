import os

os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import urllib.request
import cv2
import mediapipe as mp
import numpy as np


#URL = "http://10.134.12.225:8080/shot.jpg"
URL = "http://192.168.1.15:8080/shot.jpg"


#---Hand traking Configs------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options_cam1 = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)
options_cam2 = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

landmarker_cam1 = HandLandmarker.create_from_options(options_cam1)
landmarker_cam2 = HandLandmarker.create_from_options(options_cam2)


#---Calibration Configs------

board = cv2.aruco.CharucoBoard(
    (7, 5),
    25,
    18,
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
)


data = np.load("camera_calibration.npz")
K = data["camera_matrix"]
dist = data["distortion"]

detector = cv2.aruco.CharucoDetector(board)

last_timestamp_ms = 0

cap = cv2.VideoCapture(0)

# ------3D PLOTTING matplotlib
import matplotlib.pyplot as plt

plt.ion()
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")

# ----------HAND STATE----------------
Hand_State = [0,0,0,0,0] # 0: Extended. 1: Curled

Finger_Name = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def distance(a, b):
    return np.linalg.norm(
        np.array([a.x, a.y, a.z]) -
        np.array([b.x, b.y, b.z])
    )

def process_and_display(frame, landmarker, timestamp_ms, window_name):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w, _ = frame.shape
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            for start_connection, end_connection in HAND_CONNECTIONS:
                start_idx = start_connection.start if hasattr(start_connection, 'start') else start_connection
                end_idx = end_connection.end if hasattr(end_connection, 'end') else end_connection
                cv2.line(frame, pts[start_idx], pts[end_idx], (0, 255, 0), 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)

 


    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.imshow(window_name, frame)

    ax.clear()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(-0.2, 0.2)

    if result.hand_world_landmarks:
        for hand_world_landmarks in result.hand_world_landmarks:
            xs = [lm.x for lm in hand_world_landmarks]
            ys = [lm.y for lm in hand_world_landmarks]
            zs = [lm.z for lm in hand_world_landmarks]

            for start_connection, end_connection in HAND_CONNECTIONS:
                start_idx = start_connection.start if hasattr(start_connection, 'start') else start_connection
                end_idx = end_connection.end if hasattr(end_connection, 'end') else end_connection
                ax.plot(
                    [xs[start_idx], xs[end_idx]],
                    [ys[start_idx], ys[end_idx]],
                    [zs[start_idx], zs[end_idx]],
                    color="g",
                    linewidth=2,
                )

            ax.scatter(xs, ys, zs, color="r", s=20)
    
        fig.canvas.draw()
        fig.canvas.flush_events()
        #plt.pause(0.001)

        #----CHECK HAND STATE------- else:
            #------------ NOTE --------------------------
                #DO THE PRESENCE CHECK FOR EACH FINGER
        hand = result.hand_world_landmarks[0]

        def dist(a, b):
            return np.linalg.norm(
                np.array([a.x, a.y, a.z]) -
                np.array([b.x, b.y, b.z])
            )

        # Hand scale
        # Wrist -> Middle MCP
        hand_scale = dist(hand[0], hand[9])

        # Palm center
        # Average of the four MCP joints
        palm = np.mean(
            [
                [hand[5].x,  hand[5].y,  hand[5].z],
                [hand[9].x,  hand[9].y,  hand[9].z],
                [hand[13].x, hand[13].y, hand[13].z],
                [hand[17].x, hand[17].y, hand[17].z],
            ],
            axis=0
        )

        def palm_distance(index):
            tip = np.array([
                hand[index].x,
                hand[index].y,
                hand[index].z
            ])

            return np.linalg.norm(tip - palm) / hand_scale

        
        finger_tips = {
            1: 8,    # index
            2: 12,   # middle
            3: 16,   # ring
            4: 20    # pinky
        }
        
        os.system("clear")
        for finger_num, tip_index in finger_tips.items():

            r = palm_distance(tip_index)

            if r < 0.65:
                Hand_State[finger_num] = 1       # CURLED

            elif r > 0.85:
                Hand_State[finger_num] = 0       # EXTENDED

            else:
                Hand_State[finger_num] = 2       # NEUTRAL

            print(
                Finger_Name[finger_num],
                "ratio =", round(r, 3),
                "state =", Hand_State[finger_num]
            )


def find_position(frame):
    corners, ids, _, _ = detector.detectBoard(frame)

    if ids is None or len(ids) < 6:
        print(f"Only {0 if ids is None else len(ids)} corners detected")
        return

    obj, img = board.matchImagePoints(corners, ids)

    if obj is None or len(obj) < 6:
        return

    ok, rvec, tvec = cv2.solvePnP(
        obj,
        img,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if ok:
        R, _ = cv2.Rodrigues(rvec)

        # Camera position in board coordinates
        C = -R.T @ tvec

        x, y, z = (C / 1000).flatten()   # because board is in mm

        print(
            f"Camera: X={x:.3f} "
            f"Y={y:.3f} "
            f"Z={z:.3f} m"
        )

        xs.append(x)
        ys.append(y)
        zs.append(z)

        ax.clear()

        ax.plot(xs, ys, zs, "r-")
        ax.scatter(x, y, z, c="red", s=100)

        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0, 3)

        plt.draw()
        plt.pause(0.001)
 
try:
  while True:
    try:
      req = urllib.request.urlopen(URL, timeout=2)
      img_arr = np.array(bytearray(req.read()), dtype=np.uint8)
      frame_cam1 = cv2.imdecode(img_arr, -1)
    except Exception as e:
      print(f"Frame fetch error: {e}")
      continue

    success, frame_cam2 = cap.read()

    if frame_cam1 is None or not success or frame_cam2 is None:
      continue

    # Single uniform timestamp synchronized across both streams
    sync_timestamp_ms = int(time.time() * 1000)
    if sync_timestamp_ms <= last_timestamp_ms:
        sync_timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = sync_timestamp_ms

    process_and_display(frame_cam1, landmarker_cam1, sync_timestamp_ms, "Camera 1 (IP Webcam)")
         
    
    #process_and_display(frame_cam2, landmarker_cam2, sync_timestamp_ms, "Camera 2 (Laptop-Cam)")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker_cam1.close()
    landmarker_cam2.close()
