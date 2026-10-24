import os
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import time
import mediapipe as mp
import matplotlib.pyplot as plt

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
 
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

last_timestamp_ms = 0

plt.ion()
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

while cap.isOpened():
    success, frame = cap.read()
    if not success or frame is None:
        continue

    # Strictly increasing timestamp check
    timestamp_ms = int(time.time() * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = timestamp_ms

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    
    if result.hand_world_landmarks:
        ax.clear()
        
        colors = ['r', 'b']
        for hand_idx, hand_landmarks in enumerate(result.hand_world_landmarks):
            wrist = hand_landmarks[0]
            x, y, z = wrist.x, wrist.y, wrist.z

            color = colors[hand_idx % len(colors)]
            ax.scatter(x, z, -y, c=color, marker='o', s=60, label=f"Hand {hand_idx} Wrist")

        ax.set_xlim([-0.3, 0.3])
        ax.set_ylim([-0.3, 0.3])
        ax.set_zlim([-0.3, 0.3])

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_zlabel('Y (m)')
        ax.set_title("Real-Time Wrist 3D Coordinates (Index 0)")
        ax.legend()

        plt.draw()
        plt.pause(0.001)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
plt.close()
landmarker.close() 
