import socket
import time
import math

import pygame
import matplotlib.pyplot as plt
import numpy as np
import ikpy.chain
from ikpy.link import OriginLink, URDFLink

ESP32_IP = "192.168.4.1"
PORT = 8888

# Physical servo reference angles from the measured arm pose.
HOME_CH14 = 135.0  # base yaw
HOME_CH13 = 80.0   # base pitch
HOME_CH15 = 120.0  # elbow

LINK_1 = 0.115
LINK_2 = 0.100
BASE_HEIGHT = 0.080

# Keep commands inside physical servo limits.
SERVO_MIN = 0.0
SERVO_MAX = 180.0

# Flip a direction to -1 if that physical servo moves opposite to the preview.
DIR_CH14 = 1
DIR_CH13 = 1
DIR_CH15 = 1

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_cmd(payload):
    sock.sendto(payload.encode("utf-8"), (ESP32_IP, PORT))


def servo_angle(home, joint_radians, direction=1):
    return max(SERVO_MIN, min(SERVO_MAX, home + direction * math.degrees(joint_radians)))


# IKPy joint indices:
# [0] origin, [1] yaw/CH14, [2] pitch/CH13, [3] elbow/CH15, [4] fixed tool
#
# The mounting rotations are chosen so that at zero joint displacement:
#   CH13=80°, CH15=120° -> link 1 vertical, link 2 horizontal.
# Link dimensions match the supplied measurements and URDF reference.
HOME_CH14 = 135.0
HOME_CH13 = 80.0
HOME_CH15 = 120.0

arm_chain = ikpy.chain.Chain(
    name="calibrated_arm",
    links=[
        OriginLink(),

        # CH14: base yaw
        URDFLink(
            name="base_yaw_ch14",
            origin_translation=[0, 0, 0.03],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1],
            bounds=(
                math.radians(0 - HOME_CH14),
                math.radians(180 - HOME_CH14)
            )
        ),

        # CH13: base pitch; mounting orientation is separate
        URDFLink(
            name="base_pitch_ch13",
            origin_translation=[0, 0, 0.05],
            origin_orientation=[0, -math.pi / 2, 0],
            rotation=[0, 1, 0],
            bounds=(
                math.radians(0 - HOME_CH13),
                math.radians(180 - HOME_CH13)
            )
        ),

        # CH15: elbow
        URDFLink(
            name="elbow_ch15",
            origin_translation=[0.115, 0, 0],
            origin_orientation=[0, math.pi / 2, 0],
            rotation=[0, 1, 0],
            bounds=(
                math.radians(0 - HOME_CH15),
                math.radians(180 - HOME_CH15)
            )
        ),

        # Fixed end effector
        URDFLink(
            name="tool",
            origin_translation=[0.100, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 0],
            bounds=(0, 0)
        )
    ],
    active_links_mask=[False, True, True, True, False]
)# Joint displacements are relative to the physical reference angles above.
current_angles = [0.0, 0.0, 0.0, 0.0, 0.0]

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller detected.")
    pygame.quit()
    raise SystemExit

controller = pygame.joystick.Joystick(0)
controller.init()

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


def draw_arm(angles, target=None):
    ax.clear()
    ax.set_xlim(-0.27, 0.27)
    ax.set_ylim(-0.27, 0.27)
    ax.set_zlim(0, 0.35)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("IK arm preview — blue = current, red = target")

    points = [
        arm_chain.forward_kinematics(angles, full_kinematics=True)[i][:3, 3]
        for i in range(len(arm_chain.links))
    ]
    xyz = np.array(points)
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "-o", linewidth=5, markersize=6)

    end = arm_chain.forward_kinematics(angles)[:3, 3]
    ax.scatter([end[0]], [end[1]], [end[2]], c="blue", s=70)

    if target is not None:
        ax.scatter([target[0]], [target[1]], [target[2]], c="red", s=90)
        ax.plot([target[0], target[0]], [target[1], target[1]],
                [0, target[2]], "r--", alpha=0.5)

    plt.draw()
    plt.pause(0.01)


def move_to(target_xyz):
    global current_angles

    initial = list(current_angles)
    initial[1] = math.atan2(target_xyz[1], target_xyz[0])

    target_angles = arm_chain.inverse_kinematics(
        target_position=target_xyz,
        initial_position=initial,
        max_iter=1000
    )

    # Reject solutions that put any modeled joint below the table.
    test_points = [
        arm_chain.forward_kinematics(target_angles, full_kinematics=True)[i][:3, 3]
        for i in range(1, len(arm_chain.links))
    ]
    if any(point[2] < 0.0 for point in test_points):
        print("Move rejected: solution puts part of the arm below the table.")
        return

    steps = 30
    start_angles = current_angles.copy()

    for step in range(1, steps + 1):
        alpha = step / steps
        interpolated = [
            start + (target - start) * alpha
            for start, target in zip(start_angles, target_angles)
        ]

        # Convert relative IK angles to absolute physical servo angles.
        deg14 = servo_angle(HOME_CH14, interpolated[1], DIR_CH14)
        deg13 = servo_angle(HOME_CH13, interpolated[2], DIR_CH13)
        deg15 = servo_angle(HOME_CH15, interpolated[3], DIR_CH15)

        # Send all three joints. The ESP32 should also slew-limit servo motion.
        send_cmd(f"14:{deg14:.2f}")
        send_cmd(f"13:{deg13:.2f}")
        send_cmd(f"15:{deg15:.2f}")

        draw_arm(interpolated, target_xyz)
        time.sleep(0.035)

    current_angles = target_angles.tolist()


send_cmd("ARM")
time.sleep(1.5)

running = True
try:
    while running:
        pygame.event.pump()

        x_raw = controller.get_axis(0) * 0.8
        y_raw = -controller.get_axis(1) * 0.8
        z_raw = -controller.get_axis(3) * 0.8

        target_x = x_raw * 0.15
        target_y = y_raw * 0.15
        target_z = max(0.05, (z_raw * 0.10) + 0.15)
        target = np.array([target_x, target_y, target_z])

        draw_arm(current_angles, target)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 10:
                    move_to(target)

                elif event.button == 2:
                    send_cmd("ARM")
                    current_angles = [0.0, 0.0, 0.0, 0.0, 0.0]

                elif event.button == 1:
                    send_cmd("OFF")

except KeyboardInterrupt:
    pass
finally:
    send_cmd("OFF")
    sock.close()
    plt.ioff()
    plt.close(fig)
    pygame.quit()
