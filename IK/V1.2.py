import math
import socket
import time
from pathlib import Path

import pygame
import matplotlib.pyplot as plt
import numpy as np
import ikpy.chain


ESP32_IP = "192.168.4.1"
PORT = 8888
URDF_PATH = Path(__file__).resolve().with_name("Robot.urdf")

STARTUP = {0: 70, 1: 90, 4: 90, 13: 90, 14: 90, 15: 150}

# channel: (channel, home servo angle, direction, servo degrees per URDF degree)
# The 1.5 conversion applies ONLY to CH15.
JOINTS = {
    "joint_14": (14, 90.0,  1.0, 1.0),
    "joint_13": (13, 90.0, -1.0, 1.0),
    "joint_15": (15, 150.0, 1.0, 1.5),
    "joint_4":  (4, 90.0, -1.0, 1.0),
}

LIMITS = {
    14: (0, 180),
    13: (0, 90),
    15: (90, 150),
    4: (0, 90),
}

FOLLOW_BUTTON = 10
ARM_BUTTON = 2
OFF_BUTTON = 1

DEADZONE = 0.10
MOVE_SPEED = 0.08
UPDATE_RATE = 1 / 30
IK_TOLERANCE = 0.015

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send(channel, angle):
    sock.sendto(
        f"{channel}:{angle:.2f}".encode(),
        (ESP32_IP, PORT)
    )


def send_startup():
    for channel, angle in STARTUP.items():
        send(channel, angle)


if not URDF_PATH.is_file():
    raise FileNotFoundError(f"Cannot find {URDF_PATH}")

arm = ikpy.chain.Chain.from_urdf_file(
    str(URDF_PATH),
    base_elements=["base_link"],
)

names = [link.name for link in arm.links]
missing = set(JOINTS) - set(names)

if missing:
    raise ValueError(f"Missing URDF joints: {missing}")

arm.active_links_mask = [
    link.name in JOINTS for link in arm.links
]

indices = {name: names.index(name) for name in JOINTS}

# URDF q=0 is the physical startup pose:
# CH14=90, CH13=90, CH15=150, CH4=90.
angles = np.zeros(len(arm.links), dtype=float)

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise SystemExit("No controller detected.")

controller = pygame.joystick.Joystick(0)
controller.init()

sock.sendto(b"ARM", (ESP32_IP, PORT))
time.sleep(0.3)
send_startup()

target = arm.forward_kinematics(angles)[:3, 3].copy()

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlim(-0.40, 0.40)
ax.set_ylim(-0.40, 0.40)
ax.set_zlim(0.0, 0.45)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Blue: robot | Red: target")

arm_line, = ax.plot([], [], [], "-o", linewidth=4, markersize=6)
end_point = ax.scatter([], [], [], c="blue", s=65)
target_point = ax.scatter([], [], [], c="red", s=85)
target_line, = ax.plot([], [], [], "r--", alpha=0.5)


def get_commands(solution):
    commands = {}

    for name, (channel, home, direction, scale) in JOINTS.items():
        q_deg = math.degrees(solution[indices[name]])

        # CH15 alone has a 1.5:1 servo-to-URDF angle conversion.
        # All other joints use a direct 1:1 conversion.
        servo = home + direction * q_deg / scale

        low, high = LIMITS[channel]

        if not low - 0.01 <= servo <= high + 0.01:
            raise ValueError(
                f"CH{channel} solution {servo:.1f} outside {low}-{high}"
            )

        commands[channel] = float(np.clip(servo, low, high))

    return commands


def solve_ik(candidate_target):
    global angles

    solution = arm.inverse_kinematics(
        target_position=candidate_target,
        initial_position=angles.copy(),
        max_iter=60,
    )

    solution = np.asarray(solution, dtype=float)

    if not np.all(np.isfinite(solution)):
        print("IK returned invalid angles")
        return False

    reached = arm.forward_kinematics(solution)[:3, 3]
    error = float(np.linalg.norm(reached - candidate_target))

    if error > IK_TOLERANCE:
        print(f"Target unreachable: error {error * 100:.1f} cm")
        return False

    commands = get_commands(solution)

    angles = solution.copy()

    for channel, value in commands.items():
        send(channel, value)

    return True


def plot_arm():
    frames = arm.forward_kinematics(
        angles,
        full_kinematics=True
    )
    points = np.array([frame[:3, 3] for frame in frames])

    arm_line.set_data(points[:, 0], points[:, 1])
    arm_line.set_3d_properties(points[:, 2])

    end = points[-1]
    end_point._offsets3d = ([end[0]], [end[1]], [end[2]])
    target_point._offsets3d = ([target[0]], [target[1]], [target[2]])

    target_line.set_data(
        [end[0], target[0]],
        [end[1], target[1]]
    )
    target_line.set_3d_properties([end[2], target[2]])

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


print("Loaded Robot.urdf")
print("Startup: CH14=90, CH13=90, CH15=150, CH4=90")
print("CH15: 150 servo degrees = perpendicular; 90 = straight.")
print("CH14, CH13 and CH4 use normal 1:1 angle conversion.")
print("Hold button 10 to move the target.")
print("Blue = robot, red = target.")

last_update = 0.0

try:
    while True:
        pygame.event.pump()

        if controller.get_button(OFF_BUTTON):
            sock.sendto(b"OFF", (ESP32_IP, PORT))
            time.sleep(0.25)

        if controller.get_button(ARM_BUTTON):
            sock.sendto(b"ARM", (ESP32_IP, PORT))
            time.sleep(0.2)
            send_startup()

            angles[:] = 0.0
            target = arm.forward_kinematics(angles)[:3, 3].copy()
            time.sleep(0.25)

        now = time.monotonic()

        if controller.get_button(FOLLOW_BUTTON):
            if now - last_update >= UPDATE_RATE:
                dt = min(now - last_update, 0.05) if last_update else UPDATE_RATE
                last_update = now

                movement = np.array([
                    controller.get_axis(0),
                    -controller.get_axis(1),
                    -controller.get_axis(3),
                ])

                movement[np.abs(movement) < DEADZONE] = 0.0

                if np.any(movement):
                    candidate = target + movement * MOVE_SPEED * dt

                    candidate[0] = np.clip(candidate[0], -0.35, 0.35)
                    candidate[1] = np.clip(candidate[1], -0.35, 0.35)
                    candidate[2] = np.clip(candidate[2], 0.0, 0.40)

                    # Only accept the new target if IK can reach it.
                    if solve_ik(candidate):
                        target = candidate

        else:
            last_update = now

        plot_arm()
        time.sleep(0.005)

except KeyboardInterrupt:
    pass

finally:
    sock.sendto(b"OFF", (ESP32_IP, PORT))
    sock.close()
    plt.close("all")
    pygame.quit()

