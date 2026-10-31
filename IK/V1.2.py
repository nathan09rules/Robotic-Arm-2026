

import math
import socket
import time
import threading
from pathlib import Path

import pygame
import matplotlib.pyplot as plt
import numpy as np
import ikpy.chain
from ikpy.link import OriginLink, URDFLink


ESP32_IP = "192.168.4.1"
PORT = 8888
URDF_PATH = Path(__file__).resolve().with_name("Robot.urdf")

JOINT_CONFIG = {
    "joint_14": {"channel": 14, "home": 90.0, "direction": 1},
    "joint_13": {"channel": 13, "home": 80.0, "direction": 1},
    "joint_15": {"channel": 15, "home": 90.0, "direction": 1},
    "joint_4":  {"channel": 4,  "home": 90.0, "direction": 1},
}

SERVO_COMMAND_LIMITS = {
    14: (0.0, 180.0),
    13: (20.0, 180.0),
    15: (0.0, 180.0),
    4: (0.0, 180.0),
}

IK_LIMIT_OVERRIDES = {
    "joint_13": (-60.0, 60.0),
}

FOLLOW_BUTTON = 10
STEPS_PER_UPDATE = 3
CARTESIAN_STEP_SIZE = 0.005
IK_MAX_ITER = 80
IK_TOLERANCE = 0.012

TARGET_XY_RANGE = 0.15
TARGET_Z_DOWN_LIMIT = 0.10
TARGET_Z_UP_LIMIT = 0.30
CONTROLLER_DEADZONE = 0.08
LOOP_DELAY = 0.01


# SERVO STATE
servo_angles = {
    14: 90.0,
    13: 70.0,
    15: 120.0,
    4: 90.0,
}

saved_poses = []
manual_mode = False
command_lock = threading.Lock()


def build_arm_chain():
    if not URDF_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find {URDF_PATH.name}. Put Robot.urdf beside this Python file."
        )

    imported_chain = ikpy.chain.Chain.from_urdf_file(
        str(URDF_PATH),
        base_elements=["base_link"],
        name="urdf_arm",
    )

    imported_names = [link.name for link in imported_chain.links]
    required = {"joint_13", "joint_15", "joint_4"}
    missing = required.difference(imported_names)

    if missing:
        raise ValueError(
            "These joints were not found in Robot.urdf: "
            + ", ".join(sorted(missing))
        )

    base_yaw = URDFLink(
        name="joint_14",
        origin_translation=[0.0, 0.0, 0.0],
        origin_orientation=[0.0, 0.0, 0.0],
        rotation=[0.0, 0.0, 1.0],
        bounds=(math.radians(-90.0), math.radians(90.0)),
    )

    links = [OriginLink(), base_yaw, *imported_chain.links[1:]]

    for link in links:
        if link.name in IK_LIMIT_OVERRIDES:
            lower, upper = IK_LIMIT_OVERRIDES[link.name]
            old_lower, old_upper = link.bounds

            link.bounds = (
                math.radians(lower) if lower is not None else old_lower,
                math.radians(upper) if upper is not None else old_upper,
            )

    active_names = set(JOINT_CONFIG)
    active_mask = [link.name in active_names for link in links]

    arm = ikpy.chain.Chain(
        name="robot_arm",
        links=links,
        active_links_mask=active_mask,
    )

    joint_indices = {link.name: i for i, link in enumerate(arm.links)}

    for name in JOINT_CONFIG:
        if name not in joint_indices:
            raise ValueError(f"Could not locate {name} in the IK chain.")

    print("Loaded Robot.urdf")
    print("Active joints: CH14, CH13, CH15, CH4")

    return arm, joint_indices


arm_chain, joint_indices = build_arm_chain()
current_angles = np.zeros(len(arm_chain.links), dtype=float)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_cmd(payload):
    sock.sendto(payload.encode("utf-8"), (ESP32_IP, PORT))


def servo_command(joint_name, joint_radians):
    config = JOINT_CONFIG[joint_name]
    channel = config["channel"]

    angle = (
        config["home"]
        + config["direction"] * math.degrees(joint_radians)
    )

    low, high = SERVO_COMMAND_LIMITS[channel]

    if not low <= angle <= high:
        raise ValueError(
            f"CH{channel}={angle:.1f} outside safe command range "
            f"({low:.0f} to {high:.0f})."
        )

    return angle


def get_servo_commands(angles):
    return {
        config["channel"]: servo_command(name, angles[joint_indices[name]])
        for name, config in JOINT_CONFIG.items()
    }


def send_pose(angles):
    commands = get_servo_commands(angles)

    with command_lock:
        for channel, angle in commands.items():
            send_cmd(f"{channel}:{angle:.2f}")
            servo_angles[channel] = angle


# MANUAL MODE

def manual_command(channel, angle):
    try:
        channel = int(channel)
        angle = float(angle)

        if channel not in SERVO_COMMAND_LIMITS:
            print("Invalid channel.")
            return

        if not math.isfinite(angle):
            print("Invalid angle.")
            return

        low, high = SERVO_COMMAND_LIMITS[channel]

        if not low <= angle <= high:
            print(f"CH{channel} must be between {low} and {high}.")
            return

        with command_lock:
            send_cmd(f"{channel}:{angle:.2f}")
            servo_angles[channel] = angle

        # Synchronize IK state with the manual servo position.
        config = next(
            (name for name, data in JOINT_CONFIG.items()
             if data["channel"] == channel),
            None
        )

        if config:
            radians = math.radians(
                (angle - JOINT_CONFIG[config]["home"])
                / JOINT_CONFIG[config]["direction"]
            )
            current_angles[joint_indices[config]] = radians

        print(f"CH{channel} -> {angle:.2f}")

    except (ValueError, TypeError):
        print("Use format: 14:100")


def save_pose():
    pose = servo_angles.copy()
    saved_poses.append(pose)

    print(f"Pose {len(saved_poses)} saved:")
    print(pose)


def print_poses():
    print("\nSAVED PROCEDURE")
    print("Procedures = [")

    for pose in saved_poses:
        commands = [
            [channel, angle]
            for channel, angle in pose.items()
        ]

        print(f"    {commands},")

    print("]")
    print()


def clear_poses():
    saved_poses.clear()
    print("All saved poses cleared.")


def terminal_input():
    global manual_mode

    while True:
        try:
            command = input("\nCommand: ").strip()

            if command.lower() == "m":
                manual_mode = not manual_mode
                print("Manual mode:", manual_mode)

            elif command.lower() == "save":
                save_pose()

            elif command.lower() == "list":
                print_poses()

            elif command.lower() == "clear":
                clear_poses()

            elif command.lower() == "exit":
                break

            elif ":" in command:
                if manual_mode:
                    channel, angle = command.split(":", 1)
                    manual_command(channel, angle)
                else:
                    print("Enable manual mode first (M).")

            else:
                print("Commands: M, channel:angle, save, list, clear, exit")

        except (EOFError, KeyboardInterrupt):
            break


# PYGAME

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller detected.")
    sock.close()
    pygame.quit()
    raise SystemExit

controller = pygame.joystick.Joystick(0)
controller.init()

print("\nCONTROLS")
print("Button 10 = IK following")
print("Button 2  = ARM")
print("Button 1  = OFF")
print("Button 3  = Toggle manual mode")
print("Button 4  = Save pose")
print("Button 5  = Print poses")
print("Button 6  = Clear poses")
print("Keyboard M = Toggle manual mode")
print("Terminal   = CH:ANGLE (e.g. 14:100)")

threading.Thread(target=terminal_input, daemon=True).start()


# MATPLOTLIB

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlim(-0.30, 0.30)
ax.set_ylim(-0.30, 0.30)
ax.set_zlim(-0.15, 0.35)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Robot Arm IK / Manual Mode")

arm_line, = ax.plot([], [], [], "-o", linewidth=5, markersize=6)
end_point = ax.scatter([], [], [], c="blue", s=70)
target_point = ax.scatter([], [], [], c="red", s=90)
target_line, = ax.plot([], [], [], "r--", alpha=0.5)

plt.show(block=False)


def draw_arm(angles, target=None):
    frames = arm_chain.forward_kinematics(
        angles,
        full_kinematics=True
    )

    points = np.array([frame[:3, 3] for frame in frames])

    arm_line.set_data(points[:, 0], points[:, 1])
    arm_line.set_3d_properties(points[:, 2])

    end = points[-1]
    end_point._offsets3d = ([end[0]], [end[1]], [end[2]])

    if target is not None:
        target_point._offsets3d = (
            [target[0]], [target[1]], [target[2]]
        )

        target_line.set_data(
            [target[0], target[0]],
            [target[1], target[1]]
        )

        target_line.set_3d_properties([0.0, target[2]])

    else:
        target_point._offsets3d = ([], [], [])
        target_line.set_data([], [])
        target_line.set_3d_properties([])

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


# IK

def deadzone(value):
    return 0.0 if abs(value) < CONTROLLER_DEADZONE else value


def read_target():
    x_raw = deadzone(controller.get_axis(0))
    y_raw = deadzone(-controller.get_axis(1))
    z_axis = deadzone(controller.get_axis(3))

    target_x = x_raw * TARGET_XY_RANGE
    target_y = y_raw * TARGET_XY_RANGE

    if z_axis >= 0.0:
        target_z = -z_axis * TARGET_Z_DOWN_LIMIT
    else:
        target_z = -z_axis * TARGET_Z_UP_LIMIT

    return np.array(
        [target_x, target_y, target_z],
        dtype=float
    )


def follow_target(target_xyz):
    global current_angles

    target_xyz = np.asarray(target_xyz, dtype=float)

    if target_xyz.shape != (3,) or not np.all(np.isfinite(target_xyz)):
        return

    seed = current_angles.copy()

    current_xyz = arm_chain.forward_kinematics(seed)[:3, 3]
    difference = target_xyz - current_xyz
    distance = float(np.linalg.norm(difference))

    if distance < 0.001:
        return

    direction = difference / distance

    steps = min(
        STEPS_PER_UPDATE,
        max(1, int(math.ceil(distance / CARTESIAN_STEP_SIZE)))
    )

    for step in range(1, steps + 1):
        travel = min(step * CARTESIAN_STEP_SIZE, distance)
        waypoint = current_xyz + direction * travel

        try:
            candidate = arm_chain.inverse_kinematics(
                target_position=waypoint,
                initial_position=seed,
                max_iter=IK_MAX_ITER,
            )

            candidate = np.asarray(candidate, dtype=float)

            if not np.all(np.isfinite(candidate)):
                break

            reached = arm_chain.forward_kinematics(candidate)[:3, 3]
            error = float(np.linalg.norm(reached - waypoint))

            if error > IK_TOLERANCE:
                break

            commands = get_servo_commands(candidate)

        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            break

        with command_lock:
            for channel, angle in commands.items():
                send_cmd(f"{channel}:{angle:.2f}")
                servo_angles[channel] = angle

        current_angles = candidate.copy()
        seed = candidate.copy()


# MAIN

send_cmd("ARM")
time.sleep(0.5)

running = True
last_draw = 0.0

try:
    while running:
        pygame.event.pump()

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    manual_mode = not manual_mode
                    print("MANUAL MODE:", manual_mode)

            elif event.type == pygame.JOYBUTTONDOWN:

                if event.button == 2:
                    send_cmd("ARM")
                    print("ARM command sent.")

                elif event.button == 1:
                    send_cmd("OFF")
                    print("OFF command sent.")

                elif event.button == 3:
                    manual_mode = not manual_mode
                    print("MANUAL MODE:", manual_mode)

                elif event.button == 0:
                    save_pose()

                elif event.button == 9:
                    print_poses()

                elif event.button == 11:
                    clear_poses()
                print(event.button)

        target = read_target()

        if not manual_mode and controller.get_button(FOLLOW_BUTTON):
            follow_target(target)

        now = time.monotonic()

        if now - last_draw >= 0.03:
            draw_arm(current_angles, target)
            last_draw = now

        time.sleep(LOOP_DELAY)

except KeyboardInterrupt:
    pass

finally:
    send_cmd("OFF")
    sock.close()
    plt.ioff()
    plt.close(fig)
    pygame.quit()

