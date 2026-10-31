import math
import socket
import time
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

# ESP32 firmware accepts servo commands from 0 to 180 degrees.
SERVO_COMMAND_LIMITS = {
    14: (0.0, 180.0),
    13: (20.0, 180.0),
    15: (0.0, 180.0),
    4:  (0.0, 180.0),
}

IK_LIMIT_OVERRIDES = {
    "joint_13": (-60.0, 60.0),
}

# Receding-horizon settings: calculate and execute only these few steps,
# then read the controller again and re-plan.
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
            "These joints were not found in Robot.urdf: " + ", ".join(sorted(missing))
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
    print("Following: hold controller button 10")
    return arm, joint_indices


arm_chain, joint_indices = build_arm_chain()
current_angles = np.zeros(len(arm_chain.links), dtype=float)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_cmd(payload):
    sock.sendto(payload.encode("utf-8"), (ESP32_IP, PORT))


def servo_command(joint_name, joint_radians):
    config = JOINT_CONFIG[joint_name]
    channel = config["channel"]
    angle = config["home"] + config["direction"] * math.degrees(joint_radians)
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
    # Send every joint target immediately, without waiting between channels.
    # The ESP32 updates its channels independently, so they move concurrently.
    commands = get_servo_commands(angles)
    for channel, angle in commands.items():
        send_cmd(f"{channel}:{angle:.2f}")


pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller detected.")
    sock.close()
    pygame.quit()
    raise SystemExit

controller = pygame.joystick.Joystick(0)
controller.init()

print("Controls:")
print("  Hold button 10 = continuously follow joystick XYZ")
print("  Button 2       = ARM / reset software pose")
print("  Button 1       = OFF")

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-0.30, 0.30)
ax.set_ylim(-0.30, 0.30)
ax.set_zlim(-0.15, 0.35)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Continuous IK — blue = arm, red = target")

arm_line, = ax.plot([], [], [], "-o", linewidth=5, markersize=6)
end_point = ax.scatter([], [], [], c="blue", s=70)
target_point = ax.scatter([], [], [], c="red", s=90)
target_line, = ax.plot([], [], [], "r--", alpha=0.5)
plt.show(block=False)


def draw_arm(angles, target=None):
    frames = arm_chain.forward_kinematics(angles, full_kinematics=True)
    points = np.array([frame[:3, 3] for frame in frames])

    arm_line.set_data(points[:, 0], points[:, 1])
    arm_line.set_3d_properties(points[:, 2])

    end = points[-1]
    end_point._offsets3d = ([end[0]], [end[1]], [end[2]])

    if target is not None:
        target_point._offsets3d = ([target[0]], [target[1]], [target[2]])
        target_line.set_data([target[0], target[0]], [target[1], target[1]])
        target_line.set_3d_properties([0.0, target[2]])
    else:
        target_point._offsets3d = ([], [], [])
        target_line.set_data([], [])
        target_line.set_3d_properties([])

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


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

    return np.array([target_x, target_y, target_z], dtype=float)


def follow_target(target_xyz):
    """Execute only the next few Cartesian steps, then re-plan next loop."""
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
        max(1, int(math.ceil(distance / CARTESIAN_STEP_SIZE))),
    )

    moved = False

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

            # Validate all servo limits before sending this pose.
            commands = get_servo_commands(candidate)

        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            break

        # Update all four servo targets for this waypoint.
        for channel, angle in commands.items():
            send_cmd(f"{channel}:{angle:.2f}")

        current_angles = candidate.copy()
        seed = candidate.copy()
        moved = True

    if not moved:
        return


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

            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 2:
                    send_cmd("ARM")
                    current_angles = np.zeros(len(arm_chain.links), dtype=float)
                    print("ARM command sent. Software pose reset.")

                elif event.button == 1:
                    send_cmd("OFF")
                    print("OFF command sent.")

        target = read_target()

        # Button 10 is checked every loop, not just on button-down.
        if controller.get_button(FOLLOW_BUTTON):
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
