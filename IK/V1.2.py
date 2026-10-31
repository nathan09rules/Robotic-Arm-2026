

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

SERVO_COMMAND_LIMITS = {
    14: (0.0, 180.0),
    13: (20.0, 180.0),
    15: (0.0, 180.0),
    4: (0.0, 180.0),
}

IK_LIMIT_OVERRIDES = {
    "joint_13": (-60.0, 60.0),
}

# CONTROLLER SETTINGS
FOLLOW_BUTTON = 10
CONTROLLER_DEADZONE = 0.12

# Target movement speed (metres per second)
TARGET_SPEED_XY = 0.08
TARGET_SPEED_Z = 0.06

# Target workspace limits
TARGET_XY_RANGE = 0.3
TARGET_Z_DOWN_LIMIT = 0.10
TARGET_Z_UP_LIMIT = 0.30

# IK SETTINGS
STEPS_PER_UPDATE = 3
CARTESIAN_STEP_SIZE = 0.005
IK_MAX_ITER = 80
IK_TOLERANCE = 0.012

LOOP_DELAY = 0.01


def build_arm_chain():
    if not URDF_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find {URDF_PATH.name}. Put Robot.urdf beside this file."
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
        bounds=(math.radians(-90), math.radians(90)),
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

    active_mask = [
        link.name in active_names
        for link in links
    ]

    arm = ikpy.chain.Chain(
        name="robot_arm",
        links=links,
        active_links_mask=active_mask,
    )

    joint_indices = {
        link.name: i
        for i, link in enumerate(arm.links)
    }

    for name in JOINT_CONFIG:
        if name not in joint_indices:
            raise ValueError(f"Could not locate {name} in IK chain.")

    print("Loaded Robot.urdf")
    print("Active joints: CH14, CH13, CH15, CH4")
    print("Hold button 10 to follow target")

    return arm, joint_indices


arm_chain, joint_indices = build_arm_chain()

current_angles = np.zeros(len(arm_chain.links), dtype=float)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_cmd(payload):
    sock.sendto(
        payload.encode("utf-8"),
        (ESP32_IP, PORT)
    )


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
            f"CH{channel}={angle:.1f} outside safe range "
            f"({low:.0f} to {high:.0f})."
        )

    return angle


def get_servo_commands(angles):
    return {
        config["channel"]: servo_command(
            name,
            angles[joint_indices[name]]
        )
        for name, config in JOINT_CONFIG.items()
    }


def send_pose(angles):
    commands = get_servo_commands(angles)

    for channel, angle in commands.items():
        send_cmd(f"{channel}:{angle:.2f}")


# --------------------------------------------------
# CONTROLLER
# --------------------------------------------------

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller detected.")
    sock.close()
    pygame.quit()
    raise SystemExit

controller = pygame.joystick.Joystick(0)
controller.init()

print("\nControls:")
print("  Left joystick X = Move target X")
print("  Left joystick Y = Move target Y")
print("  Right joystick Y = Move target Z")
print("  Button 10        = Follow target")
print("  Button 2         = ARM / reset software pose")
print("  Button 1         = OFF")


def process_axis(value):
    if abs(value) < CONTROLLER_DEADZONE:
        return 0.0

    return math.copysign(
        (abs(value) - CONTROLLER_DEADZONE)
        / (1.0 - CONTROLLER_DEADZONE),
        value
    )


def read_target(target, dt):
    x = process_axis(controller.get_axis(0))
    y = process_axis(-controller.get_axis(1))
    z = process_axis(controller.get_axis(3))

    target[0] += x * TARGET_SPEED_XY * dt
    target[1] += y * TARGET_SPEED_XY * dt
    target[2] -= z * TARGET_SPEED_Z * dt

    target[0] = np.clip(
        target[0],
        -TARGET_XY_RANGE,
        TARGET_XY_RANGE
    )

    target[1] = np.clip(
        target[1],
        -TARGET_XY_RANGE,
        TARGET_XY_RANGE
    )

    target[2] = np.clip(
        target[2],
        -TARGET_Z_DOWN_LIMIT,
        TARGET_Z_UP_LIMIT
    )

    return target


# --------------------------------------------------
# PLOT
# --------------------------------------------------

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlim(-0.30, 0.30)
ax.set_ylim(-0.30, 0.30)
ax.set_zlim(-0.15, 0.35)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

ax.set_title("Velocity Controlled IK")

arm_line, = ax.plot(
    [], [], [],
    "-o",
    linewidth=5,
    markersize=6
)

end_point = ax.scatter([], [], [], c="blue", s=70)
target_point = ax.scatter([], [], [], c="red", s=90)

target_line, = ax.plot(
    [], [], [],
    "r--",
    alpha=0.5
)

plt.show(block=False)


def draw_arm(angles, target=None):
    frames = arm_chain.forward_kinematics(
        angles,
        full_kinematics=True
    )

    points = np.array([
        frame[:3, 3]
        for frame in frames
    ])

    arm_line.set_data(
        points[:, 0],
        points[:, 1]
    )

    arm_line.set_3d_properties(points[:, 2])

    end = points[-1]

    end_point._offsets3d = (
        [end[0]],
        [end[1]],
        [end[2]]
    )

    if target is not None:
        target_point._offsets3d = (
            [target[0]],
            [target[1]],
            [target[2]]
        )

        target_line.set_data(
            [target[0], target[0]],
            [target[1], target[1]]
        )

        target_line.set_3d_properties(
            [0.0, target[2]]
        )

    else:
        target_point._offsets3d = ([], [], [])
        target_line.set_data([], [])
        target_line.set_3d_properties([])

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


# --------------------------------------------------
# INVERSE KINEMATICS
# --------------------------------------------------

def follow_target(target_xyz):
    global current_angles

    target_xyz = np.asarray(target_xyz, dtype=float)

    if (
        target_xyz.shape != (3,)
        or not np.all(np.isfinite(target_xyz))
    ):
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
        max(
            1,
            int(math.ceil(distance / CARTESIAN_STEP_SIZE))
        )
    )

    for step in range(1, steps + 1):

        travel = min(
            step * CARTESIAN_STEP_SIZE,
            distance
        )

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

            reached = arm_chain.forward_kinematics(
                candidate
            )[:3, 3]

            error = float(
                np.linalg.norm(reached - waypoint)
            )

            if error > IK_TOLERANCE:
                break

            commands = get_servo_commands(candidate)

        except (
            ValueError,
            RuntimeError,
            np.linalg.LinAlgError
        ):
            break

        for channel, angle in commands.items():
            send_cmd(f"{channel}:{angle:.2f}")

        current_angles = candidate.copy()
        seed = candidate.copy()


# --------------------------------------------------
# INITIALISE
# --------------------------------------------------

send_cmd("ARM")
time.sleep(0.5)

# Target begins at the arm's current end-effector position.
target = arm_chain.forward_kinematics(
    current_angles
)[:3, 3].copy()

last_time = time.monotonic()
last_draw = 0.0

print("\nTarget initialised:", target)


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

running = True

try:
    while running:

        pygame.event.pump()

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.JOYBUTTONDOWN:

                if event.button == 2:
                    send_cmd("ARM")

                    current_angles = np.zeros(
                        len(arm_chain.links),
                        dtype=float
                    )

                    target = arm_chain.forward_kinematics(
                        current_angles
                    )[:3, 3].copy()

                    print("ARM command sent. Pose reset.")

                elif event.button == 1:
                    send_cmd("OFF")
                    print("OFF command sent.")

        now = time.monotonic()

        dt = min(now - last_time, 0.05)
        last_time = now

        # Update target using joystick velocity.
        target = read_target(target, dt)

        # Follow target while button 10 is held.
        if controller.get_button(FOLLOW_BUTTON):
            follow_target(target)
            print(target)

        # Update plot.
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

