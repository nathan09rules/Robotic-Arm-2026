import asyncio
import math
import socket
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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

STEPS_PER_UPDATE = 3
CARTESIAN_STEP_SIZE = 0.005
IK_MAX_ITER = 80
IK_TOLERANCE = 0.012
LOOP_DELAY = 0.01

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_move_lock = asyncio.Lock()
_armed = False


def send_cmd(payload):
    sock.sendto(payload.encode("utf-8"), (ESP32_IP, PORT))


def build_arm_chain():
    if not URDF_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find {URDF_PATH.name}. Put Robot.urdf beside Arm.py."
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


# -------------------- PLOT --------------------

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlim(-0.30, 0.30)
ax.set_ylim(-0.30, 0.30)
ax.set_zlim(-0.15, 0.35)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Robotic Arm — Blue: Arm | Red: Target")

arm_line, = ax.plot(
    [], [], [],
    "-o",
    linewidth=5,
    markersize=6,
    color="blue"
)

arm_endpoint = ax.scatter([], [], [], color="blue", s=50)

target_point = ax.scatter([], [], [], color="red", s=70)

target_guide, = ax.plot(
    [], [],
    [],
    "--",
    color="red",
    linewidth=1
)


def draw_arm(angles, target=None):
    frames = arm_chain.forward_kinematics(
        angles,
        full_kinematics=True
    )

    points = np.array([frame[:3, 3] for frame in frames])

    arm_line.set_data(points[:, 0], points[:, 1])
    arm_line.set_3d_properties(points[:, 2])

    endpoint = points[-1]
    arm_endpoint._offsets3d = (
        [endpoint[0]],
        [endpoint[1]],
        [endpoint[2]]
    )

    if target is not None:
        x, y, z = target

        target_point._offsets3d = ([x], [y], [z])

        target_guide.set_data([x, x], [y, y])
        target_guide.set_3d_properties([0.0, z])
    else:
        target_point._offsets3d = ([], [], [])
        target_guide.set_data([], [])
        target_guide.set_3d_properties([])

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


draw_arm(current_angles)
plt.show(block=False)


# -------------------- SERVO CONTROL --------------------

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
        config["channel"]: servo_command(
            name, angles[joint_indices[name]]
        )
        for name, config in JOINT_CONFIG.items()
    }


def arm():
    global _armed

    if not _armed:
        send_cmd("ARM")
        _armed = True


def disarm():
    global _armed

    send_cmd("OFF")
    _armed = False


# -------------------- MOVEMENT --------------------

async def move_to_target(target_xyz, traking=False):
    global current_angles

    target = np.asarray(target_xyz, dtype=float)

    if target.shape != (3,) or not np.all(np.isfinite(target)):
        return []

    async with _move_lock:
        seed = current_angles.copy()
        current_xyz = arm_chain.forward_kinematics(seed)[:3, 3]

        difference = target - current_xyz
        distance = float(np.linalg.norm(difference))

        draw_arm(current_angles, target)

        if distance < 0.001:
            return [("Target", tuple(target))]

        direction = difference / distance

        total_steps = max(
            1, int(math.ceil(distance / CARTESIAN_STEP_SIZE))
        )

        if traking:
            total_steps = min(STEPS_PER_UPDATE, total_steps)

        moved = False

        for step in range(1, total_steps + 1):
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

            arm()

            for channel, angle in commands.items():
                send_cmd(f"{channel}:{angle:.2f}")

            current_angles = candidate.copy()
            seed = candidate.copy()
            moved = True

            draw_arm(current_angles, target)

            await asyncio.sleep(LOOP_DELAY)

        final_xyz = arm_chain.forward_kinematics(current_angles)[:3, 3]
        remaining = float(np.linalg.norm(target - final_xyz))

        if not moved and remaining > IK_TOLERANCE:
            return []

        if not traking and remaining > IK_TOLERANCE:
            print(
                f"Target not fully reached. Remaining distance: "
                f"{remaining:.4f} m"
            )
            return []

        return [("Target", tuple(target))]


# -------------------- PROCEDURES --------------------

async def run_procedure(steps):
    history = []

    for step in steps:
        if not step:
            continue

        if step[0] in ("IK", "Target"):
            if len(step) < 4:
                raise ValueError(
                    "IK/Target steps must contain X, Y and Z."
                )

            history.extend(
                await move_to_target(step[1:4], traking=False)
            )

        elif len(step) == 2:
            channel, angle = step

            if channel not in SERVO_COMMAND_LIMITS:
                raise ValueError(f"Unsupported servo channel: {channel}")

            low, high = SERVO_COMMAND_LIMITS[channel]
            angle = float(angle)

            if not low <= angle <= high:
                raise ValueError(
                    f"CH{channel} angle must be between {low} and {high}."
                )

            arm()
            send_cmd(f"{channel}:{angle:.2f}")

            current_angles[joint_indices[
                next(name for name, config in JOINT_CONFIG.items()
                     if config["channel"] == channel)
            ]] = math.radians(
                (angle - JOINT_CONFIG[
                    next(name for name, config in JOINT_CONFIG.items()
                         if config["channel"] == channel)
                ]["home"])
                / JOINT_CONFIG[
                    next(name for name, config in JOINT_CONFIG.items()
                         if config["channel"] == channel)
                ]["direction"]
            )

            draw_arm(current_angles)
            history.append(("Servo", (channel, angle)))

        elif len(step) == 1:
            delay = float(step[0])

            if delay < 0:
                raise ValueError("Pause duration cannot be negative.")

            await asyncio.sleep(delay)

        else:
            raise ValueError(f"Unknown procedure step: {step}")

    return history


def close():
    disarm()
    sock.close()
    plt.close(fig)
