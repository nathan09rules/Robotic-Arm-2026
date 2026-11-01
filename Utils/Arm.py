import math
import socket
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ikpy.chain
from ikpy.link import OriginLink, URDFLink

ESP32_IP = "192.168.4.1"
PORT = 8888
URDF_PATH = Path(__file__).resolve().with_name("Robot.urdf")

JOINT_CONFIG = {
    "joint_14": {"channel": 14, "home": 90.0, "direction": 1, "offset": 90.0},
    "joint_13": {"channel": 13, "home": 80.0, "direction": 1, "offset": 0.0},
    "joint_15": {"channel": 15, "home": 90.0, "direction": 1, "offset": 0.0},
    "joint_4":  {"channel": 4,  "home": 90.0, "direction": 1, "offset": 0.0},
}

SERVO_COMMAND_LIMITS = {
    14: (0.0, 180.0),
    13: (20.0, 180.0),
    15: (0.0, 180.0),
    4:  (0.0, 180.0),
}

IK_LIMIT_OVERRIDES = {
    "joint_13": (-60.0, 60.0),
}

STEPS_PER_UPDATE = 30
CARTESIAN_STEP_SIZE = 0.03
IK_MAX_ITER = 150
IK_TOLERANCE = 0.02
LOOP_DELAY = 0.01

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_armed = False
_last_target = None
_target_queue = []

def send_cmd(payload):
    sock.sendto(payload.encode("utf-8"), (ESP32_IP, PORT))

def build_arm_chain():
    if not URDF_PATH.is_file():
        raise FileNotFoundError(f"Could not find {URDF_PATH.name}.")

    imported_chain = ikpy.chain.Chain.from_urdf_file(
        str(URDF_PATH),
        base_elements=["base_link"],
        name="urdf_arm",
    )

    imported_names = [link.name for link in imported_chain.links]
    required = {"joint_13", "joint_15", "joint_4"}
    missing = required.difference(imported_names)
    if missing:
        raise ValueError("Missing joints in URDF: " + ", ".join(sorted(missing)))

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

    for link in links:
        if link.name in active_names:
            link.bounds = (math.radians(-180.0), math.radians(180.0))

    arm = ikpy.chain.Chain(
        name="robot_arm",
        links=links,
        active_links_mask=active_mask,
    )
    joint_indices = {link.name: i for i, link in enumerate(arm.links)}
    for name in JOINT_CONFIG:
        if name not in joint_indices:
            raise ValueError(f"Could not locate {name} in IK chain.")

    return arm, joint_indices

arm_chain, joint_indices = build_arm_chain()
current_angles = np.zeros(len(arm_chain.links), dtype=float)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-0.30, 0.30)
ax.set_ylim(-0.30, 0.30)
ax.set_zlim(-0.15, 0.35)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Continuous IK")

arm_line, = ax.plot([], [], [], "-o", linewidth=5, markersize=6)
end_point = ax.scatter([], [], [], c="blue", s=70)
target_point = ax.scatter([], [], [], c="red", s=90)
target_line, = ax.plot([], [], [], "r--", alpha=0.5)
plt.show(block=False)

def draw_arm():
    global current_angles, _last_target
    plt.figure(fig.number)
    frames = arm_chain.forward_kinematics(current_angles, full_kinematics=True)
    points = np.array([frame[:3, 3] for frame in frames])

    arm_line.set_data(points[:, 0], points[:, 1])
    arm_line.set_3d_properties(points[:, 2])

    end = points[-1]
    end_point._offsets3d = ([end[0]], [end[1]], [end[2]])

    if _last_target is not None:
        target_point._offsets3d = ([_last_target[0]], [_last_target[1]], [_last_target[2]])
        target_line.set_data([_last_target[0], _last_target[0]], [_last_target[1], _last_target[1]])
        target_line.set_3d_properties([0.0, _last_target[2]])
    else:
        target_point._offsets3d = ([], [], [])
        target_line.set_data([], [])
        target_line.set_3d_properties([])

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)

draw_arm()

def servo_command(joint_name, joint_radians):
    config = JOINT_CONFIG[joint_name]
    channel = config["channel"]
    offset = config.get("offset", 0.0)
    angle = config["home"] + config["direction"] * math.degrees(joint_radians) + offset
    low, high = SERVO_COMMAND_LIMITS[channel]
    if not low <= angle <= high:
        raise ValueError(f"CH{channel}={angle:.1f} outside range ({low} to {high}).")
    return angle

def get_servo_commands(angles):
    return {
        config["channel"]: servo_command(name, angles[joint_indices[name]])
        for name, config in JOINT_CONFIG.items()
    }

def print_servo_angles(angles):
    commands = get_servo_commands(angles)
    angle_str = ", ".join([f"CH{ch}: {ang:.1f}°" for ch, ang in commands.items()])
    print(f"Servo Angles -> {angle_str}")

def arm():
    global _armed
    if not _armed:
        send_cmd("ARM")
        _armed = True
        time.sleep(0.5)

def disarm():
    global _armed
    _armed = False

def move_to_target(target_xyz, traking=True):
    global current_angles, _last_target, _target_queue

    target_xyz = np.asarray(target_xyz, dtype=float)
    if target_xyz.shape != (3,) or not np.all(np.isfinite(target_xyz)):
        print("Error: Target coordinates are invalid/non-finite.")
        return []

    _last_target = target_xyz
    seed = current_angles.copy()
    current_xyz = arm_chain.forward_kinematics(seed)[:3, 3]
    difference = target_xyz - current_xyz
    distance = float(np.linalg.norm(difference))

    if distance < 0.001:
        return [("Target", tuple(target_xyz))]

    direction = difference / distance
    steps_count = max(1, int(math.ceil(distance / CARTESIAN_STEP_SIZE)))

    temp_queue = []
    reached_any = False

    for step in range(1, steps_count + 1):
        travel = min(step * CARTESIAN_STEP_SIZE, distance)
        waypoint = current_xyz + direction * travel

        try:
            # Prioritize Joint 14 (Base Rotation) first by setting its angle to face waypoint directly
            j14_idx = joint_indices["joint_14"]
            desired_base_angle = math.atan2(waypoint[1], waypoint[0])
            seed[j14_idx] = desired_base_angle

            candidate = arm_chain.inverse_kinematics(
                target_position=waypoint,
                initial_position=seed,
                max_iter=IK_MAX_ITER,
            )
            candidate = np.asarray(candidate, dtype=float)

            if not np.all(np.isfinite(candidate)):
                break

            commands = get_servo_commands(candidate)
            temp_queue.append((candidate, commands))
            seed = candidate.copy()
            reached_any = True

        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            break

    if not reached_any:
        print(f"Error: Arm could not make any progress toward target {tuple(target_xyz)}.")
    else:
        if len(temp_queue) < steps_count:
            print(f"Warning: Arm stopped short, but moved as close as possible to {tuple(target_xyz)}.")
        _target_queue.clear()
        _target_queue.extend(temp_queue)

    return [("Target", tuple(target_xyz))]

def step_interpolation():
    global current_angles, _target_queue
    if not _target_queue:
        return

    candidate, commands = _target_queue.pop(0)
    arm()
    for channel, angle in commands.items():
        send_cmd(f"{channel}:{angle:.2f}")

    current_angles = candidate.copy()
    if not _target_queue:
        print_servo_angles(current_angles)

def run_procedure(steps):
    history = []
    for step in steps:
        if not step:
            continue
        if isinstance(step, (int, float)):
            time.sleep(float(step))
        elif step[0] in ("IK", "Target"):
            if len(step) < 4:
                raise ValueError("IK/Target steps must contain X, Y and Z.")
            history.extend(move_to_target(step[1:4], traking=True))
        elif len(step) == 2:
            channel, angle = step
            low, high = SERVO_COMMAND_LIMITS[channel]
            angle = float(angle)
            arm()
            send_cmd(f"{channel}:{angle:.2f}")
            history.append(("Servo", (channel, angle)))
        elif len(step) == 1:
            try:
                time.sleep(float(step[0]))
            except ValueError:
                pass
    return history

def close():
    disarm()
    sock.close()
    plt.ioff()
    plt.close(fig)
