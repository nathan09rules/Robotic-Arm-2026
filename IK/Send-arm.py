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

# The URDF's reference pose is CH13=80, CH15=90, CH4=90.
# CH14 is added below because it is not present in the supplied URDF.
JOINT_CONFIG = {
    "joint_14": {"channel": 14, "home": 90.0, "direction": 1},
    "joint_13": {"channel": 13, "home": 80.0, "direction": 1},
    "joint_15": {"channel": 15, "home": 90.0, "direction": 1},
    "joint_4":  {"channel": 4,  "home": 90.0, "direction": 1},
}

# Command-side safety limits. CH4 remains capped at 180 degrees.
# CH13 is allowed down to 50 degrees to match the wider IK range below;
# verify this travel is mechanically safe before running the arm.
SERVO_COMMAND_LIMITS = {
    14: (0.0, 200.0),
    13: (20.0, 200.0),
    15: (0.0, 200.0),
    4:  (0.0, 200.0),
}

# The uploaded URDF's CH13 lower limit (-20 degrees relative to its
# reference) came only from photographed poses, not a measured hard stop.
# This override permits q13=-30 degrees, i.e. servo command CH13=50 degrees.
# Do NOT use this extension unless the physical arm has been checked for it.
IK_LIMIT_OVERRIDES = {
    "joint_13": (-60.0, 60.0),
}

MOVE_STEPS = 30
MOVE_DELAY = 0.035

# Cartesian straight-line movement settings.
# A 5 mm waypoint spacing makes the end-effector follow a sequence of
# closely spaced IK targets instead of interpolating joint angles directly.
LINEAR_STEP_SIZE = 0.005
LINEAR_IK_TOLERANCE = 0.008
LINEAR_MAX_STEPS = 120
LINEAR_BUTTON = 3  # pygame button 3 is commonly Y / Triangle

# Asymmetric Z-stick range: stick down reaches -0.10 m; stick up reaches +0.30 m.
TARGET_Z_DOWN_LIMIT = 0.10
TARGET_Z_UP_LIMIT = 0.30
CONTROLLER_DEADZONE = 0.08


def build_arm_chain():
    if not URDF_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find {URDF_PATH.name}. Put Robot.urdf beside this Python file."
        )

    # Import the shoulder -> elbow -> wrist -> end-effector chain from the URDF.
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

    # The supplied URDF starts at base_link (the CH13 pivot) and has no CH14.
    # Add CH14 as a yaw joint before the imported URDF links.
    base_yaw = URDFLink(
        name="joint_14",
        origin_translation=[0.0, 0.0, 0.0],
        origin_orientation=[0.0, 0.0, 0.0],
        rotation=[0.0, 0.0, 1.0],
        bounds=(math.radians(-90.0), math.radians(90.0)),
    )

    links = [OriginLink(), base_yaw, *imported_chain.links[1:]]

    # Apply explicitly documented IK-only limit overrides while retaining
    # the joint origins, axes, link lengths and other limits from Robot.urdf.
    for link in links:
        if link.name in IK_LIMIT_OVERRIDES:
            lower_deg, upper_deg = IK_LIMIT_OVERRIDES[link.name]
            old_lower, old_upper = link.bounds
            link.bounds = (
                math.radians(lower_deg) if lower_deg is not None else old_lower,
                math.radians(upper_deg) if upper_deg is not None else old_upper,
            )

    active_names = set(JOINT_CONFIG)
    active_links_mask = [link.name in active_names for link in links]

    arm = ikpy.chain.Chain(
        name="robot_arm",
        links=links,
        active_links_mask=active_links_mask,
    )

    joint_indices = {link.name: i for i, link in enumerate(arm.links)}
    for joint_name in JOINT_CONFIG:
        if joint_name not in joint_indices:
            raise ValueError(f"Could not locate {joint_name} in the imported IK chain.")

    print("Loaded Robot.urdf")
    print("IK chain:", " -> ".join(link.name for link in arm.links))
    print("Active joints: CH14, CH13, CH15, CH4")
    print("CH13 IK range: -30 to 0 degrees (servo command 50 to 80 degrees)")
    print("CH4 command safety cap: 180 degrees")
    print("WARNING: verify CH13=50 degrees is mechanically safe before moving.")
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
            f"CH{channel} command {angle:.1f} is outside the safety range "
            f"({low:.0f} to {high:.0f})."
        )
    return angle


def get_servo_commands(angles):
    return {
        config["channel"]: servo_command(name, angles[joint_indices[name]])
        for name, config in JOINT_CONFIG.items()
    }


pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller detected.")
    sock.close()
    pygame.quit()
    raise SystemExit

controller = pygame.joystick.Joystick(0)
controller.init()
print("Controls: button 10 = point move, button 3 = straight-line move,")
print("          button 2 = ARM/reset model, button 1 = OFF")

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-0.30, 0.30)
ax.set_ylim(-0.30, 0.30)
ax.set_zlim(-0.15, 0.35)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("URDF arm preview — blue = arm, red = target")

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


def move_to(target_xyz):
    global current_angles
    print(target_xyz)
    target_xyz = np.asarray(target_xyz, dtype=float)
    initial = current_angles.copy()

    # Give IK a useful starting yaw, limited to CH14's modeled +/-90 degree range.
    yaw_index = joint_indices["joint_14"]
    yaw_bounds = arm_chain.links[yaw_index].bounds
    desired_yaw = math.atan2(target_xyz[1], target_xyz[0])
    initial[yaw_index] = np.clip(desired_yaw, yaw_bounds[0], yaw_bounds[1])

    target_angles = arm_chain.inverse_kinematics(
        target_position=target_xyz,
        initial_position=initial,
        max_iter=1000,
    )

    target_angles = np.asarray(target_angles, dtype=float)
    reached = arm_chain.forward_kinematics(target_angles)[:3, 3]
    error = float(np.linalg.norm(reached - target_xyz))


    # Check all four physical servo commands before moving anything.
    try:
        get_servo_commands(target_angles)
    except ValueError as exc:
        print(f"Move rejected: {exc}")
        return

    start_angles = current_angles.copy()

    for step in range(1, MOVE_STEPS + 1):
        alpha = step / MOVE_STEPS
        interpolated = start_angles + (target_angles - start_angles) * alpha

        try:
            commands = get_servo_commands(interpolated)
        except ValueError as exc:
            print(f"Move stopped: {exc}")
            return

        for channel, angle in commands.items():
            send_cmd(f"{channel}:{angle:.2f}")

        draw_arm(interpolated, target_xyz)
        time.sleep(MOVE_DELAY)

    current_angles = target_angles.copy()


def move_linear_to(target_xyz):
    """Move the end-effector along a straight Cartesian line to target_xyz."""
    global current_angles

    target_xyz = np.asarray(target_xyz, dtype=float)
    if target_xyz.shape != (3,) or not np.all(np.isfinite(target_xyz)):
        print("Linear move rejected: target must be a finite [x, y, z] position.")
        return

    # The modeled current pose is the start of the insertion path.
    start_xyz = arm_chain.forward_kinematics(current_angles)[:3, 3].copy()
    distance = float(np.linalg.norm(target_xyz - start_xyz))

    if distance < 0.001:
        print("Linear move skipped: target is already at the current position.")
        return

    steps = max(1, int(math.ceil(distance / LINEAR_STEP_SIZE)))
    if steps > LINEAR_MAX_STEPS:
        print(
            f"Linear move rejected: {steps} waypoints required; "
            f"limit is {LINEAR_MAX_STEPS}."
        )
        return

    print(
        f"Linear move: {distance:.3f} m, {steps} waypoints, "
        f"from {np.round(start_xyz, 3)} to {np.round(target_xyz, 3)}"
    )

    # Plan and validate the WHOLE path first. This prevents the arm from
    # starting an insertion and then discovering that a later point is unreachable.
    planned_path = []
    seed_angles = current_angles.copy()
    yaw_index = joint_indices["joint_14"]
    yaw_bounds = arm_chain.links[yaw_index].bounds

    for step in range(1, steps + 1):
        alpha = step / steps
        waypoint = start_xyz + (target_xyz - start_xyz) * alpha

        # Use the previous IK result as the next initial guess so the solver
        # stays on the same solution branch as it follows the line.
        if math.hypot(waypoint[0], waypoint[1]) > 1e-6:
            desired_yaw = math.atan2(waypoint[1], waypoint[0])
            seed_angles[yaw_index] = np.clip(
                desired_yaw, yaw_bounds[0], yaw_bounds[1]
            )

        try:
            waypoint_angles = arm_chain.inverse_kinematics(
                target_position=waypoint,
                initial_position=seed_angles,
                max_iter=1000,
            )
            waypoint_angles = np.asarray(waypoint_angles, dtype=float)

            reached = arm_chain.forward_kinematics(waypoint_angles)[:3, 3]
            error = float(np.linalg.norm(reached - waypoint))
            if not np.all(np.isfinite(waypoint_angles)) or error > LINEAR_IK_TOLERANCE:
                joint_state_parts = []
                for name in ("joint_13", "joint_15", "joint_4"):
                    config = JOINT_CONFIG[name]
                    command_angle = (
                        config["home"]
                        + config["direction"]
                        * math.degrees(waypoint_angles[joint_indices[name]])
                    )
                    joint_state_parts.append(
                        f"CH{config['channel']}={command_angle:.1f} deg"
                    )
                joint_state = ", ".join(joint_state_parts)
                print(
                    f"Linear move rejected at waypoint {step}/{steps}: "
                    f"IK error {error * 1000:.1f} mm (allowed "
                    f"{LINEAR_IK_TOLERANCE * 1000:.1f} mm). "
                    f"Closest IK commands: {joint_state}. No movement was sent."
                )
                return

            commands = get_servo_commands(waypoint_angles)
        except (ValueError, RuntimeError) as exc:
            print(
                f"Linear move rejected at waypoint {step}/{steps}: {exc}. "
                "No movement was sent."
            )
            return

        planned_path.append((waypoint_angles.copy(), commands, waypoint.copy()))
        seed_angles = waypoint_angles.copy()

    # All Cartesian waypoints and servo angles are valid; execute the path.
    for waypoint_angles, commands, waypoint in planned_path:
        for channel, angle in commands.items():
            send_cmd(f"{channel}:{angle:.2f}")

        draw_arm(waypoint_angles, target_xyz)
        time.sleep(MOVE_DELAY)

    current_angles = planned_path[-1][0].copy()
    print("Straight-line move complete.")


send_cmd("ARM")
time.sleep(1.5)

running = True
try:
    while running:
        pygame.event.pump()

        x_raw = controller.get_axis(0) * 1.0
        y_raw = -controller.get_axis(1) * 1.0
        z_axis = controller.get_axis(3)

        target_x = x_raw * 0.15
        target_y = y_raw * 0.15
        # Stick up (negative pygame axis) = +0.30 m; stick down = -0.10 m.
        if z_axis >= 0.0:
            target_z = -z_axis * TARGET_Z_DOWN_LIMIT
        else:
            target_z = -z_axis * TARGET_Z_UP_LIMIT
        target = np.array([target_x, target_y, target_z])

        draw_arm(current_angles, target)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 10:
                    move_to(target)

                elif event.button == LINEAR_BUTTON:
                    move_linear_to(target)

                elif event.button == 2:
                    send_cmd("ARM")
                    current_angles = np.zeros(len(arm_chain.links), dtype=float)

                elif event.button == 1:
                    send_cmd("OFF")

        time.sleep(0.01)

except KeyboardInterrupt:
    pass
finally:
    send_cmd("OFF")
    sock.close()
    plt.ioff()
    plt.close(fig)
    pygame.quit()
