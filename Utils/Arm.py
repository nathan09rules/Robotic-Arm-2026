

import socket
import asyncio
import math
import numpy as np
import matplotlib.pyplot as plt
import ikpy.chain
from ikpy.link import OriginLink, URDFLink

ESP32_IP = "192.168.4.1"
PORT = 8888

HOME_CH14 = 135.0
HOME_CH13 = 80.0
HOME_CH15 = 120.0

LINK_1 = 0.115
LINK_2 = 0.100

SERVO_MIN = 0.0
SERVO_MAX = 180.0

DIR_CH14 = 1
DIR_CH13 = 1
DIR_CH15 = 1

SERVO_SPEED = 30.0
STEP_TIME = 0.03
STEP_SIZE = SERVO_SPEED * STEP_TIME

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_cmd(payload):
    sock.sendto(payload.encode("utf-8"), (ESP32_IP, PORT))


def servo_angle(home, radians, direction=1):
    return max(
        SERVO_MIN,
        min(SERVO_MAX, home + direction * math.degrees(radians))
    )


arm_chain = ikpy.chain.Chain(
    name="calibrated_arm",
    links=[
        OriginLink(),
        URDFLink(
            name="base_yaw_ch14",
            origin_translation=[0, 0, 0.03],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1],
            bounds=(
                math.radians(SERVO_MIN - HOME_CH14),
                math.radians(SERVO_MAX - HOME_CH14)
            )
        ),
        URDFLink(
            name="base_pitch_ch13",
            origin_translation=[0, 0, 0.05],
            origin_orientation=[0, -math.pi / 2, 0],
            rotation=[0, 1, 0],
            bounds=(
                math.radians(SERVO_MIN - HOME_CH13),
                math.radians(SERVO_MAX - HOME_CH13)
            )
        ),
        URDFLink(
            name="elbow_ch15",
            origin_translation=[LINK_1, 0, 0],
            origin_orientation=[0, math.pi / 2, 0],
            rotation=[0, 1, 0],
            bounds=(
                math.radians(SERVO_MIN - HOME_CH15),
                math.radians(SERVO_MAX - HOME_CH15)
            )
        ),
        URDFLink(
            name="tool",
            origin_translation=[LINK_2, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 0],
            bounds=(0, 0)
        )
    ],
    active_links_mask=[False, True, True, True, False]
)

current_angles = [0.0] * 5

channel_states = {
    14: HOME_CH14,
    13: HOME_CH13,
    15: HOME_CH15
}

send_cmd("ARM")


# Plot
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set(
    xlim=(-0.27, 0.27),
    ylim=(-0.27, 0.27),
    zlim=(0, 0.35),
    xlabel="X (m)",
    ylabel="Y (m)",
    zlabel="Z (m)",
    title="Arm preview: blue = end effector, red = target"
)

arm_line, = ax.plot([], [], [], "-o", linewidth=5, markersize=6)
end_cursor = ax.scatter([], [], [], c="blue", s=70)
target_cursor = ax.scatter([], [], [], c="red", s=90)
target_guide, = ax.plot([], [], [], "r--", alpha=0.5)


def draw_arm(angles, target=None, tracking=True):
    transforms = arm_chain.forward_kinematics(
        angles, full_kinematics=True
    )
    points = np.array([t[:3, 3] for t in transforms])

    arm_line.set_data(points[:, 0], points[:, 1])
    arm_line.set_3d_properties(points[:, 2])

    end = points[-1]
    end_cursor._offsets3d = ([end[0]], [end[1]], [end[2]])

    if target is not None:
        x, y, z = target
        target_cursor._offsets3d = ([x], [y], [z])
        target_cursor.set_color("red" if tracking else "orange")
        target_guide.set_data([x, x], [y, y])
        target_guide.set_3d_properties([0, z])

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


def get_current_position():
    return arm_chain.forward_kinematics(current_angles)[:3, 3]


async def smooth_move(targets):
    starts = dict(channel_states)
    distance = max(
        abs(targets[ch] - starts[ch]) for ch in targets
    )
    steps = max(1, math.ceil(distance / STEP_SIZE))

    for i in range(1, steps + 1):
        alpha = i / steps

        for ch, target in targets.items():
            angle = starts[ch] + (target - starts[ch]) * alpha
            await asyncio.to_thread(
                send_cmd, f"{ch}:{angle:.2f}"
            )
            channel_states[ch] = angle

            if ch == 14:
                current_angles[1] = math.radians(
                    (angle - HOME_CH14) * DIR_CH14
                )
            elif ch == 13:
                current_angles[2] = math.radians(
                    (angle - HOME_CH13) * DIR_CH13
                )
            elif ch == 15:
                current_angles[3] = math.radians(
                    (angle - HOME_CH15) * DIR_CH15
                )

        draw_arm(current_angles, latest_target)
        await asyncio.sleep(STEP_TIME)


latest_target = None
worker_task = None
target_lock = None


async def move_to_target(target_xyz, traking=False):
    global latest_target, worker_task, target_lock

    target = np.asarray(target_xyz, dtype=float)

    if target.shape != (3,) or not np.all(np.isfinite(target)):
        return []

    latest_target = target.copy()

    if not traking:
        draw_arm(current_angles, target, tracking=False)
        return []

    if target[2] < 0.05:
        return []

    if target_lock is None:
        target_lock = asyncio.Lock()

    # Serialize IK and physical movements.
    async with target_lock:
        target = latest_target.copy()

        initial = list(current_angles)
        initial[1] = math.atan2(target[1], target[0])

        angles = await asyncio.to_thread(
            arm_chain.inverse_kinematics,
            target_position=target,
            initial_position=initial,
            max_iter=1000
        )

        if not np.all(np.isfinite(angles)):
            print("IK failed: invalid angles")
            return []

        commands = {
            14: servo_angle(HOME_CH14, angles[1], DIR_CH14),
            13: servo_angle(HOME_CH13, angles[2], DIR_CH13),
            15: servo_angle(HOME_CH15, angles[3], DIR_CH15)
        }

        # Verify the pose represented by the commanded servo angles.
        check_angles = list(angles)
        check_angles[1] = math.radians(
            (commands[14] - HOME_CH14) * DIR_CH14
        )
        check_angles[2] = math.radians(
            (commands[13] - HOME_CH13) * DIR_CH13
        )
        check_angles[3] = math.radians(
            (commands[15] - HOME_CH15) * DIR_CH15
        )

        actual = arm_chain.forward_kinematics(check_angles)[:3, 3]
        error = float(np.linalg.norm(actual - target))

        if error > 0.015:
            print(
                f"Target may be unreachable: error={error:.3f} m, "
                f"target={target}, calculated={actual}"
            )
            draw_arm(current_angles, target)
            return []

        print("Target:", target)
        print("Servo commands:", commands)
        print(f"Predicted endpoint error: {error:.4f} m")

        await smooth_move(commands)

        current_angles[:] = check_angles
        draw_arm(current_angles, target)

        return [
            (14, commands[14]),
            (13, commands[13]),
            (15, commands[15]),
            ("IK", tuple(target))
        ]


async def set_channel(channel, rotation):
    if channel not in channel_states:
        raise ValueError(f"Unsupported channel: {channel}")

    rotation = max(SERVO_MIN, min(SERVO_MAX, float(rotation)))
    await smooth_move({channel: rotation})
    draw_arm(current_angles)
    return [(channel, rotation)]


async def run_procedure(steps):
    history = []

    for step in steps:
        if step[0] == "IK":
            history.extend(
                await move_to_target(step[1:4], traking=True)
            )
        else:
            history.extend(
                await set_channel(step[0], step[1])
            )

    return history

