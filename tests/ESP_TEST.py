
import math
import socket
import time

import numpy as np
import ikpy.chain
from ikpy.link import OriginLink, URDFLink

# ---------------- Connection ----------------
ESP32_IP = "192.168.4.1"
PORT = 8888
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_cmd(payload):
    sock.sendto(payload.encode("utf-8"), (ESP32_IP, PORT))

# ---------------- Servo calibration ----------------
HOME_CH14 = 135.0
HOME_CH13 = 80.0
HOME_CH15 = 120.0

SERVO_MIN = 0.0
SERVO_MAX = 180.0

LINK_1 = 0.115
LINK_2 = 0.100

# ---------------- IK chain ----------------
arm_chain = ikpy.chain.Chain(
    name="calibrated_arm",
    links=[
        OriginLink(),
        URDFLink(
            name="base_yaw_ch14",
            origin_translation=[0, 0, 0.03],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1],
            bounds=(math.radians(-HOME_CH14), math.radians(180 - HOME_CH14)),
        ),
        URDFLink(
            name="base_pitch_ch13",
            origin_translation=[0, 0, 0.05],
            origin_orientation=[0, -math.pi / 2, 0],
            rotation=[0, 1, 0],
            bounds=(math.radians(-HOME_CH13), math.radians(180 - HOME_CH13)),
        ),
        URDFLink(
            name="elbow_ch15",
            origin_translation=[LINK_1, 0, 0],
            origin_orientation=[0, math.pi / 2, 0],
            rotation=[0, 1, 0],
            bounds=(math.radians(-HOME_CH15), math.radians(180 - HOME_CH15)),
        ),
        URDFLink(
            name="tool",
            origin_translation=[LINK_2, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 0],
            bounds=(0, 0),
        ),
    ],
    active_links_mask=[False, True, True, True, False],
)

# ---------------- Procedure syntax ----------------
# [channel, angle]       -> move one physical servo to an absolute angle
# [x, y, z]              -> move end effector using inverse kinematics (metres)
# [-1, seconds]          -> pause for seconds
# [-1]                   -> pause for 1 second
# [-2]                   -> ARM
# [-3]                   -> OFF
#
# Use explicit helpers if a 3-number step could be ambiguous:
# {"type": "ik", "xyz": [x, y, z]}
# {"type": "servo", "channel": 14, "angle": 135}
# {"type": "wait", "seconds": 1}
#
# A list with exactly 3 numbers is interpreted as IK [x, y, z].
# Manual servo steps should always have exactly 2 numbers.

Procedures = {
    "claw": [
        [0, 40],       # change channel 0 to 40 degrees
        [-1, 1],        # wait 1 second
        [0, 100],
    ],
    "Insertion": [
        [0.07199707, -0.09552612, 0.2],
        [-1, 5],
        [0.07199707, -0.09552612, 0.1496875],
        [-1, 5],
        [0, 70],
        [0.10870239, 0.04941284, 0.1496875],
    ],
}

# Track the current joint displacement used as the IK initial position.
current_angles = [0.0, 0.0, 0.0, 0.0, 0.0]
current_servo_angles = {
    14: HOME_CH14,
    13: HOME_CH13,
    15: HOME_CH15,
}

# How smoothly each movement is sent. ESP32 should also slew-limit.
MOVE_STEPS = 30
STEP_DELAY = 0.035


def clamp_servo(angle):
    return max(SERVO_MIN, min(SERVO_MAX, float(angle)))


def servo_to_joint(channel, servo_angle):
    """Convert an absolute servo angle to the IK joint displacement."""
    servo_angle = clamp_servo(servo_angle)
    if channel == 14:
        return 1, math.radians(servo_angle - HOME_CH14)
    if channel == 13:
        return 2, math.radians(servo_angle - HOME_CH13)
    if channel == 15:
        return 3, math.radians(servo_angle - HOME_CH15)
    raise ValueError(f"Channel {channel} is not an IK joint (valid: 13, 14, 15).")


def send_servo(channel, angle):
    """Move an absolute servo channel angle; channels 13/14/15 update IK state."""
    angle = clamp_servo(angle)
    send_cmd(f"{int(channel)}:{angle:.2f}")
    current_servo_angles[int(channel)] = angle

    if int(channel) in (13, 14, 15):
        index, displacement = servo_to_joint(int(channel), angle)
        current_angles[index] = displacement


def move_ik(x, y, z):
    """Solve IK and move the three calibrated joints to an XYZ target."""
    global current_angles

    target = np.array([float(x), float(y), float(z)], dtype=float)
    if not np.all(np.isfinite(target)):
        raise ValueError("IK target must contain finite numbers.")
    if target[2] < 0:
        raise ValueError("IK target Z must be >= 0.")

    initial = list(current_angles)
    initial[1] = math.atan2(target[1], target[0])

    solution = arm_chain.inverse_kinematics(
        target_position=target,
        initial_position=initial,
        max_iter=1000,
    )

    # Confirm IK actually reached the target closely enough.
    achieved = arm_chain.forward_kinematics(solution)[:3, 3]
    error = float(np.linalg.norm(achieved - target))
    if error > 0.025:
        print(f"Warning: target may be unreachable; position error = {error:.3f} m")

    points = [
        arm_chain.forward_kinematics(solution, full_kinematics=True)[i][:3, 3]
        for i in range(1, len(arm_chain.links))
    ]
    if any(point[2] < 0 for point in points):
        raise ValueError("IK solution rejected: a joint/link origin is below Z=0.")

    start = current_angles.copy()
    steps = max(1, MOVE_STEPS)

    for step in range(1, steps + 1):
        alpha = step / steps
        interpolated = [
            a + (b - a) * alpha for a, b in zip(start, solution)
        ]

        ch14 = clamp_servo(HOME_CH14 + math.degrees(interpolated[1]))
        ch13 = clamp_servo(HOME_CH13 + math.degrees(interpolated[2]))
        ch15 = clamp_servo(HOME_CH15 + math.degrees(interpolated[3]))

        send_cmd(f"14:{ch14:.2f}")
        send_cmd(f"13:{ch13:.2f}")
        send_cmd(f"15:{ch15:.2f}")
        current_servo_angles.update({14: ch14, 13: ch13, 15: ch15})
        time.sleep(STEP_DELAY)

    current_angles = list(solution)
    print(
        f"IK -> ({x:.3f}, {y:.3f}, {z:.3f}) m; "
        f"error={error:.3f} m; "
        f"CH14={current_servo_angles[14]:.1f}, "
        f"CH13={current_servo_angles[13]:.1f}, "
        f"CH15={current_servo_angles[15]:.1f}"
    )


def execute_step(step):
    """Interpret a single compact procedure step."""
    if isinstance(step, dict):
        kind = step.get("type")
        if kind == "ik":
            move_ik(*step["xyz"])
        elif kind == "servo":
            send_servo(step["channel"], step["angle"])
        elif kind == "wait":
            time.sleep(max(0.0, float(step["seconds"])))
        elif kind == "arm":
            send_cmd("ARM")
        elif kind == "off":
            send_cmd("OFF")
        else:
            raise ValueError(f"Unknown procedure step type: {kind!r}")
        return

    if not isinstance(step, (list, tuple)):
        raise TypeError(f"Invalid procedure step: {step!r}")

    values = list(step)

    if len(values) == 1 and values[0] == -1:
        time.sleep(1.0)
    elif len(values) == 2 and values[0] == -1:
        time.sleep(max(0.0, float(values[1])))
    elif len(values) == 1 and values[0] == -2:
        send_cmd("ARM")
    elif len(values) == 1 and values[0] == -3:
        send_cmd("OFF")
    elif len(values) == 2:
        channel, angle = values
        send_servo(int(channel), angle)
    elif len(values) == 3:
        move_ik(*values)
    else:
        raise ValueError(f"Unrecognized procedure step: {step!r}")


def run_procedure(name, repeat=1):
    if name not in Procedures:
        raise KeyError(f"Unknown procedure {name!r}. Available: {list(Procedures)}")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")

    print(f"Running procedure: {name}")
    for _ in range(repeat):
        for index, step in enumerate(Procedures[name], start=1):
            print(f"  Step {index}/{len(Procedures[name])}: {step}")
            execute_step(step)
    print(f"Procedure complete: {name}")


if __name__ == "__main__":
    try:
        send_cmd("ARM")
        time.sleep(1.0)

        # Choose the procedure to run here:
        run_procedure("Insertion")

        # Leave the arm powered/holding position after completion.
        # Uncomment if you want to disable it automatically:
        send_cmd("OFF")
    except KeyboardInterrupt:
        send_cmd("OFF")
    except Exception as exc:
        print(f"Procedure failed: {exc}")
        send_cmd("OFF")
        raise
    finally:
        sock.close()

