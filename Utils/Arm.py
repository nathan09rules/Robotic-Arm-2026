import socket
import time
import matplotlib.pyplot as plt
import math
import numpy as np
import ikpy.chain
from ikpy.link import OriginLink, URDFLink

ESP32_IP = "192.168.4.1"
PORT = 8888
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_cmd(payload: str):
    sock.sendto(payload.encode('utf-8'), (ESP32_IP, PORT))

arm_chain = ikpy.chain.Chain(name='arm', links=[
    OriginLink(),
    URDFLink(
        name="base_yaw",
        origin_translation=[0, 0, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 1],
        bounds=(-math.pi, math.pi)
    ),
    URDFLink(
        name="base_pitch",
        origin_translation=[0, 0, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(-math.pi/2, math.pi/2)
    ),
    URDFLink(
        name="shoulder",
        origin_translation=[0.105, 0, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(0.0, math.pi * 0.8)
    ),
    URDFLink(
        name="end_effector",
        origin_translation=[0.11, 0, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 0]
    )
])
current_angles = [0, 0, 0, 0, 0]
channel_states = {}

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlim(-0.2, 0.2)
ax.set_ylim(-0.2, 0.2)
ax.set_zlim(0, 0.25)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

send_cmd("ARM")
time.sleep(1.5)

def get_current_position():
    return arm_chain.forward_kinematics(current_angles)[:3, 3]

def set_channel(channel, rotation):
    channel_states[channel] = rotation
    send_cmd(f"{channel}:{rotation:.2f}")
    return [(channel, rotation)]

def move_to_target(target_xyz):
    global current_angles
    target_x, target_y, target_z = target_xyz

    if target_z < 0.05:
        return []

    desired_yaw = math.atan2(target_y, target_x)
    init_pos = list(current_angles)
    init_pos[1] = desired_yaw

    target_angles = ikpy.chain.Chain.inverse_kinematics(arm_chain, [target_x, target_y, target_z], initial_position=init_pos)

    deg1 = 90 + math.degrees(target_angles[1])
    deg2 = math.degrees(target_angles[3])

    channel_states[14] = deg1
    channel_states[15] = deg2

    ax.cla()
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(0, 0.25)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    arm_chain.plot(target_angles, ax)
    curr_pos = arm_chain.forward_kinematics(target_angles)[:3, 3]
    ax.scatter([target_x], [target_y], [target_z], c='r', s=100)
    ax.scatter([curr_pos[0]], [curr_pos[1]], [curr_pos[2]], c='b', s=100)
    plt.draw()
    plt.pause(0.02)

    send_cmd(f"14:{deg1:.2f}")
    #send_cmd(f"15:{deg2:.2f}")

    current_angles = target_angles
    return [(14, deg1), (15, deg2), ("IK", (target_x, target_y, target_z))]

def run_procedure(steps):
    history = []
    for step in steps:
        if step[0] == "IK":
            history.extend(move_to_target((step[1], step[2], step[3])))
        else:
            history.extend(set_channel(step[0], step[1]))
    return history
