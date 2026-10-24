import socket
import time
import pygame
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

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller detected.")
    exit()

controller = pygame.joystick.Joystick(0)
controller.init()

arm_chain = ikpy.chain.Chain(name='arm', links=[
    OriginLink(),
    URDFLink(
        name="base_yaw",
        origin_translation=[0, 0, 0],
        origin_orientation=[0, -math.pi/2, 0],
        rotation=[0, 0, 1]
    ),
    URDFLink(
        name="base_pitch",
        origin_translation=[0, 0, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0]
    ),
    URDFLink(
        name="shoulder",
        origin_translation=[0.105, 0, 0],
        origin_orientation=[0, 0, math.pi/2],
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

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlim(-0.2, 0.2)
ax.set_ylim(-0.2, 0.2)
ax.set_zlim(0, 0.25)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

running = True

send_cmd("ARM")
time.sleep(1.5)

try:
    while running:
        pygame.event.pump()
        x_raw = controller.get_axis(0)
        y_raw = -controller.get_axis(1)
        z_raw = -controller.get_axis(3)

        target_x = x_raw * 0.15
        target_y = y_raw * 0.15
        target_z = max(0.05, (z_raw * 0.1) + 0.05)

        curr_pos = arm_chain.forward_kinematics(current_angles)[:3, 3]

        ax.cla()
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0, 0.25)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        arm_chain.plot(current_angles, ax)
        ax.scatter([target_x], [target_y], [target_z], c='r', s=100)
        ax.scatter([curr_pos[0]], [curr_pos[1]], [curr_pos[2]], c='b', s=100)
        
        ax.plot([target_x, target_x], [target_y, target_y], [0, target_z], 'r--', alpha=0.5)
        ax.plot([curr_pos[0], curr_pos[0]], [curr_pos[1], curr_pos[1]], [0, curr_pos[2]], 'b--', alpha=0.5)

        plt.draw()
        plt.pause(0.01)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 10:
                    desired_yaw = math.atan2(target_y, target_x)
                    init_pos = list(current_angles)
                    init_pos[1] = desired_yaw

                    target_angles = arm_chain.inverse_kinematics([target_x, target_y, target_z], initial_position=init_pos)
                    
                    fk_test = [arm_chain.forward_kinematics(target_angles, full_kinematics=True)[i][:3, 3] for i in range(len(arm_chain.links))]
                    #if any(pt[2] < 0 for pt in fk_test):
                    #    print("Target leads to arm below table. Ignored.")
                    #    continue

                    steps = 30
                    for step in range(steps + 1):
                        alpha = step / steps
                        interpolated = [c + (t - c) * alpha for c, t in zip(current_angles, target_angles)]
                        curr_pos = arm_chain.forward_kinematics(interpolated)[:3, 3]
                        
                        ax.cla()
                        ax.set_xlim(-0.2, 0.2)
                        ax.set_ylim(-0.2, 0.2)
                        ax.set_zlim(0, 0.25)
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_zlabel('Z')
                        arm_chain.plot(interpolated, ax)
                        ax.scatter([target_x], [target_y], [target_z], c='r', s=100)
                        ax.scatter([curr_pos[0]], [curr_pos[1]], [curr_pos[2]], c='b', s=100)
                        ax.plot([target_x, target_x], [target_y, target_y], [0, target_z], 'r--', alpha=0.5)
                        ax.plot([curr_pos[0], curr_pos[0]], [curr_pos[1], curr_pos[1]], [0, curr_pos[2]], 'b--', alpha=0.5)
                        plt.draw()
                        plt.pause(0.02)
                        
                        send_cmd(f"14:{math.degrees(interpolated[1]):.2f}")
                        #send_cmd(f"13:{math.degrees(interpolated[2]):.2f}")
                        #send_cmd(f"15:{math.degrees(interpolated[3]):.2f}")
                        print(math.degrees(interpolated[1]))
                    current_angles = target_angles

                if event.button == 2:
                    send_cmd("ARM")
                if event.button == 1:
                    send_cmd("OFF")

except KeyboardInterrupt:
    send_cmd("OFF")
finally:
    send_cmd("OFF")
    plt.ioff()
    pygame.quit()
