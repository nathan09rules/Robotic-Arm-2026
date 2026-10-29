import socket
import time
import pygame
import matplotlib.pyplot as plt
import math

ESP32_IP = "192.168.4.1"
PORT = 8888
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_cmd(payload: str):
    sock.sendto(payload.encode('utf-8'), (ESP32_IP, PORT))

Procedures = {
    "claw": [
        [0, 60],
        [-1, 3],
        [0, 100]
    ],
    "mv": [
        [4, 120],
        [0, 60],
        [-1, 2],
        [4, 70],
        [0, 100]
    ],
    "NOT": [
        [4, 120],
        [0, 60],
        [-1, 2],
        [4, 70],
        ["IK", 0.3, 0.4, 0.1],
        [0, 100]    
    ]
}

def run_procedure(name_or_steps):
    steps = Procedures.get(name_or_steps, name_or_steps)
    for step in steps:
        if not step:
            continue
        if step[0] == -1:
            duration = step[1]
            send_cmd(f"-1:{duration:.2f}")
            start_time = time.time()
            while time.time() - start_time < duration:
                plt.pause(0.05)
        else:
            ch, val = step[0], step[1]
            send_cmd(f"{ch}:{val:.2f}")

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller detected.")
    exit()

controller = pygame.joystick.Joystick(0)
controller.init()

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_axis_off()

scatter = ax.scatter([0], [0], [0], c='r', s=100)
running = True

send_cmd("ARM")
time.sleep(1.5)

try:
    while running:
        pygame.event.pump()
        x = controller.get_axis(0)
        y = -controller.get_axis(1)
        z = -controller.get_axis(3)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 2:
                    send_cmd("ARM")
                elif event.button == 1:
                    send_cmd("OFF")
                elif event.button == 3:
                    run_procedure("claw")
                elif event.button == 4:
                    run_procedure("mv")
                print(event.button)

except KeyboardInterrupt:
    send_cmd("OFF")
finally:
    send_cmd("OFF")
    plt.ioff()
    pygame.quit()
