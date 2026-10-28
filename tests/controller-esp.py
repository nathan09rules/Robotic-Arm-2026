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
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

scatter = ax.scatter([0], [0], [0], c='r', s=100)
running = True

print("Waking up Wi-Fi network bridge...")
send_cmd("ARM")
time.sleep(1.5)

try:
    while running:
        pygame.event.pump()
        x = controller.get_axis(0)
        y = -controller.get_axis(1)
        z = -controller.get_axis(3)

        scatter._offsets3d = ([x], [y], [z])
        plt.draw()
        plt.pause(0.01)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 10:
                    bearing_rad = math.atan2(x, y)
                    bearing_deg = (math.degrees(bearing_rad) + 360) % 360
                    print(f"X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f} | North Bearing: {bearing_deg:.2f}°")
                    send_cmd(f"14:{bearing_deg:.2f}")

                if event.button == 2:
                    print("ARM")
                    send_cmd("ARM")
                if event.button == 1:
                    print("OFF")
                    send_cmd("OFF")

                print(event.button)

except KeyboardInterrupt:
    send_cmd("OFF")
finally:
    send_cmd("OFF")
    plt.ioff()
    pygame.quit()
