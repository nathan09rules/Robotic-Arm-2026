import socket
import asyncio
import math
import numpy as np
import matplotlib.pyplot as plt

ESP32_IP = "192.168.4.1"
PORT = 8888

X_LIMIT = (-0.25, 0.25)
Y_LIMIT = (-0.25, 0.25)
Z_LIMIT = (0.0, 0.5)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_cmd(payload):
    sock.sendto(payload.encode("utf-8"), (ESP32_IP, PORT))


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set(
    xlim=X_LIMIT,
    ylim=Y_LIMIT,
    zlim=Z_LIMIT,
    xlabel="X (m)",
    ylabel="Y (m)",
    zlabel="Z (m)",
    title="Target Plotter with Bounds"
)


def draw_bounding_box():
    x0, x1 = X_LIMIT
    y0, y1 = Y_LIMIT
    z0, z1 = Z_LIMIT

    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for edge in edges:
        p1 = corners[edge[0]]
        p2 = corners[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "k--", alpha=0.3)


draw_bounding_box()

target_cursor = ax.scatter([], [], [], c="red", s=90)
target_guide, = ax.plot([], [], [], "r--", alpha=0.5)


def draw_target(target, tracking=True):
    if target is not None:
        x, y, z = target
        target_cursor._offsets3d = ([x], [y], [z])
        target_cursor.set_color("red" if tracking else "orange")
        target_guide.set_data([x, x], [y, y])
        target_guide.set_3d_properties([0, z])

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


async def move_to_target(target_xyz, traking=False):
    target = np.asarray(target_xyz, dtype=float)

    if target.shape != (3,) or not np.all(np.isfinite(target)):
        return []

    if not (X_LIMIT[0] <= target[0] <= X_LIMIT[1] and
            Y_LIMIT[0] <= target[1] <= Y_LIMIT[1] and
            Z_LIMIT[0] <= target[2] <= Z_LIMIT[1]):
        print(f"Target out of bounds: {target}")
        return []

    print("Plotting target within bounds:", target)
    draw_target(target, tracking=traking)
    return [("Target", tuple(target))]


async def run_procedure(steps):
    history = []

    for step in steps:
        if step[0] == "IK" or step[0] == "Target":
            history.extend(
                await move_to_target(step[1:4], traking=True)
            )

    return history
