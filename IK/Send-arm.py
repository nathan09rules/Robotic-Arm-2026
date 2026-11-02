#!/usr/bin/env python3
import math
import os
import socket
import sys
import threading
import time
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')  # Prevents GUI event loop / epoll terminal conflicts
import matplotlib.pyplot as plt
import numpy as np

ESP32_IP = "192.168.4.1"
PORT = 8888
HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "Robot.urdf")

STARTUP = {0: 70, 1: 90, 4: 90, 13: 90, 14: 90, 15: 150}
ACTIVE = ["joint_14", "joint_13", "joint_15", "joint_4"]
SERVO_LIMITS = {14: (0, 180), 13: (60, 120), 15: (90, 150), 4: (0, 90)}

Procedure = {
    "Grab" : [
        {4: 0.0, 13: 90.0, 14: 41.66, 15: 96.06},
        {0: 60},
        {4: 0.0, 13: 94.9, 14: 70.6, 15: 150.0},
        {0: 90},
        {4: 0.0, 13: 90.0, 14: 41.66, 15: 96.06},
        {4: 14.34, 13: 97.0, 14: 132.62, 15: 91.29},
        {4: 2.83, 13: 103.1, 14: 132.62, 15: 131.39},
        {0: 60},
    ],
    "P": [
        {4: 0.0, 13: 90.0, 14: 54.38, 15: 136.87},
        {4: 6.64, 13: 91.86, 14: 46.98, 15: 128.58},
        {4: 4.94, 13: 90.02, 14: 19.49, 15: 116.32},
    ],
    "MV" : [
        {4: 0.0, 13: 90.0, 14: 30.55, 15: 143.96},
        {4: 0.0, 13: 90.0, 14: 76.56, 15: 150.0},
        {4: 0.0, 13: 70.0, 14: 76.56, 15: 115.82},
        {0: 90},
        {0: 60}
    ],
    "Grip" : [
        {0: 60},
        {0: 90},
        {0: 60},
        {0: 90},
    ],
    "Moving" : [
        {4: 2.26, 13: 72.19, 14: 45.49, 15: 134.56, 0:60},
        {4: 7.96, 13: 98.59, 14: 45.49, 15: 147.51},
        {0: 90},
        {4: 0.0, 13: 99.05, 14: 100.04, 15: 150.0},
        {4: 12.43, 13: 100.16, 14: 140.75, 15: 145.7},
        {4: 13.68, 13: 76.01, 14: 140.75, 15: 124.41},
        {4: 15.73, 13: 73.66, 14: 40.39, 15: 127.21},
        {4: 36.71, 13: 103.04, 14: 21.7, 15: 127.88},
    ]
}

def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p),
                            math.cos(y), math.sin(y))
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp,   cp*sr,            cp*cr]])

def _rot(axis, q):
    x, y, z = axis
    c, s, C = math.cos(q), math.sin(q), 1 - math.cos(q)
    return np.array([[c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
                     [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
                     [z*x*C - y*s, z*y*C + x*s, c + z*z*C]])

class UrdfChain:
    def __init__(self, path=URDF, root="base_link", tip="end_effector_link"):
        joints = {}
        for j in ET.parse(path).getroot().findall("joint"):
            o, ax, lim = j.find("origin"), j.find("axis"), j.find("limit")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [0, 0, 1]
            T = np.eye(4)
            T[:3, :3] = _rpy(*rpy)
            T[:3, 3] = xyz
            joints[j.find("child").get("link")] = dict(
                name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                T=T, axis=np.array(axis) / np.linalg.norm(axis),
                lo=float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
                hi=float(lim.get("upper")) if lim is not None and lim.get("upper") else None)
        chain, link = [], tip
        while link != root:
            chain.append(joints[link])
            link = joints[link]["parent"]
        self.chain = chain[::-1]
        self.names = [j["name"] for j in self.chain]
        idx = [self.names.index(n) for n in ACTIVE]
        self.lower = np.array([self.chain[i]["lo"] for i in idx])
        self.upper = np.array([self.chain[i]["hi"] for i in idx])
        self.fixed = [j["name"] for j in self.chain if j["type"] == "fixed"]

    def frames(self, q):
        qd = dict(zip(ACTIVE, q))
        T, out = np.eye(4), []
        for j in self.chain:
            T = T @ j["T"]
            if j["type"] == "revolute":
                R = np.eye(4)
                R[:3, :3] = _rot(j["axis"], qd[j["name"]])
                T = T @ R
            out.append(T.copy())
        return out

    def fk(self, q):
        return self.frames(q)[-1][:3, 3]

def servo_to_urdf(s):
    return np.radians([s[14] - 90, (s[13] - 90) * 1.5, (s[15] - 90) * 1.5, 90 - s[4]])

class ArmState:
    def __init__(self, chain):
        self.chain = chain
        self.reset()

    def reset(self):
        self.q = servo_to_urdf(STARTUP)
        self.target = self.chain.fk(self.q).copy()

current_servos = STARTUP.copy()

def run_procedure(Prod, chain, state, send_cmd):
    global current_servos
    if Prod in Procedure:
        print(f"Running procedure: {Prod}")
        for pose in Procedure[Prod]:
            try:
                current_servos.update(pose)
                for ch, val in pose.items():
                    send_cmd(f"{ch}:{val:.2f}")
                time.sleep(3)
            except Exception as e:
                print(f"Error in step: {e}")
        print("Procedure completed.")

def main(urdf_path):
    chain = UrdfChain(urdf_path)
    state = ArmState(chain)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_cmd(msg):
        sock.sendto(msg.encode(), (ESP32_IP, PORT))

    def send_startup():
        send_cmd("ARM")
        time.sleep(0.3)
        for ch, a in STARTUP.items():
            send_cmd(f"{ch}:{a}")

    send_startup()
    print("Commands:\n  A: ARM\n  F: OFF\n  Grab, P, MV, Grip, Moving: Procedures\n  exit: Quit")

    try:
        while True:
            cmd = input(">>> ").strip()
            if cmd == "A":
                send_cmd("ARM")
                print("Arm ON")
            elif cmd == "F":
                send_cmd("OFF")
                print("Arm OFF")
            elif cmd in Procedure:
                run_procedure(cmd, chain, state, send_cmd)
            elif cmd == "exit":
                break
            else:
                print("Unknown command")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        send_cmd("OFF")
        sock.close()

if __name__ == "__main__":
    path = sys.argv[sys.argv.index("--urdf") + 1] if "--urdf" in sys.argv else URDF
    main(path)
