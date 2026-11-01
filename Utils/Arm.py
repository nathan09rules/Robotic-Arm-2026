#!/usr/bin/env python3
"""Utils/Arm.py - URDF-driven, joint-limit-safe arm control (drop-in for main.py).

Public API (unchanged): current_angles, draw_arm, move_to_target,
step_interpolation, arm, send_cmd, run_procedure.

Three separate stages:
  1. URDF model     UrdfChain      FK straight from Robot.urdf: T = T_origin * Rot(axis, q)
  2. Bounded solver solve_ik       scipy least_squares, hard lower/upper bounds from the URDF
  3. Servo mapping  urdf_to_servo  hardware calibration only

Unreachable targets never raise: the solver's closest in-bounds pose is used.
Robot frame: with all joints at 0 the arm points up (+Z); bending (CH13/15/4)
swings it towards -Y; CH14 yaws it about Z (+-90 deg).
"""
import math
import os
import socket
import time
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

# ------------------------------------------------------------------ settings
ESP32_IP = "192.168.4.1"
PORT = 8888
HERE = os.path.dirname(os.path.abspath(__file__))

URDF_CANDIDATES = [
    os.path.join(HERE, "Robot.urdf"),
    os.path.join(HERE, "..", "Robot.urdf"),
    os.path.join(HERE, "Utils", "Robot.urdf"),
    os.path.join(HERE, "..", "Utils", "Robot.urdf"),
    "/home/nathanpd/Documents/26/Robotic-Arm/IK/Robot.urdf",
]

CARTESIAN_STEP_SIZE = 0.03      # m between interpolated waypoints
MAX_WAYPOINTS = 12              # cap solver work per move_to_target call
IK_TOLERANCE = 0.015            # "exact" if within 1.5 cm (informational only)

STARTUP = {0: 70, 1: 90, 4: 90, 13: 90, 14: 90, 15: 150}
SEND_ORDER = (14, 13, 15, 4)
ACTIVE = ["joint_14", "joint_13", "joint_15", "joint_4"]
SERVO_LIMITS = {14: (0, 180), 13: (90, 150), 15: (90, 150), 4: (0, 90)}


# ------------------------------------------------------------------ stage 1: URDF
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
    def __init__(self, path, root="base_link", tip="end_effector_link"):
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
        while link != root:                       # walk tip -> root, then reverse
            chain.append(joints[link])
            link = joints[link]["parent"]
        self.chain = chain[::-1]
        self.names = [j["name"] for j in self.chain]
        idx = [self.names.index(n) for n in ACTIVE]          # raises if a joint is missing
        self.lower = np.array([self.chain[i]["lo"] for i in idx])
        self.upper = np.array([self.chain[i]["hi"] for i in idx])

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

    def points(self, q):                           # base + every joint origin
        return np.array([np.zeros(3)] + [f[:3, 3] for f in self.frames(q)])


# ------------------------------------------------------------------ stage 3: servo mapping
def urdf_to_servo(q):
    d = [math.degrees(v) for v in q]               # CH13/CH15: 60 servo-deg per 90 URDF-deg
    return {14: 90 + d[0], 13: 90 + d[1] / 1.5, 15: 90 + d[2] / 1.5, 4: 90 - d[3]}


def servo_to_urdf(s):
    return np.radians([s[14] - 90, (s[13] - 90) * 1.5, (s[15] - 90) * 1.5, 90 - s[4]])


def servo_ok(s, eps=1e-6):
    return all(lo - eps <= s[c] <= hi + eps for c, (lo, hi) in SERVO_LIMITS.items())


# ------------------------------------------------------------------ stage 2: bounded solver
class Solution:
    def __init__(self, q, err, exact, in_bounds):
        self.q, self.err, self.exact, self.in_bounds = q, err, exact, in_bounds


def solve_ik(chain, target, q_prev, tol=IK_TOLERANCE, reg=1e-3, max_nfev=60, n_seeds=5):
    """min ||FK(q)-target||^2 + reg*||q-q_prev||^2   subject to  lower <= q <= upper.

    Bounds go straight into least_squares (trf), which keeps every iterate inside
    them: the result is in-bounds by construction, nothing is clamped afterwards.
    Unreachable target -> closest in-bounds configuration (exact=False).
    """
    lo, hi = chain.lower, chain.upper
    target = np.asarray(target, float)
    q_prev = np.clip(q_prev, lo, hi)

    def resid(q):
        return np.concatenate([chain.fk(q) - target, reg * (q - q_prev)])

    # yaw 0 points the arm along -Y, so direction (x, y) needs yaw = atan2(x, -y)
    yaw = math.atan2(target[0], -target[1]) if abs(target[0]) + abs(target[1]) > 1e-9 else 0.0
    seeds = [q_prev, np.array([yaw, 0.5, 1.0, 0.5]), 0.5 * (lo + hi),
             lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo)][:max(1, n_seeds)]
    results = []
    for k, s in enumerate(seeds):
        r = least_squares(resid, np.clip(s, lo, hi), bounds=(lo, hi), method="trf",
                          max_nfev=max_nfev, xtol=1e-10, ftol=1e-10)
        results.append((r.x, float(np.linalg.norm(chain.fk(r.x) - target))))
        if k == 0 and results[0][1] < 1e-3:
            break                                   # continuity seed already on target
    best = min(e for _, e in results)
    q, err = min(((x, e) for x, e in results if e <= best + 5e-4),
                 key=lambda t: np.linalg.norm(t[0] - q_prev))   # prefer least joint motion
    ok = bool(np.all(np.isfinite(q)) and np.all(q >= lo) and np.all(q <= hi))
    return Solution(q, err, exact=(ok and err <= tol), in_bounds=ok)


# ------------------------------------------------------------------ state
def _find_urdf():
    for p in URDF_CANDIDATES:
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError("Could not find Robot.urdf. Tried:\n  " + "\n  ".join(URDF_CANDIDATES))


chain = UrdfChain(_find_urdf())                       # built once
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_armed = False
_last_target = None
_target_queue = []                                    # [(q, servo_commands), ...]

# Joint angles (rad) in ACTIVE order: joint_14, joint_13, joint_15, joint_4.
# Mutated in place so `from Utils.Arm import current_angles` stays valid.
current_angles = servo_to_urdf(STARTUP).astype(float)


def send_cmd(payload):
    try:
        sock.sendto(payload.encode("utf-8"), (ESP32_IP, PORT))
    except OSError as e:                              # network down: never crash
        print("UDP send failed:", e)


def arm():
    """Send ARM and the startup pose once (the model assumes the arm starts there)."""
    global _armed
    if not _armed:
        send_cmd("ARM")
        _armed = True
        time.sleep(0.5)
        for channel, angle in STARTUP.items():
            send_cmd(f"{channel}:{angle}")
        time.sleep(0.5)


# ------------------------------------------------------------------ plot
fig = plt.figure(num="Robotic Arm 3D")
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-0.4, 0.4); ax.set_ylim(-0.4, 0.4); ax.set_zlim(0, 0.45)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
_p0 = chain.points(current_angles)
arm_line, = ax.plot(_p0[:, 0], _p0[:, 1], _p0[:, 2], "-o", color="blue", linewidth=5, markersize=6)
target_point, = ax.plot([], [], [], "o", color="red", markersize=9)
target_line, = ax.plot([], [], [], "--", color="red", alpha=0.5)


def draw_arm():
    """Plot the robot from the same URDF forward kinematics the solver uses."""
    try:
        pts = chain.points(current_angles)
        arm_line.set_data(pts[:, 0], pts[:, 1])
        arm_line.set_3d_properties(pts[:, 2])
        if _last_target is not None:
            t = _last_target
            target_point.set_data([t[0]], [t[1]])
            target_point.set_3d_properties([t[2]])
            target_line.set_data([t[0], t[0]], [t[1], t[1]])
            target_line.set_3d_properties([0.0, t[2]])
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    except Exception as e:                            # plotting must never stop the arm
        print("draw_arm error:", e)


# ------------------------------------------------------------------ motion
def move_to_target(target_xyz, traking=True):
    """Plan a straight Cartesian path to target_xyz as a queue of joint poses.

    Every waypoint uses the bounded solver's result (exact or closest).
    """
    global _last_target
    try:
        target = np.asarray(target_xyz, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            return []
        _last_target = target

        q = current_angles.copy()
        start = chain.fk(q)
        diff = target - start
        distance = float(np.linalg.norm(diff))
        if distance < 0.001:
            return [("Target", tuple(target))]

        n = min(MAX_WAYPOINTS, max(1, int(math.ceil(distance / CARTESIAN_STEP_SIZE))))
        queue = []
        for i in range(1, n + 1):
            last = (i == n)   # intermediate waypoints: warm-start only (fast); final one: multi-start
            sol = solve_ik(chain, start + diff * (i / n), q,
                           n_seeds=4 if last else 1, max_nfev=60 if last else 25)
            cmds = urdf_to_servo(sol.q)
            if not (sol.in_bounds and servo_ok(cmds)):   # cannot happen with bounded solve
                break
            queue.append((sol.q.copy(), cmds))
            q = sol.q
        if queue:
            _target_queue.clear()
            _target_queue.extend(queue)
    except Exception as e:                            # never crash: keep current plan
        print("move_to_target error:", e)
    return [("Target", tuple(np.asarray(target_xyz, dtype=float).tolist()))] \
        if np.ndim(target_xyz) == 1 and len(target_xyz) == 3 else []


def step_interpolation():
    """Send the next queued pose (call once per main-loop tick)."""
    if not _target_queue:
        return
    try:
        q, commands = _target_queue.pop(0)
        arm()
        for channel in SEND_ORDER:
            send_cmd(f"{channel}:{commands[channel]:.2f}")
        current_angles[:] = q
    except Exception as e:
        print("step_interpolation error:", e)


def run_procedure(steps):
    history = []
    for step in steps:
        if not step:
            continue
        if isinstance(step, (int, float)):
            time.sleep(float(step))
        elif isinstance(step, (tuple, list)) and step[0] in ("IK", "Target"):
            if len(step) < 4:
                raise ValueError("IK/Target steps must contain X, Y and Z.")
            history.extend(move_to_target(step[1:4], traking=True))
        elif isinstance(step, (tuple, list)) and len(step) == 2:
            channel, angle = step
            angle = float(angle)
            arm()
            send_cmd(f"{channel}:{angle:.2f}")
            history.append(("Servo", (channel, angle)))
        elif isinstance(step, (tuple, list)) and len(step) == 1:
            try:
                time.sleep(float(step[0]))
            except ValueError:
                pass
    return history
