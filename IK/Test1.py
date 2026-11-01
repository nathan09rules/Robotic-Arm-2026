import math
import os
import socket
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares

ESP32_IP = "192.168.4.1"
PORT = 8888
HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "Robot.urdf")

MOVE_SPEED = 0.08
DEADZONE = 0.1
IK_TOLERANCE = 0.015
PLOT_EVERY = 4

STARTUP = {0: 70, 1: 90, 4: 90, 13: 90, 14: 90, 15: 150}
SEND_ORDER = (14, 13, 15, 4)
ACTIVE = ["joint_14", "joint_13", "joint_15", "joint_4"]
SERVO_LIMITS = {14: (0, 180), 13: (90, 150), 15: (90, 150), 4: (0, 90)}


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

    def points(self, q):
        return np.array([np.zeros(3)] + [f[:3, 3] for f in self.frames(q)])


def urdf_to_servo(q):
    d = [math.degrees(v) for v in q]
    return {14: 90 + d[0], 13: 90 + d[1] / 1.5, 15: 90 + d[2] / 1.5, 4: 90 - d[3]}


def servo_to_urdf(s):
    return np.radians([s[14] - 90, (s[13] - 90) * 1.5, (s[15] - 90) * 1.5, 90 - s[4]])


def servo_ok(s, eps=1e-6):
    return all(lo - eps <= s[c] <= hi + eps for c, (lo, hi) in SERVO_LIMITS.items())


class Solution:
    def __init__(self, q, err, exact, in_bounds):
        self.q, self.err, self.exact, self.in_bounds = q, err, exact, in_bounds


def solve_ik(chain, target, q_prev, tol=IK_TOLERANCE, reg=1e-3, max_nfev=60):
    lo, hi = chain.lower, chain.upper
    target = np.asarray(target, float)
    q_prev = np.clip(q_prev, lo, hi)

    def resid(q):
        return np.concatenate([chain.fk(q) - target, reg * (q - q_prev)])

    yaw = math.atan2(-target[0], target[1]) if abs(target[0]) + abs(target[1]) > 1e-9 else 0.0
    seeds = [q_prev, np.array([yaw, 0.5, 1.0, 0.5]), 0.5 * (lo + hi),
             lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo)]
    results = []
    for k, s in enumerate(seeds):
        r = least_squares(resid, np.clip(s, lo, hi), bounds=(lo, hi), method="trf",
                          max_nfev=max_nfev, xtol=1e-10, ftol=1e-10)
        results.append((r.x, float(np.linalg.norm(chain.fk(r.x) - target))))
        if k == 0 and results[0][1] < tol:
            break
    best = min(e for _, e in results)
    q, err = min(((x, e) for x, e in results if e <= best + 5e-4),
                   key=lambda t: np.linalg.norm(t[0] - q_prev))
    ok = bool(np.all(np.isfinite(q)) and np.all(q >= lo) and np.all(q <= hi))
    return Solution(q, err, exact=(ok and err <= tol), in_bounds=ok)


class ArmState:
    def __init__(self, chain):
        self.chain = chain
        self.reset()

    def reset(self):
        self.q = servo_to_urdf(STARTUP)
        self.target = self.chain.fk(self.q).copy()

    def move(self, movement, dt):
        try:
            new_target = self.target + np.asarray(movement) * MOVE_SPEED * dt
            sol = solve_ik(self.chain, new_target, self.q)
            cmds = urdf_to_servo(sol.q)
            if not (sol.in_bounds and servo_ok(cmds)):
                return None, "solver result invalid, pose kept"
            self.q = sol.q
            self.target = new_target if sol.exact else self.chain.fk(self.q).copy()
            return cmds, "ok" if sol.exact else f"closest (err {sol.err*100:.1f} cm)"
        except Exception as e:
            return None, f"solver error ({e}), pose kept"


def main(urdf_path):
    import pygame
    import matplotlib.pyplot as plt

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

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise SystemExit("No controller connected.")
    pad = pygame.joystick.Joystick(0)
    pad.init()
    print("Controller:", pad.get_name())

    def axis(v):
        return 0 if abs(v) < DEADZONE else v

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-0.4, 0.4); ax.set_ylim(-0.4, 0.4); ax.set_zlim(0, 0.45)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Robotic Arm IK")
    p = chain.points(state.q)
    robot, = ax.plot(p[:, 0], p[:, 1], p[:, 2], "-o", color="blue", linewidth=4)
    tgt, = ax.plot([state.target[0]], [state.target[1]], [state.target[2]], "o", color="red", markersize=9)
    plt.show(block=False)

    def redraw():
        p = chain.points(state.q)
        robot.set_data_3d(p[:, 0], p[:, 1], p[:, 2])
        tgt.set_data_3d([state.target[0]], [state.target[1]], [state.target[2]])
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    send_startup()
    print("Arm started.\nButton 10: Move target\nButton 2: Save pose (servos 4, 13, 14, 15)\nButton 3: Print procedure\nButton 1: Turn off")

    recorded_procedure = []
    btn2_prev = False
    btn3_prev = False

    prev, tick, last = time.monotonic(), 0, ""
    try:
        while plt.fignum_exists(fig.number):
            try:
                pygame.event.pump()
                now = time.monotonic()
                dt = min(now - prev, 0.05)
                prev = now

                b2 = pad.get_button(2)
                if b2 and not btn2_prev:
                    current_servos = urdf_to_servo(state.q)
                    pose = {ch: round(current_servos[ch], 2) for ch in [4, 13, 14, 15]}
                    recorded_procedure.append(pose)
                    print(f"Pose saved: {pose}")
                btn2_prev = b2

                b3 = pad.get_button(3)
                if b3 and not btn3_prev:
                    print("\n--- Recorded Procedure ---")
                    for i, pose in enumerate(recorded_procedure):
                        print({pose})
                    print("--------------------------\n")
                btn3_prev = b3

                if pad.get_button(1):
                    send_cmd("OFF"); print("Arm disabled."); time.sleep(0.3)
                if pad.get_button(10):
                    mv = np.array([axis(pad.get_axis(0)), axis(-pad.get_axis(1)), axis(-pad.get_axis(3))])
                    if mv.any():
                        cmds, status = state.move(mv, dt)
                        if cmds:
                            for ch in SEND_ORDER:
                                send_cmd(f"{ch}:{cmds[ch]:.2f}")
                        if status != last:
                            print(status); last = status

                tick += 1
                if tick % PLOT_EVERY == 0:
                    redraw()
                plt.pause(0.001)
                time.sleep(0.002)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if str(e) != last:
                    print("loop error (continuing):", e); last = str(e)
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("Stopping controller.")
    finally:
        send_cmd("OFF")
        sock.close()
        pygame.quit()
        plt.close("all")


def selftest(urdf_path):
    ch = UrdfChain(urdf_path)
    bad = 0
    print("A. chain:", ch.names, "| fixed:", ch.fixed)
    print("   limits deg:", np.degrees(ch.lower).round(2), np.degrees(ch.upper).round(2))

    print("B. joint mapping")
    for c, u, exp in [(14, 0, 90), (14, 90, 180), (13, 0, 90), (13, 90, 150),
                      (15, 0, 90), (15, 90, 150), (4, 0, 90), (4, 90, 0)]:
        q = np.zeros(4); q[ACTIVE.index(f"joint_{c}")] = math.radians(u)
        got = urdf_to_servo(q)[c]
        good = abs(got - exp) < 1e-9; bad += not good
        print(f"   CH{c:<2} {u:>3} -> {got:6.1f} (exp {exp}) {'OK' if good else 'FAIL'}")

    print("C. FK")
    for label, s in [("startup", STARTUP), ("straight", {14: 90, 13: 90, 15: 90, 4: 90})]:
        print(f"   {label}: tip =", ch.fk(servo_to_urdf(s)).round(4), " points:",
              ch.points(servo_to_urdf(s)).round(3).tolist())

    print("D. IK")
    q0 = servo_to_urdf(STARTUP)
    for t in [(0.10, -0.20, 0.10), (0, 0, 0.37), (0, -0.15, 0.20), (0, 0, 0.60), (0.5, 0.5, 0)]:
        s = solve_ik(ch, t, q0)
        inb = bool(np.all(s.q >= ch.lower) and np.all(s.q <= ch.upper))
        sv = urdf_to_servo(s.q); bad += not (inb and servo_ok(sv))
        print(f"   {t}: exact={s.exact} err={s.err*100:.2f}cm in_bounds={inb} servo_ok={servo_ok(sv)}")

    print("E. round trip")
    rng = np.random.default_rng(1); worst = 0.0
    for _ in range(25):
        q = rng.uniform(ch.lower, ch.upper)
        s = solve_ik(ch, ch.fk(q), q0)
        worst = max(worst, s.err); bad += not s.in_bounds
    print(f"   25 random configs, worst error {worst*1000:.3f} mm")

    print("F. stress: 2000 random stick steps at V1.2 speed")
    st = ArmState(ch); mv = rng.uniform(-1, 1, 3)
    n = {"ok": 0, "closest": 0, "rejected": 0}; jump = 0.0; t0 = time.time()
    for i in range(2000):
        if i % 50 == 0:
            mv = rng.uniform(-1, 1, 3)
        q_before = st.q.copy()
        cmds, status = st.move(mv, 0.02)
        assert np.all(st.q >= ch.lower) and np.all(st.q <= ch.upper)
        assert cmds is None or servo_ok(cmds)
        n["ok" if status == "ok" else "closest" if status.startswith("closest") else "rejected"] += 1
        jump = max(jump, float(np.degrees(np.abs(st.q - q_before)).max()))
    print(f"   {n} | worst accepted joint step {jump:.1f} deg | {(time.time()-t0)/2000*1000:.1f} ms/solve")

    try:
        from ikpy.chain import Chain
        c = Chain.from_urdf_file(urdf_path)
        names = [l.name for l in c.links]; w = 0.0
        for _ in range(20):
            q = rng.uniform(ch.lower, ch.upper); full = [0.0] * len(names)
            for nme, v in zip(ACTIVE, q):
                full[names.index(nme)] = v
            w = max(w, float(np.linalg.norm(c.forward_kinematics(full)[:3, 3] - ch.fk(q))))
        print(f"G. IKPy FK cross-check: worst difference {w*1000:.4f} mm")
    except ImportError:
        print("G. IKPy not installed: cross-check skipped")
    print("SELFTEST", "PASSED" if not bad else f"FAILED ({bad})")


if __name__ == "__main__":
    path = sys.argv[sys.argv.index("--urdf") + 1] if "--urdf" in sys.argv else URDF
    selftest(path) if "--selftest" in sys.argv else main(path)
