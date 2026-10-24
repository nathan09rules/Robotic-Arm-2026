#!/usr/bin/env python3
"""RoverArm persistent TCP + IKPy controller. Read README.md before motion.

No connection, initialization, plotting, or reconnect operation sends a movement.
M6 is a gripper actuator, held at its synchronized command during Cartesian IK.
All positions shown are commanded/modelled positions, not encoder measurements.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import queue
import re
import socket
import threading
import time
from typing import Optional

import numpy as np
from ikpy.chain import Chain
from ikpy.link import DHLink, OriginLink
 

@dataclass(frozen=True)
class ServoCalibration:
    name: str
    channel: int
    neutral: float
    minimum: Optional[float]
    maximum: Optional[float]
    direction: int
    joint_index: Optional[int]


@dataclass(frozen=True)
class DHParameters:
    name: str
    d: Optional[float]       # metres along previous z
    a: Optional[float]       # metres along common normal
    alpha: Optional[float]   # radians, twist about common normal
    theta: Optional[float]   # radians, fixed DH angular offset


# ====================== USER CONFIGURATION ======================
HOST, PORT = "192.168.4.1", 9999
SERVO_CALIBRATION_CONFIRMED = False
GEOMETRY_CONFIRMED = False
# None means UNMEASURED. Enter measured safe ranges, matching the firmware.
# The +1 directions and 90-degree neutrals below are provisional defaults.
SERVO_CONFIG = (
    ServoCalibration("M1", 14, 90.0, None, None, +1, 1),
    ServoCalibration("M2", 13, 90.0, None, None, +1, 2),
    ServoCalibration("M3", 15, 90.0, None, None, +1, 3),
    ServoCalibration("M4",  4, 90.0, None, None, +1, 4),
    ServoCalibration("M5",  1, 90.0, None, None, +1, 5),
    ServoCalibration("M6",  0, 90.0, None, None, +1, None),
)
# Standard DH: Rz(q + theta) Tz(d) Tx(a) Rx(alpha).
# First three rows are the user's APPROXIMATE geometry: verify all of them.
# theta=0 chooses a provisional DH zero; measure alignment to the servo neutral.
DH_CONFIG = (
    DHParameters("base",          0.010, 0.000, math.pi / 2, 0.0),
    DHParameters("shoulder",      0.000, 0.105, 0.0,         0.0),
    DHParameters("back_forearm",  0.000, 0.105, 0.0,         0.0),
    DHParameters("front_forearm", None,  None,  None,        None),
    DHParameters("wrist",         None,  None,  None,        None),
)
# Tool is a fixed link, NOT M6 rotation. The stated ~0.05 m tool length does
# not establish its DH axis/offset. Measure the complete wrist-to-tip transform.
APPROX_TOOL_LENGTH_M = 0.05  # reference only; not inserted along an invented axis
TOOL_DH = DHParameters("tool_tip", None, None, None, None)
# Optional extra mechanical joint bounds in radians, intersected with servo bounds.
# None means use calibrated servo-derived bounds, not unlimited joint travel.
JOINT_LIMITS_RAD = {1: None, 2: None, 3: None, 4: None, 5: None}

STEP_METRES = 0.002
POSITION_TOLERANCE_M = 0.003
MIN_SERVO_CHANGE_DEG = 0.10
MAX_SERVO_STEP_DEG = 10.0  # additional software guard; not a measured safe limit
MIN_COMMAND_INTERVAL_S = 0.15  # below firmware's 10 Hz cap
CONNECT_TIMEOUT_S = 2.0
RESPONSE_TIMEOUT_S = 1.5
HEARTBEAT_INTERVAL_S = 0.5
RECONNECT_INTERVAL_S = 2.0
# ================================================================


class ArmError(RuntimeError):
    """An error that can be reported without inventing a replacement command."""


class ConfigurationError(ArmError): pass
class ConnectionFailure(ArmError): pass
class Disconnected(ArmError): pass
class ProtocolError(ArmError): pass
class IKFailure(ArmError): pass
class UnreachableTarget(ArmError): pass
class JointLimitViolation(ArmError): pass
class ServoLimitViolation(ArmError): pass
class MotionDisabled(ArmError): pass


class CommandRejected(ArmError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(f"ESP32 rejected command: {code}")


DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)\Z")
STATUSES = {"READY", "KILLED", "DISABLED", "FAULT"}
ERROR_CODES = {"INVALID_COMMAND", "LIMIT", "KILLED", "DISABLED", "FAULT", "RATE_LIMIT"}
JOGS = {"a": (0, -1), "d": (0, 1), "w": (1, 1), "s": (1, -1),
        "i": (2, 1), "k": (2, -1)}


def parse_positions(line: str) -> np.ndarray:
    if not line.startswith("POS:"):
        raise ProtocolError(f"Expected POS as first line; received {line!r}")
    fields = line[4:].split(",")
    if len(fields) != 6 or any(DECIMAL.fullmatch(x) is None for x in fields):
        raise ProtocolError(f"Malformed POS: {line!r}")
    angles = np.array([float(x) for x in fields])
    if not np.isfinite(angles).all() or np.any(angles < 0) or np.any(angles > 180):
        raise ProtocolError("POS contains non-finite or out-of-envelope servo angles")
    return angles


def servo_to_joint(angle: float, calibration: ServoCalibration) -> float:
    return calibration.direction * math.radians(angle - calibration.neutral)


def joint_to_servo(angle: float, calibration: ServoCalibration) -> float:
    return calibration.neutral + calibration.direction * math.degrees(angle)


def validate_calibration(calibration: ServoCalibration) -> None:
    c = calibration
    if c.direction not in (-1, 1):
        raise ConfigurationError(f"{c.name}: direction must be +1 or -1")
    if c.minimum is None or c.maximum is None:
        raise ConfigurationError(f"{c.name}: enter measured servo minimum and maximum")
    if not all(math.isfinite(x) for x in (c.neutral, c.minimum, c.maximum)):
        raise ConfigurationError(f"{c.name}: calibration must be finite")
    if not (0 <= c.minimum <= c.maximum <= 180 and 0 <= c.neutral <= 180):
        raise ConfigurationError(f"{c.name}: require 0 <= minimum <= maximum <= 180; "
                                 "neutral must lie in 0..180")


def validate_servo(angle: float, c: ServoCalibration) -> None:
    validate_calibration(c)
    if not math.isfinite(angle) or not c.minimum <= angle <= c.maximum:
        raise ServoLimitViolation(f"{c.name}: {angle:.4f} deg outside [{c.minimum}, {c.maximum}]")


def make_dh_link(parameters: DHParameters, bounds: tuple, active: bool) -> DHLink:
    p = parameters
    if any(x is None for x in (p.d, p.a, p.alpha, p.theta)):
        raise ConfigurationError(f"Measure DH parameters for {p.name}: {p}")
    if not all(math.isfinite(x) for x in (p.d, p.a, p.alpha, p.theta)):
        raise ConfigurationError(f"Non-finite DH parameters for {p.name}")
    link = DHLink(name=p.name, d=p.d, a=p.a, alpha=p.alpha, theta=p.theta,
                  bounds=bounds, use_symbolic_matrix=False, length=math.hypot(p.d, p.a))
    # IKPy 3.4.2 DHLink drops name/bounds and joint metadata in its constructor.
    # Restore these explicitly; IKPy's optimizer reads link.bounds.
    link.name, link.bounds = p.name, bounds
    link.joint_type = "revolute" if active else "fixed"
    link.has_rotation = active
    link.has_translation = False
    return link


class ArmModel:
    """Five positioning joints plus a fixed tool; gripper opening is independent."""

    def __init__(self, servos=SERVO_CONFIG, dh=DH_CONFIG, tool=TOOL_DH,
                 joint_limits=None, *, confirmed=False):
        if not confirmed:
            raise ConfigurationError("Confirm measured geometry and servo calibration before IK; "
                                     "use --monitor or --joint-test during commissioning")
        if len(servos) != 6 or len(dh) != 5:
            raise ConfigurationError("Expected six motors and five positioning DH joints")
        if [c.joint_index for c in servos] != [1, 2, 3, 4, 5, None]:
            raise ConfigurationError("Expected M1..M5 joint indices 1..5; M6 must be None")
        self.servos = tuple(servos)
        self.bounds = {}
        joint_limits = JOINT_LIMITS_RAD if joint_limits is None else joint_limits
        links = [OriginLink()]
        for c in servos:
            validate_calibration(c)
            if c.joint_index is None:
                continue
            lo, hi = sorted((servo_to_joint(c.minimum, c), servo_to_joint(c.maximum, c)))
            extra = joint_limits.get(c.joint_index)
            if extra is not None:
                if len(extra) != 2 or not all(math.isfinite(x) for x in extra):
                    raise ConfigurationError(f"Invalid joint bounds for {c.name}")
                lo, hi = max(lo, extra[0]), min(hi, extra[1])
            if lo >= hi:
                raise ConfigurationError(f"{c.name}: IK needs a nonzero safe joint interval")
            self.bounds[c.joint_index] = (lo, hi)
            links.append(make_dh_link(dh[c.joint_index - 1], (lo, hi), True))
        links.append(make_dh_link(tool, (0.0, 0.0), False))
        self.chain = Chain(name="RoverArm", links=links,
                           active_links_mask=[False, True, True, True, True, True, False])

    def validate_joints(self, joints) -> np.ndarray:
        q = np.asarray(joints, dtype=float)
        if q.shape != (len(self.chain.links),) or not np.isfinite(q).all():
            raise IKFailure("IK/FK joint vector has wrong length or non-finite values")
        for index, (lo, hi) in self.bounds.items():
            if not lo <= q[index] <= hi:
                raise JointLimitViolation(f"Joint {index}: {q[index]:.6f} rad outside [{lo}, {hi}]")
        if q[0] != 0 or q[-1] != 0:
            raise IKFailure("Origin and fixed tool angles must remain zero")
        return q

    def servos_to_joints(self, positions) -> np.ndarray:
        if len(positions) != 6:
            raise ProtocolError("Expected exactly six servo positions")
        q = np.zeros(len(self.chain.links))
        for angle, c in zip(positions, self.servos):
            validate_servo(float(angle), c)
            if c.joint_index is not None:
                q[c.joint_index] = servo_to_joint(float(angle), c)
        return self.validate_joints(q)

    def joints_to_servos(self, joints, previous_servos) -> np.ndarray:
        q = self.validate_joints(joints)
        result = np.array(previous_servos, dtype=float, copy=True)
        for index, c in enumerate(self.servos):
            if c.joint_index is not None:
                result[index] = joint_to_servo(q[c.joint_index], c)
            validate_servo(float(result[index]), c)
        return result  # M6 stays at its synchronized commanded angle

    def forward(self, joints, *, full=False) -> np.ndarray:
        self.validate_joints(joints)
        frames = np.asarray(self.chain.forward_kinematics(joints, full_kinematics=full))
        if not np.isfinite(frames).all():
            raise IKFailure("Forward kinematics produced non-finite coordinates")
        return frames if full else frames[:3, 3].copy()

    def solve(self, target, current_joints, current_servos):
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise UnreachableTarget("XYZ target must contain three finite values in metres")
        self.validate_joints(current_joints)
        try:
            result = self.chain.inverse_kinematics(
                target_position=target, initial_position=np.array(current_joints, copy=True))
        except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            raise IKFailure(f"IKPy solver failed: {exc}") from exc
        if result is None:
            raise IKFailure("IKPy returned no solution")
        self.validate_joints(result)
        servos = self.joints_to_servos(result, current_servos)
        error = float(np.linalg.norm(self.forward(result) - target))
        if error > POSITION_TOLERANCE_M:
            raise UnreachableTarget(f"Target unreachable within limits, or solver did not converge: "
                                    f"FK error {error:.6f} m > {POSITION_TOLERANCE_M:.6f} m")
        # Validate exactly what will go on the wire, including decimal rounding.
        wire_servos = np.array([float(f"{x:.2f}") for x in servos])
        wire_joints = self.servos_to_joints(wire_servos)
        error = float(np.linalg.norm(self.forward(wire_joints) - target))
        if error > POSITION_TOLERANCE_M:
            raise UnreachableTarget(f"Rounded command FK error {error:.6f} m exceeds tolerance")
        return wire_servos, wire_joints


class ESP32Connection:
    """One socket, one in-flight request, LF framing, bounded response deadlines."""

    def __init__(self, host=HOST, port=PORT):
        self.host, self.port = host, port
        self.sock = None
        self.buffer = bytearray()
        self.status = "DISCONNECTED"

    def close(self):
        if self.sock is not None:
            self.sock.close()
        self.sock = None
        self.buffer.clear()
        self.status = "DISCONNECTED"

    def read_line(self, deadline: float) -> str:
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                if newline > 127:
                    raise ProtocolError("ESP32 line exceeds 127 bytes")
                raw = bytes(self.buffer[:newline])
                del self.buffer[:newline + 1]
                if any(x < 32 or x > 126 for x in raw):
                    raise ProtocolError("ESP32 line contains invalid ASCII/control characters")
                return raw.decode("ascii")
            if len(self.buffer) > 127:
                raise ProtocolError("ESP32 sent an overlong/incomplete line")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Disconnected("ESP32 response timeout; command outcome may be unknown")
            if self.sock is None:
                raise Disconnected("TCP socket is closed")
            self.sock.settimeout(remaining)
            try:
                data = self.sock.recv(256)
            except (TimeoutError, OSError) as exc:
                raise Disconnected(f"ESP32 read failed: {exc}") from exc
            if not data:
                raise Disconnected("ESP32 closed the TCP connection")
            self.buffer.extend(data)

    def consume_status(self, line: str) -> bool:
        if not line.startswith("STATUS:"):
            return False
        status = line[7:]
        if status not in STATUSES:
            raise ProtocolError(f"Unknown status: {line!r}")
        self.status = status
        return True

    def connect(self) -> np.ndarray:
        self.close()
        try:
            self.sock = socket.create_connection((self.host, self.port), CONNECT_TIMEOUT_S)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError as exc:
            self.close()
            raise ConnectionFailure(f"Cannot connect to {self.host}:{self.port}: {exc}") from exc
        deadline = time.monotonic() + RESPONSE_TIMEOUT_S
        positions = parse_positions(self.read_line(deadline))
        if not self.consume_status(self.read_line(deadline)):
            raise ProtocolError("Expected STATUS immediately after initial POS")
        return positions

    def request(self, command: str, expected="OK") -> None:
        if self.sock is None:
            raise Disconnected("No TCP connection")
        if "\n" in command or "\r" in command or len(command) > 127:
            raise ProtocolError("Invalid outbound command framing")
        deadline = time.monotonic() + RESPONSE_TIMEOUT_S
        self.sock.settimeout(RESPONSE_TIMEOUT_S)
        try:
            self.sock.sendall((command + "\n").encode("ascii"))
        except OSError as exc:
            raise Disconnected(f"TCP send failed; command outcome may be unknown: {exc}") from exc
        while True:
            line = self.read_line(deadline)
            if self.consume_status(line):
                continue
            if line.startswith("ERROR:"):
                code = line[6:]
                if code not in ERROR_CODES:
                    raise ProtocolError(f"Unknown error response: {line!r}")
                if code in {"KILLED", "DISABLED", "FAULT"}:
                    self.status = code
                raise CommandRejected(code)
            if line != expected:
                raise ProtocolError(f"Expected {expected!r}, received {line!r}")
            return

    def heartbeat(self):
        self.request("PING", "PONG")

    def send_servos(self, positions):
        if self.status != "READY":
            raise MotionDisabled(f"ESP32 status is {self.status}")
        if len(positions) != 6 or not np.isfinite(positions).all():
            raise ProtocolError("SET needs six finite angles")
        if any(x < 0 or x > 180 for x in positions):
            raise ServoLimitViolation("SET angle outside absolute 0..180 envelope")
        # IK may take time: consume any safety transition before sending movement.
        self.heartbeat()
        if self.status != "READY":
            raise MotionDisabled(f"ESP32 status is {self.status}; movement discarded")
        self.request("SET:" + ",".join(f"{x:.2f}" for x in positions))


class ArmController:
    """State is committed only after OK; failed commands never advance the target."""

    def __init__(self, connection, model=None, *, joint_test=False):
        self.connection, self.model, self.joint_test = connection, model, joint_test
        self.servos = self.joints = self.target = None
        self.last_move = -math.inf

    def synchronize(self, positions):
        # A reconnect replaces ALL virtual state with ESP32's current command state.
        joints = self.model.servos_to_joints(positions) if self.model else None
        target = self.model.forward(joints) if self.model else None
        self.servos = np.array(positions, copy=True)
        self.joints, self.target = joints, target

    def check_motion(self):
        if self.servos is None or self.connection.status != "READY":
            raise MotionDisabled("Movement requires synchronization and ESP32 STATUS:READY")
        if time.monotonic() - self.last_move < MIN_COMMAND_INTERVAL_S:
            raise MotionDisabled("Movement rate limited; issue a fresh request after a short pause")

    def move_xyz(self, target):
        self.check_motion()
        if self.model is None:
            raise MotionDisabled("Cartesian mode requires measured, confirmed geometry/calibration")
        servos, joints = self.model.solve(target, self.joints, self.servos)
        delta = float(np.max(np.abs(servos - self.servos)))
        if delta < MIN_SERVO_CHANGE_DEG:
            return "Below servo-change threshold; previous target/state retained"
        if delta > MAX_SERVO_STEP_DEG:
            raise MotionDisabled(f"IK requires a {delta:.2f}-degree jump; limit is "
                                 f"{MAX_SERVO_STEP_DEG:.2f}. Request a closer target")
        self.connection.send_servos(servos)
        self.servos, self.joints = servos, joints
        self.target = np.array(target, dtype=float, copy=True)
        self.last_move = time.monotonic()
        return f"Accepted XYZ {self.target.round(5).tolist()}"

    def move_motor(self, index: int, angle: float):
        self.check_motion()
        if not self.joint_test:
            raise MotionDisabled("Individual servo commands require --joint-test")
        if index not in range(6):
            raise ServoLimitViolation("Motor index must be 1..6")
        angle = float(f"{angle:.2f}")
        validate_servo(angle, SERVO_CONFIG[index])
        if abs(angle - self.servos[index]) > MAX_SERVO_STEP_DEG:
            raise MotionDisabled("Single-joint request exceeds MAX_SERVO_STEP_DEG")
        self.connection.request(f"M{index + 1}:{angle:.2f}")
        self.servos[index] = angle
        self.last_move = time.monotonic()
        return f"Accepted M{index + 1}:{angle:.2f}"


@dataclass(frozen=True)
class Snapshot:
    epoch: int = 0
    connected: bool = False
    status: str = "DISCONNECTED"
    servos: tuple = ()
    joints: tuple = ()
    target: tuple = ()
    message: str = "Connecting; movement unavailable"


class Worker:
    """Network/IK thread keeps matplotlib responsive; no movement backlog/replay."""

    def __init__(self, controller):
        self.controller = controller
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.pending = queue.Queue(maxsize=1)
        self.view = Snapshot()
        self.busy = False
        self.epoch = 0
        self.thread = threading.Thread(target=self.run, name="RoverArm-network", daemon=True)

    def publish(self, message=None, connected=True):
        c = self.controller
        with self.lock:
            self.view = Snapshot(self.epoch, connected, c.connection.status,
                                 tuple(c.servos) if c.servos is not None else (),
                                 tuple(c.joints) if c.joints is not None else (),
                                 tuple(c.target) if c.target is not None else (),
                                 message or self.view.message)
        if message:
            print(message, flush=True)

    def submit(self, kind, payload=None):
        with self.lock:
            if not self.view.connected:
                return "Disconnected/synchronizing: request discarded"
            if self.view.status != "READY" and kind != "rearm":
                return f"Motion disabled: {self.view.status}"
            if self.busy:
                return "A request is in progress; this input was discarded"
            self.busy = True
            self.pending.put_nowait((self.view.epoch, kind, payload))
        return "Request submitted"

    def clear_pending(self):
        with self.lock:
            while not self.pending.empty():
                self.pending.get_nowait()
            self.busy = False

    def execute(self, kind, payload):
        c = self.controller
        # Refresh asynchronous safety state immediately before processing a request.
        c.connection.heartbeat()
        if kind == "rearm":
            c.connection.request("REARM")
            return "Rearmed without pulses. Check physical pose before requesting movement"
        if c.connection.status != "READY":
            raise MotionDisabled(f"ESP32 reports {c.connection.status}; request discarded")
        if kind == "jog":
            if c.target is None:
                raise MotionDisabled("Cartesian model is unavailable")
            candidate = c.target.copy()
            axis, sign = JOGS[payload]
            candidate[axis] += sign * STEP_METRES
            return c.move_xyz(candidate)
        if kind == "xyz":
            return c.move_xyz(payload)
        if kind == "motor":
            return c.move_motor(*payload)
        raise ProtocolError(f"Unknown local request: {kind}")

    def run(self):
        c = self.controller
        try:
            while not self.stop.is_set():
                try:
                    positions = c.connection.connect()
                    c.synchronize(positions)
                    self.epoch += 1
                    self.clear_pending()
                    self.publish(f"Synchronized {positions.tolist()}; status {c.connection.status}. "
                                 "Waiting for a NEW movement request")
                    heartbeat_at = 0.0
                    while not self.stop.is_set():
                        if time.monotonic() >= heartbeat_at:
                            old = c.connection.status
                            c.connection.heartbeat()
                            heartbeat_at = time.monotonic() + HEARTBEAT_INTERVAL_S
                            if old != c.connection.status:
                                self.clear_pending()
                                self.publish(f"ESP32 status changed to {c.connection.status}")
                        try:
                            epoch, kind, payload = self.pending.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        try:
                            if epoch != self.epoch:
                                self.publish("Discarded a request from an earlier connection")
                            else:
                                self.publish(self.execute(kind, payload))
                        except (ConnectionFailure, Disconnected, ProtocolError):
                            raise  # uncertain transaction: close and resynchronize, never replay
                        except ArmError as exc:
                            self.publish(f"{type(exc).__name__}: {exc}")
                        finally:
                            with self.lock:
                                self.busy = False
                except ArmError as exc:
                    c.connection.close()
                    self.publish(f"{type(exc).__name__}: {exc}. Reconnecting; pending movement discarded",
                                 connected=False)
                    self.clear_pending()
                    self.stop.wait(RECONNECT_INTERVAL_S)
        except Exception as exc:
            # Unexpected programming/library failures are visible and stop the worker.
            c.connection.close()
            self.publish(f"Fatal {type(exc).__name__}: {exc}; controller stopped", connected=False)
            self.stop.set()
        finally:
            c.connection.close()

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=CONNECT_TIMEOUT_S + RESPONSE_TIMEOUT_S + 1)


def run_console(worker):
    print("Commands: status | xyz X Y Z | a d w s i k | m1 ANGLE ... m6 ANGLE "
          "(--joint-test) | rearm | quit\nXYZ units: metres. Jog keys require Enter here.")
    while not worker.stop.is_set():
        try:
            fields = input("arm> ").strip().lower().split()
        except (EOFError, KeyboardInterrupt):
            break
        if not fields:
            continue
        command = fields[0]
        if command in {"quit", "exit", "q"}:
            break
        if command == "status" and len(fields) == 1:
            print(worker.view)
            continue
        try:
            if command == "xyz" and len(fields) == 4:
                result = worker.submit("xyz", [float(x) for x in fields[1:]])
            elif command in JOGS and len(fields) == 1:
                result = worker.submit("jog", command)
            elif re.fullmatch("m[1-6]", command) and len(fields) == 2:
                result = worker.submit("motor", (int(command[1]) - 1, float(fields[1])))
            elif command == "rearm" and len(fields) == 1:
                result = worker.submit("rearm")
            else:
                result = "Unknown command or wrong number of arguments"
            print(result)
        except ValueError as exc:
            print(f"Invalid numeric input: {exc}")


def run_plot(worker, model):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox

    # Remove matplotlib shortcuts which conflict with Cartesian keys (e.g. s/k).
    for name in list(plt.rcParams):
        if name.startswith("keymap."):
            plt.rcParams[name] = [x for x in plt.rcParams[name] if x not in JOGS]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.25)
    status_text = fig.text(0.08, 0.02, "Waiting for synchronized state", fontsize=9, wrap=True)
    boxes = [TextBox(fig.add_axes([0.12 + i * 0.23, 0.14, 0.18, 0.045]), label, initial="")
             for i, label in enumerate(("X m ", "Y m ", "Z m "))]
    move_button = Button(fig.add_axes([0.80, 0.14, 0.13, 0.045]), "Move XYZ")
    last_view = [None]
    held = set()

    def move(_):
        try:
            target = [float(box.text) for box in boxes]
        except ValueError:
            status_text.set_text("Enter three finite XYZ values in metres")
            return
        status_text.set_text(worker.submit("xyz", target))

    def key_press(event):
        key = (event.key or "").lower()
        # Typing in an XYZ field must not also jog the robot.
        if any(box.capturekeystrokes for box in boxes) or key not in JOGS or key in held:
            return
        held.add(key)
        status_text.set_text(worker.submit("jog", key))

    def refresh():
        view = worker.view
        if view == last_view[0]:
            return
        previous = last_view[0]
        last_view[0] = view
        ax.clear()
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.set_title(f"RoverArm — commanded state — {view.status}\n"
                     "A/D: X  •  W/S: Y  •  I/K: Z  •  release between presses")
        if view.joints:
            frames = model.forward(np.array(view.joints), full=True)
            points = frames[:, :3, 3]
            ax.plot(points[:, 0], points[:, 1], points[:, 2], "o-", label="Links / joints")
            ax.scatter(*points[0], marker="s", s=70, label="Base")
            ax.scatter(*points[-1], s=75, label="Current model tip")
            ax.scatter(*view.target, marker="x", s=90, color="red", label="Requested target")
            for index in range(1, 6):
                ax.text(*points[index], f" M{index}")
            extent = max(0.1, float(np.max(np.abs(points))), float(np.max(np.abs(view.target)))) * 1.2
            ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent); ax.set_zlim(-extent, extent)
            ax.set_box_aspect((1, 1, 1))
            ax.legend(loc="upper left", fontsize=8)
            # Only synchronization initializes entry boxes; editing a box never moves.
            if previous is None or previous.epoch != view.epoch:
                held.clear()
                for box, value in zip(boxes, view.target):
                    box.set_val(f"{value:.5f}")
        status_text.set_text(view.message)
        fig.canvas.draw_idle()

    move_button.on_clicked(move)
    fig.canvas.mpl_connect("key_press_event", key_press)
    fig.canvas.mpl_connect("key_release_event", lambda event: held.discard((event.key or "").lower()))
    timer = fig.canvas.new_timer(interval=100)
    timer.add_callback(refresh)
    timer.start()
    refresh()
    plt.show()
    timer.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--console", action="store_true", help="Use terminal controls instead of matplotlib")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--monitor", action="store_true", help="Synchronize/monitor without a kinematic model")
    modes.add_argument("--joint-test", action="store_true", help="Test individual calibrated servos; no IK")
    args = parser.parse_args()
    try:
        model = None if (args.monitor or args.joint_test) else ArmModel(
            confirmed=GEOMETRY_CONFIRMED and SERVO_CALIBRATION_CONFIRMED)
    except ConfigurationError as exc:
        parser.exit(2, f"ConfigurationError: {exc}\nSee the configuration section and README.md.\n")
    worker = Worker(ArmController(ESP32Connection(args.host, args.port), model,
                                 joint_test=args.joint_test))
    worker.start()
    try:
        if args.console or model is None:
            run_console(worker)
        else:
            run_plot(worker, model)
    except KeyboardInterrupt:
        print("Stopping controller; firmware retains its latest commanded positions")
    finally:
        worker.close()


if __name__ == "__main__":
    main()
