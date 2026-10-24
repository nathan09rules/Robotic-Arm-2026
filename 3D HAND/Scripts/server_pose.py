import bpy
import socket
import threading
import queue

command_queue = queue.Queue()
initial_rotations = {}

armature = bpy.data.objects.get("Armature")
if armature:
    for bone in armature.pose.bones:
        initial_rotations[bone.name] = bone.rotation_euler.copy()

def run_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', 9999))
    server.listen(1)
    while True:
        try:
            conn, addr = server.accept()
            data = conn.recv(1024).decode().strip()
            if data == "RESET":
                command_queue.put("RESET")
                conn.sendall(b"OK")
            elif data.startswith("GET:"):
                bone_name = data.split(":")[1]
                rot = (0.0, 0.0, 0.0)
                if armature and bone_name in armature.pose.bones:
                    rot = tuple(armature.pose.bones[bone_name].rotation_euler)
                conn.sendall(f"{rot[0]},{rot[1]},{rot[2]}".encode())
            elif ":" in data:
                command_queue.put(data)
                conn.sendall(b"OK")
            conn.close()
        except:
            pass

if not any(t.name == "SocketServerThread" for t in threading.enumerate()):
    threading.Thread(target=run_server, daemon=True, name="SocketServerThread").start()

def process_queue():
    while not command_queue.empty():
        data = command_queue.get()
        if data == "RESET" and armature:
            for bone_name, rot in initial_rotations.items():
                if bone_name in armature.pose.bones:
                    armature.pose.bones[bone_name].rotation_euler = rot.copy()
        else:
            parts = data.strip().split(":")
            if len(parts) == 2:
                bone_name = parts[0]
                angles = [float(x) for x in parts[1].split(",")]
                if len(angles) == 3 and armature and bone_name in armature.pose.bones:
                    bone = armature.pose.bones[bone_name]
                    bone.rotation_mode = 'XYZ'
                    bone.rotation_euler = angles
        
        if armature:
            armature.update_tag()
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
    return 0.1

if not bpy.app.timers.is_registered(process_queue):
    bpy.app.timers.register(process_queue)