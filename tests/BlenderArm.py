import socket

def send_pose(bone_name, x, y, z):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('127.0.0.1', 9999))
    client.sendall(f"{bone_name}:{x},{y},{z}".encode())
    client.close()

def get_bone_rotation(bone_name):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('127.0.0.1', 9999))
    client.sendall(f"GET:{bone_name}".encode())
    response = client.recv(1024).decode()
    client.close()
    return tuple(float(v) for v in response.split(","))

def finger(finger, state):
    if finger == "THUMB":
        if state == 2:
            send_pose("4", 0.75, 0.0, 0.0)
        if state == 0:
            send_pose("4", 1.2 ,0.0 ,0.0)
    


"""

4 (0.75, 0.0, 0.0)
3 (0.0, 0.0, 0.0)
2 (0.0, 0.0, 0.0)


"""
