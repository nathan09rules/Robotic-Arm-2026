import socket
import time

ESP32_IP = "192.168.4.1"
PORT = 8888
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

Procedures = {
    "claw":[
        [0,60],
        [-1, 3],
        [0,100],
    ],
    "mv":[
    [4,120],
    [0,60],
    [-1,2],
    [4,70],
    [0,100]
        ]
    }

def send_cmd(payload: str):
    print(payload)
    if payload in Procedures:
        current = Procedures[payload]
        for i in current:
            print(i)
            if i[0] == -1:
                time.sleep(i[1])
            else:
                send_cmd(f"{i[0]}:{i[1]}")
    else:
        sock.sendto(payload.encode('utf-8'), (ESP32_IP, PORT))

def process_msg(usr_msg: str):
    msg = usr_msg.strip()
    if not msg:
        return
    if msg.upper() == "OFF":
        send_cmd("OFF")
        return
    send_cmd(msg)
