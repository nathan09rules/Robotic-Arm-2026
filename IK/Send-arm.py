import math
import ikpy.chain
from ikpy.link import OriginLink, URDFLink

arm_chain = ikpy.chain.Chain(name='arm', links=[
    OriginLink(),
    URDFLink(
        name="base_yaw",
        origin_translation=[0, 0, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 1],
        bounds=(-math.pi, math.pi)
    ),
    URDFLink(
        name="base_pitch",
        origin_translation=[0, 0, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(-math.pi/2, math.pi/2)
    ),
    URDFLink(
        name="shoulder",
        origin_translation=[0.105, 0, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(0.0, math.pi * 0.8)
    ),
    URDFLink(
        name="end_effector",
        origin_translation=[0.11, 0, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 0]
    )
])

def InvK(posx, posy, posz):
    if posz < 0.0:
        raise ValueError("Position below table.")
    
    if math.sqrt(posx**2 + posy**2 + posz**2) > 0.215:
        raise ValueError("Position out of reach.")

    init_pos = [0, math.atan2(posy, posx), 0, 0, 0]
    target_angles = arm_chain.inverse_kinematics([posx, posy, posz], initial_position=init_pos)
    
    fk_test = [arm_chain.forward_kinematics(target_angles, full_kinematics=True)[i][:3, 3] for i in range(len(arm_chain.links))]
    if any(pt[2] < 0 for pt in fk_test):
        raise ValueError("Link below table.")

    return [
        [14, round(90 + math.degrees(target_angles[1]), 2)],
        [13, round(90 + math.degrees(target_angles[2]), 2)],
        [15, round(math.degrees(target_angles[3]), 2)]
    ]
