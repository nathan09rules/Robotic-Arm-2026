import bpy

armature = bpy.data.objects.get("Armature")
if armature:
    for name in ["4", "3", "2"]:
        if name in armature.pose.bones:
            print(name, armature.pose.bones[name].rotation_euler[:])