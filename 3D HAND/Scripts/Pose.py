import bpy

bpy.ops.wm.open_mainfile(filepath="/home/nathanpd/Documents/26/Robotic-Arm/3D HAND/source/RiggedLowpolyHand.blend")
armature = bpy.data.objects["Armature"]

for i, bone in enumerate(armature.data.bones):
    print(i, bone.name, bone.parent.name if bone.parent else "None")