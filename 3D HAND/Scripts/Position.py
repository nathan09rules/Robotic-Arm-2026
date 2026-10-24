import bpy

# Open your base file
bpy.ops.wm.open_mainfile(filepath="/home/nathanpd/Documents/26/Robotic-Arm/3D HAND/source/RiggedLowpolyHand.blend")

# Ensure the armature is selected and active
armature = bpy.data.objects.get("Armature")
if armature:
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

# Export to GLB
bpy.ops.export_scene.gltf(filepath="/home/nathanpd/Documents/26/Robotic-Arm/3D HAND/source/hand.glb", export_format='GLB')
print("Successfully exported to hand.glb!")