import bpy
my_path = ".../SVarM/MNIST_heightmaps/"
from os import listdir
ls = listdir(my_path)
print(ls)
for name in ls:
    bpy.ops.import_mesh.ply(filepath=str(my_path + "{}".format(name)))
    selection_name = bpy.context.selected_objects
    print(selection_name[0].name)
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')
    bpy.ops.object.modifier_add(type='REMESH')
    bpy.context.object.modifiers["Remesh"].mode = 'SMOOTH'
    bpy.context.object.modifiers["Remesh"].octree_depth = 6
    bpy.ops.object.modifier_add(type='TRIANGULATE')
    bpy.context.object.modifiers["Triangulate"].min_vertices = 4
    bpy.ops.object.modifier_add(type='SMOOTH')
    bpy.context.object.modifiers["Smooth"].iterations = 10
    bpy.ops.export_mesh.ply(filepath=str(".../MNIST_remeshed/{}.ply".format(selection_name[0].name)),check_existing=True, filter_glob="*.ply", use_mesh_modifiers=True, use_normals=False, use_uv_coords=False, use_colors=False, global_scale=1.0, axis_forward='Y', axis_up='Z')
    bpy.ops.object.delete() 