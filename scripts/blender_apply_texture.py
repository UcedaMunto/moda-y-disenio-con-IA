import bpy
import sys
from pathlib import Path


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_model(model_path: str) -> None:
    suffix = Path(model_path).suffix.lower()
    if suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=model_path)
    elif suffix == ".glb" or suffix == ".gltf":
        bpy.ops.import_scene.gltf(filepath=model_path)
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=model_path)
    else:
        raise ValueError(f"Unsupported model format: {suffix}")


def ensure_uvs() -> None:
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    for obj in mesh_objects:
        if obj.data.uv_layers and len(obj.data.uv_layers) > 0:
            continue
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
        bpy.ops.object.mode_set(mode="OBJECT")


def assign_texture(texture_path: str) -> None:
    image = bpy.data.images.load(texture_path)
    material = bpy.data.materials.new(name="FabricMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    output = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    tex = nodes.new(type="ShaderNodeTexImage")
    tex.image = image

    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    for obj in mesh_objects:
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)


def export_model(output_path: str) -> None:
    suffix = Path(output_path).suffix.lower()
    if suffix == ".obj":
        bpy.ops.wm.obj_export(filepath=output_path, export_selected_objects=False)
    elif suffix == ".glb":
        bpy.ops.export_scene.gltf(filepath=output_path, export_format="GLB")
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def main() -> None:
    args = sys.argv
    if "--" not in args:
        raise ValueError("Expected Blender args separator '--'")

    custom = args[args.index("--") + 1 :]
    if len(custom) != 3:
        raise ValueError("Usage: blender -b -P script.py -- model texture output")

    model_path, texture_path, output_path = custom
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    clear_scene()
    import_model(model_path)
    ensure_uvs()
    assign_texture(texture_path)
    export_model(output_path)


if __name__ == "__main__":
    main()
