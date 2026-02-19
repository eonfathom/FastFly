"""
Convert TuragaLab/flybody MuJoCo model to GLB for Three.js.

Parses the MuJoCo XML to get body hierarchy and transforms,
loads OBJ meshes, assembles fly in rest pose, and exports GLB.

Usage:
    pip install trimesh numpy
    python convert_flybody.py
"""

import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from pathlib import Path
import json

# Paths
REPO_DIR = Path("mujoco_menagerie_tmp/flybody")
XML_PATH = REPO_DIR / "fruitfly.xml"
ASSETS_DIR = REPO_DIR / "assets"
OUTPUT_PATH = Path("static/models/flybody.glb")

# MuJoCo material colors (RGBA)
MATERIAL_COLORS = {
    "body":          [172, 89, 36, 255],
    "red":           [204, 7, 0, 255],
    "ocelli":        [33, 12, 4, 255],
    "black":         [10, 10, 10, 255],
    "bristle-brown": [10, 10, 10, 255],
    "lower":         [204, 156, 98, 255],
    "brown":         [52, 20, 7, 255],
    "membrane":      [137, 175, 204, 102],  # translucent
}

DEFAULT_MESH_SCALE = 0.1  # from <default><mesh scale="0.1 0.1 0.1"/>


def quat_to_matrix(q):
    """MuJoCo quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def make_transform(pos, quat):
    """Build 4x4 homogeneous transform from position and quaternion."""
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(quat)
    T[:3, 3] = pos
    return T


def parse_vec(s, default):
    """Parse a space-separated number string, return list of floats."""
    if s is None:
        return list(default)
    return [float(x) for x in s.split()]


def get_color_for_mesh(mesh_name, geom_material=None):
    """Determine RGBA color based on mesh name suffix or geom material attribute."""
    if geom_material and geom_material in MATERIAL_COLORS:
        return MATERIAL_COLORS[geom_material]
    # Infer from mesh name
    for suffix in ["membrane", "red", "ocelli", "black", "bristle-brown", "brown", "lower"]:
        if suffix in mesh_name:
            return MATERIAL_COLORS[suffix]
    return MATERIAL_COLORS["body"]


def parse_xml():
    """Parse MuJoCo XML and return list of (mesh_name, obj_file, world_transform, material)."""
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    # Build mesh name → file mapping
    mesh_files = {}
    for mesh_el in root.findall(".//asset/mesh"):
        name = mesh_el.get("name")
        filename = mesh_el.get("file")
        scale_str = mesh_el.get("scale")
        if scale_str:
            scale = [float(x) for x in scale_str.split()]
        else:
            scale = [DEFAULT_MESH_SCALE] * 3
        mesh_files[name] = {"file": filename, "scale": scale}

    # Traverse body hierarchy
    results = []

    def traverse_body(elem, parent_world_transform):
        pos = parse_vec(elem.get("pos"), [0, 0, 0])
        quat = parse_vec(elem.get("quat"), [1, 0, 0, 0])

        local_T = make_transform(pos, quat)
        world_T = parent_world_transform @ local_T

        # Process geoms with meshes
        for geom in elem.findall("geom"):
            mesh_name = geom.get("mesh")
            if mesh_name is None or mesh_name not in mesh_files:
                continue

            # Skip collision geoms
            geom_class = geom.get("class", "")
            if "collision" in geom_class:
                continue

            geom_pos = parse_vec(geom.get("pos"), [0, 0, 0])
            geom_quat = parse_vec(geom.get("quat"), [1, 0, 0, 0])
            geom_T = make_transform(geom_pos, geom_quat)

            final_T = world_T @ geom_T

            material = geom.get("material")
            body_name = elem.get("name", "unknown")

            results.append({
                "mesh_name": mesh_name,
                "body_name": body_name,
                "obj_file": mesh_files[mesh_name]["file"],
                "scale": mesh_files[mesh_name]["scale"],
                "transform": final_T,
                "material": material,
            })

        # Recurse into child bodies
        for child in elem.findall("body"):
            traverse_body(child, world_T)

    worldbody = root.find(".//worldbody")
    for body in worldbody.findall("body"):
        traverse_body(body, np.eye(4))

    return results


def convert():
    print("Parsing MuJoCo XML...")
    geom_list = parse_xml()
    print(f"  Found {len(geom_list)} visual geoms")

    scene = trimesh.Scene()
    loaded = 0
    skipped = 0

    for entry in geom_list:
        obj_path = ASSETS_DIR / entry["obj_file"]
        if not obj_path.exists():
            print(f"  WARNING: {obj_path} not found, skipping")
            skipped += 1
            continue

        try:
            mesh = trimesh.load(str(obj_path), force="mesh")
        except Exception as e:
            print(f"  WARNING: Failed to load {obj_path}: {e}")
            skipped += 1
            continue

        # Apply mesh scale
        s = entry["scale"]
        scale_matrix = np.diag([s[0], s[1], s[2], 1.0])
        mesh.apply_transform(scale_matrix)

        # Apply world transform
        mesh.apply_transform(entry["transform"])

        # Assign color
        color = get_color_for_mesh(entry["mesh_name"], entry["material"])
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
        mesh.visual.face_colors = np.tile(color, (len(mesh.faces), 1))

        # Use a descriptive node name: body_name/mesh_name
        node_name = f"{entry['body_name']}__{entry['mesh_name']}"
        scene.add_geometry(mesh, node_name=node_name)
        loaded += 1

    print(f"  Loaded {loaded} meshes, skipped {skipped}")

    # Transform: MuJoCo (Z-up, X-forward) → Three.js (Y-up, Z-forward)
    # new_x = -old_y, new_y = old_z, new_z = old_x
    coord_transform = np.array([
        [0, -1, 0, 0],
        [0,  0, 1, 0],
        [1,  0, 0, 0],
        [0,  0, 0, 1],
    ], dtype=float)

    # Apply transforms to each geometry individually (scene.apply_transform can fail)
    for name, geom in scene.geometry.items():
        geom.apply_transform(coord_transform)

    # Compute bounding box and scale to desired size
    bounds = scene.bounds
    extent = bounds[1] - bounds[0]
    center = (bounds[0] + bounds[1]) / 2
    print(f"  Bounding box extent: {extent}")
    print(f"  Center: {center}")

    # Target: fly body ~3.5 units long (matching existing procedural fly)
    max_extent = max(extent)
    target_size = 3.5
    scale_factor = target_size / max_extent
    print(f"  Scale factor: {scale_factor:.2f}")

    # Center, scale, and ground-align
    centering = np.eye(4)
    centering[:3, 3] = -center
    scaling = np.eye(4)
    scaling[:3, :3] *= scale_factor
    combined = scaling @ centering

    for name, geom in scene.geometry.items():
        geom.apply_transform(combined)

    # Move fly so legs touch ground (GROUND_Y = -0.9 in scene)
    new_bounds = scene.bounds
    lowest_y = new_bounds[0][1]
    ground_offset = np.eye(4)
    ground_offset[1, 3] = -0.9 - lowest_y
    for name, geom in scene.geometry.items():
        geom.apply_transform(ground_offset)

    final_bounds = scene.bounds
    print(f"  Final bounds: min={final_bounds[0]}, max={final_bounds[1]}")

    # Export
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    glb_data = scene.export(file_type="glb")
    with open(OUTPUT_PATH, "wb") as f:
        f.write(glb_data)

    file_size = OUTPUT_PATH.stat().st_size
    print(f"\nExported to {OUTPUT_PATH} ({file_size / 1024:.1f} KB)")

    # Also export a metadata JSON with node names for the frontend
    meta = {
        "nodes": [
            {
                "node_name": f"{e['body_name']}__{e['mesh_name']}",
                "body_name": e["body_name"],
                "mesh_name": e["mesh_name"],
                "material": e["material"],
            }
            for e in geom_list
            if (ASSETS_DIR / e["obj_file"]).exists()
        ]
    }
    meta_path = OUTPUT_PATH.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    convert()
