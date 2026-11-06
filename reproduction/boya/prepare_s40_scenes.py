"""Create an explicitly approximate S40 Boya scene manifest from pinned assets.

The source triangle mesh is converted losslessly to OBJ; existing DGN2 convex
pieces remain a separately measured collision approximation. No learned scores,
grip labels, DGN_5k scale or selection policy is consulted.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def placement(visual_vertices, collision_vertices, xy=(-.75, .1)):
    """Fixed native axes, common support plane; never choose pose by outcome."""
    visual_vertices = np.asarray(visual_vertices, float)
    collision_vertices = np.asarray(collision_vertices, float)
    if any(v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all()
           for v in [visual_vertices, collision_vertices]):
        raise ValueError('Finite nonempty triangle mesh vertices required')
    T = np.eye(4)
    T[:2, 3] = np.asarray(xy) - (visual_vertices[:, :2].min(0) + visual_vertices[:, :2].max(0)) / 2
    T[2, 3] = -min(visual_vertices[:, 2].min(), collision_vertices[:, 2].min())
    return T


def verify_mesh_conversion(source, converted):
    a, b = np.asarray(source.vertices), np.asarray(converted.vertices)
    if a.shape != b.shape or not np.array_equal(source.faces, converted.faces):
        raise ValueError('OBJ conversion changed triangle topology or ordering')
    err = float(np.abs(a - b).max())
    if err > 2e-12:
        raise ValueError('OBJ conversion changed source mesh coordinates')
    return err


def validate_inventory(inventory):
    if inventory.get('schema') != 'anydex_graspnet_s40_asset_inventory_v1':
        raise ValueError('Pinned S40 inventory required')
    ids = inventory['object_ids']
    rows = inventory['objects']
    if len(ids) != 40 or len(set(ids)) != 40 or sorted(r['graspnet_object_id'] for r in rows) != sorted(ids):
        raise ValueError('All original S40 identities must be retained')
    if inventory['training_scene_range'] != [0, 99] or len(inventory['scenes']) != 100:
        raise ValueError('S40 derivation must bind all100 train scene identity lists')
    for row in inventory['scenes']:
        if sha(row['path']) != row['sha256']:
            raise ValueError('Source training scene identity changed')
    for row in rows:
        for source in row['files'].values():
            if sha(source['path']) != source['sha256']:
                raise ValueError('Source object changed: ' + source['path'])
    return sorted(rows, key=lambda r: r['graspnet_object_id'])


def package(row, asset_root, run_dir, inventory_sha):
    oid = 'graspnet_' + row['graspnet_directory']
    source_mesh = Path(row['files']['nontextured.ply']['path'])
    source = source_mesh.parent
    root = asset_root / oid
    for sub in ['mesh', 'urdf/meshes', 'info']:
        (root / sub).mkdir(parents=True, exist_ok=False)
    mesh = trimesh.load_mesh(source_mesh, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not np.isfinite(mesh.vertices).all():
        raise ValueError('Expected finite original source triangles')
    output_mesh = root / 'mesh/simplified.obj'
    # Filename compatibility with the read-only backend; absolutely no simplification.
    output_mesh.write_text(trimesh.exchange.obj.export_obj(mesh, include_normals=False,
                            include_color=False, include_texture=False, digits=16))
    converted = trimesh.load_mesh(output_mesh, process=False)
    conversion_error = verify_mesh_conversion(mesh, converted)
    source_urdf = source / 'coacd/coacd.urdf'
    xml = ET.parse(source_urdf)
    all_source = {str(source_mesh): sha(source_mesh), str(source_urdf): sha(source_urdf)}
    parts = {}
    collision_vertices = []
    for node in xml.getroot().findall('.//geometry/mesh'):
        scale = np.array([float(x) for x in node.get('scale', '1 1 1').split()])
        if not np.array_equal(scale, np.ones(3)):
            raise ValueError('Never inherit a collision scale other than1')
        src = (source_urdf.parent / node.attrib['filename']).resolve()
        if src.parent != source_urdf.parent:
            raise ValueError('Unexpected source convex part path')
        dst = root / 'urdf/meshes' / src.name
        if src.name not in parts:
            shutil.copyfile(src, dst)
            if sha(src) != sha(dst):
                raise ValueError('Collision copy changed source bytes')
            m = trimesh.load_mesh(dst, process=False)
            if not isinstance(m, trimesh.Trimesh) or not np.isfinite(m.vertices).all():
                raise ValueError('Invalid collision mesh')
            parts[src.name] = dict(path=str(dst), sha256=sha(dst), scale='1 1 1')
            all_source[str(src)] = sha(src)
        node.set('filename', 'meshes/' + src.name)
    for node in xml.getroot().findall('.//collision'):
        origin = node.find('origin')
        if origin is not None and any(float(x) != 0 for k in ['xyz', 'rpy'] for x in origin.get(k, '0 0 0').split()):
            raise ValueError('Nonzero local collision transform needs explicit conversion')
        node_mesh = node.find('geometry/mesh')
        m = trimesh.load_mesh(root / 'urdf' / node_mesh.attrib['filename'], process=False)
        collision_vertices.append(np.asarray(m.vertices))
    if not collision_vertices:
        raise ValueError('No packaged convex geometry')
    # Fixed source joints must not introduce unaccounted geometry transforms.
    for joint in xml.getroot().findall('joint'):
        origin = joint.find('origin')
        if joint.get('type') != 'fixed' or (origin is not None and any(float(x) != 0 for k in ['xyz', 'rpy'] for x in origin.get(k, '0 0 0').split())):
            raise ValueError('Collision package has unsupported articulated transform')
    collision_vertices = np.concatenate(collision_vertices)
    urdf = root / 'urdf/coacd.urdf'
    xml.write(urdf, encoding='unicode')
    T = placement(mesh.vertices, collision_vertices)
    visual_world = np.asarray(mesh.vertices) + T[:3, 3]
    collision_world = collision_vertices + T[:3, 3]
    visual_bounds = np.asarray(mesh.bounds)
    collision_bounds = np.array([collision_vertices.min(0), collision_vertices.max(0)])
    pose = root / 'info/tabletop_pose.json'
    save(pose, [np.r_[T[:3, 3], 1., 0., 0., 0.].tolist()])
    metadata = root / 'info/source_geometry.json'
    geometry = dict(source_triangle_vertices=len(mesh.vertices), source_triangles=len(mesh.faces),
                    lossless_OBJ_vertex_max_error_m=conversion_error, source_visual_bounds_m=visual_bounds.tolist(),
                    collision_union_bounds_m=collision_bounds.tolist(),
                    collision_vs_visual_AABB_delta_m=(collision_bounds - visual_bounds).tolist(),
                    collision_vs_visual_max_AABB_delta_m=float(np.abs(collision_bounds - visual_bounds).max()),
                    visual_minimum_world_z_m=float(visual_world[:, 2].min()),
                    collision_minimum_world_z_m=float(collision_world[:, 2].min()),
                    geometry_boundary='AABB discrepancy is measured, not a Hausdorff bound. DGN2 collision parts are an explicit approximate proxy, not verified original AnyDex CAD. Native20um import audit is distinct from this approximation.',
                    physics_mass_kg=.1, physics_friction=.6,
                    physical_parameters_source='Fixed Boya simulation approximation, not paper real object properties',
                    input_sha256=all_source)
    save(metadata, geometry)
    record = dict(id=oid, object_id=oid, pilot_id=oid, graspnet_object_id=row['graspnet_object_id'],
        split='train', source_split='graspnet_train_S40', source_family='GraspNet_DGN2_derived',
        object_root=str(root), mesh_path=str(output_mesh), mesh_sha256=sha(output_mesh),
        mesh_group_sha256=sha(source_mesh), urdf_path=str(urdf), urdf_sha256=sha(urdf),
        pose_path=str(pose), pose_sha256=sha(pose), metadata_path=str(metadata), metadata_sha256=sha(metadata),
        scale=1., geometry_scaled_on_disk=False, object_mass_kg=.1, object_friction=.6,
        original_AnyDex_CAD_identity_verified=False, source_visual_mesh=str(source_mesh),
        source_visual_mesh_sha256=sha(source_mesh), source_collision_urdf=str(source_urdf),
        source_collision_urdf_sha256=sha(source_urdf), source_inventory_sha256=inventory_sha,
        source_input_sha256=all_source, convex_parts=list(parts.values()),
        selected_source_pose_index=0, source_pose_quaternion_order='wxyz',
        world_T_object=T.tolist(), world_pose_xyzw=np.r_[T[:3, 3], 0., 0., 0., 1.].tolist(),
        world_AABB_m=[visual_world.min(0).tolist(), visual_world.max(0).tolist()],
        world_extents_m=np.ptp(visual_world, axis=0).tolist(), geometry_approximation=geometry,
        source_placement_rule='Native model axes, visual AABB XY center(-.75,.10), minimum of original visual and convex union Z at table0. No outcome/class/learned score used. Static support only, no dynamic stability certificate.',
        preflight_status='ready', preflight_issues=[],
        observation_json=str(run_dir/'observations'/oid/'observation.json'),
        representation_dir=str(run_dir/'representations'/oid),
        grasp_outcome_consulted=False, prior_grasp_pose_consulted=False,
        original_scene_pose_reproduced=False)
    save(root/'PROVENANCE.json', record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--asset-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4, choices=range(1, 9))
    a = p.parse_args()
    a.inventory, a.asset_root, a.output_dir, a.run_dir = [v.resolve() for v in (a.inventory, a.asset_root, a.output_dir, a.run_dir)]
    if a.asset_root.exists() or a.output_dir.exists():
        raise ValueError('New output and asset directories required; preserve every old run')
    start = time.monotonic()
    rows = validate_inventory(json.loads(a.inventory.read_text()))
    digest = sha(a.inventory)
    a.output_dir.mkdir(parents=True)
    a.asset_root.mkdir(parents=True)
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            records = list(pool.map(lambda r: package(r, a.asset_root, a.run_dir, digest), rows))
        camera = dict(width=1280, height=720, fx=640., fy=640., eye_world_m=[-.75,-.20,.50],
                      target_world_m=[-.75,.10,.06], up_world=[0.,0.,1.], depth_range_m=[.20,.65],
                      focal_length=24., horizontal_aperture=48., clipping_range_m=[.1,1.])
        manifest = dict(schema='anydex_boya_s40_scene_manifest_v1', source_inventory=str(a.inventory),
            source_inventory_sha256=digest, source_active_list_sha256=digest,
            source_active_list_boundary='Compatibility key binds S40 inventory; never DGN4330 active list',
            source_dataset_root=str(a.asset_root), active_object_count=40, scale=1.,
            objects=records, camera=camera,
            split_boundary='All original S40 identities reserved for adaptation training. No test or fake10-validation mesh list; EGAD is separate and unused.',
            scene_boundary='Isolated synthetic native-axis scenes, approximate collision/mass/friction; not paper real cluttered scenes or original hand task. Source asset identities remain GraspNet-derived copies.',
            original_AnyDex_CAD_identity_verified=False, original_scene_pose_reproduced=False,
            physics_steps=0, grasp_attempts=0, training_steps=0)
        save(a.output_dir/'objects.json', manifest)
        summary = dict(schema='anydex_boya_s40_scene_preparation_v1', status='complete',
            objects=40, train_objects=40, validation_objects=0, test_objects_used=0, scale=1.,
            source_inventory_sha256=digest, scene_manifest=str(a.output_dir/'objects.json'),
            scene_manifest_sha256=sha(a.output_dir/'objects.json'), asset_root=str(a.asset_root),
            max_source_to_OBJ_vertex_error_m=max(r['geometry_approximation']['lossless_OBJ_vertex_max_error_m'] for r in records),
            max_collision_AABB_delta_m=max(r['geometry_approximation']['collision_vs_visual_max_AABB_delta_m'] for r in records),
            all_asset_input_files_sha256_bound=True, source_script_sha256=sha(__file__),
            original_AnyDex_CAD_identity_verified=False, original_scene_pose_reproduced=False,
            ignored_DGN5k_scope_rules=True, new_physics_steps=0, new_training_steps=0,
            elapsed_s=time.monotonic()-start)
        save(a.output_dir/'SCENE_PREPARATION.json', summary)
        print(json.dumps(summary, indent=2))
    except BaseException as e:
        save(a.output_dir/'FAILED.json', dict(error=repr(e), elapsed_s=time.monotonic()-start,
            physics_steps=0, training_steps=0, partial_assets_retained=True))
        raise


if __name__ == '__main__':
    main()
