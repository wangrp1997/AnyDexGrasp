"""Frozen multi-object development scene: native settling and one RTX snapshot.

Every object remains dynamic and receives its own declared weight. Observation
and execution are explicitly separate cold starts at measured poses, never a
claim of continuous same-process collection or a paper hardware benchmark.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CANON = Path('/home/wangrenpeng/CanonDex')
DT = .004
STEPS = 750
OBJECTS = [('graspnet_000', [-.75, .10]), ('graspnet_002', [-.63, .10]),
           ('graspnet_005', [-.89, .10])]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def verify(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError('Frozen scene dependency changed: ' + path)


def transform(value):
    a = np.asarray(value, float)
    if (a.shape != (4, 4) or not np.isfinite(a).all()
            or not np.allclose(a[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(a[:3, :3]), 1, atol=1e-8)):
        raise ValueError('Proper finite SE3 required')
    return a


def bin_boxes(spec):
    """World boxes for the axis-aligned bin, shared with source ray checks."""
    T = transform(spec['world_T_bin'])
    if not np.allclose(T[:3, :3], np.eye(3), atol=1e-12, rtol=0):
        raise ValueError('This frozen development bin must be axis aligned')
    h = np.asarray(spec['inner_half_extents_m'], float)
    wall, floor = spec['wall_thickness_m'], spec['floor_thickness_m']
    if h.shape != (3,) or not np.isfinite([*h, wall, floor]).all() or min(*h, wall, floor) <= 0:
        raise ValueError('Positive finite bin dimensions required')
    x, y, z = h
    rows = [('floor', [0, 0, -z-floor/2], [2*x+2*wall, 2*y+2*wall, floor])]
    for sign in [-1, 1]:
        side = 'minus' if sign == -1 else 'plus'
        rows.append((f'x_{side}', [sign*(x+wall/2), 0, 0], [wall, 2*y+2*wall, 2*z]))
        rows.append((f'y_{side}', [0, sign*(y+wall/2), 0], [2*x, wall, 2*z]))
    return [(name, (T[:3, 3]+offset).tolist(), size) for name, offset, size in rows]


def validate(scene):
    if scene['schema'] != 'anydex_boya_author_scene_v1':
        raise ValueError('Independent author scene schema required')
    rows = scene['objects']
    ids = [r['object_id'] for r in rows]
    if len(rows) < 2 or len(set(ids)) != len(ids) or scene['target_object_id'] not in ids:
        raise ValueError('Distinct co-located objects and compatibility primary required')
    for row in rows:
        if Path(row['object_id']).name != row['object_id'] or row['object_id'] in ('.', '..'):
            raise ValueError('Path-safe object identity required')
        transform(row['world_T_object'])
        if (row['scale'] != 1 or row['mass_kg'] != .1 or row['friction'] != .6
                or sha(row['source_mesh']) != row['source_mesh_sha256']):
            raise ValueError('Frozen source geometry and native .1kg/.6 profile required')
        if row.get('collision_refinement') and sha(row['collision_refinement']) != row['collision_refinement_sha256']:
            raise ValueError('Frozen collision refinement changed')
    bin_boxes(scene['bin'])
    verify(scene['input_sha256'])
    verify(scene['source_sha256'])


def build(source_manifest, native_summary, backend_path, output_dir):
    source = read(source_manifest)
    native = {r['object_id']: r for r in read(native_summary)['results']}
    backend = read(backend_path)
    byid = {r['object_id']: r for r in source['objects']}
    inputs = {str(p.resolve()): sha(p) for p in [source_manifest, native_summary, backend_path]}
    rows = []
    for identity, xy in OBJECTS:
        old = byid[identity]
        T = transform(old['world_T_object']).copy()
        T[:2, 3] = xy
        row = dict(object_id=identity, object_root=old['object_root'], scale=1.,
                   world_T_object=T.tolist(), mass_kg=.1, friction=.6,
                   source_mesh=old['mesh_path'], source_mesh_sha256=old['mesh_sha256'],
                   collision_refinement=native[identity].get('repair_path'),
                   collision_refinement_sha256=native[identity].get('repair_sha256'))
        rows.append(row)
        for p in Path(row['object_root']).rglob('*'):
            if p.is_file():
                inputs[str(p.resolve())] = sha(p)
        for name in native[identity]['artifacts']:
            inputs[name] = sha(name)
    inputs[backend['physics_config']] = sha(backend['physics_config'])
    sources = {str(Path(__file__).resolve()): sha(__file__)}
    for folder, pattern in [(CANON/'canondex', '**/*.py'), (CANON/'scripts/research', '*.py')]:
        for path in folder.glob(pattern):
            sources[str(path)] = sha(path)
    for name in ['settle_s40_scenes.py', 'render_s40_observations.py', 'author_transport.py']:
        path = Path(__file__).with_name(name)
        sources[str(path)] = sha(path)
    Tbin = np.eye(4)
    Tbin[:3, 3] = [-.40, -.25, .10]
    scene = dict(schema='anydex_boya_author_scene_v1', scene_id='development_three_20261114',
                 target_object_id=OBJECTS[0][0], objects=rows,
                 target_semantics='Compatibility primary only; transport may deliver any original tabletop body',
                 bin=dict(world_T_bin=Tbin.tolist(), inner_half_extents_m=[.15, .15, .10],
                          floor_thickness_m=.01, wall_thickness_m=.01, friction=.6),
                 camera=source['camera'], physics_config=backend['physics_config'],
                 dt_s=DT, initialization_steps=STEPS, seed=20261114,
                 input_sha256=inputs, source_sha256=sources,
                 scope='Three previously used S40 development objects; one fixed co-located placement, not random-scene coverage or external generalization',
                 state_boundary='Initialize once, settle all bodies together, capture without integration; execution cold-recreates measured poses with explicit zero twist',
                 training_started=False, grasp_attempts=0)
    validate(scene)
    output_dir.mkdir(parents=True, exist_ok=False)
    save(output_dir/'scene.json', scene)
    return scene


def capture(scene_path, output_dir):
    scene = read(scene_path)
    validate(scene)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    progress = dict(status='starting', scene_sha256=sha(scene_path), physics_ticks=0,
                    initialization_object_steps=0, grasp_attempts=0, CGR_forwards=0)
    save(output_dir/'PROGRESS.json', progress)
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    os.environ.setdefault('CANONDEX_COLLISION_PARTITION_CACHE', str(ROOT/'outputs/boya_collision_partition_cache'))
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, enable_cameras=True, device='cpu',
                           visualizer=None, kit_args='--/log/file=/dev/null')
    exit_code = 1
    try:
        sys.path[:0] = [str(CANON), str(CANON/'scripts/research')]
        import torch
        import warp as wp
        import trimesh
        import scipy.io
        from scipy.spatial.transform import Rotation
        from PIL import Image
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import Camera, CameraCfg
        from pxr import Usd, UsdGeom, UsdPhysics
        from carb.settings import get_settings
        from run_template_arm_positive import object_asset
        from canondex.grasp.physics import load_physics_config
        from canondex.scene.floor import spawn_floor_and_light
        from canondex.scene import grasp_object
        from canondex.grasp.arm_resident_scene import restore_declared_object_state
        from canondex.scene.contact_readback import contact_data_to_host
        from author_transport import decode_contacts
        from template_gripper_native_audit import audit_scene
        from settle_s40_scenes import namespaced_spawn, window_metrics
        from render_s40_observations import look_at_ros, ray_diagnostic
        from types import SimpleNamespace
        profile = load_physics_config(scene['physics_config'])
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device='cpu', dt=DT,
                gravity=(0., 0., 0.), physics=profile.scene_cfg(), use_fabric=False, visualizer_cfgs=[]))
        # Lab's context constructor sets this flag. Enable direct native
        # contact views after construction, before reset and every physics tick.
        get_settings().set_bool('/physics/disableContactProcessing', False)
        spawn_floor_and_light()
        UsdGeom.Xform.Define(sim.stage, '/World/Robot')
        UsdGeom.Xform.Define(sim.stage, '/World/AuthorScene')
        bodies, roots, initial = [], [], []
        for row in scene['objects']:
            asset = object_asset(SimpleNamespace(object_root=Path(row['object_root'])),
                                 np.asarray(row['world_T_object']), row['scale'])
            path = '/World/AuthorScene/' + row['object_id']
            body = namespaced_spawn(grasp_object, path)(sim, asset, physics_profile=profile,
                    collision_refinement=read(row['collision_refinement']) if row['collision_refinement'] else None)
            bodies.append(body)
            roots.append(path)
            initial.append(asset.pose_xyzw)
        sim_utils.bind_physics_material('/World/ground', roots[0]+'Material')
        for name, center, size in bin_boxes(scene['bin']):
            cfg = sim_utils.CuboidCfg(size=tuple(size), collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.6, dynamic_friction=.6, restitution=0.),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.3, .4, .6), roughness=.5))
            cfg.func('/World/AuthorBin/'+name, cfg, translation=tuple(center))
        cam = scene['camera']
        T = look_at_ros(cam['eye_world_m'], cam['target_world_m'], cam['up_world'])
        camera = Camera(CameraCfg(prim_path='/World/observation_camera', width=cam['width'], height=cam['height'],
                update_period=0, data_types=['rgb', 'distance_to_image_plane'], update_latest_camera_pose=True,
                spawn=sim_utils.PinholeCameraCfg(focal_length=cam['focal_length'], horizontal_aperture=cam['horizontal_aperture'],
                        clipping_range=tuple(cam['clipping_range_m']))))
        sim.reset()
        camera.set_world_poses(torch.tensor([T[:3, 3].tolist()], dtype=torch.float32),
                               torch.tensor([Rotation.from_matrix(T[:3, :3]).as_quat().tolist()], dtype=torch.float32), convention='ros')
        material_readback = []
        for obj, pose in zip(bodies, initial):
            obj.update(DT)
            restore_declared_object_state(sim, obj, pose)
            values = obj.root_view.get_material_properties().numpy().copy()
            values[..., :2], values[..., 2] = .6, 0.
            indices = wp.array(np.arange(len(values), dtype=np.int32), dtype=wp.int32, device='cpu')
            obj.root_view.set_material_properties(wp.array(values, dtype=wp.float32, device='cpu'), indices)
            actual = obj.root_view.get_material_properties().numpy()
            mass = float(obj.root_view.get_masses().numpy().ravel()[0])
            if not np.allclose(actual, values, atol=1e-7, rtol=0) or mass != float(np.float32(.1)):
                raise ValueError('Native multi-object mass/material mismatch')
            material_readback.append(dict(mass_kg=mass, friction=float(actual[..., 0].min())))
            obj.permanent_wrench_composer.reset()
            obj.permanent_wrench_composer.add_forces_and_torques_index(
                    forces=torch.tensor([[[0., 0., -.98]]], dtype=torch.float32, device=obj.device), is_global=True)
        (output_dir/'native').mkdir()
        native = audit_scene(sim, output_dir/'native', ['all'], roots=tuple(roots), fast=True)
        if not native['within_20um']:
            raise ValueError('Native20um object import gate failed')
        static_paths = [str(prim.GetPath()) for parent in ['/World/ground', '/World/AuthorBin']
                        for prim in Usd.PrimRange(sim.stage.GetPrimAtPath(parent))
                        if prim.HasAPI(UsdPhysics.CollisionAPI)]
        save(output_dir/'CONTACT_SENSOR_CONFIG.json', dict(static_collision_paths=static_paths,
                disable_contact_processing=get_settings().get_as_bool('/physics/disableContactProcessing')))
        if not static_paths or get_settings().get_as_bool('/physics/disableContactProcessing'):
            raise ValueError('Explicit enabled native contact processing and collision paths required')
        targets = roots+static_paths
        capacity = 262144
        contacts = sim.physics_sim_view.create_rigid_contact_view(roots,
                filter_patterns=[targets]*len(roots), max_contact_data_count=capacity)
        poses, velocities, contact_rows = [], [], []
        def observe():
            poses.append(np.stack([o.data.root_link_pose_w.torch[0].cpu().numpy().copy() for o in bodies]))
            velocities.append(np.stack([o.data.root_link_vel_w.torch[0].cpu().numpy().copy() for o in bodies]))
        observe()
        for step in range(STEPS):
            for obj in bodies:
                obj.write_data_to_sim()
            before = sim.get_physics_step_count()
            sim.step(render=False)
            progress['physics_ticks'] += 1
            progress['initialization_object_steps'] += len(bodies)
            if sim.get_physics_step_count() != before+1:
                raise RuntimeError('Unexpected multi-object physics clock')
            for obj in bodies:
                obj.update(DT)
            observe()
            decoded = decode_contacts(roots, targets,
                    contact_data_to_host(contacts, DT, capacity, compact=True), capacity)
            contact_rows.append(dict(step=step+1, contacts=decoded))
        np.savez_compressed(output_dir/'initialization_trace.npz', pose_xyzw=np.asarray(poses), velocity=np.asarray(velocities))
        save(output_dir/'initialization_contacts.json', contact_rows)
        stability = [window_metrics(np.asarray(poses)[:, i], np.asarray(velocities)[:, i]) for i in range(len(bodies))]
        if not all(r['stable'] for r in stability):
            raise ValueError('Fixed 750-step common-scene settling did not meet frozen stationarity gate')
        floor_paths = set(p for p in static_paths if p.startswith('/World/ground'))
        final_contacts = contact_rows[-250:]
        contact_evidence = {path:dict(
                final_window_supported_ticks=sum(any(c['bodies'][0] == path and c['bodies'][1] in floor_paths
                    and c['normal_force_n'] > .001 for c in row['contacts']) for row in final_contacts),
                contact_rows=sum(sum(c['bodies'][0] == path for c in row['contacts']) for row in contact_rows))
                for path in roots}
        save(output_dir/'CONTACT_SENSOR_CHECK.json', dict(contact_processing_enabled=True,
                rigid_body_paths=roots, target_collision_paths=targets, evidence=contact_evidence))
        if any(r['final_window_supported_ticks'] == 0 for r in contact_evidence.values()):
            raise ValueError('Native contact stream did not observe support for every settled dynamic body')
        observed = copy.deepcopy(scene)
        world_meshes = []
        for i, row in enumerate(observed['objects']):
            pose = poses[-1][i]
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
            matrix[:3, 3] = pose[:3]
            row.update(world_T_object=matrix.tolist(), observed_pose_xyzw=pose.tolist(),
                       observed_twist=velocities[-1][i].tolist(), initialization_stability=stability[i])
            mesh = trimesh.load_mesh(row['source_mesh'], process=False)
            mesh.apply_transform(matrix)
            world_meshes.append(mesh)
        for _, center, size in bin_boxes(scene['bin']):
            mesh = trimesh.creation.box(extents=size)
            mesh.apply_translation(center)
            world_meshes.append(mesh)
        observed.update(source_scene_path=str(scene_path), source_scene_sha256=sha(scene_path),
                        initial_state_boundary='Measured common-scene pose; independent execution cold recreation at this pose and zero twist',
                        initialization_physics_ticks=STEPS, initialization_object_steps=STEPS*len(bodies))
        save(output_dir/'observed_scene.json', observed)
        count_before = sim.get_physics_step_count()
        for _ in range(16):
            sim.render()
            camera.update(DT, force_recompute=True)
        if sim.get_physics_step_count() != count_before:
            raise RuntimeError('Observation rendering integrated physics')
        def cpu(value):
            return (value.torch if hasattr(value, 'torch') else value).detach().cpu().numpy()
        rgb = cpu(camera.data.output['rgb'])[0, :, :, :3]
        depth = cpu(camera.data.output['distance_to_image_plane'])[0].squeeze()
        np.savez_compressed(output_dir/'native_rgbd.npz', rgb=rgb, depth_m=depth)
        K = cpu(camera.data.intrinsic_matrices)[0]
        expected_K = [[cam['fx'], 0, cam['width']/2], [0, cam['fy'], cam['height']/2], [0, 0, 1]]
        if (rgb.shape != (cam['height'], cam['width'], 3) or depth.shape != (cam['height'], cam['width'])
                or not np.allclose(K, expected_K, atol=.002, rtol=0)):
            raise ValueError('Native RGBD shape/intrinsics changed')
        actual = np.eye(4)
        actual[:3, :3] = Rotation.from_quat(cpu(camera.data.quat_w_ros)[0]).as_matrix()
        actual[:3, 3] = cpu(camera.data.pos_w)[0]
        if np.max(abs(actual-T)) > 2e-6:
            raise ValueError('Native camera pose differs from declared look-at')
        ray = ray_diagnostic(trimesh.util.concatenate(world_meshes), depth, K, actual)
        save(output_dir/'render_geometry_check.json', ray)
        if not ray['diagnostic_pass']:
            raise ValueError('Multi-object source first-hit ray check failed')
        near, far = cam['depth_range_m']
        mask = np.isfinite(depth) & (depth >= near) & (depth <= far)
        Image.fromarray(rgb.astype(np.uint8)).save(output_dir/'color.png')
        Image.fromarray(np.where(mask, np.rint(depth*1000), 0).astype(np.uint16)).save(output_dir/'depth.png')
        scipy.io.savemat(output_dir/'meta.mat', dict(intrinsic_matrix=K, factor_depth=np.array([[1000.]])))
        observation = dict(schema='anydex_boya_author_scene_observation_v1', status='rendered',
                simulator='Isaac Sim / PhysX', renderer='Native RTX', T_WC=actual.tolist(),
                intrinsic_matrix=K.tolist(), depth_factor=1000., object_pose_as_network_input=False,
                scene_json=str(output_dir/'observed_scene.json'), scene_sha256=sha(output_dir/'observed_scene.json'),
                scene_id=scene['scene_id'], object_ids=[r['object_id'] for r in scene['objects']],
                files={n:dict(path=str(output_dir/n), sha256=sha(output_dir/n)) for n in ['color.png', 'depth.png', 'meta.mat']},
                valid_depth_pixels=int(mask.sum()), grasp_attempts=0, render_physics_steps=0,
                initialization_physics_ticks=STEPS, initialization_object_steps=STEPS*len(bodies),
                state_boundary=observed['initial_state_boundary'])
        save(output_dir/'observation.json', observation)
        validate(scene)
        progress.update(status='complete', wall_s=time.monotonic()-started, stability=stability,
                native_within_20um=True, native_hulls=native['hulls'], materials=material_readback,
                observation_json=str(output_dir/'observation.json'), observation_sha256=sha(output_dir/'observation.json'),
                geometry_check_sha256=sha(output_dir/'render_geometry_check.json'), render_physics_steps=0)
        progress['initialization_contact_sensor_check_sha256'] = sha(output_dir/'CONTACT_SENSOR_CHECK.json')
        save(output_dir/'RESULTS.json', progress)
        exit_code = 0
    except BaseException:
        progress.update(status='failed', error=traceback.format_exc(), wall_s=time.monotonic()-started)
        save(output_dir/'ERROR.json', progress)
        raise
    finally:
        save(output_dir/'PROGRESS.json', progress)
        launcher.app.close(exit_code=exit_code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build')
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--native-summary', type=Path, required=True)
    p.add_argument('--backend', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p = sub.add_parser('capture')
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.source_manifest.resolve(), args.native_summary.resolve(), args.backend.resolve(), args.output_dir.resolve())
    else:
        capture(args.scene.resolve(), args.output_dir.resolve())


if __name__ == '__main__':
    main()
