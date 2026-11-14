"""Independent read-only scene/trace/source-mesh/native-hull/RGBD audit.

No Isaac, controller, CGR model or simulator is imported. Empty contact streams
remain an explicit failed evidence gate even when snapshot geometry is valid.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import scipy.io
from scipy.spatial import ConvexHull
import trimesh


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rotation(quaternion):
    q = np.asarray(quaternion, float)
    if q.shape[-1] != 4 or not np.isfinite(q).all(): raise ValueError('Finite XYZW quaternion required')
    norm = np.linalg.norm(q, axis=-1)
    if np.max(abs(norm-1)) > 1e-4: raise ValueError('Native quaternion is not unit length')
    x, y, z, w = np.moveaxis(q/norm[..., None], -1, 0)
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=-1).reshape(q.shape[:-1]+(3, 3))


def stability(pose, velocity):
    p, v = np.asarray(pose, float)[-250:], np.asarray(velocity, float)[-250:]
    q = p[:, 3:]/np.linalg.norm(p[:, 3:], axis=1)[:, None]
    result = dict(max_linear_speed_m_s=float(np.linalg.norm(v[:, :3], axis=1).max()),
        max_angular_speed_rad_s=float(np.linalg.norm(v[:, 3:], axis=1).max()),
        max_window_translation_m=float(np.linalg.norm(p[:, :3]-p[-1, :3], axis=1).max()),
        max_window_rotation_rad=float((2*np.arccos(np.clip(abs(q@q[-1]), 0, 1))).max()))
    result['stable'] = all(result[k] <= threshold for k, threshold in zip(result, [.02, .5, .0002, .005]))
    return result


def bin_geometry(spec):
    frame = np.asarray(spec['world_T_bin'], float)
    if not np.array_equal(frame[:3, :3], np.eye(3)): raise ValueError('Current capture bin is axis aligned')
    x, y, z = spec['inner_half_extents_m']; wall = spec['wall_thickness_m']; floor = spec['floor_thickness_m']
    boxes = [([0, 0, -z-floor/2], [2*x+2*wall, 2*y+2*wall, floor]),
             ([-x-wall/2, 0, 0], [wall, 2*y+2*wall, 2*z]),
             ([x+wall/2, 0, 0], [wall, 2*y+2*wall, 2*z]),
             ([0, -y-wall/2, 0], [2*x, wall, 2*z]),
             ([0, y+wall/2, 0], [2*x, wall, 2*z])]
    triangles = []
    for center, size in boxes:
        mesh = trimesh.creation.box(extents=size)
        triangles.append(mesh.triangles+frame[:3, 3]+center)
    return np.concatenate(triangles), [dict(center_world_m=(frame[:3, 3]+center).tolist(), extents_m=size) for center, size in boxes]


def first_depth(triangles, origin, direction):
    """Moller-Trumbore first positive intersection; ray camera-Z equals one."""
    a, b, c = np.moveaxis(triangles, 1, 0)
    e1, e2 = b-a, c-a
    h = np.cross(direction, e2); determinant = np.einsum('ij,ij->i', e1, h)
    valid = abs(determinant) > 1e-14
    inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=valid)
    s = origin-a; u = np.einsum('ij,ij->i', s, h)*inverse
    q = np.cross(s, e1); v = q@direction*inverse
    t = np.einsum('ij,ij->i', e2, q)*inverse
    valid &= (u >= -1e-10)&(v >= -1e-10)&(u+v <= 1+1e-10)&(t > 1e-10)
    mesh_depth = float(t[valid].min()) if valid.any() else np.inf
    ground_depth = -origin[2]/direction[2] if direction[2] < -1e-9 else np.inf
    return (min(mesh_depth, ground_depth), bool(mesh_depth < ground_depth),
            bool(np.isfinite(mesh_depth) and abs(mesh_depth-ground_depth) <= 1e-9))


def convex_distance(points, vertices):
    # Work at unit scale so trimesh's fixed small-triangle tolerance cannot
    # collapse millimetre hull facets into segments and inflate distances.
    center = vertices.mean(axis=0); scale = float(np.ptp(vertices, axis=0).max())
    if scale <= 0: raise ValueError('Degenerate cooked/source convex hull')
    query, local = (points-center)/scale, (vertices-center)/scale
    hull = ConvexHull(local)
    signed = query@hull.equations[:, :3].T+hull.equations[:, 3]
    outside = signed.max(axis=1) > 1e-12
    if not outside.any(): return 0.
    mesh = trimesh.Trimesh(vertices=local, faces=hull.simplices, process=False)
    return float(trimesh.proximity.closest_point_naive(mesh, query[outside])[1].max()*scale)


def audit(capture, output, frozen_source=None, frozen_dependencies=None):
    began = time.monotonic()
    capture, output = Path(capture).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError('Fresh independent audit output required')
    receipt = read(capture/'RESULTS.json'); observed = read(capture/'observed_scene.json')
    observation = read(capture/'observation.json'); scene_path = Path(observed['source_scene_path']); scene = read(scene_path)
    if receipt['status'] != 'complete' or sha(scene_path) != receipt['scene_sha256'] or sha(scene_path) != observed['source_scene_sha256']:
        raise ValueError('Original successful scene/receipt binding changed')
    if (sha(capture/'observed_scene.json') != observation['scene_sha256']
            or sha(capture/'observation.json') != receipt['observation_sha256']
            or sha(capture/'render_geometry_check.json') != receipt['geometry_check_sha256']):
        raise ValueError('Observed scene / RGBD / ray receipt hashes disagree')
    source_resolution, all_bindings = {}, {}
    archives = {str(Path(p).resolve()): Path(q).resolve() for p, q in (frozen_dependencies or {}).items()}
    if frozen_source:
        source_names = [p for p in scene['source_sha256'] if p.endswith('/author_scene.py')]
        if len(source_names) != 1: raise ValueError('Expected one frozen capture source')
        archives[source_names[0]] = Path(frozen_source).resolve()
    unknown_archives = set(archives)-set(scene['source_sha256'])
    if unknown_archives: raise ValueError('Archive mappings must refer to frozen source paths: '+str(unknown_archives))
    for label in ('input_sha256', 'source_sha256'):
        for path, expected in scene[label].items():
            actual_path = Path(path)
            if str(actual_path) in archives:
                actual_path = archives[str(actual_path)]
            if sha(actual_path) != expected:
                raise ValueError('Frozen dependency changed without matching archived source: '+path)
            source_resolution[path] = str(actual_path); all_bindings[str(actual_path)] = expected
        if observed[label] != scene[label]: raise ValueError('Observation rewrote source scene bindings')
    capture_source = Path(source_resolution[next(p for p in scene['source_sha256'] if p.endswith('/author_scene.py'))])
    tree = ast.parse(capture_source.read_text())
    capture_function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'capture')
    wrench_calls = [n for n in ast.walk(capture_function) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_forces_and_torques_index']
    if len(wrench_calls) != 1 or '-0.98' not in ast.unparse(wrench_calls[0]): raise ValueError('Frozen per-body weight code changed')
    helper_imported = any(isinstance(n, ast.ImportFrom) and n.module == 'author_transport' for n in ast.walk(capture_function))
    helper_frozen = any(p.endswith('/author_transport.py') for p in scene['source_sha256'])
    with np.load(capture/'initialization_trace.npz', allow_pickle=False) as data:
        poses, velocities = data['pose_xyzw'], data['velocity']
    steps, count = receipt['physics_ticks'], len(scene['objects'])
    if (steps != 750 or poses.shape != (steps+1, count, 7) or velocities.shape != (steps+1, count, 6)
            or not np.isfinite(poses).all() or not np.isfinite(velocities).all()
            or receipt['initialization_object_steps'] != count*steps or receipt['render_physics_steps'] != 0
            or receipt['grasp_attempts'] != 0 or receipt['CGR_forwards'] != 0):
        raise ValueError('Shared clock / finite initialization trace / zero-grasp boundary mismatch')
    object_rows, mesh_triangles = [], []
    for i, row in enumerate(scene['objects']):
        final = observed['objects'][i]
        if row['object_id'] != final['object_id']: raise ValueError('Observed body order changed')
        for key, value in row.items():
            if key != 'world_T_object' and final[key] != value: raise ValueError('Observed body changed physical asset parameter: '+key)
        initial = np.asarray(row['world_T_object']); measured = np.eye(4)
        measured[:3, :3] = rotation(poses[-1, i, 3:]); measured[:3, 3] = poses[-1, i, :3]
        initial_position_error = float(abs(poses[0, i, :3]-initial[:3, 3]).max())
        initial_rotation_error = float(abs(rotation(poses[0, i, 3:])-initial[:3, :3]).max())
        if initial_position_error > 1e-7 or initial_rotation_error > 1e-6:
            raise ValueError('Recorded initial body pose differs from declared source frame')
        if (not np.array_equal(poses[-1, i], final['observed_pose_xyzw'])
                or not np.array_equal(velocities[-1, i], final['observed_twist'])
                or np.max(abs(measured-final['world_T_object'])) > 1e-12):
            raise ValueError('Captured observation does not bind final measured body pose/twist')
        recomputed_stability = stability(poses[:, i], velocities[:, i])
        for saved in (receipt['stability'][i], final['initialization_stability']):
            if any(abs(recomputed_stability[k]-saved[k]) > 1e-12 for k in recomputed_stability):
                raise ValueError('Independent stationarity differs from saved result')
        if not recomputed_stability['stable']: raise ValueError('Saved body does not pass original stationarity limits')
        native_material = receipt['materials'][i]
        if native_material['mass_kg'] != float(np.float32(.1)) or native_material['friction'] != float(np.float32(.6)):
            raise ValueError('Saved native material/mass readback differs from frozen common profile')
        if sha(row['source_mesh']) != row['source_mesh_sha256']: raise ValueError('Source object triangles changed')
        mesh = trimesh.load_mesh(row['source_mesh'], process=False)
        vertices = np.asarray(mesh.vertices)*row['scale']
        R = rotation(poses[:, i, 3:]); bottom = np.empty(len(poses))
        for start in range(0, len(poses), 64):
            bottom[start:start+64] = (vertices@R[start:start+64, 2, :].T+poses[start:start+64, i, 2]).min(axis=0)
        world_vertices = vertices@measured[:3, :3].T+measured[:3, 3]
        mesh_triangles.append(world_vertices[np.asarray(mesh.faces)])
        object_rows.append(dict(object_id=row['object_id'], source_vertices=len(vertices),
            initial_position_error_m=initial_position_error, initial_rotation_matrix_error=initial_rotation_error,
            observed_pose_matrix_error=float(abs(measured-final['world_T_object']).max()),
            independent_stability=recomputed_stability, mass_and_friction_readback=native_material,
            source_bottom_initial_m=float(bottom[0]), source_bottom_final_m=float(bottom[-1]),
            source_bottom_min_m=float(bottom.min()), source_bottom_max_m=float(bottom.max()),
            observed_source_world_bounds_m=[world_vertices.min(axis=0).tolist(), world_vertices.max(axis=0).tolist()]))
    native = read(capture/'native/native_audit.json'); maximum_source = maximum_cooked = 0.
    owners = {}; independently_checked = 0
    with np.load(capture/'native/native_hulls.npz', allow_pickle=False) as data:
        for i, row in enumerate(native['results']):
            source = data[f'hull_{i:04d}_source']; cooked = data[f'hull_{i:04d}_cooked']
            if not np.isfinite(source).all() or not np.isfinite(cooked).all(): raise ValueError('Nonfinite source/cooked hull')
            s2c, c2s = convex_distance(source, cooked), convex_distance(cooked, source)
            if s2c > row['source_outside_cooked_bound_m']+1e-9 or c2s > row['cooked_outside_source_bound_m']+1e-9:
                raise ValueError('Saved native20um bounds underestimate independent closest-triangle distance')
            maximum_source = max(maximum_source, s2c); maximum_cooked = max(maximum_cooked, c2s)
            owners[row['owner']] = owners.get(row['owner'], 0)+1; independently_checked += 1
    if (independently_checked != native['hulls'] or native['hulls'] != receipt['native_hulls']
            or not native['within_20um'] or max(maximum_source, maximum_cooked) > 20e-6):
        raise ValueError('Native geometry import preservation exceeds original 20um gate')
    contacts = read(capture/'initialization_contacts.json')
    if len(contacts) != steps or [r['step'] for r in contacts] != list(range(1, steps+1)):
        raise ValueError('Contact trace does not cover the shared initialization clock')
    full_stream = 'contacts' in contacts[0]
    supported = {}; totals = {}; contact_count = 0
    for row in object_rows:
        path = '/World/AuthorScene/'+row['object_id']; final_window = contacts[-250:]
        if full_stream:
            supported[path] = sum(any(c['bodies'][0] == path and c['bodies'][1].startswith('/World/ground') and c['normal_force_n'] > .001
                                      for c in tick['contacts']) for tick in final_window)
            totals[path] = sum(sum(c['bodies'][0] == path for c in tick['contacts']) for tick in contacts)
        else:
            supported[path] = sum(any(c.get('body_path') == path and c['normal_force_n'] > .001 for c in tick['ground_contacts']) for tick in final_window)
            totals[path] = sum(sum(c.get('body_path') == path for c in tick['ground_contacts']) for tick in contacts)
    for tick in contacts:
        for c in tick['contacts'] if full_stream else tick['ground_contacts']:
            if not np.isfinite([c['normal_force_n'], c['separation_m'], *c['position_world_m']]).all(): raise ValueError('Nonfinite saved contact')
            contact_count += 1
    usable_contacts = full_stream and all(n > 0 for n in supported.values())
    sensor_path = capture/'CONTACT_SENSOR_CHECK.json'
    if sensor_path.exists():
        sensor = read(sensor_path)
        if not sensor['contact_processing_enabled']: raise ValueError('Native contact processing was not enabled')
        for path in supported:
            if sensor['evidence'][path] != dict(final_window_supported_ticks=supported[path], contact_rows=totals[path]):
                raise ValueError('Independent support/contact counts differ from sensor receipt')
    color = np.asarray(Image.open(capture/'color.png')); depth = np.asarray(Image.open(capture/'depth.png'))
    for item in observation['files'].values():
        if sha(item['path']) != item['sha256']: raise ValueError('Saved RGBD/meta changed')
    K = np.asarray(observation['intrinsic_matrix']); meta = scipy.io.loadmat(capture/'meta.mat')
    if not np.array_equal(K, meta['intrinsic_matrix']) or float(meta['factor_depth'].item()) != observation['depth_factor']:
        raise ValueError('Network camera files differ from saved intrinsic matrix/depth factor')
    camera = scene['camera']; near, far = camera['depth_range_m']
    if color.shape != (camera['height'], camera['width'], 3) or depth.shape != (camera['height'], camera['width']):
        raise ValueError('RGBD image dimensions changed')
    if int(np.count_nonzero(depth)) != observation['valid_depth_pixels']: raise ValueError('Depth validity count changed')
    eye, target, up = [np.asarray(camera[k], float) for k in ('eye_world_m', 'target_world_m', 'up_world')]
    z = (target-eye)/np.linalg.norm(target-eye); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
    expected = np.eye(4); expected[:3, :3] = np.column_stack((x, y, z)); expected[:3, 3] = eye
    T = np.asarray(observation['T_WC'])
    if abs(T-expected).max() > 2e-6: raise ValueError('Independent look-at camera differs from native observed camera')
    bin_triangles, bin_boxes = bin_geometry(scene['bin']); triangles = np.concatenate(mesh_triangles+[bin_triangles])
    ray = read(capture/'render_geometry_check.json'); independent_depth = []; hits = []; ground_ties = []
    for pixel in np.asarray(ray['pixels_xy']):
        ray_camera = np.r_[np.linalg.solve(K[:2, :2], pixel+.5-K[:2, 2]), 1.]
        distance, hit, tie = first_depth(triangles, T[:3, 3], T[:3, :3]@ray_camera)
        independent_depth.append(distance); hits.append(hit); ground_ties.append(tie)
    expected_saved = np.asarray([np.inf if d is None else d for d in ray['expected_source_first_depth_m']])
    hit_differences = np.flatnonzero(np.asarray(hits) != ray['source_first_hit_is_object'])
    if (not np.allclose(independent_depth, expected_saved, atol=1e-9, rtol=0)
            or any(not ground_ties[i] for i in hit_differences)):
        raise ValueError('Independent source-triangle/bin/ground first-depth rays differ')
    pixels = np.asarray(ray['pixels_xy']); decoded_depth = depth[pixels[:, 1], pixels[:, 0]]/observation['depth_factor']
    actual_ray = np.asarray(ray['observed_rtx_depth_m'])
    ray_mask = np.isfinite(actual_ray)&(actual_ray >= near)&(actual_ray <= far)
    if not np.array_equal(decoded_depth > 0, ray_mask): raise ValueError('PNG range mask disagrees with native sampled rays')
    quantization_error = float(abs(decoded_depth[ray_mask]-actual_ray[ray_mask]).max())
    if quantization_error > .0005001: raise ValueError('Saved depth PNG disagrees with native pre-quantization ray values')
    native_ray_difference = abs(np.asarray(independent_depth)-actual_ray)
    raw_depth_paths = list(capture.glob('*depth*.npy'))+list(capture.glob('*depth*.npz'))
    if (capture/'native_rgbd.npz').exists(): raw_depth_paths.append(capture/'native_rgbd.npz')
    raw_depth_checks = []
    for path in raw_depth_paths:
        data = np.load(path, allow_pickle=False)
        values = data if isinstance(data, np.ndarray) else data['depth_m'] if 'depth_m' in data else data[data.files[0]]
        if values.shape != depth.shape: raise ValueError('Raw RTX depth dimensions changed')
        raw_values = values[pixels[:, 1], pixels[:, 0]]
        if not np.array_equal(raw_values, actual_ray.astype(raw_values.dtype)): raise ValueError('Raw RTX depth differs from ray capture')
        mask = np.isfinite(values)&(values >= near)&(values <= far)
        # Preserve the camera float32 quantization and frozen range mask.
        expected_png = np.zeros(values.shape, dtype=np.uint16)
        expected_png[mask] = np.rint(values[mask]*observation['depth_factor']).astype(np.uint16)
        if not np.array_equal(expected_png, depth): raise ValueError('Full saved native depth does not reconstruct network PNG')
        full_color = None
        if not isinstance(data, np.ndarray) and 'rgb' in data:
            full_color = bool(np.array_equal(data['rgb'].astype(np.uint8), color))
            if not full_color: raise ValueError('Full saved native RGB differs from network PNG')
        raw_depth_checks.append(dict(path=str(path), sha256=sha(path), sampled_rays_exact=True,
            full_depth_png_exact=True, full_rgb_png_exact=full_color,
            valid_pixels=int(mask.sum()), image_pixels=int(mask.size)))
    artifact_paths = [p for p in capture.rglob('*') if p.is_file()]
    artifacts = {str(p): sha(p) for p in artifact_paths}
    if any(sha(path) != digest for path, digest in all_bindings.items()): raise ValueError('Frozen dependency changed during independent audit')
    result = dict(schema='anydex_boya_author_scene_independent_audit_v1', status='complete',
        snapshot_validated=True, initialization_contact_evidence_usable=usable_contacts,
        all_required_scene_evidence_validated=bool(usable_contacts and (not helper_imported or helper_frozen)),
        capture_dir=str(capture), original_scene_path=str(scene_path),
        independent_method='NumPy quaternion FK/source vertex bottom, source/cooked convex closest triangle distances, Moller-Trumbore camera rays; no simulation imports',
        physics_ticks_read=steps, object_steps_read=count*steps, objects=object_rows,
        existing_frozen_input_count=len(scene['input_sha256']), existing_frozen_source_count=len(scene['source_sha256']),
        existing_frozen_dependencies_unchanged=True, resolved_archived_sources={p:q for p,q in source_resolution.items() if p != q},
        new_contact_decoder_imported=helper_imported, new_contact_decoder_in_scene_source_lock=helper_frozen,
        weight_evidence=dict(applied_per_body_world_n=[0, 0, -.98], frozen_source_weight_call=ast.unparse(wrench_calls[0]),
            boundary='Frozen applied-force/write/update source and actual material/mass readback; saved trace has no independent native force-composer readback'),
        native_geometry=dict(hulls_independently_checked=independently_checked, owners=owners,
            maximum_source_to_cooked_closest_triangle_distance_m=maximum_source,
            maximum_cooked_to_source_closest_triangle_distance_m=maximum_cooked, native20um_pass=True,
            boundary='Source/cooked import preservation; convex proxies remain approximations and runtime shape poses/contact solver are separate'),
        contacts=dict(schema='full_object_object_floor_bin' if full_stream else 'ground_only',
            total_recorded_contacts=contact_count, final_250_supported_ticks=supported, total_rows_by_body=totals,
            evidence_usable=usable_contacts, boundary='Empty stream does not establish absence of physical contact'),
        camera=dict(pose_matrix_max_error=float(abs(T-expected).max()), rays_independently_checked=len(independent_depth),
            source_object_or_bin_first_hits=int(sum(hits)), native_depth_median_error_m=float(np.median(native_ray_difference)),
            saved_source_object_or_bin_first_hits=ray['source_object_first_hits'],
            ground_coplanar_first_hit_ties=int(sum(ground_ties)),
            equivalent_ground_bin_hit_classification_differences=hit_differences.tolist(),
            native_depth_max_error_m=float(native_ray_difference.max()), png_quantization_max_error_m=quantization_error,
            rays_in_network_depth_range=int(ray_mask.sum()), rays_masked_from_network_png=int((~ray_mask).sum()),
            raw_depth_files_checked=raw_depth_checks, bin_boxes=bin_boxes),
        source_and_input_sha256=all_bindings, capture_artifact_sha256=artifacts,
        audit_source_sha256=sha(__file__), new_physics_steps=0, new_CGR_forwards=0, new_training_updates=0,
        limitations=['Fixed three previously used development objects, not random-scene coverage or external generalization',
            'Observed measured pose is cold-recreated with zero twist for execution; no continuous observation-to-execution process claim',
            'This scene audit contains no grasp, transport, training labels or physical success rate'],
        audit_wall_s=time.monotonic()-began)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status', 'snapshot_validated', 'initialization_contact_evidence_usable',
        'physics_ticks_read', 'object_steps_read', 'native_geometry', 'audit_wall_s')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frozen-source', type=Path)
    parser.add_argument('--frozen-dependency', action='append', default=[], metavar='ORIGINAL=ARCHIVE',
                        help='Use only an explicit SHA-identical archived frozen source dependency')
    args = parser.parse_args()
    dependency_pairs = [value.split('=', 1) for value in args.frozen_dependency]
    if any(len(pair) != 2 for pair in dependency_pairs): parser.error('Frozen dependencies require ORIGINAL=ARCHIVE')
    if len(dict(dependency_pairs)) != len(dependency_pairs): parser.error('Duplicate frozen dependency mapping')
    audit(args.capture, args.output, args.frozen_source, dict(dependency_pairs))
