"""Read-only resident capture/input comparison; never import a simulator/model."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
from scipy.io import loadmat
import trimesh

import author_collection_batch as B
from audit_author_scene import rotation, stability, first_depth


def need(condition, message):
    if not condition: raise ValueError(message)


class Inputs:
    def __init__(self): self.files = {}
    def bind(self, path):
        path = Path(path).resolve(); digest = B.sha(path)
        if str(path) in self.files: need(self.files[str(path)] == digest, 'Input changed during audit')
        self.files[str(path)] = digest
        return path
    def read(self, path): return B.read(self.bind(path))
    def verify(self): B.verify(B.normalized_bindings(self.files))


def capture_audit(capture):
    """One full gzip decode; endpoints use every source vertex/triangle."""
    began = time.monotonic(); capture = Path(capture).resolve(); inputs = Inputs()
    result = inputs.read(capture/'RESULTS.json'); observation = inputs.read(capture/'observation.json')
    observed = inputs.read(capture/'observed_scene.json'); scene_path = inputs.bind(observed['source_scene_path'])
    scene = B.read(scene_path); ids = [r['object_id'] for r in scene['objects']]
    need(len(ids) == 3 and len(set(ids)) == 3 and scene['bin'] is None, 'Three unique bin-free bodies required')
    need(scene['original_object_ids'] == observed['original_object_ids'] == observation['object_ids'] == ids, 'All-body order differs')
    need(scene['task'] == observed['task'] == B.TASK, 'Lift task differs')
    need(result['status'] == 'complete' and result == inputs.read(capture/'PROGRESS.json'), 'Capture incomplete')
    need(result['physics_ticks'] == observation['initialization_physics_ticks'] == 750, 'Initialization clock differs')
    need(result['initialization_object_steps'] == observation['initialization_object_steps'] == 2250, 'Object-step clock differs')
    need(result['render_physics_steps'] == observation['render_physics_steps'] == 0, 'Rendering integrated physics')
    need(result['grasp_attempts'] == result['CGR_forwards'] == 0 and scene['dt_s'] == .004, 'Capture crossed task boundary')
    need(result['scene_sha256'] == observed['source_scene_sha256'] == B.sha(scene_path), 'Scene binding differs')
    for name, key in [('observed_scene.json', 'scene_sha256'), ('observation.json', 'observation_sha256'),
                      ('render_geometry_check.json', 'geometry_check_sha256'), ('CONTACT_SENSOR_CHECK.json', 'initialization_contact_sensor_check_sha256')]:
        path = inputs.bind(capture/name)
        saved = observation[key] if name == 'observed_scene.json' else result[key]
        need(B.sha(path) == saved, 'Capture artifact hash differs: '+name)
    for kind in ('input_sha256', 'source_sha256'):
        need(observed[kind] == scene[kind], 'Observed source bindings differ')
        for path, digest in scene[kind].items(): need(B.sha(inputs.bind(path)) == digest, 'Frozen scene dependency differs')
    source = next(Path(p) for p in scene['source_sha256'] if p.endswith('/author_scene.py'))
    function = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'capture')
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == 'add_forces_and_torques_index']
    need(len(calls) == 1 and '-0.98' in ast.unparse(calls[0]), 'Original per-body weight source differs')
    with np.load(inputs.bind(capture/'initialization_trace.npz'), allow_pickle=False) as data:
        poses, velocity = data['pose_xyzw'].astype(float), data['velocity'].astype(float)
    need(poses.shape == (751,3,7) and velocity.shape == (751,3,6), 'Trace clock/shape differs')
    need(np.isfinite(poses).all() and np.isfinite(velocity).all(), 'Trace nonfinite')
    need(np.max(abs(np.linalg.norm(poses[:,:,3:], axis=2)-1)) < 1e-4, 'Nonunit trace quaternion')
    metrics = []; triangles = []
    for i, row in enumerate(scene['objects']):
        actual = observed['objects'][i]; need(actual['object_id'] == row['object_id'], 'Body order differs')
        for key, value in row.items():
            if key != 'world_T_object': need(actual[key] == value, 'Body asset/material changed: '+key)
        initial = np.asarray(row['world_T_object']); measured = np.eye(4)
        measured[:3,:3] = rotation(poses[-1,i,3:]); measured[:3,3] = poses[-1,i,:3]
        need(np.max(abs(poses[0,i,:3]-initial[:3,3])) < 1e-7, 'Initial source position differs')
        need(np.max(abs(rotation(poses[0,i,3:])-initial[:3,:3])) < 1e-6, 'Initial source rotation differs')
        need(np.array_equal(poses[-1,i], actual['observed_pose_xyzw']) and np.array_equal(velocity[-1,i], actual['observed_twist']), 'Measured final state differs')
        need(np.max(abs(measured-actual['world_T_object'])) < 1e-12, 'Final SE3 differs')
        recomputed = stability(poses[:,i], velocity[:,i])
        need(recomputed['stable'], 'Initialization unstable')
        for saved in (result['stability'][i], actual['initialization_stability']):
            need(all(abs(recomputed[k]-saved[k]) <= 1e-12 for k in recomputed), 'Saved stability differs')
        material = result['materials'][i]
        need(material == dict(mass_kg=float(np.float32(.1)), friction=float(np.float32(.6))), 'Native material readback differs')
        need(B.sha(inputs.bind(row['source_mesh'])) == row['source_mesh_sha256'], 'Source mesh changed')
        if row.get('source_visual_mesh'):
            need(B.sha(inputs.bind(row['source_visual_mesh'])) == row['source_visual_mesh_sha256'], 'Original source identity changed')
        mesh = trimesh.load_mesh(row['source_mesh'], process=False)
        vertices = np.asarray(mesh.vertices)*row['scale']; faces = np.asarray(mesh.faces)
        need(vertices.ndim == 2 and vertices.shape[1] == 3 and np.isfinite(vertices).all(), 'Source vertices invalid')
        need(faces.ndim == 2 and faces.shape[1] == 3 and faces.min() >= 0 and faces.max() < len(vertices), 'Source triangles invalid')
        world = vertices@measured[:3,:3].T+measured[:3,3]
        triangles.append(world[faces])
        metrics.append(dict(object_id=row['object_id'], source_vertices=len(vertices), source_triangles=len(faces),
            observed_source_bounds_m=[world.min(0).tolist(), world.max(0).tolist()], recomputed_stability=recomputed))

    sensor = inputs.read(capture/'CONTACT_SENSOR_CHECK.json'); config = inputs.read(capture/'CONTACT_SENSOR_CONFIG.json')
    journal = inputs.read(capture/'CONTACT_JOURNAL.json'); compressed = inputs.bind(journal['path'])
    need(journal['status'] == 'complete' and journal['frames'] == 750 and journal['sha256'] == B.sha(compressed), 'Incomplete journal binding')
    roots = ['/World/AuthorScene/'+object_id for object_id in ids]; floor = set(config['static_collision_paths'])
    need(floor and all(p.startswith('/World/ground') for p in floor), 'Unexpected bin/static body')
    need(sensor['contact_processing_enabled'] is True and config['disable_contact_processing'] is False, 'Contact processing disabled')
    need(journal['original_object_paths'] == sensor['rigid_body_paths'] == roots, 'Journal body order differs')
    need(set(sensor['target_collision_paths']) == set(roots)|floor, 'Contact targets differ')
    counts = Counter(); supports = Counter(); contacts = 0; logical = hashlib.sha256(); max_force = 0.; min_separation = 0.
    with gzip.open(compressed, 'rb') as stream:
        for step, line in enumerate(stream, 1):
            logical.update(line); frame = json.loads(line)
            need(frame['step'] == step and step <= 750, 'Contact clock differs')
            supported = set()
            for contact in frame['contacts']:
                body, other = contact['bodies']; force = contact['normal_force_n']; separation = contact['separation_m']
                need(body in roots and other in set(roots)|floor, 'Foreign contact pair')
                need(np.isfinite([force, separation, *contact['position_world_m'], *contact['raw_normal_world']]).all() and force >= 0, 'Contact numerical value invalid')
                counts[body] += 1; contacts += 1; max_force = max(max_force, force); min_separation = min(min_separation, separation)
                if other in floor and force > .001: supported.add(body)
            if step > 500: supports.update(supported)
    need(step == 750 and logical.hexdigest() == journal['logical_jsonl_sha256'] and contacts == journal['contacts'], 'Full journal content/count differs')
    for root in roots:
        need(dict(final_window_supported_ticks=supports[root], contact_rows=counts[root]) == sensor['evidence'][root] == journal['evidence'][root], 'Independent contact/support evidence differs')
        need(supports[root] == 250, 'Missing final-window table support')

    native = inputs.read(capture/'native/native_audit.json'); max_plane_error = 0.; max_delta = 0.; native_owners = Counter()
    need(native['complete'] and native['within_20um'] and native['physics_steps'] == 0, 'Native import gate missing')
    need(native['hulls'] == result['native_hulls'] == len(native['results']), 'Native hull count differs')
    with np.load(inputs.bind(capture/'native/native_hulls.npz'), allow_pickle=False) as data:
        need(len(data.files) == 3*native['hulls'], 'Native array count differs')
        for i, row in enumerate(native['results']):
            src, cooked, planes = [data[f'hull_{i:04d}'+suffix] for suffix in ('_source','_cooked','_native_planes_body')]
            need(src.shape == (row['source_vertices'],3) and cooked.shape == (row['cooked_vertices'],3) and planes.shape == (row['cooked_polygons'],4), 'Native array shape differs')
            need(all(np.isfinite(a).all() for a in (src,cooked,planes)), 'Native hull nonfinite')
            need(row['owner'] in roots and row['path'].startswith(row['owner']+'/'), 'Foreign native owner')
            need(max(row['source_outside_cooked_bound_m'],row['cooked_outside_source_bound_m']) <= 2e-5, 'Saved native20um gate failed')
            error = float(max(0.,(cooked@planes[:,:3].T+planes[:,3]).max()))
            max_plane_error = max(max_plane_error,error); max_delta = max(max_delta,abs(error-row['max_cooked_vertex_outside_native_plane_m']))
            native_owners[row['owner']] += 1
    need(max_delta < 1e-12, 'Independent native plane inequality differs')
    for root in roots:
        need(native_owners[root] == native['schemas'][root]['enabled_collision_shapes'] and native['schemas'][root]['disabled_collision_shapes'] == 0, 'Native body shapes differ')

    with np.load(inputs.bind(capture/'native_rgbd.npz'), allow_pickle=False) as data: rgb, depth = data['rgb'], data['depth_m']
    color = np.asarray(Image.open(inputs.bind(capture/'color.png'))); png = np.asarray(Image.open(inputs.bind(capture/'depth.png')))
    near, far = scene['camera']['depth_range_m']; mask = np.isfinite(depth)&(depth >= near)&(depth <= far)
    need(np.array_equal(rgb,color) and np.array_equal(np.where(mask,np.rint(depth*1000),0).astype(np.uint16),png), 'Native RGBD/PNG quantization differs')
    need(int(mask.sum()) == observation['valid_depth_pixels'], 'Depth validity differs')
    K = np.asarray(observation['intrinsic_matrix']); meta = loadmat(inputs.bind(capture/'meta.mat'))
    need(np.array_equal(meta['intrinsic_matrix'],K) and meta['factor_depth'].item() == observation['depth_factor'] == 1000., 'Meta payload differs')
    for entry in observation['files'].values(): need(B.sha(inputs.bind(entry['path'])) == entry['sha256'], 'Network RGBD binding differs')
    ray = inputs.read(capture/'render_geometry_check.json'); triangles = np.concatenate(triangles)
    T = np.asarray(observation['T_WC']); expected = []; hit = []
    for pixel in np.asarray(ray['pixels_xy']):
        direction = T[:3,:3]@np.r_[np.linalg.solve(K[:2,:2],pixel+.5-K[:2,2]),1.]
        value, object_hit, tie = first_depth(triangles,T[:3,3],direction)
        expected.append(value); hit.append(object_hit)
    saved = np.asarray([np.inf if d is None else d for d in ray['expected_source_first_depth_m']])
    need(np.allclose(expected,saved,atol=1e-10,rtol=0), 'Independent full-source first-depth rays differ')
    pixels = np.asarray(ray['pixels_xy']); actual = depth[pixels[:,1],pixels[:,0]]
    need(np.array_equal(actual,ray['observed_rtx_depth_m']), 'Native sampled RGBD rays differ')
    inputs.verify()
    return dict(status='pass',capture=str(capture),object_ids=ids,source_meshes=metrics,physics_ticks=750,initialization_object_steps=2250,
        all_contact_frames=750,all_contacts=contacts,per_body_contact_rows=dict(counts),final250_table_support_ticks=dict(supports),
        minimum_contact_separation_m=min_separation,maximum_contact_force_n=max_force,native_hulls=native['hulls'],
        native_per_body_hulls=dict(native_owners),native_plane_inequality_max_m=max_plane_error,native_plane_vs_receipt_error_m=max_delta,
        native_scope='Every saved source/cooked/plane array checked; original convex-distance optimizer not rerun; native20um is import fidelity',
        independent_full_source_rays=len(expected),independent_ray_depth_max_error_m=float(np.max(abs(np.asarray(expected)-saved))),
        rgbd_and_meta_exact=True,inputs_sha256=inputs.files,wall_s=time.monotonic()-began,
        independent_audit_Kit_launches=0,independent_audit_physics_steps=0,independent_audit_model_forwards=0,
        cost_scope='Read-only diagnostic audit performed alongside the actual run; not a collector built-in stage')


def array_compare(left, right):
    rows = {}; exact = True
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        need(a.files == b.files, 'Array key/order differs')
        for key in a.files:
            x,y = a[key],b[key]
            same = x.dtype == y.dtype and x.shape == y.shape and hashlib.sha256(np.ascontiguousarray(x)).digest() == hashlib.sha256(np.ascontiguousarray(y)).digest()
            rows[key] = dict(exact=same,dtype=str(x.dtype),shape=list(x.shape)); exact &= same
    return dict(exact=exact,arrays=rows,left=B.binding(left),right=B.binding(right))


def compare_run(root):
    root = Path(root).resolve(); protocol = B.read(root/'RESIDENT_COLLECTION_PROTOCOL.json')
    source = Path(protocol['source_batch_protocol']['path']).parent
    rows = []
    for i,row in zip(protocol['source_scene_indices'],protocol['queue']['scenes']):
        name = f'{i:07d}_{row["scene_id"]}'; current = root/'scenes'/name; old = source/'scenes'/name
        out = dict(source_queue_index=i,scene_id=row['scene_id'],capture={},representation={},gestures={})
        for filename in ('initialization_trace.npz','native_rgbd.npz'):
            out['capture'][filename] = array_compare(current/'capture'/filename,old/'capture'/filename)
        for filename in ('raw_camera_candidates.npz','representation_outputs.npz'):
            out['representation'][filename] = array_compare(root/'representation/representations'/row['scene_id']/filename,
                old/'representation/representations'/row['scene_id']/filename)
        for g in B.GESTURES:
            new = current/f'g{g}'; prior = old/f'g{g}'
            result = new/'LIFT_PIPELINE_RESULTS.json'
            if not result.exists(): out['gestures'][str(g)] = dict(status='pending'); continue
            fresh = B.read(result); previous = B.read(prior/'LIFT_PIPELINE_RESULTS.json')
            record = dict(status=fresh['status'],reference_status=previous['status'],candidate_sha256=fresh.get('candidate_sha256'),
                reference_candidate_sha256=previous.get('candidate_sha256'), candidate_exact=fresh.get('candidate_sha256')==previous.get('candidate_sha256'))
            for relative in ('pool/candidates.npz','arm/native/commands.npz','lift_bound/commands.npz'):
                if (new/relative).exists() and (prior/relative).exists(): record[relative] = array_compare(new/relative,prior/relative)
            out['gestures'][str(g)] = record
        rows.append(out)
    return dict(status='snapshot',protocol=B.binding(root/'RESIDENT_COLLECTION_PROTOCOL.json'),reference_protocol=protocol['source_batch_protocol'],
        scenes=rows,scope='Path/JSON timestamps are not compared; array dtype/shape/C-order bytes are compared. Pending stages remain explicit.',
        audit_Kit_launches=0,audit_physics_steps=0,audit_model_forwards=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    capture = sub.add_parser('capture'); capture.add_argument('--capture-dir',type=Path,required=True)
    compare = sub.add_parser('compare'); compare.add_argument('--run-dir',type=Path,required=True)
    for p in (capture,compare): p.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); need(not args.output.exists(),'Fresh audit output required')
    began = time.monotonic()
    try: result = capture_audit(args.capture_dir) if args.command == 'capture' else compare_run(args.run_dir)
    except Exception: result = dict(status='failed',error=traceback.format_exc(),wall_s=time.monotonic()-began)
    result.update(audit_source=B.binding(__file__))
    B.save(args.output,result,exclusive=True)
    print(json.dumps({k:result[k] for k in ('status','wall_s','error') if k in result},indent=2))
    if result['status'] == 'failed': raise SystemExit(1)


if __name__ == '__main__': main()
