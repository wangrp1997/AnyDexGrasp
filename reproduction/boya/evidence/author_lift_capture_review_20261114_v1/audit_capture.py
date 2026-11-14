"""Read existing capture artifacts only; no simulator, planning, or mutation."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[4]
CAPTURE = ROOT/'outputs/boya_full_data_lift_20261114_v1/scene/capture'
OUT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def json_rows(path):
    """Bounded-memory parser for the original large JSON top-level array."""
    decoder = json.JSONDecoder()
    with Path(path).open() as handle:
        buffer = handle.read(2*1024*1024).lstrip()
        assert buffer.startswith('['); buffer = buffer[1:]
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(','): buffer = buffer[1:].lstrip()
            if buffer.startswith(']'):
                assert not (buffer[1:]+handle.read()).strip()
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = handle.read(2*1024*1024)
                if not chunk: raise
                buffer += chunk
                continue
            yield value
            buffer = buffer[end:]


def main():
    began = time.monotonic()
    scene_path = CAPTURE.parent/'scene.json'
    scene, observed = read(scene_path), read(CAPTURE/'observed_scene.json')
    result, observation = read(CAPTURE/'RESULTS.json'), read(CAPTURE/'observation.json')
    files = [p for p in CAPTURE.rglob('*') if p.is_file()]+[scene_path]
    before = {str(p):sha(p) for p in sorted(files)}
    assert scene['schema'] == observed['schema'] == 'anydex_boya_author_lift_scene_v1'
    assert scene['task'] == observed['task'] == 'declared_boya_clutter_lift_v1'
    assert scene['bin'] is None and observed['bin'] is None
    ids = [r['object_id'] for r in scene['objects']]
    roots = ['/World/AuthorScene/'+i for i in ids]
    assert ids == ['graspnet_000', 'graspnet_002', 'graspnet_005']
    assert scene['original_object_ids'] == observed['original_object_ids'] == observation['object_ids'] == ids
    assert result['status'] == 'complete' and result == read(CAPTURE/'PROGRESS.json')
    assert result['physics_ticks'] == observation['initialization_physics_ticks'] == 750
    assert result['initialization_object_steps'] == observation['initialization_object_steps'] == 2250
    assert result['render_physics_steps'] == observation['render_physics_steps'] == 0
    assert result['grasp_attempts'] == result['CGR_forwards'] == 0
    assert scene['dt_s'] == .004 and scene['initialization_steps'] == 750
    assert result['scene_sha256'] == sha(scene_path) == observed['source_scene_sha256']
    assert result['observation_sha256'] == sha(CAPTURE/'observation.json')
    assert result['geometry_check_sha256'] == sha(CAPTURE/'render_geometry_check.json')
    assert result['initialization_contact_sensor_check_sha256'] == sha(CAPTURE/'CONTACT_SENSOR_CHECK.json')
    assert observation['scene_sha256'] == sha(CAPTURE/'observed_scene.json')
    assert observation['object_pose_as_network_input'] is False
    locks = {**scene['input_sha256'], **scene['source_sha256']}
    assert observed['input_sha256'] == scene['input_sha256'] and observed['source_sha256'] == scene['source_sha256']
    for name, digest in locks.items(): assert sha(name) == digest, name
    for name, binding in observation['files'].items():
        assert Path(binding['path']) == CAPTURE/name and sha(binding['path']) == binding['sha256']
    for material in result['materials']:
        assert material['mass_kg'] == float(np.float32(.1)) and material['friction'] == float(np.float32(.6))

    with np.load(CAPTURE/'initialization_trace.npz', allow_pickle=False) as data:
        poses, twists = data['pose_xyzw'].astype(float), data['velocity'].astype(float)
    assert poses.shape == (751,3,7) and twists.shape == (751,3,6)
    assert np.isfinite(poses).all() and np.isfinite(twists).all()
    assert np.max(abs(np.linalg.norm(poses[:,:,3:],axis=2)-1)) < 1e-4
    recomputed, pose_errors = [], []
    for index, row in enumerate(observed['objects']):
        window, velocity = poses[-250:,index], twists[-250:,index]
        q = window[:,3:]/np.linalg.norm(window[:,3:],axis=1)[:,None]
        metrics = dict(max_linear_speed_m_s=float(np.linalg.norm(velocity[:,:3],axis=1).max()),
            max_angular_speed_rad_s=float(np.linalg.norm(velocity[:,3:],axis=1).max()),
            max_window_translation_m=float(np.linalg.norm(window[:,:3]-window[-1,:3],axis=1).max()),
            max_window_rotation_rad=float((2*np.arccos(np.clip(abs(q@q[-1]),0,1))).max()))
        assert all(np.isclose(metrics[k], result['stability'][index][k],atol=1e-12,rtol=0) for k in metrics)
        assert metrics['max_linear_speed_m_s'] <= .02 and metrics['max_angular_speed_rad_s'] <= .5
        assert metrics['max_window_translation_m'] <= .0002 and metrics['max_window_rotation_rad'] <= .005
        assert result['stability'][index]['stable'] is True
        assert np.array_equal(poses[-1,index],row['observed_pose_xyzw'])
        assert np.array_equal(twists[-1,index],row['observed_twist'])
        T=np.eye(4);T[:3,:3]=Rotation.from_quat(poses[-1,index,3:]).as_matrix();T[:3,3]=poses[-1,index,:3]
        pose_errors.append(float(abs(T-np.asarray(row['world_T_object'])).max()))
        assert pose_errors[-1] < 1e-12
        recomputed.append(dict(object_id=row['object_id'],**metrics))

    contact_cfg, check = read(CAPTURE/'CONTACT_SENSOR_CONFIG.json'), read(CAPTURE/'CONTACT_SENSOR_CHECK.json')
    assert contact_cfg == dict(static_collision_paths=['/World/ground/collisionPlane'],disable_contact_processing=False)
    assert check['contact_processing_enabled'] is True and check['rigid_body_paths'] == roots
    assert check['target_collision_paths'] == roots+['/World/ground/collisionPlane']
    counts = dict.fromkeys(roots,0); supports = dict.fromkeys(roots,0)
    totals = {root:[] for root in roots}; contact_count = row_count = 0
    minimum_separation, maximum_force = 0.,0.
    for row_count, frame in enumerate(json_rows(CAPTURE/'initialization_contacts.json'),1):
        assert frame['step'] == row_count
        supported=set(); forces=dict.fromkeys(roots,0.)
        for contact in frame['contacts']:
            body, other = contact['bodies']
            assert body in roots and other in check['target_collision_paths']
            assert not any('Bin' in part for part in contact['bodies'])
            assert all(math.isfinite(v) for v in contact['position_world_m']+contact['raw_normal_world']+[contact['separation_m'],contact['normal_force_n']])
            assert contact['normal_force_n'] >= 0.
            counts[body]+=1; contact_count+=1
            minimum_separation=min(minimum_separation,contact['separation_m'])
            maximum_force=max(maximum_force,contact['normal_force_n'])
            if other == '/World/ground/collisionPlane':
                forces[body]+=contact['normal_force_n']
                if contact['normal_force_n'] > .001: supported.add(body)
        if row_count > 500:
            for body in supported: supports[body]+=1
            for body in roots: totals[body].append(forces[body])
    assert row_count == 750
    for root in roots:
        assert supports[root] == check['evidence'][root]['final_window_supported_ticks'] == 250
        assert counts[root] == check['evidence'][root]['contact_rows']

    native = read(CAPTURE/'native/native_audit.json')
    assert native['complete'] and native['within_20um'] and native['physics_steps'] == 0
    assert native['hulls'] == result['native_hulls'] == len(native['results']) == 1633
    assert set(native['schemas']) == set(roots)
    assert native['source_sha256'] == sha(ROOT.parent/'CanonDex/scripts/research/template_gripper_native_audit.py')
    per_body = {root:sum(r['owner']==root for r in native['results']) for root in roots}
    plane_error = max_delta = 0.
    with np.load(CAPTURE/'native/native_hulls.npz', allow_pickle=False) as data:
        assert len(data.files) == 3*native['hulls']
        for index,row in enumerate(native['results']):
            key=f'hull_{index:04d}';src,cooked,planes=[data[key+s] for s in ('_source','_cooked','_native_planes_body')]
            assert src.shape == (row['source_vertices'],3) and cooked.shape == (row['cooked_vertices'],3)
            assert planes.shape == (row['cooked_polygons'],4)
            assert all(np.isfinite(v).all() for v in (src,cooked,planes))
            assert row['owner'] in roots and row['path'].startswith(row['owner']+'/')
            assert max(row['source_outside_cooked_bound_m'],row['cooked_outside_source_bound_m']) <= 2e-5
            error=float(max(0.,(cooked@planes[:,:3].T+planes[:,3]).max()))
            max_delta=max(max_delta,abs(error-row['max_cooked_vertex_outside_native_plane_m']))
            plane_error=max(plane_error,error)
        assert max_delta < 1e-12
    assert plane_error == native['max_cooked_vertex_outside_native_plane_m']
    for root in roots:
        assert per_body[root] == native['schemas'][root]['enabled_collision_shapes']
        assert native['schemas'][root]['disabled_collision_shapes'] == 0

    with np.load(CAPTURE/'native_rgbd.npz',allow_pickle=False) as data: rgb,depth=data['rgb'],data['depth_m']
    assert rgb.shape == (720,1280,3) and depth.shape == (720,1280)
    assert np.array_equal(rgb,np.asarray(Image.open(CAPTURE/'color.png')))
    near,far=scene['camera']['depth_range_m'];mask=np.isfinite(depth)&(depth>=near)&(depth<=far)
    depth_png=np.where(mask,np.rint(depth*1000),0).astype(np.uint16)
    assert np.array_equal(depth_png,np.asarray(Image.open(CAPTURE/'depth.png')))
    assert int(mask.sum()) == observation['valid_depth_pixels']
    K=np.asarray(observation['intrinsic_matrix']);assert np.array_equal(K,loadmat(CAPTURE/'meta.mat')['intrinsic_matrix'])
    assert float(loadmat(CAPTURE/'meta.mat')['factor_depth'][0,0]) == observation['depth_factor'] == 1000.
    # Independent Moller-Trumbore intersections against source triangles plus z=0.
    triangles=[]
    for row in observed['objects']:
        mesh=trimesh.load_mesh(row['source_mesh'],process=False);T=np.asarray(row['world_T_object'])
        vertices=np.asarray(mesh.vertices)*row['scale'];world=vertices@T[:3,:3].T+T[:3,3]
        triangles.append(world[np.asarray(mesh.faces)])
    tri=np.concatenate(triangles);a=tri[:,0];e1=tri[:,1]-a;e2=tri[:,2]-a
    T=np.asarray(observation['T_WC']);origin=T[:3,3];offset=origin-a
    ray=read(CAPTURE/'render_geometry_check.json');expected=[];objects=[]
    for u,v in ray['pixels_xy']:
        direction=T[:3,:3]@np.array([(u+.5-K[0,2])/K[0,0],(v+.5-K[1,2])/K[1,1],1.])
        p=np.cross(np.broadcast_to(direction,e2.shape),e2);det=(e1*p).sum(1)
        good=abs(det)>1e-14;inverse=np.divide(1.,det,out=np.zeros_like(det),where=good)
        s=(offset*p).sum(1)*inverse;q=np.cross(offset,e1);t=q@direction*inverse;distance=(e2*q).sum(1)*inverse
        good &= (s>=-1e-10)&(t>=-1e-10)&(s+t<=1+1e-10)&(distance>1e-10)
        object_depth=float(distance[good].min()) if np.any(good) else float('inf')
        floor_depth=float(-origin[2]/direction[2]) if direction[2]<0 else float('inf')
        expected.append(min(object_depth,floor_depth));objects.append(object_depth<floor_depth)
    expected=np.asarray(expected);declared=np.asarray(ray['expected_source_first_depth_m'])
    assert np.max(abs(expected-declared)) < 1e-10
    assert objects == ray['source_first_hit_is_object'] and sum(objects)==37
    pixels=np.asarray(ray['pixels_xy']);rtx=depth[pixels[:,1],pixels[:,0]].astype(float)
    assert np.array_equal(rtx,ray['observed_rtx_depth_m'])
    absolute=np.abs(expected-rtx)
    assert np.isclose(absolute.max(),ray['max_absolute_depth_error_m'],atol=1e-10,rtol=0)
    after={str(p):sha(p) for p in sorted(files)}
    assert before == after
    summary=dict(schema='anydex_boya_author_lift_capture_independent_audit_v1',status='pass',
        capture_path=str(CAPTURE),task=scene['task'],bin=None,original_object_ids=ids,
        capture_physics_ticks=750,capture_initialization_object_steps=2250,capture_render_physics_steps=0,
        trace_shape=list(poses.shape),recomputed_stability=recomputed,max_observed_pose_transform_error=float(max(pose_errors)),
        contact_frames=row_count,all_contacts=contact_count,per_body_contact_rows=counts,
        final_250_ground_supported_ticks=supports,
        final_250_mean_ground_force_n={root:float(np.mean(v)) for root,v in totals.items()},
        max_contact_penetration_m=-minimum_separation,max_single_contact_force_n=maximum_force,
        static_collision_paths=contact_cfg['static_collision_paths'],no_bin_in_all_saved_contact_pairs=True,
        native_hulls=1633,native_per_body_hulls=per_body,native_reported_20um_gate=True,
        native_max_source_outside_cooked_bound_m=native['max_source_outside_cooked_bound_m'],
        native_max_cooked_outside_source_bound_m=native['max_cooked_outside_source_bound_m'],
        all_native_arrays_finite=True,native_plane_inequality_recomputed_max_m=plane_error,
        native_plane_inequality_vs_receipt_max_delta_m=max_delta,
        native_scope='Saved cooked/source arrays and receipt checked; original convex-distance solver not rerun; 20um is import fidelity rather than exact source CAD/solver validation',
        source_input_bindings_verified=len(locks),source_bindings=len(scene['source_sha256']),input_bindings=len(scene['input_sha256']),
        capture_files_before_after_sha256=before,
        source_sha256={str(Path(__file__).resolve()):sha(__file__)},
        rgb_png_exact=True,depth_png_exact=True,meta_exact=True,valid_depth_pixels=int(mask.sum()),
        independent_rays=len(expected),independent_source_first_object_hits=sum(objects),
        independent_vs_declared_source_first_depth_max_error_m=float(abs(expected-declared).max()),
        independent_vs_rtx_max_absolute_depth_error_m=float(absolute.max()),
        audit_Kit_launches=0,audit_physics_steps=0,audit_IK_calls=0,audit_CGR_forwards=0,
        audit_grasp_attempts=0,audit_labels=0,audit_training_updates=0,wall_s=time.monotonic()-began)
    (OUT/'CAPTURE_AUDIT.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:summary[key] for key in ('status','all_contacts','contact_frames','native_hulls','source_input_bindings_verified','independent_rays','wall_s')}))


if __name__ == '__main__': main()
