#!/usr/bin/env python3
"""Fixed EGAD12 external assets, before scores/outcomes; CPU-only preparation.

Native source OBJ coordinates are multiplied by .001 m/native-unit, a declared
simulation assumption rather than AnyDex printing calibration. No recentering,
resizing to the hand, mesh simplification, pose search or result-based selection.
The old S40 deployment/model sources are not imported or changed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import trimesh
from prepare_s40_scenes import placement, verify_mesh_conversion

ROWS = ('A', 'C', 'E', 'G')
COLS = (0, 3, 6)
FIXED_IDS = tuple(f'{r}{c}' for r in ROWS for c in COLS)
UNIT_M = .001
REPO = Path(__file__).resolve().parents[2]
EXCLUSION_REFS = (
    REPO/'outputs/s40_stable_supplement_20261107_v2/collection_resume_v3/objects.json',
    REPO/'outputs/boya_pilot8_20261105_v1/object_selection.json',
    REPO/'outputs/collection40_20261105_v1/selection_v2/object_selection.json')
PARAMETERS = dict(threshold=.01, resolution=2000, mcts_iterations=75,
    mcts_nodes=20, mcts_max_depth=3, max_convex_hull=128,
    preprocess_mode='auto', preprocess_resolution=50, pca=False, merge=True,
    decimate=False, max_ch_vertex=256, extrude=False, extrude_margin=.01,
    apx_mode='ch', seed=20261107)
WORKER_TIMEOUT_S = 300
CAMERA = dict(width=1280, height=720, fx=640., fy=640.,
    eye_world_m=[-.75,-.20,.50], target_world_m=[-.75,.10,.06], up_world=[0.,0.,1.],
    depth_range_m=[.20,.65], focal_length=24., horizontal_aperture=48., clipping_range_m=[.1,1.])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def validate_inventory(inventory):
    expected = [f'{r}{c}' for r in 'ABCDEFG' for c in range(7)]
    rows = inventory.get('objects', [])
    if (inventory.get('schema') != 'anydex_official_egad_eval_asset_inventory_v1'
            or inventory.get('object_count') != 49
            or inventory.get('object_ids') != expected
            or len(rows) != 49 or [r['object_id'] for r in rows] != expected):
        raise ValueError('Require all49 original EGAD identities in pinned grid order')
    for row in rows:
        if sha(row['path']) != row['sha256']:
            raise ValueError('Original EGAD source changed: ' + row['object_id'])
    by_id = {r['object_id']: r for r in rows}
    return [by_id[oid] for oid in FIXED_IDS]


def meter_geometry_sha(mesh, scale=1.):
    h=hashlib.sha256()
    h.update(np.asarray(np.asarray(mesh.vertices)*scale,dtype='<f8').tobytes())
    h.update(np.asarray(mesh.faces,dtype='<i8').tobytes())
    return h.hexdigest()


def exclusion_receipt(paths):
    references=[];source_hashes=set();meter_hashes=set()
    for path in paths:
        doc=json.loads(Path(path).read_text());references.append(dict(path=str(path.resolve()),sha256=sha(path)))
        for row in doc['objects']:
            for key in ('mesh_sha256','mesh_group_sha256','source_mesh_sha256','source_visual_mesh_sha256'):
                if row.get(key):source_hashes.add(row[key])
            m=trimesh.load_mesh(row['mesh_path'],process=False)
            meter_hashes.add(meter_geometry_sha(m,row['scale']))
    return dict(manifest_refs=references,source_hashes=sorted(source_hashes),meter_geometry_hashes=sorted(meter_hashes),
        comparison='Source/exported OBJ SHA plus float64 meter vertices and int64 triangle-order digest; no shape-similarity exclusion')


def scaled_visual(source):
    mesh = trimesh.load_mesh(source, process=False)
    if (not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces)
            or not np.isfinite(mesh.vertices).all()):
        raise ValueError('Original finite triangle mesh required')
    scaled = trimesh.Trimesh(vertices=np.asarray(mesh.vertices) * UNIT_M,
                            faces=np.asarray(mesh.faces).copy(), process=False)
    if not np.array_equal(scaled.faces, mesh.faces):
        raise ValueError('Source topology changed')
    return mesh, scaled


def export_urdf(root, parts):
    robot = ET.Element('robot', name=root.name)
    link = ET.SubElement(robot, 'link', name='base_link')
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'origin', xyz='0 0 0', rpy='0 0 0')
    ET.SubElement(inertial, 'mass', value='.1')
    # Backend derives COM/inertia from unchanged source triangles, not this hint.
    ET.SubElement(inertial, 'inertia', ixx='1e-4', ixy='0', ixz='0', iyy='1e-4', iyz='0', izz='1e-4')
    visual = ET.SubElement(link, 'visual')
    ET.SubElement(visual, 'origin', xyz='0 0 0', rpy='0 0 0')
    ET.SubElement(ET.SubElement(visual, 'geometry'), 'mesh', filename='../mesh/simplified.obj', scale='1 1 1')
    for part in parts:
        collision = ET.SubElement(link, 'collision', name=Path(part['path']).stem)
        ET.SubElement(collision, 'origin', xyz='0 0 0', rpy='0 0 0')
        ET.SubElement(ET.SubElement(collision, 'geometry'), 'mesh',
                      filename='meshes/' + Path(part['path']).name, scale='1 1 1')
    path = root / 'urdf/coacd.urdf'
    ET.ElementTree(robot).write(path, encoding='unicode')
    return path


def package(job, components, runtime):
    """Export original scaled visual and measured convex approximation."""
    root = Path(job['object_root'])
    for sub in ('mesh', 'urdf/meshes', 'info'):
        (root / sub).mkdir(parents=True, exist_ok=False)
    source, visual = scaled_visual(job['source']['path'])
    meter_digest=meter_geometry_sha(visual)
    if (job['source']['sha256']in job['exclusion']['source_hashes']
            or meter_digest in job['exclusion']['meter_geometry_hashes']):
        raise ValueError('External source or scaled geometry overlaps prior manifests')
    visual_path = root / 'mesh/simplified.obj'
    visual_path.write_text(trimesh.exchange.obj.export_obj(visual,
        include_normals=False, include_color=False, include_texture=False, digits=16))
    conversion_error = verify_mesh_conversion(visual, trimesh.load_mesh(visual_path, process=False))
    if sha(visual_path)in job['exclusion']['source_hashes']:
        raise ValueError('Exported meter OBJ overlaps prior manifest mesh SHA')
    parts, vertices = [], []
    for i, (v, f) in enumerate(components):
        m = trimesh.Trimesh(np.asarray(v, float), np.asarray(f, np.int32), process=False)
        if (not len(m.faces) or not np.isfinite(m.vertices).all()
                or not m.is_convex or not m.is_watertight or abs(m.volume) <= 1e-14):
            raise ValueError('Invalid CoACD convex component')
        path = root / 'urdf/meshes' / f'coacd_convex_piece_{i:04d}.obj'
        path.write_text(trimesh.exchange.obj.export_obj(m, include_normals=False,
            include_color=False, include_texture=False, digits=16))
        check = trimesh.load_mesh(path, process=False)
        verify_mesh_conversion(m, check)
        vertices.append(np.asarray(check.vertices))
        parts.append(dict(path=str(path), sha256=sha(path), scale='1 1 1',
            vertices=len(m.vertices), triangles=len(m.faces), volume_m3=float(abs(m.volume)),
            convex=True, watertight=True))
    if not parts:
        raise ValueError('CoACD emitted no convex parts')
    convex = np.concatenate(vertices)
    urdf = export_urdf(root, parts)
    T = placement(visual.vertices, convex)
    pose = root / 'info/tabletop_pose.json'
    save(pose, [np.r_[T[:3, 3], 1., 0., 0., 0.].tolist()])
    source_bounds = np.asarray(visual.bounds)
    collision_bounds = np.array([convex.min(0), convex.max(0)])
    geometry = dict(source_native_units_to_m=UNIT_M,
        unit_scope='Uniform simulation assumption; not original AnyDex printed scale',
        source_triangle_vertices=len(source.vertices), source_triangles=len(source.faces),
        lossless_scaled_OBJ_vertex_max_error_m=conversion_error,
        source_visual_bounds_m=source_bounds.tolist(), collision_union_bounds_m=collision_bounds.tolist(),
        collision_vs_visual_AABB_delta_m=(collision_bounds-source_bounds).tolist(),
        collision_vs_visual_max_AABB_delta_m=float(np.abs(collision_bounds-source_bounds).max()),
        source_visual_volume_m3=float(abs(visual.volume)),
        collision_component_volume_sum_m3=sum(p['volume_m3'] for p in parts),
        collision_component_volume_sum_is_union_volume=False,
        source_visual_watertight=bool(visual.is_watertight),
        visual_minimum_world_z_m=float(visual.vertices[:,2].min()+T[2,3]),
        collision_minimum_world_z_m=float(convex[:,2].min()+T[2,3]),
        convex_part_count=len(parts), coacd_parameters=PARAMETERS, coacd_runtime=runtime,
        geometry_boundary='Measured AABB and convexity only. Not Hausdorff, cavity/handle fidelity or shape-preservation certificate. Native20um cooking audit is a separate step.',
        physics_mass_kg=.1, physics_friction=.6,
        physical_parameters_source='Common Boya simulation approximation, not paper real EGAD properties')
    metadata = root / 'info/source_geometry.json'
    save(metadata, geometry)
    save(root / 'info/simplified.json', dict(gravity_center=visual.center_mass.tolist(),
        mass=float(visual.mass), inertia=visual.moment_inertia.tolist(),
        source='Original triangles uniformly scaled only; density1 geometric properties'))
    world_visual = np.asarray(visual.vertices) + T[:3,3]
    oid = root.name
    record = dict(id=oid, object_id=oid, pilot_id=oid, egad_object_id=job['source']['object_id'],
        split='external_validation', source_split='official_EGAD49_external_grid12', source_family='EGAD_official_eval',
        object_root=str(root), mesh_path=str(visual_path), mesh_sha256=sha(visual_path),
        mesh_group_sha256=job['source']['sha256'], urdf_path=str(urdf), urdf_sha256=sha(urdf),
        pose_path=str(pose), pose_sha256=sha(pose), metadata_path=str(metadata), metadata_sha256=sha(metadata),
        scale=1., source_native_units_to_m=UNIT_M, geometry_scaled_on_disk=True,
        object_mass_kg=.1, object_friction=.6, original_AnyDex_CAD_identity_verified=False,
        source_visual_mesh=job['source']['path'], source_visual_mesh_sha256=job['source']['sha256'],
        source_mesh_sha256=job['source']['sha256'],meter_geometry_sha256=meter_digest,
        source_inventory_sha256=job['inventory_sha256'],
        source_input_sha256={job['source']['path']:job['source']['sha256']}, convex_parts=parts,
        selected_source_pose_index=0, source_pose_quaternion_order='wxyz', world_T_object=T.tolist(),
        world_pose_xyzw=np.r_[T[:3,3],0.,0.,0.,1.].tolist(),
        world_AABB_m=[world_visual.min(0).tolist(),world_visual.max(0).tolist()],
        world_extents_m=np.ptp(world_visual,axis=0).tolist(), geometry_approximation=geometry,
        source_placement_rule='Native axes; visual AABB XY center(-.75,.10)m; min visual/collision union Z at table0. Static support only, no stability certificate. Common settle before observation is separate.',
        preflight_status='ready', preflight_issues=[],
        observation_json=str(Path(job['run_dir'])/'observations'/oid/'observation.json'),
        representation_dir=str(Path(job['run_dir'])/'representations'/oid),
        grasp_outcome_consulted=False, prior_grasp_pose_consulted=False,
        original_scene_pose_reproduced=False)
    save(root/'PROVENANCE.json', record)
    return record


def worker(jobfile):
    job = json.loads(Path(jobfile).read_text())
    started = time.monotonic()
    receipt = dict(schema='anydex_boya_egad_external_asset_result_v1', object_id=job['object_id'],
        complete=False, source=job['source'], physics_steps=0, network_forwards=0, training_steps=0)
    try:
        import coacd
        runtime = dict(python_module=str(Path(coacd.__file__).resolve()), python_sha256=sha(coacd.__file__),
            native_library=str(Path(coacd._lib._name).resolve()), native_sha256=sha(coacd._lib._name),
            API_signature=str(inspect.signature(coacd.run_coacd)))
        if sha(job['source']['path']) != job['source']['sha256']:
            raise ValueError('Original source changed')
        _, visual = scaled_visual(job['source']['path'])
        start = time.monotonic()
        parts = coacd.run_coacd(coacd.Mesh(np.asarray(visual.vertices),np.asarray(visual.faces)), **PARAMETERS)
        runtime['decomposition_wall_s'] = time.monotonic()-start
        record = package(job, parts, runtime)
        receipt.update(complete=True,status='ready',record=record)
    except Exception as exc:
        receipt.update(status='asset_preparation_failed',error=repr(exc),traceback=traceback.format_exc())
    receipt['wall_s'] = time.monotonic()-started
    save(Path(jobfile).with_name('RESULT.json'), receipt)
    return 0 if receipt['complete'] else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path)
    p.add_argument('--asset-root',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--workers',type=int,choices=range(1,5),default=4)
    p.add_argument('--worker-job',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker_job:
        return worker(args.worker_job)
    if any(getattr(args,k) is None for k in ('inventory','asset_root','output_dir','run_dir')):
        p.error('inventory, asset-root, output-dir and run-dir are required')
    args.inventory,args.asset_root,args.output_dir,args.run_dir=[v.resolve() for v in
        (args.inventory,args.asset_root,args.output_dir,args.run_dir)]
    if args.asset_root.exists() or args.output_dir.exists():
        raise ValueError('Fresh asset/output directories required; failed assets are never overwritten')
    rows=validate_inventory(json.loads(args.inventory.read_text()))
    digest=sha(args.inventory)
    exclusion=exclusion_receipt(EXCLUSION_REFS)
    args.run_dir.mkdir(parents=True,exist_ok=True)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    args.asset_root.mkdir(parents=True,exist_ok=False)
    selection=dict(schema='anydex_boya_egad_external_selection_v1',fixed_before_scores_or_outcomes=True,
        selection_rule='Official49 grid rows A,C,E,G crossed with columns0,3,6 in row-major order',
        object_ids=['egad_'+x for x in FIXED_IDS],source_object_ids=list(FIXED_IDS),
        source_inventory=str(args.inventory),source_inventory_sha256=digest,
        source_native_units_to_m=UNIT_M,parameters=PARAMETERS,workers=args.workers,
        worker_timeout_s=WORKER_TIMEOUT_S,script=str(Path(__file__).resolve()),script_sha256=sha(__file__),
        placement_source=str(Path(__file__).with_name('prepare_s40_scenes.py')),
        placement_source_sha256=sha(Path(__file__).with_name('prepare_s40_scenes.py')),
        unit_scope='Uniform simulation units assumption; not original AnyDex printing calibration',
        source_files={r['path']:r['sha256'] for r in rows},training_or_scores_or_outcomes_read=False,
        exclusion_manifest_refs=exclusion['manifest_refs'],exclusion_comparison=exclusion['comparison'])
    save(args.output_dir/'SELECTION.json',selection)
    started=time.monotonic()
    def one(row):
        oid='egad_'+row['object_id'];directory=args.output_dir/oid;directory.mkdir()
        job=dict(object_id=oid,source=row,inventory_sha256=digest,object_root=str(args.asset_root/oid),run_dir=str(args.run_dir),exclusion=exclusion)
        path=directory/'JOB.json';save(path,job)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        try:
            with (directory/'coacd.log').open('xb') as log:
                done=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker-job',str(path)],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=WORKER_TIMEOUT_S)
            receipt=json.loads((directory/'RESULT.json').read_text())
            if done.returncode and receipt.get('status')=='ready':
                raise ValueError('Worker returncode differs from ready receipt')
        except Exception as exc:
            receipt=dict(schema='anydex_boya_egad_external_asset_result_v1',object_id=oid,complete=False,
                status='asset_preparation_failed',error=repr(exc),source=row,physics_steps=0,training_steps=0,network_forwards=0,
                partial_assets_retained=True)
            save(directory/'RESULT.json',receipt)
        return receipt
    completed={}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(one,row):row for row in rows}
        for future in as_completed(futures):
            receipt=future.result();completed[receipt['object_id']]=receipt
            save(args.output_dir/'PROGRESS.json',dict(status='preparing',completed=len(completed),total=12,
                ready=sum(x.get('complete',False) for x in completed.values()),objects=list(completed.values()),
                wall_s=time.monotonic()-started,physics_steps=0,network_forwards=0,training_steps=0))
            print(json.dumps({k:receipt[k] for k in ('object_id','status','complete')}),flush=True)
    records=[]
    for row in rows:
        oid='egad_'+row['object_id'];result=completed[oid]
        if result['complete']:
            records.append(result['record'])
        else:
            records.append(dict(id=oid,object_id=oid,pilot_id=oid,egad_object_id=row['object_id'],split='external_validation',
                source_family='EGAD_official_eval',object_root=str(args.asset_root/oid),scale=1.,
                source_native_units_to_m=UNIT_M,source_visual_mesh=row['path'],source_visual_mesh_sha256=row['sha256'],
                source_inventory_sha256=digest,mesh_group_sha256=row['sha256'],source_mesh_sha256=row['sha256'],
                mesh_sha256=None,preflight_status='asset_preparation_failed',
                preflight_issues=[result['error']],observation_json=str(args.run_dir/'observations'/oid/'observation.json'),
                representation_dir=str(args.run_dir/'representations'/oid)))
    for row in rows:
        if sha(row['path'])!=row['sha256']:raise ValueError('Original source changed during preparation')
    manifest=dict(schema='anydex_boya_external_scene_manifest_v1',source_inventory=str(args.inventory),
        source_inventory_sha256=digest,source_active_list_sha256=digest,
        source_active_list_boundary='Compatibility key binds official49 EGAD inventory, not S40 or DGN list',
        source_dataset_root=str(args.asset_root),active_object_count=12,scale=1.,objects=records,camera=CAMERA,
        approved_object_ids=['egad_'+x for x in FIXED_IDS],exclusion_manifest_refs=exclusion['manifest_refs'],
        exclusion_comparison=exclusion['comparison'],
        split_boundary='Fixed external EGAD12; never used to train/select heads or tune policy; keep all12 including asset failures',
        source_native_units_to_m=UNIT_M,selection=str(args.output_dir/'SELECTION.json'),selection_sha256=sha(args.output_dir/'SELECTION.json'),
        scene_boundary='Native-axis isolated synthetic scenes; explicit CoACD/mass/friction/unit approximation, not original AnyDex real scene or printed scale',
        original_AnyDex_CAD_identity_verified=False,original_scene_pose_reproduced=False,
        physics_steps=0,grasp_attempts=0,training_steps=0,network_forwards=0)
    save(args.output_dir/'objects.json',manifest)
    summary=dict(schema='anydex_boya_egad_external_preparation_v1',status='complete',complete=True,
        total_objects=12,ready_objects=sum(x['preflight_status']=='ready' for x in records),
        asset_failed_objects=[x['object_id']for x in records if x['preflight_status']!='ready'],
        selection_sha256=manifest['selection_sha256'],scene_manifest=str(args.output_dir/'objects.json'),
        scene_manifest_sha256=sha(args.output_dir/'objects.json'),asset_root=str(args.asset_root),
        source_units_to_m=UNIT_M,all12_identities_retained=True,script_sha256=sha(__file__),wall_s=time.monotonic()-started,
        physics_steps=0,grasp_attempts=0,training_steps=0,network_forwards=0)
    save(args.output_dir/'SCENE_PREPARATION.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    return 0 if summary['ready_objects']==12 else 2


if __name__=='__main__':
    raise SystemExit(main())
