"""Freeze source-bound S40 clutter scenes without capture, planning or labels.

The old three-object builder stays unchanged. Each scene belongs entirely to
one original object split; placement failure consumes its place in the queue.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANON = Path('/home/wangrenpeng/CanonDex')
TASK = 'declared_boya_clutter_lift_v1'
QUEUE_SCHEMA = 'anydex_boya_author_collection_scene_queue_v1'
SPEC_SCHEMA = 'anydex_boya_author_collection_scene_spec_v1'
S40 = {0,2,5,7,8,9,11,14,15,17,18,20,21,22,26,27,29,30,34,36,
       37,38,40,41,43,44,46,48,51,52,56,57,58,60,61,62,63,66,69,70}
POLICY = dict(schema='anydex_boya_author_collection_scene_policy_v1',
    objects_per_scene=3, xy_AABB_world_m=[[-.95,-.02],[-.58,.23]],
    minimum_projected_AABB_gap_m=.003, source_mesh_bottom_world_m=.005,
    orientation='Independent world-Z uniform yaw[0,2pi) times source settled orientation',
    layout_attempts_per_object=256, object_replacement_after_failure=False,
    alternate_yaws_after_failure=False, dt_s=.004, initialization_steps=750,
    mass_kg=.1, friction=.6, bin=None, gestures=[1,6,7,8],
    candidate_attempts_per_gesture_scene=1, executions_per_gesture_scene=1,
    fit_target_per_gesture=1000, internal_target_per_gesture=200,
    labels_from_placement_null=False, labels_from_planning_null=False,
    old_435_reused=False, EGAD_training_objects=0,
    sampling_scope='Fixed-budget source-geometry placement, no outcome or arm reachability selection',
    settle_rule='Original author_scene.capture:750 ticks, original stationarity/contact/native20um gates; failure stays a scene null without resampling')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def verify(values):
    for path, expected in values.items():
        if sha(path) != expected:
            raise ValueError('Frozen scene dependency changed: '+path)


def rigid(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4,4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0,0,0,1], atol=1e-10, rtol=0)
            or not np.allclose(matrix[:3,:3].T@matrix[:3,:3], np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3,:3]),1,atol=1e-8,rtol=0)):
        raise ValueError('Proper finite source transform required')
    return matrix


def _execution_sources():
    paths = [Path(__file__).resolve()]
    paths += [HERE/name for name in ('author_scene.py','author_transport.py','author_lift.py',
                                     'settle_s40_scenes.py','render_s40_observations.py')]
    paths += sorted((CANON/'canondex').rglob('*.py'))
    paths += sorted((CANON/'scripts/research').glob('*.py'))
    return {str(path):sha(path) for path in paths}


def _catalog(source_manifest, native_summary, backend_path, split_manifest):
    inputs = {name:binding(path) for name,path in dict(source_manifest=source_manifest,
        native_summary=native_summary,backend=backend_path,split_manifest=split_manifest).items()}
    source, native, backend, splits = [read(inputs[name]['path'])
        for name in ('source_manifest','native_summary','backend','split_manifest')]
    # The declared resident profile spells out every field; native receipts
    # bind its canonical value, whereas input_sha256 binds exact file bytes.
    profile_identity=hashlib.sha256(json.dumps(read(backend['physics_config']),
        sort_keys=True,separators=(',',':')).encode()).hexdigest()
    rows = source['objects']
    if (source.get('schema') != 'anydex_boya_s40_scene_manifest_v1' or len(rows) != 40
            or {r.get('graspnet_object_id') for r in rows} != S40
            or any(r.get('scale') != 1. or r.get('split') != 'train' for r in rows)):
        raise ValueError('All40 original-scale outer-training S40 assets required')
    byid = {r['object_id']:r for r in rows}
    if (len(byid) != 40 or any(not re.fullmatch(r'graspnet_\d{3}',name) for name in byid)
            or any(r['object_id'] != f"graspnet_{r['graspnet_object_id']:03d}" for r in rows)):
        raise ValueError('Distinct S40 identities required; no foreign training objects')
    groups = splits.get('internal_mesh_groups', {})
    if set(groups) != {'fit','internal_check'} or [len(groups[k]) for k in ('fit','internal_check')] != [32,8]:
        raise ValueError('Original frozen32-fit/8-internal identity manifest required')
    split_ids, all_ids, all_digests = {}, set(), set()
    for name, original in [('fit','fit'),('internal','internal_check')]:
        ids = []
        for item in groups[original]:
            identity = item['object_id']; row = byid.get(identity)
            if (row is None or identity in all_ids or item['mesh_sha256'] in all_digests
                    or row['source_visual_mesh_sha256'] != item['mesh_sha256']):
                raise ValueError('Changed, duplicated or leaking original object split')
            ids.append(identity); all_ids.add(identity); all_digests.add(item['mesh_sha256'])
        split_ids[name] = sorted(ids)
    if all_ids != set(byid):
        raise ValueError('Split manifest must cover all40 source identities exactly')
    native_rows = {r['object_id']:r for r in native['results']}
    if native.get('complete') is not True or set(native_rows) != set(byid):
        raise ValueError('Complete all40 source native asset summary required')
    locks = {v['path']:v['sha256'] for v in inputs.values()}
    records = {}
    for identity, row in byid.items():
        root = Path(row['object_root']).resolve(); mesh = Path(row['mesh_path']).resolve()
        if sha(mesh) != row['mesh_sha256'] or sha(row['source_visual_mesh']) != row['source_visual_mesh_sha256']:
            raise ValueError('Changed source visual geometry: '+identity)
        if not row.get('settling_ready'):
            raise ValueError('Declared source settled orientation required: '+identity)
        rigid(row['world_T_object'])
        urdf = root/'urdf/coacd.urdf'
        parts = []
        for collision in ET.parse(urdf).findall('.//collision'):
            geometry = collision.find('geometry/mesh');origin = collision.find('origin')
            if geometry is None:
                raise ValueError('This source catalog requires mesh collision parts')
            origin_xyz = np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            origin_rpy = np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            scale = np.fromstring(geometry.get('scale','1 1 1'),sep=' ')
            if (origin_xyz.shape!=(3,) or origin_rpy.shape!=(3,) or scale.shape!=(3,)
                    or np.any(origin_xyz!=0) or np.any(origin_rpy!=0) or np.any(scale!=1)):
                raise ValueError('Source collision parts must remain untransformed unit-scale geometry')
            parts.append((urdf.parent/geometry.get('filename')).resolve())
        if not parts:
            raise ValueError('Nonempty source collision parts required')
        native_row = native_rows[identity]; result_path = Path(native_row['result_json']).resolve()
        result = read(result_path)
        if result.get('complete') is not True or result.get('original_or_refined_within_20um') is not True:
            raise ValueError('Existing source native20um preflight must be clear: '+identity)
        native_inputs=result['binding']['inputs']
        if (native_inputs.get('object_id')!=identity or native_inputs.get('scale')!=1.
                or native_inputs.get('physics_profile_sha256')!=profile_identity):
            raise ValueError('Native preflight identity, scale or physics profile differs: '+identity)
        for relative,expected in native_inputs['source_files'].items():
            if sha(root/relative) != expected:
                raise ValueError('Asset no longer matches its native preflight: '+identity)
        for relative,expected in result.get('artifact_sha256',{}).items():
            artifact=Path(relative)
            if not artifact.is_absolute():artifact=Path(result['cache_directory'])/artifact
            if sha(artifact) != expected:
                raise ValueError('Changed native preflight artifact: '+identity)
        files = [p for p in root.rglob('*') if p.is_file()]+[Path(row['source_visual_mesh']),result_path]
        files += [Path(p) for p in native_row['artifacts']]
        repair = native_row.get('repair_path')
        if repair:
            if sha(repair) != native_row['repair_sha256']:
                raise ValueError('Changed native collision refinement')
            files.append(Path(repair))
        asset_locks = {str(p.resolve()):sha(p) for p in files}
        locks.update(asset_locks)
        records[identity] = dict(object_id=identity, object_root=str(root), source_mesh=str(mesh),
            source_mesh_sha256=row['mesh_sha256'], source_object_digest=row['source_visual_mesh_sha256'],
            source_visual_mesh=str(Path(row['source_visual_mesh']).resolve()),
            source_settled_R=rigid(row['world_T_object'])[:3,:3].tolist(),
            collision_part_paths=list(map(str,parts)), collision_refinement=repair,
            collision_refinement_sha256=native_row.get('repair_sha256'), input_sha256=asset_locks)
    locks[backend['physics_config']] = sha(backend['physics_config'])
    return dict(schema='anydex_boya_author_collection_scene_catalog_v1',task=TASK,
        inputs=inputs, input_sha256=locks, source_sha256=_execution_sources(),
        split_object_ids=split_ids, objects=records, camera=source['camera'],
        physics_config=backend['physics_config'], old_labels_loaded=False,
        source_scope='S40 GraspNet identities from local DGN2-derived geometry; not verified AnyDex L144 CAD')


def _vertices(path):
    import trimesh
    mesh = trimesh.load_mesh(path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('Finite nonempty source vertices required: '+str(path))
    return vertices


def _geometry(record, cache):
    identity = record['object_id']
    if identity not in cache:
        visual = _vertices(record['source_mesh'])
        collision = np.concatenate([_vertices(p) for p in record['collision_part_paths']])
        cache[identity] = (visual,collision)
    return cache[identity]


def gap_clear(a, b, gap):
    """AABB separation on at least one XY axis, including the declared gap."""
    a,b = np.asarray(a),np.asarray(b)
    separation = np.maximum(b[0,:2]-a[1,:2],a[0,:2]-b[1,:2])
    # Only absorb floating subtraction roundoff (under2e-15m in this rig),
    # so a decimal exactly3mm gap has the same decision on either side.
    roundoff = 8*np.finfo(float).eps*max(1.,float(np.max(np.abs([a,b]))))
    return bool(np.any(separation >= gap-roundoff))


def sample_scene(catalog, split, index, seed, *, geometry_cache=None):
    """Fixed object draw/yaw, at most256 XY attempts/body, no replacement."""
    if split not in ('fit','internal') or type(index) is not int or index < 0 or type(seed) is not int or seed < 0:
        raise ValueError('Explicit split and nonnegative integer index/seed required')
    rng = np.random.default_rng(np.random.SeedSequence([seed,0 if split=='fit' else 1,index]))
    identities = rng.choice(catalog['split_object_ids'][split], POLICY['objects_per_scene'], replace=False).tolist()
    yaws = rng.uniform(0.,2*np.pi,len(identities))
    domain = np.asarray(POLICY['xy_AABB_world_m']); placed,rows = [],[]
    cache = geometry_cache if geometry_cache is not None else {}
    failed = False
    for identity,yaw in zip(identities,yaws):
        record = catalog['objects'][identity]
        c,s = np.cos(yaw),np.sin(yaw)
        rotation = np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])@np.asarray(record['source_settled_R'])
        visual,collision = _geometry(record,cache)
        v,cverts = visual@rotation.T,collision@rotation.T
        low = np.minimum(v.min(axis=0),cverts.min(axis=0)); high = np.maximum(v.max(axis=0),cverts.max(axis=0))
        offset_z = POLICY['source_mesh_bottom_world_m']-float(v[:,2].min())
        row = dict(object_id=identity,source_object_digest=record['source_object_digest'],yaw_world_rad=float(yaw),
            R_world_object=rotation.tolist(), attempts=0,world_T_object=None,source_world_AABB_m=None,proxy_world_AABB_m=None)
        if failed:
            row['placement_status']='not_attempted_after_scene_null';rows.append(row);continue
        lower,upper = domain[0]-low[:2],domain[1]-high[:2]
        if (upper<lower).any() or low[2]+offset_z < 0:
            row['placement_status']='footprint_or_proxy_floor_incompatible';failed=True;rows.append(row);continue
        for attempt in range(1,POLICY['layout_attempts_per_object']+1):
            xy = rng.uniform(lower,upper); translation=np.r_[xy,offset_z]
            bounds=np.array([low+translation,high+translation]); row['attempts']=attempt
            if all(gap_clear(bounds,other,POLICY['minimum_projected_AABB_gap_m']) for other in placed):
                T=np.eye(4);T[:3,:3]=rotation;T[:3,3]=translation
                row.update(placement_status='placed',world_T_object=T.tolist(),
                    source_world_AABB_m=np.array([v.min(axis=0)+translation,v.max(axis=0)+translation]).tolist(),
                    proxy_world_AABB_m=bounds.tolist())
                placed.append(bounds);break
        else:
            row['placement_status']='fixed_layout_budget_exhausted';failed=True
        rows.append(row)
    return dict(schema=SPEC_SCHEMA,task=TASK,scene_id=f's40_lift_{split}_{index:05d}_s{seed}',
        split=split,scene_index=index,seed=seed,rng=dict(bit_generator='PCG64',seed_sequence_entropy=[seed,0 if split=='fit' else 1,index]),
        policy=copy.deepcopy(POLICY),source_object_ids=identities,
        source_object_digests={name:catalog['objects'][name]['source_object_digest'] for name in identities},
        objects=rows,status='placement_null' if failed else 'ready',
        physics_steps=0,planning_calls=0,new_labels=0,placement_null_is_training_label=False)


def generate_queue(source_manifest,native_summary,backend_path,split_manifest,output_dir,*,
                   seed=2026111401,fit_scene_count=8,internal_scene_count=2):
    if any(type(n) is not int or n<0 for n in (seed,fit_scene_count,internal_scene_count)) or fit_scene_count+internal_scene_count == 0:
        raise ValueError('Positive finite scene budget and nonnegative integer seed required')
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    catalog=_catalog(source_manifest,native_summary,backend_path,split_manifest)
    save(output/'CATALOG.json',catalog);catalog_binding=binding(output/'CATALOG.json')
    protocol=dict(schema=QUEUE_SCHEMA,task=TASK,seed=seed,policy=copy.deepcopy(POLICY),
        inputs=catalog['inputs'],sources={p:dict(path=p,sha256=h) for p,h in catalog['source_sha256'].items()},
        split_manifest=catalog['inputs']['split_manifest'],
        catalog=catalog_binding,split_object_ids=catalog['split_object_ids'],
        scene_counts=dict(fit=fit_scene_count,internal=internal_scene_count),
        maximum_gesture_scene_attempts=4*(fit_scene_count+internal_scene_count),
        sampler_retries_after_null=0,source_only_pure_preparation=True,
        target_is_capacity_requirement_not_current_labels=True)
    save(output/'QUEUE_PROTOCOL.json',protocol)  # Before any random scene draw or geometry placement.
    entries=[];cache={}
    for split,count in [('fit',fit_scene_count),('internal',internal_scene_count)]:
        for index in range(count):
            spec=sample_scene(catalog,split,index,seed,geometry_cache=cache)
            spec['catalog']=catalog_binding
            path=output/'specs'/(spec['scene_id']+'.json');save(path,spec)
            entries.append(dict(scene_id=spec['scene_id'],split=split,scene_spec=binding(path),
                source_object_ids=spec['source_object_ids'],source_object_digests=spec['source_object_digests'],
                status=spec['status'],max_candidates_per_gesture=1,max_executions_per_gesture=1))
    verify(catalog['input_sha256']);verify(catalog['source_sha256'])
    queue=dict(protocol,protocol=binding(output/'QUEUE_PROTOCOL.json'),scenes=entries,
        scene_builder=binding(__file__)|dict(command='build',argv_template=[sys.executable,'-B',str(Path(__file__).resolve()),
            'build','--scene-spec','{scene_spec}','--output-dir','{output_dir}']),
        status='complete',placement_null_count=sum(r['status']=='placement_null' for r in entries),
        physics_steps=0,planning_calls=0,new_labels=0,formal_collection_started=False)
    save(output/'QUEUE.json',queue)
    return queue


def build_scene(scene_spec,output_dir):
    """Validate one frozen layout and emit the unchanged capture input schema."""
    scene_spec=Path(scene_spec).resolve();spec=read(scene_spec)
    if spec.get('schema') != SPEC_SCHEMA or spec.get('task') != TASK or spec['policy'] != POLICY:
        raise ValueError('Frozen explicit source collection scene spec required')
    catalog=read(spec['catalog']['path'])
    if sha(spec['catalog']['path']) != spec['catalog']['sha256']:
        raise ValueError('Scene catalog changed')
    verify(catalog['input_sha256']);verify(catalog['source_sha256'])
    split=spec['split'];ids=spec['source_object_ids']
    if len(ids)!=3 or len(set(ids))!=3 or not set(ids)<=set(catalog['split_object_ids'][split]):
        raise ValueError('All original bodies must belong exclusively to this scene split')
    recomputed=sample_scene(catalog,split,spec['scene_index'],spec['seed'])
    if {k:v for k,v in spec.items() if k!='catalog'} != recomputed:
        raise ValueError('Changed frozen deterministic placement')
    if spec['status'] != 'ready':
        raise ValueError('placement_null is terminal; no scene or label may be fabricated')
    rows=[];inputs={str(scene_spec):sha(scene_spec),spec['catalog']['path']:spec['catalog']['sha256']}
    for placed in spec['objects']:
        old=catalog['objects'][placed['object_id']];inputs.update(old['input_sha256'])
        rows.append(dict(object_id=old['object_id'],object_root=old['object_root'],scale=1.,
            world_T_object=placed['world_T_object'],mass_kg=.1,friction=.6,
            source_mesh=old['source_mesh'],source_mesh_sha256=old['source_mesh_sha256'],
            source_visual_mesh=old['source_visual_mesh'],source_visual_mesh_sha256=old['source_object_digest'],
            mesh_group_sha256=old['source_object_digest'],
            collision_refinement=old['collision_refinement'],collision_refinement_sha256=old['collision_refinement_sha256']))
    inputs[catalog['physics_config']]=catalog['input_sha256'][catalog['physics_config']]
    scene=dict(schema='anydex_boya_author_lift_scene_v1',task=TASK,scene_id=spec['scene_id'],
        target_object_id=ids[0],original_object_ids=ids,objects=rows,bin=None,
        target_semantics='Compatibility primary only; any original body can satisfy the lift task',
        camera=catalog['camera'],physics_config=catalog['physics_config'],dt_s=.004,
        initialization_steps=750,seed=spec['seed'],input_sha256=inputs,source_sha256=catalog['source_sha256'],
        source_collection_split=split,dataset_split=split,
        all_source_mesh_sha256=[catalog['objects'][name]['source_object_digest'] for name in ids],
        source_object_digests=spec['source_object_digests'],
        collection_scene_spec=binding(scene_spec),collection_scene_policy=POLICY,
        scope='New S40 same-split clutter placement; existing declared Boya simulation, not L144/hardware reproduction',
        state_boundary='Initialize all bodies together, original750tick settling; measured-pose capture then independent cold recreation',
        task_boundary='Original lift only; no bin, transport, release or extra lift hold',training_started=False,grasp_attempts=0)
    # This imports the existing validator only; no capture or simulator import.
    import author_scene
    author_scene.validate(scene)
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    save(output/'scene.json',scene)
    return scene


def main():
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('generate')
    for name in ('source-manifest','native-summary','backend','split-manifest','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,default=2026111401)
    p.add_argument('--fit-scene-count',type=int,required=True)
    p.add_argument('--internal-scene-count',type=int,required=True)
    p=subs.add_parser('build');p.add_argument('--scene-spec',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='generate':
        result=generate_queue(a.source_manifest,a.native_summary,a.backend,a.split_manifest,a.output_dir,
            seed=a.seed,fit_scene_count=a.fit_scene_count,internal_scene_count=a.internal_scene_count)
        print(json.dumps(dict(queue=str(a.output_dir.resolve()/'QUEUE.json'),scenes=len(result['scenes']),placement_nulls=result['placement_null_count'])))
    else:
        result=build_scene(a.scene_spec,a.output_dir)
        print(json.dumps(dict(scene=str(a.output_dir.resolve()/'scene.json'),scene_id=result['scene_id'])))


if __name__=='__main__':main()
