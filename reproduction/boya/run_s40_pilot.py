"""Bounded S40 Boya collection smoke, independent of the old DGN score heads.

This is a declared simulation port, not the author's hardware protocol. It
uses the two numerically first S40 source IDs, four fixed Boya gestures and
extra-depth0: eight slots. Preparation is default; --execute authorizes each
ready slot once. Old output directories cannot be resumed or overwritten.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_s40_scenes import sha, save

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path('/home/wangrenpeng/CanonDex/scripts/research/preprocess_native_object.py')
IDS = ('graspnet_000', 'graspnet_002')
GESTURES = (1, 6, 7, 8)


def read(path):
    return json.loads(Path(path).read_text())


def verify(files):
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError('Frozen input or implementation changed: ' + path)


def run_command(command, log):
    started = time.monotonic()
    with Path(log).open('xb') as f:
        p = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    result = dict(argv=list(map(str, command)), returncode=p.returncode, log=str(log),
                  log_sha256=sha(log), wall_s=time.monotonic()-started)
    save(str(log)+'.receipt.json', result)
    if p.returncode:
        raise RuntimeError('Stage failed; retained log: ' + str(log))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--backend-config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--native-deps', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    for name in ['scene_manifest','output_dir','calibration','backend_config','checkpoint','native_deps']:
        setattr(args, name, getattr(args, name).resolve())
    out = args.output_dir
    if out.exists() and (set(p.name for p in out.iterdir()) != {'scenes'} or not args.scene_manifest.is_relative_to(out/'scenes')):
        raise ValueError('Fresh pilot stage required; only its completed scenes directory may preexist')
    manifest = read(args.scene_manifest)
    if (manifest.get('schema') != 'anydex_boya_s40_scene_manifest_v1' or manifest['scale'] != 1.
            or len(manifest['objects']) != 40 or any(r['split'] != 'train' for r in manifest['objects'])):
        raise ValueError('Complete native-scale S40 training manifest required')
    selected = [r for r in manifest['objects'] if r['object_id'] in IDS]
    if tuple(r['object_id'] for r in selected) != IDS:
        raise ValueError('Fixed predeclared source-ID selection changed')
    # Consumer observation/representation paths must bind this exact fresh run.
    for row in selected:
        if Path(row['observation_json']) != out/'observations'/row['object_id']/'observation.json':
            raise ValueError('Manifest observation does not bind the new run')
        if Path(row['representation_dir']) != out/'representations'/row['object_id']:
            raise ValueError('Manifest CGR destination does not bind the new run')
    out.mkdir(parents=True, exist_ok=True)
    (out/'logs').mkdir()
    source_paths = list((ROOT/'reproduction/boya').glob('*.py')) + [NATIVE]
    # Include every read-only native implementation, not merely wrapper scripts.
    source_paths += list(Path('/home/wangrenpeng/CanonDex/canondex').rglob('*.py'))
    source_paths += list(Path('/home/wangrenpeng/CanonDex/scripts/research').glob('*.py'))
    sources = {str(p.resolve()): sha(p) for p in source_paths}
    inputs = {str(p): sha(p) for p in [args.scene_manifest, args.calibration, args.backend_config, args.checkpoint]}
    backend = read(args.backend_config)
    for key in ['robot_urdf','reference_root','hand_urdf','hand_scene','package','physics_config','diagnostic_config','arm_collision_refinement','reference_plan']:
        p = Path(backend[key])
        if p.is_file():
            inputs[str(p.resolve())] = sha(p)
    for row in selected:
        inputs.update(row['source_input_sha256'])
        for pk, hk in [('mesh_path','mesh_sha256'),('urdf_path','urdf_sha256'),('pose_path','pose_sha256'),('metadata_path','metadata_sha256')]:
            inputs[row[pk]] = row[hk]
        for part in row['convex_parts']:
            inputs[part['path']] = part['sha256']
    protocol = dict(schema='anydex_boya_s40_pilot_v1', object_ids=list(IDS), gesture_ids=list(GESTURES),
        extra_depth_indices=[0], slots=8, candidate_limit=500, geometry_keep=8, max_arm_plans=4,
        physics_device='cpu', execution_mode='serial_resident_per_object', object_execution_workers=2,
        execute_authorized=args.execute, source_manifest=str(args.scene_manifest),
        input_sha256=inputs, source_sha256=sources, representation_device=args.device,
        collection_head_loaded=False, score_threshold_used=False,
        deployment_score_first_used=False, new_training_updates=0,
        protocol_boundary='Fixed coverage Boya simulation collection port: existing both-flip geometry, fixed extra-depth0, complete-arm gates, isolated single-object scenes and nominal position closure differ from released DH3 hardware collection. No old DGN decision weights, no held-out EGAD input.',
        dataset_boundary='S40 derived mesh identities, all40 train; 000/002 chosen by numeric source ID before any grasp result. Original physical parameters, exact author CAD and real cluttered scenes remain unverified.',
        nominal_steps_per_attempt=2100, ordinary_negative_continues=True, null_is_not_negative=True,
        source_scale=1., started_unix=time.time())
    verify(inputs); verify(sources)
    save(out/'PILOT_PROTOCOL.json', protocol)
    progress = dict(schema='anydex_boya_s40_pilot_progress_v1', status='starting', stages={},
                    protocol=str(out/'PILOT_PROTOCOL.json'), protocol_sha256=sha(out/'PILOT_PROTOCOL.json'),
                    new_training_updates=0, decision_head_loaded=False)
    started = time.monotonic()

    def publish(status):
        progress.update(status=status, elapsed_s=time.monotonic()-started)
        save(out/'PROGRESS.json', progress)
        print('S40_PILOT '+json.dumps(dict(status=status, elapsed_s=progress['elapsed_s'])), flush=True)

    def command(name, values):
        return run_command([sys.executable,'-B',*map(str,values)], out/'logs'/(name+'.log'))

    try:
        publish('rendering')
        progress['stages']['observations'] = command('observations', [ROOT/'reproduction/boya/render_s40_observations.py',
            '--objects-json',args.scene_manifest,'--output-dir',out/'observations','--object-ids',*IDS])
        collection = dict(schema='anydex_boya_s40_fixed_pilot_manifest_v1', objects=selected,
                          scene_manifest_sha256=sha(args.scene_manifest))
        save(out/'pilot_objects.json', collection)
        publish('representation')
        progress['stages']['representation'] = command('representation',[ROOT/'reproduction/boya/run_collection_representation.py',
            '--manifest',out/'pilot_objects.json','--run-dir',out,'--device',args.device,'--checkpoint',args.checkpoint,
            '--native-deps',args.native_deps,'--seed','20261106'])
        verify(inputs); verify(sources)
        publish('native_assets')
        def asset(row):
            oid = row['object_id']
            receipt = command('native_'+oid,[NATIVE,'--object-root',row['object_root'],'--scale','1',
                '--physics-config',backend['physics_config'],'--cache-dir',out/'native_cache','--output-dir',out/'native_assets'/oid,
                '--max-failed-partitions','32'])
            result = read(out/'native_assets'/oid/'result.json')
            if not result.get('complete') or not result.get('original_or_refined_within_20um') or result['trial_physics_steps'] != 0:
                raise ValueError('Native preflight did not pass unchanged gate')
            return oid,dict(receipt=receipt,result=result)
        with ThreadPoolExecutor(max_workers=2) as executor:
            native = dict(executor.map(asset, selected))
        progress['stages']['native_assets'] = {k:v['receipt'] for k,v in native.items()}
        publish('geometry')
        slots = [dict(object_id=oid, gesture_id=g, extra_depth_index=0,
                      trial_id=oid+'_g'+str(g)+'_d0') for oid in IDS for g in GESTURES]
        def geometry(slot):
            oid, g, tid = slot['object_id'],slot['gesture_id'],slot['trial_id']
            dest = out/'pools'/oid/('g'+str(g))/'d0'
            receipt = command('geometry_'+tid,[ROOT/'reproduction/boya/build_collection_pool.py',
                '--observation',out/'observations'/oid/'observation.json','--frame-dir',out/'representations'/oid,
                '--calibration',args.calibration,'--gesture-id',g,'--extra-depth-index','0',
                '--candidate-limit','500','--keep','8','--output-dir',dest])
            return tid, receipt
        with ThreadPoolExecutor(max_workers=2) as executor:
            progress['stages']['geometry'] = dict(executor.map(geometry, slots))
        verify(inputs); verify(sources)
        publish('planning')
        jobs = []
        for slot in slots:
            oid = slot['object_id']
            job = dict(**slot, split='train',candidate_dir=str(out/'pools'/oid/('g'+str(slot['gesture_id']))/'d0'),
                       observation_json=str(out/'observations'/oid/'observation.json'),
                       source_npz=str(out/'representations'/oid/'raw_camera_candidates.npz'),
                       calibration=str(args.calibration),max_arm_plans=4)
            native_result = native[oid]['result']
            if native_result['repair_required']:
                # Published cache provides the exact source-union repair identity.
                rp = Path(native_result['cache_directory'])/'repair.json'
                job.update(collision_refinement=str(rp),collision_refinement_sha256=sha(rp))
            jobs.append(job)
        save(out/'planning_jobs.json',dict(backend_config=str(args.backend_config),jobs=jobs))
        progress['stages']['planning'] = command('planning',[ROOT/'reproduction/boya/prepare_collection_queue.py',
            '--jobs-json',out/'planning_jobs.json','--output-dir',out/'planning','--workers','2'])
        queue = read(out/'planning/queue_manifest.json')
        if not queue['complete'] or queue['errors'] or not queue['inputs_unchanged']:
            raise ValueError('Planning queue has an infrastructure problem')
        progress.update(ready_slots=len(queue['trials']), preparation_null_slots=len(queue['nulls']))
        verify(inputs); verify(sources)
        if args.execute and queue['trials']:
            publish('executing')
            def execute(oid):
                rows = [r for r in queue['trials'] if r['object_id'] == oid]
                if not rows:
                    return oid, dict(no_ready_plans=True,physics_steps=0,training_records=[])
                manifest_path = out/(oid+'_queue.json')
                subqueue = dict(queue,trials=rows,nulls=[r for r in queue['nulls'] if r['object_id']==oid])
                save(manifest_path,subqueue)
                dest = out/'execution'/oid
                receipt = command('execution_'+oid,[ROOT/'reproduction/boya/execute_collection_queue.py',
                    '--manifest',manifest_path,'--output-dir',dest,'--device','cpu','--mode','serial','--batch-size','16'])
                actual = read(dest/'RESULTS.json')
                if actual.get('infrastructure_error') or actual.get('status') == 'infrastructure_error':
                    raise RuntimeError('Execution infrastructure failure retained: '+str(dest))
                return oid,dict(receipt=receipt,result=actual)
            with ThreadPoolExecutor(max_workers=2) as executor:
                progress['stages']['execution'] = dict(executor.map(execute, IDS))
        verify(inputs); verify(sources)
        progress.update(inputs_unchanged=True,sources_unchanged=True)
        publish('complete' if args.execute else 'prepared_not_executed')
    except BaseException as error:
        progress.update(error=repr(error),partial_outputs_preserved=True,no_automatic_retry=True)
        publish('paused_infrastructure')
        raise


if __name__ == '__main__':
    main()
