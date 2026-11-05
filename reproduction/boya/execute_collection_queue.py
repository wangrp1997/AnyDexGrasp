#!/usr/bin/env python3
"""Execute source-bound AnyDex/Boya plans through the read-only native backend.

Serial: one resident native scene per homogeneous wave. Parallel: <=16 isolated
lanes per wave. Both modes start a new Kit process for each wave; cross-wave
scene reuse is not implemented. No cross-wave resident GPU service is claimed. This tool never
prepares/reorders grasp candidates, updates controls, or repairs assets.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents if (p/'reproduction/boya/trial_record.py').is_file()), Path('/home/wangrenpeng/AnyDexGrasp'))
CANON = Path('/home/wangrenpeng/CanonDex')
SCHEMA = 'anydex_boya_collection_queue_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def load(path): return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def native_imports():
    sys.path[:0] = [str(CANON), str(CANON/'scripts/research')]
    from canondex.grasp.arm_parallel import cohort_key
    return cohort_key


def trial_record_module():
    # Loading this file by absolute path also permits zero-physics /tmp tests.
    import importlib.util
    path = ROOT/'reproduction/boya/trial_record.py'
    if not path.is_file():
        path = Path('/home/wangrenpeng/AnyDexGrasp/reproduction/boya/trial_record.py')
    spec = importlib.util.spec_from_file_location('anydex_queue_trial_record', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def check_hashes(files):
    changed = [p for p, expected in files.items() if not Path(p).is_file() or sha(p) != expected]
    if changed: raise ValueError('Frozen inputs/source changed: ' + ', '.join(changed[:6]))


def freeze_file(files, path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None and digest != expected: raise ValueError('SHA mismatch: ' + str(path))
    if str(path) in files and files[str(path)] != digest: raise ValueError('Conflicting frozen input')
    files[str(path)] = digest
    return digest


def check_trial(row, inputs):
    """Fail before Kit if identity, canonical binding or original commands differ."""
    tr = trial_record_module()
    for key in ('trial_id', 'plan_dir', 'binding_json', 'commands_sha256',
                'trial_binding_sha256', 'object_id', 'gesture_id', 'extra_depth_index'):
        if key not in row: raise ValueError('Trial missing key: ' + key)
    identity = row.get('output_identity', row['trial_id'])
    if not isinstance(identity, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}', identity) is None:
        raise ValueError('Unsafe output identity')
    path = Path(row['plan_dir']).resolve(); binding_path = Path(row['binding_json']).resolve()
    plan = load(path/'plan.json'); binding = load(binding_path)
    tr.check_binding(binding)
    if plan.get('complete') is not True or plan.get('admitted') is not True or plan.get('approach_only'):
        raise ValueError('Queue accepts complete admitted full-lift plans only')
    digest = row['trial_binding_sha256']
    if binding['trial_binding_sha256'] != digest or plan.get('trial_binding_sha256') != digest:
        raise ValueError('Plan/binding identity mismatch')
    if (int(row['gesture_id']) != binding['multifinger_pose_finger_type']
            or int(row['gesture_id']) != plan['gesture_id']
            or int(row['extra_depth_index']) != binding['multifinger_pose_depth_type']
            or row['object_id'] != Path(plan['arguments']['object_root']).name):
        raise ValueError('Declared object/gesture/depth differs from frozen trial')
    if row['commands_sha256'] != plan['commands_sha256']:
        raise ValueError('Declared command SHA differs from plan')
    freeze_file(inputs, binding_path, row.get('binding_sha256'))
    for p, h in row.get('input_sha256', {}).items(): freeze_file(inputs, p, h)
    for name, expected in [('plan.json', None), ('commands.npz', row['commands_sha256']),
                           ('path_geometry.json', plan['path_geometry_sha256']),
                           ('lift_arm_geometry.json', plan.get('lift_arm_geometry_sha256'))]:
        freeze_file(inputs, path/name, expected)
    with np.load(path/'commands.npz', allow_pickle=False) as z:
        if str(z['trial_binding_sha256'].item()) != digest: raise ValueError('Commands do not echo binding')
        if (len(z['phase']) != 2100 or plan['total_command_steps'] != 2100
                or z['joint_target'].shape[0] != 2100
                or not {'close', 'hold', 'lift'}.issubset(set(map(str, z['phase'])))):
            raise ValueError('Exactly 2100 original approach/closure/hold/lift commands required')
        if any(z[k].dtype.hasobject for k in z.files): raise ValueError('Pickled command arrays forbidden')
    for key, hash_key in [('source_npz','source_npz_sha256'),
                          ('official_canonical_source','official_canonical_source_sha256')]:
        freeze_file(inputs, binding[key], binding[hash_key])
    for key in ('robot_urdf','hand_urdf','hand_scene','package','physics_config','diagnostic_config',
                'world_object_pose','arm_seed_file','collision_refinement'):
        if plan['arguments'].get(key): freeze_file(inputs, plan['arguments'][key])
    # Freeze the actual mesh files as well as their URDFs. Runtime native
    # audits remain unchanged and still compare every imported collision hull.
    object_root = Path(plan['arguments']['object_root'])
    urdfs = [Path(plan['arguments']['robot_urdf']), Path(plan['arguments']['hand_urdf']), object_root/'urdf/coacd.urdf']
    freeze_file(inputs, object_root/'mesh/simplified.obj')
    for urdf in urdfs:
        freeze_file(inputs, urdf)
        for node in ET.parse(urdf).getroot().findall('.//mesh'):
            filename = node.get('filename', '').removeprefix('file://')
            if not filename or filename.startswith('package://'):
                raise ValueError('Unresolved mesh URI in frozen URDF: ' + filename)
            freeze_file(inputs, urdf.parent/filename)
    for p in Path(plan['arguments']['reference_root']).rglob('*.json'): freeze_file(inputs, p)
    generation = Path(plan['arguments']['candidates'])
    for name in ('generation.json','candidates.npz'): freeze_file(inputs, generation/name)
    for p, h in load(generation/'generation.json')['input_sha256'].items(): freeze_file(inputs, p, h)
    for section in ('source_sha256','extra_input_sha256'):
        for p, h in plan.get(section, {}).items(): freeze_file(inputs, p, h)
    repair = path.parent/'arm_collision_refinements.json'
    repair_sha = freeze_file(inputs, repair) if repair.exists() else None
    return dict(row, plan_dir=str(path), binding_json=str(binding_path), output_identity=identity,
                cohort_key=native_imports()(plan), arm_collision_refinement_sha256=repair_sha)


def make_cohorts(trials, mode, batch_size):
    if type(batch_size) is not int or not 1 <= batch_size <= 16: raise ValueError('Batch size must be 1..16')
    if mode not in ('serial','parallel'): raise ValueError('Unknown execution mode')
    groups = {}
    for index, row in enumerate(trials):
        key = (row['cohort_key'], row['arm_collision_refinement_sha256'])
        groups.setdefault(key, []).append(index)
    waves = []
    for indices in groups.values():
        size = batch_size
        waves.extend(indices[i:i+size] for i in range(0, len(indices), size))
    return waves


def prepare(args):
    manifest = load(args.manifest)
    if manifest.get('complete') is not True or manifest.get('errors') or manifest.get('inputs_unchanged') is False:
        raise ValueError('Completed input preparation without infrastructure errors required')
    rows = manifest.get('trials')
    if not isinstance(rows,list) or not rows: raise ValueError('Nonempty manifest trials list required')
    inputs = {}; freeze_file(inputs, args.manifest)
    for p, h in manifest.get('source_sha256', {}).items(): freeze_file(inputs, p, h)
    if manifest.get('jobs_json'): freeze_file(inputs, manifest['jobs_json'], manifest.get('jobs_sha256'))
    for row in manifest.get('nulls', []):
        if row.get('no_candidate_json'): freeze_file(inputs, row['no_candidate_json'], row.get('no_candidate_sha256'))
    trials = [check_trial(r, inputs) for r in rows]
    for field in ('trial_id','output_identity','plan_dir','trial_binding_sha256'):
        if len({r[field] for r in trials}) != len(trials): raise ValueError('Repeated ' + field)
    # Freeze the invoked adapter and every read-only native Python dependency.
    sources = {str(Path(__file__).resolve()): sha(__file__)}
    tr = trial_record_module(); sources[str(Path(tr.__file__).resolve())] = sha(tr.__file__)
    for p in sorted([*(CANON/'canondex').rglob('*.py'),*(CANON/'scripts/research').glob('*.py')]):
        sources[str(p)] = sha(p)
    return dict(schema=SCHEMA, manifest=str(args.manifest.resolve()), manifest_sha256=sha(args.manifest),
        device=args.device,mode=args.mode,batch_size=args.batch_size,
        benchmark_only=bool(manifest.get('benchmark_only',False) or args.benchmark_only),
        paired_benchmark_id=args.paired_benchmark_id or manifest.get('paired_benchmark_id'),
        preparation_nulls=manifest.get('nulls', []),
        trials=trials,cohorts=make_cohorts(trials,args.mode,args.batch_size),
        input_sha256=inputs,source_sha256=sources,
        interpreter=sys.executable,
        scope='Original bound plans; serial resident within each <=16-plan wave; parallel shared clock within each <=16-lane wave; fresh process per wave; no online candidate generation or across-wave GPU residency')


def host_numpy(value):
    """Torch CUDA needs explicit host transfer; Warp .numpy() already transfers."""
    if hasattr(value, 'detach'): return value.detach().cpu().numpy()
    return np.asarray(value.numpy())


def worker(args):
    """Delegate scene construction/control/gates to the unchanged backend main."""
    native_imports()
    import benchmark_template_arm_parallel as backend
    work = load(args.worker_spec)
    # Only the cross-environment isolation guard's readback is adapted. Native
    # contact checks, original loops and per-trial thresholds remain untouched.
    def check_cross(self):
        count = 0
        for view in self.cross:
            pair_counts = host_numpy(view.get_contact_data(dt=self.runtime[0].runtime[0].get_physics_dt())[4])
            if pair_counts.dtype.kind not in 'iu' or np.any(pair_counts < 0):
                raise RuntimeError('Malformed cross-environment contact counts')
            value = int(pair_counts.sum())
            if value >= 16384: raise RuntimeError('Cross-environment contact buffer exhausted')
            count += value
        self.audit['max_cross_environment_contacts'] = max(self.audit['max_cross_environment_contacts'], count)
        self.audit['cuda_host_transfer_adapter'] = str(Path(__file__).resolve())
        if count: raise RuntimeError('Collision isolation failed between candidate worlds')
    backend.ParallelRuntime.check_cross = check_cross
    sys.argv = [str(CANON/'scripts/research/benchmark_template_arm_parallel.py'),
        '--plans',*[r['plan_dir'] for r in work['trials']], '--output-dir',work['output_dir'],
        '--device',work['device'],'--mode',work['mode'],'--worker']
    backend.main()


def finish_trial(row, cohort_dir, lane, trial_dir, cohort, returncode):
    """Null on missing/aborted/partial execution; only exact full traces label."""
    tr = trial_record_module(); trial_dir.mkdir(parents=True,exist_ok=False)
    binding = load(row['binding_json']); save(trial_dir/'trial_binding.json', binding)
    ev_path = cohort_dir/f'execution_{lane:03d}'/'evaluation.json'
    commands_path = cohort_dir/'plans'/f'p{lane:03d}'/'commands.npz'
    steps = 0; label = None; reason = 'missing_native_evaluation'; eligible = False
    guards = {}; phases = []
    if ev_path.exists():
        ev = load(ev_path); steps = ev.get('total_physics_steps',0)
        guards['native_scene_backend_confirmed'] = (
            type(cohort.get('native_scene',{}).get('physxScene:enableGPUDynamics')) is bool
            and cohort['native_scene']['physxScene:enableGPUDynamics'] == (cohort['device'] != 'cpu')
            and ev.get('device') == cohort['device'])
        guards['source_commands_unchanged'] = commands_path.is_file() and sha(commands_path) == row['commands_sha256']
        guards['native_echo_commands'] = ev.get('commands_sha256') == row['commands_sha256']
        guards['complete_2100_steps'] = ev.get('complete') is True and steps == 2100
        trace_path = ev_path.parent/'trace.npz'
        if trace_path.exists() and commands_path.exists():
            with np.load(trace_path,allow_pickle=False) as trace, np.load(commands_path,allow_pickle=False) as commands:
                phases = list(dict.fromkeys(map(str,trace['phase'])))
                names = list(map(str,trace['joint_names'])); columns = [names.index(str(n)) for n in commands['joint_names']]
                guards['original_target_arrays'] = bool(np.array_equal(trace['joint_target'][:,columns],commands['joint_target'].astype(np.float32)))
                guards['original_phase_arrays'] = bool(np.array_equal(trace['phase'],commands['phase']))
                guards['trace_2100_steps'] = len(trace['phase']) == 2100
                guards['original_desired_palm_arrays'] = bool(np.array_equal(trace['desired_palm'],commands['desired_palm']))
        guards['lift_present'] = 'lift' in phases
        guards['worker_completed_without_error'] = returncode == 0 and not cohort.get('error')
        guards['inputs_and_sources_unchanged'] = cohort.get('inputs_unchanged') is True and cohort.get('sources_unchanged') is True
        eligible = bool(guards and all(guards.values()))
        bound = dict(ev,trial_binding_sha256=row['trial_binding_sha256'],commands_sha256=row['commands_sha256'],
            original_native_evaluation_sha256=sha(ev_path),completed_phases=phases,label_eligible=eligible,
            backend='Isaac Sim / PhysX',queue_binding_guards=guards,
            success_definition='Unchanged native tabletop-lift task; not author transport-to-bin hardware success')
        if not eligible: bound['infrastructure_error'] = 'Queue full-execution/source/backend/trace contract not satisfied'
        save(trial_dir/'bound_evaluation.json',bound)
        if commands_path.exists() and guards.get('source_commands_unchanged'):
            record = tr.collect_record(binding,trial_dir/'bound_evaluation.json',commands_path)
            record['benchmark_only'] = cohort['benchmark_only']
            record['paired_benchmark_id'] = cohort.get('paired_benchmark_id')
            save(trial_dir/'training_record.json',record)
            label,reason = record['result'],record['outcome_status']
        else: reason = 'staged_commands_missing_or_changed'
    result = dict(trial_id=row['trial_id'],output_identity=row['output_identity'],object_id=row['object_id'],
        gesture_id=row['gesture_id'],extra_depth_index=row['extra_depth_index'],
        source_plan_dir=row['plan_dir'],binding_json=row['binding_json'],
        trial_binding_sha256=row['trial_binding_sha256'],commands_sha256=row['commands_sha256'],
        execution_dir=str(ev_path.parent),physics_steps=steps,label=label,label_eligible=eligible,
        null_reason=reason if label is None else None,guards=guards,
        benchmark_only=cohort['benchmark_only'],paired_benchmark_id=cohort.get('paired_benchmark_id'))
    save(trial_dir/'RESULTS.json',result)
    return result


def execute(args, prepared):
    root = args.output_dir.resolve()
    check_hashes(prepared['input_sha256']); check_hashes(prepared['source_sha256'])
    result = dict(schema=SCHEMA,status='running',manifest_sha256=prepared['manifest_sha256'],
        mode=args.mode,device=args.device,batch_size=args.batch_size,trials=[],cohorts=[],
        benchmark_only=prepared['benchmark_only'],paired_benchmark_id=prepared.get('paired_benchmark_id'),
        cross_wave_residency=False,serial_resident_within_wave=args.mode=='serial',
        preparation_nulls=prepared.get('preparation_nulls', []),
        planned_trial_count=len(prepared['trials'])+len(prepared.get('preparation_nulls', [])))
    began = time.monotonic(); save(root/'RESULTS.json',result)
    try:
        for i, indices in enumerate(prepared['cohorts']):
            output = root/'cohorts'/f'cohort_{i:04d}'
            rows = [prepared['trials'][j] for j in indices]
            work = dict(trials=rows,output_dir=str(output),device=args.device,mode=args.mode)
            work_path = root/'cohorts'/f'cohort_{i:04d}.WORK.json'; save(work_path,work)
            cmd = [sys.executable,'-B',str(Path(__file__).resolve()),'--worker-spec',str(work_path)]
            env = dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',PYTHONDONTWRITEBYTECODE='1',
                CANONDEX_COLLISION_PARTITION_CACHE=str(ROOT/'outputs/boya_collision_partition_cache'),
                OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='1')
            start = time.monotonic()
            with output.with_suffix('.log').open('w') as log:
                proc = subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            receipt_path = output/'result.json'
            receipt = load(receipt_path) if receipt_path.exists() else {}
            receipt.update(device=args.device,benchmark_only=prepared['benchmark_only'],
                paired_benchmark_id=prepared.get('paired_benchmark_id'))
            cohort_summary = dict(index=i,trial_indices=indices,returncode=proc.returncode,
                process_wall_s=time.monotonic()-start,receipt_path=str(receipt_path),
                actual_gpu_enabled=(receipt.get('native_scene') or {}).get('physxScene:enableGPUDynamics'),
                physics_ticks=receipt.get('physics_ticks',0),environment_steps=receipt.get('environment_steps',0),
                resident_scene=receipt.get('resident_scene'),parallel_scene=receipt.get('parallel_scene'),
                setup_s=receipt.get('setup_s'),loop_s=receipt.get('loop_s'))
            result['cohorts'].append(cohort_summary)
            for lane,row in enumerate(rows):
                result['trials'].append(finish_trial(row,output,lane,root/'trials'/row['output_identity'],receipt,proc.returncode))
            save(root/'RESULTS.json',result)
            check_hashes(prepared['input_sha256']); check_hashes(prepared['source_sha256'])
            if proc.returncode or receipt.get('error'):
                result['infrastructure_error'] = 'Worker failed; remaining queue retained as unexecuted null'
                break
        executed = {r['trial_id'] for r in result['trials']}
        for row in prepared['trials']:
            if row['trial_id'] not in executed:
                result['trials'].append(dict(trial_id=row['trial_id'],object_id=row['object_id'],gesture_id=row['gesture_id'],
                    extra_depth_index=row['extra_depth_index'],label=None,label_eligible=False,physics_steps=0,
                    null_reason='queue_not_dispatched_after_infrastructure_error',commands_sha256=row['commands_sha256'],
                    trial_binding_sha256=row['trial_binding_sha256'],benchmark_only=prepared['benchmark_only']))
        order = {r['trial_id']:i for i,r in enumerate(prepared['trials'])}
        result['trials'].sort(key=lambda r:order[r['trial_id']])
        result['status'] = 'complete' if all(r['label_eligible'] for r in result['trials']) and not result['preparation_nulls'] else 'complete_with_null'
    except BaseException:
        result.update(status='infrastructure_error',error=traceback.format_exc()); raise
    finally:
        result.update(wall_s=time.monotonic()-began,
            actual_grasps=sum(r['physics_steps']>0 for r in result['trials']),
            physical_steps=sum(r['physics_steps'] for r in result['trials']),
            positives=sum(r['label']==1 for r in result['trials']),negatives=sum(r['label']==0 for r in result['trials']),
            nulls=sum(r['label'] is None for r in result['trials'])+len(result['preparation_nulls']),
            benchmark_records_excluded_from_independent_training=prepared['benchmark_only'],
            input_sha256_unchanged=all(Path(p).is_file() and sha(p)==h for p,h in prepared['input_sha256'].items()),
            source_sha256_unchanged=all(Path(p).is_file() and sha(p)==h for p,h in prepared['source_sha256'].items()))
        save(root/'RESULTS.json',result)
        with (root/'training_records.jsonl').open('w') as f:
            for row in prepared['trials']:
                path = root/'trials'/row['output_identity']/'training_record.json'
                if path.exists():
                    record=load(path)
                    if record['label_eligible']: f.write(json.dumps(record,allow_nan=False)+'\n')


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path); p.add_argument('--output-dir',type=Path)
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    p.add_argument('--mode',choices=('serial','parallel'),default='serial')
    p.add_argument('--batch-size',type=int,default=2); p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--benchmark-only',action='store_true',help='Tag all records as paired backend diagnostics, excluded from independent training')
    p.add_argument('--paired-benchmark-id',help='Shared identity across separate CPU/GPU output runs')
    p.add_argument('--worker-spec',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker_spec: return a
    if not a.manifest or not a.output_dir: p.error('--manifest and --output-dir required')
    if not 1 <= a.batch_size <= 16: p.error('--batch-size must be 1..16')
    if a.paired_benchmark_id and not a.benchmark_only: p.error('--paired-benchmark-id requires --benchmark-only')
    return a


def main():
    args = parse_args()
    if args.worker_spec: return worker(args)
    root=args.output_dir.resolve()
    if root.exists() and not (root/'PREPARED.json').is_file(): raise ValueError('Output already exists without queue preparation')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.queue.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('Collection queue already running')
        if (root/'RESULTS.json').exists(): raise ValueError('Queue already dispatched; never replay existing outputs')
        if (root/'PREPARED.json').exists():
            prepared=load(root/'PREPARED.json')
            if (prepared['manifest_sha256'] != sha(args.manifest) or any(prepared[k] != getattr(args,k)
                    for k in ('mode','device','batch_size'))
                    or (args.benchmark_only and not prepared['benchmark_only'])
                    or (args.paired_benchmark_id and prepared['paired_benchmark_id'] != args.paired_benchmark_id)):
                raise ValueError('Prepared configuration changed')
            check_hashes(prepared['input_sha256']); check_hashes(prepared['source_sha256'])
        else:
            prepared=prepare(args);save(root/'PREPARED.json',prepared)
            (root/'cohorts').mkdir();(root/'trials').mkdir()
        if args.prepare_only:
            print(json.dumps(dict(status='prepared',trials=len(prepared['trials']),cohorts=len(prepared['cohorts']),
                prepared_sha256=sha(root/'PREPARED.json'),new_physics_steps=0)),flush=True)
            return
        execute(args,prepared)


if __name__ == '__main__': main()
