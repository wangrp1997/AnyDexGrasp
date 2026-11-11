"""Bounded rank-ordered new-pose diagnostic, never a top-1 benchmark.

Three prior failed objects, up to three previously unexecuted admitted poses each.
Reuses all frozen inputs and the accepted controlled-wrist executor. Preparation
binds actual historical candidate arrays, rankings, dependencies and job files;
execution refuses redispatch. No model update, new CGR or training writeback.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import numpy as np

ROOT = Path('/home/wangrenpeng/AnyDexGrasp')
B = ROOT/'reproduction/boya'
BASE = ROOT/'outputs/egad_external12_20261107_v1'
OLD_FLOAT = ROOT/'outputs/egad_floating_three_20261108_v1'
IDS = ('egad_A3', 'egad_A6', 'egad_E0')
POSE_KEYS = ('proposal_palm', 'proposal_q', 'proposal_q_close', 'proposal_approach_direction', 'world_T_object')
PHASES = {'prehold': 800, 'approach': 500, 'close': 250, 'hold': 50, 'lift': 500}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def arrays(pool):
    with np.load(Path(pool)/'candidates.npz', allow_pickle=False) as z:
        return {k: z[k].copy() for k in POSE_KEYS}


def same_pose(a, b):
    # Score/rank metadata can differ across heads while the physical pose is identical.
    return all(a[k].shape == b[k].shape and np.allclose(a[k], b[k], atol=2e-12, rtol=0) for k in POSE_KEYS)


def pose_digest(values):
    h = hashlib.sha256()
    for k in POSE_KEYS:
        x = np.asarray(values[k], dtype='<f8')
        h.update(k.encode())
        h.update(str(x.shape).encode())
        h.update(x.tobytes())
    return h.hexdigest()


def history():
    """Only actual historical trials contribute to the pose exclusion ledger."""
    result = []
    q = read(BASE/'planning/queue_manifest.json')
    executions = read(BASE/'execution/RESULTS.json')
    for r in executions['results']:
        oid = r['object_id']
        if oid not in IDS or r['physical_steps'] <= 0:
            continue
        selected = next(x for x in q['objects'] if x['object_id'] == oid)['selected']
        if selected['trial_id'] != r['trial']['trial_id']:
            raise ValueError('Historical execution/selected candidate binding differs')
        rank = int(selected['trial_id'].rsplit('_rank_', 1)[1])
        pool = BASE/'diagnostic'/oid/f'pool_{rank:05d}'
        result.append(dict(source_object_id=oid, candidate_dir=str(pool),
                           evaluation=str(Path(r['trial']['execution_dir'])/'evaluation.json'),
                           source_run=str(BASE), physical_steps=r['physical_steps']))
    runs = [OLD_FLOAT] + [Path(x['run']) for x in read(ROOT/'outputs/boya_alignment_20261108_v1/SUMMARY.json')['runs']]
    for run in runs:
        for r in read(run/'RESULTS.json')['results']:
            job = read(run/'jobs'/(r['object_id']+'.json'))
            oid = job.get('source_object_id', job['object_id'])
            if oid not in IDS or r['physical_steps'] <= 0:
                continue
            result.append(dict(source_object_id=oid, candidate_dir=job['candidate_dir'],
                               evaluation=r['evaluation'], source_run=str(run), physical_steps=r['physical_steps']))
    for r in result:
        r['evaluation_sha256'] = sha(r['evaluation'])
        r['pose_array_sha256'] = pose_digest(arrays(r['candidate_dir']))
    return result


def choose(admitted, prior, limit=3):
    if not 0 <= limit <= 3:
        raise ValueError('At most three fresh candidates per object')
    selected, excluded = [], []
    previous = [(r, arrays(r['candidate_dir'])) for r in prior]
    for c in sorted(admitted, key=lambda x: x['candidate_rank']):
        a = arrays(c['candidate_dir'])
        matches = [r for r, b in previous if same_pose(a, b)]
        duplicate = any(same_pose(a, arrays(x['candidate_dir'])) for x in selected)
        if matches or duplicate:
            excluded.append(dict(candidate_rank=c['candidate_rank'], candidate_dir=c['candidate_dir'],
                                 historical_evaluations=[m['evaluation'] for m in matches], duplicate_selected=duplicate))
        elif len(selected) < limit:
            selected.append(dict(c, pose_array_sha256=pose_digest(a)))
    return selected, excluded


def prepare(out):
    if out.exists():
        raise ValueError('Fresh output directory required; no overwrite')
    prior = history()
    old = read(OLD_FLOAT/'protocol.json')
    source_hashes = dict(old['source_sha256'])
    source_hashes[str(Path(__file__).resolve())] = sha(__file__)
    source_hashes[str(B/'render_egad_floating_three.py')] = sha(B/'render_egad_floating_three.py')
    input_hashes = dict(old['input_sha256'])
    def freeze(path):
        input_hashes[str(Path(path).resolve())] = sha(path)
    for r in prior:
        freeze(r['evaluation'])
        for name in ('candidates.npz', 'generation.json'):
            freeze(Path(r['candidate_dir'])/name)
    freeze(ROOT/'outputs/boya_alignment_20261108_v1/SUMMARY.json')
    freeze(BASE/'EVALUATION_FREEZE.json')
    rows = {r['object_id']: r for r in read(BASE/'objects.json')['objects']}
    cases, selection = [], []
    for oid in IDS:
        admitted = []
        freeze(BASE/'diagnostic'/oid/'RESULTS.json')
        for path in sorted((BASE/'diagnostic'/oid).glob('pool_*/generation.json')):
            generation = read(path)
            record = generation['records'][0]
            if not generation['complete'] or not record['geometry']['ok']:
                continue
            if sha(path.parent/'candidates.npz') != generation['candidates_sha256']:
                raise ValueError('Candidate NPZ changed')
            admitted.append(dict(source_object_id=oid, candidate_rank=record['global_score_rank'],
                                 candidate_dir=str(path.parent), candidate_record=record))
            freeze(path)
            freeze(path.parent/'candidates.npz')
        chosen, exclusions = choose(admitted, [r for r in prior if r['source_object_id'] == oid])
        selection.append(dict(object_id=oid, admitted_ranks=[r['candidate_rank'] for r in admitted],
                              selected_ranks=[r['candidate_rank'] for r in chosen], exclusions=exclusions))
        for c in chosen:
            c['object_id'] = f"{oid}_rank{c['candidate_rank']:05d}"
            c['object_row'] = rows[oid]
            c['old_execution'] = None
            c['selection_reason'] = 'First three still-unexecuted physical poses in original preactivation geometry-admitted rank order'
            cases.append(c)
            pool = Path(c['candidate_dir'])
            freeze(pool/'world_object_pose.json')
            for p, h in read(pool/'generation.json')['input_sha256'].items():
                if sha(p) != h:
                    raise ValueError('Candidate dependency changed: '+p)
                input_hashes[p] = h
    if not cases or len(cases) > 9:
        raise ValueError('No candidates or exceeded fixed budget')
    for section in (source_hashes, input_hashes):
        for p, h in section.items():
            if sha(p) != h:
                raise ValueError('Frozen source/input changed: '+p)
    out.mkdir(parents=True)
    (out/'jobs').mkdir()
    protocol = dict(schema='anydex_boya_candidate_diagnostic_v1', created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    purpose='Distinguish poor selected pose from absence of usable poses in a bounded fixed admitted candidate pool',
                    source_external_protocol_sha256=sha(BASE/'protocol.json'), source_external_run=str(BASE),
                    source_object_ids=list(IDS), object_ids=[c['object_id'] for c in cases], cases=cases,
                    history=prior, selection=selection, candidate_equality_atol=2e-12,
                    head_recipe='preactivation_selected_mse', ranking_reused=True, physics_attempt_budget=9,
                    planned_physics_attempts=len(cases), maximum_attempts_per_source_object=3,
                    execute_all_frozen_cases_even_after_success=True, retry=False, replacement=False,
                    benchmark_only=True, independent_generalization=False, training_writeback=False,
                    new_CGR_forwards=0, new_training_updates=0, models_changed=False,
                    robot='Original controlled-wrist Boya executor with previous A0 positive control; no arm or IK',
                    positive_control_reference=str(ROOT/'outputs/boya_contact_proxy_alignment_20261108_v2/execution/egad_A0_coarse/evaluation.json'),
                    phases_steps=PHASES, same_coarse_geometry=True, same_hand_closure=True, mass_kg=.1,
                    friction=.6, dt_s=.004, source_visual_and_object_initial_state_unchanged=True,
                    task_early_stops='Original >5mm collision / >30mm approach push retained; no retry',
                    infrastructure_policy='Stop dispatching next wave after infrastructure error; never restart an attempted case',
                    old_external_result='1/12 unchanged; this is known-object multi-candidate diagnosis, not top1/generalization',
                    source_sha256=source_hashes, input_sha256=input_hashes)
    save(out/'protocol.json', protocol)
    backend = read(B/'backend.example.json')
    jobs = {}
    for c in cases:
        j = dict(c, **{k: backend[k] for k in ('hand_urdf','hand_scene','package','physics_config','diagnostic_config')},
                 output_dir=str(out/'execution'/c['object_id']), protocol_path=str(out/'protocol.json'),
                 protocol_sha256=sha(out/'protocol.json'))
        path = out/'jobs'/(c['object_id']+'.json')
        save(path, j)
        jobs[str(path)] = sha(path)
    save(out/'PREPARED.json', dict(protocol_sha256=sha(out/'protocol.json'), jobs_sha256=jobs,
                                  physics_steps=0, planned=len(cases), selection=selection))
    return protocol


def verify(out):
    p = read(out/'protocol.json')
    prepared = read(out/'PREPARED.json')
    if sha(out/'protocol.json') != prepared['protocol_sha256']:
        raise ValueError('Protocol changed')
    for section in (p['source_sha256'], p['input_sha256'], prepared['jobs_sha256']):
        for path, h in section.items():
            if sha(path) != h:
                raise ValueError('Frozen dependency changed: '+path)
    return p


def execute(out):
    p = verify(out)
    if (out/'DISPATCH.json').exists() or (out/'RESULTS.json').exists():
        raise ValueError('Already dispatched; no replay or implicit resume')
    save(out/'DISPATCH.json', dict(protocol_sha256=sha(out/'protocol.json'), pid=os.getpid(),
                                   started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), waves=3, concurrency=3))
    began = time.monotonic()
    results, errors = {}, []
    def one(slot, c):
        oid = c['object_id']
        argv = ['taskset','-c',f'{32+8*slot}-{39+8*slot}',sys.executable,'-B',str(B/'run_boya_floating_diagnostic.py'),
                '--job',str(out/'jobs'/(oid+'.json'))]
        env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   CANONDEX_COLLISION_PARTITION_CACHE=str(ROOT/'outputs/boya_collision_partition_cache'))
        log = out/(oid+'.physics.log')
        before = time.monotonic()
        with log.open('xb') as f:
            rc = subprocess.run(argv, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=600).returncode
        evaluation = out/'execution'/oid/'evaluation.json'
        if rc or not evaluation.exists():
            raise RuntimeError(f'{oid}: process {rc}, see {log}')
        e = read(evaluation)
        if not e['complete']:
            raise RuntimeError('Incomplete trial: '+oid)
        return dict(object_id=oid, source_object_id=c['source_object_id'], candidate_rank=c['candidate_rank'],
                    evaluation=str(evaluation), evaluation_sha256=sha(evaluation), task_success=e['task_success'],
                    physical_steps=e['total_physics_steps'], task_early_stop=e['failure_reason'],
                    wall_s=time.monotonic()-before, argv=argv, process_returncode=rc)
    # Interleave objects so an infrastructure stop does not first exhaust one object.
    by_object = {oid: sorted([c for c in p['cases'] if c['source_object_id']==oid],
                            key=lambda c:c['candidate_rank']) for oid in IDS}
    queue = [by_object[oid][i] for i in range(3) for oid in IDS if len(by_object[oid])>i]
    for first in range(0, len(queue), 3):
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(one, i, c): c for i, c in enumerate(queue[first:first+3])}
            for future in as_completed(futures):
                c = futures[future]
                try:
                    results[c['object_id']] = future.result()
                except Exception as e:
                    errors.append(dict(object_id=c['object_id'], error=repr(e)))
                save(out/'PROGRESS.json', dict(status='executing', results=list(results.values()), errors=errors))
                print('CANDIDATE_TRIAL', c['object_id'], results.get(c['object_id']), flush=True)
        if errors:
            break
    report = dict(schema='anydex_boya_candidate_diagnostic_results_v1', status='complete' if not errors else 'infrastructure_error',
                  protocol_sha256=sha(out/'protocol.json'), results=[results[c['object_id']] for c in p['cases'] if c['object_id'] in results],
                  errors=errors, actual_grasps=len(results), success=sum(r['task_success'] for r in results.values()),
                  physical_steps=sum(r['physical_steps'] for r in results.values()), known_recorded_physics_steps_only=bool(errors),
                  wall_s=time.monotonic()-began, benchmark_only=True, training_writeback=False, independent_generalization=False)
    verify(out)
    save(out/'RESULTS.json', report)
    save(out/'PROGRESS.json', report)
    return report


def audit(out):
    """Recompute commands, gates and source-mesh support directly from saved traces."""
    from scipy.spatial.transform import Rotation
    import xml.etree.ElementTree as ET
    import ast
    p = verify(out)
    report = read(out/'RESULTS.json')
    if report['status'] != 'complete':
        raise ValueError('Audit expects completed experiment')
    tree = ast.parse((B/'run_boya_floating_diagnostic.py').read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='run')
    loop = next(n for n in ast.walk(run) if isinstance(n, ast.For) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='step' for x in ast.walk(n)))
    calls = {x.func.attr for x in ast.walk(loop) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)}
    assert not calls & {'write_root_link_pose_to_sim_index','write_root_link_velocity_to_sim_index','write_joint_state_to_sim'}
    output = []
    for r in report['results']:
        job = read(out/'jobs'/(r['object_id']+'.json'))
        d = Path(r['evaluation']).parent
        e = read(r['evaluation'])
        native = read(d/'native_audit.json')
        assert sha(r['evaluation']) == r['evaluation_sha256']
        assert sha(d/'trace.npz') == e['trace_sha256']
        assert sha(d/'commands.npz') == e['commands_sha256']
        with np.load(d/'trace.npz') as z:
            t = {k:z[k].copy() for k in z.files}
        with np.load(d/'commands.npz') as z:
            cmd = {k:z[k].copy() for k in z.files}
        with np.load(Path(job['candidate_dir'])/'candidates.npz') as z:
            pool = {k:z[k].copy() for k in z.files}
        n = len(t['time_s'])
        assert n == e['total_physics_steps'] == r['physical_steps']
        assert all(np.isfinite(v).all() for v in t.values() if v.dtype.kind in 'fiu')
        for k in ('joint_target','velocity_target','desired_palm','phase','time_s'):
            assert np.array_equal(t[k],cmd[k][:n]), k
        assert np.array_equal(t['time_s'],np.arange(1,n+1)*.004)
        names = list(t['joint_names'])
        motor_idx = [names.index(m) for m in pool['independent_motor_names']]
        opened, closed = pool['proposal_q'][0], pool['proposal_q_close'][0]
        wanted = np.repeat(opened[None],2100,axis=0)
        fraction = np.arange(1,251,dtype=float)/250
        wanted[1300:1550] = (1-fraction[:,None])*opened+fraction[:,None]*closed
        wanted[1550:] = closed
        assert np.array_equal(t['joint_target'][:,motor_idx],wanted[:n].astype(np.float32))
        palm = pool['proposal_palm'][0]
        direction = pool['proposal_approach_direction'][0]
        pre = palm[:3,3]-.1*direction
        desired = np.repeat(palm[None],2100,axis=0)
        desired[:800,:3,3] = pre
        s=np.arange(1,501,dtype=float)/500
        u=s*s*s*(10+s*(-15+6*s))
        desired[800:1300,:3,3] = pre+u[:,None]*.1*direction
        desired[1600:,:3,3] += u[:,None]*np.array([0,0,.1])
        assert np.allclose(t['desired_palm'],desired[:n],atol=2e-14,rtol=0)
        axes=[names.index('diagnostic_slide_'+a) for a in 'xyz']
        xyz=(desired[:n,:3,3]-pre)@palm[:3,:3]
        assert np.array_equal(t['joint_target'][:,axes],xyz.astype(np.float32))
        urdf=ET.parse(d/'boya_translation_gantry.urdf').getroot()
        limit=mimic=0.
        for joint in urdf.findall('joint'):
            name=joint.get('name')
            if name not in names:
                continue
            vals=t['joint_pos'][:,names.index(name)]
            lim=joint.find('limit')
            if lim is not None and joint.get('type') in ('revolute','prismatic'):
                limit=max(limit,float(np.maximum(np.maximum(float(lim.get('lower'))-vals,vals-float(lim.get('upper'))),0).max()))
            m=joint.find('mimic')
            if m is not None:
                delta=vals-t['joint_pos'][:,names.index(m.get('joint'))]*float(m.get('multiplier','1'))-float(m.get('offset','0'))
                mimic=max(mimic,float(abs(delta).max()))
        assert abs(limit-e['maxima']['joint_limit_violation_rad']) < 2e-7
        assert abs(mimic-e['maxima']['mimic_error_rad']) < 2e-7
        position=np.linalg.norm(t['palm_pose'][:,:3,3]-desired[:n,:3,3],axis=1)
        penetration=max(float(t['object_penetration_m'].max()),float(t['robot_penetration_m'].max()))
        assert np.isclose(position.max(),e['maxima']['palm_position_error_m'],atol=1e-12)
        assert np.isclose(penetration,e['maxima']['penetration_m'],atol=1e-12)
        assert native['within_20um'] and native['physics_steps']==0
        row=job['object_row']
        assert sha(row['mesh_path'])==row['mesh_sha256']
        vertices=np.array([[float(x) for x in line.split()[1:4]] for line in Path(row['mesh_path']).read_text().splitlines() if line.startswith('v ')])*row['scale']
        poses=t['object_pose_xyzw'].astype(float)
        rotations=Rotation.from_quat(poses[:,3:]).as_matrix()
        bottom=np.empty(n)
        for i in range(0,n,64):
            bottom[i:i+64]=(vertices@rotations[i:i+64,2,:].T+poses[i:i+64,2]).min(0)
        both=(t['finger_loads_n'][:,0]>.001)&(t['finger_loads_n'][:,1:].sum(1)>.001)
        holds=np.flatnonzero(t['phase']=='hold')
        lift=t['phase']=='lift'
        rise=poses[:,2]-poses[holds[-1],2] if len(holds) else np.zeros(n)
        qualifying=lift&(bottom>=.005)&(rise>=.005)&both
        success=bool(all(e['gates'].values()) and qualifying[-1] and rise[-1]>=.05)
        assert success==e['task_success']
        if e['tabletop_pickup']:
            pickup=e['tabletop_pickup']
            assert int(qualifying.sum())==pickup['qualifying_steps']
            assert bool(qualifying[-1])==pickup['final_state_still_lifted_and_bearing']
            assert np.isclose(bottom[-1],pickup['final_bottom_clearance_m'],atol=2e-12)
            assert np.isclose(rise[-1],pickup['final_reference_rise_m'],atol=2e-12)
        output.append(dict(object_id=r['object_id'],source_object_id=job['source_object_id'],candidate_rank=job['candidate_rank'],
                           gesture_id=job['candidate_record']['gesture_id'],physical_steps=n,task_success=success,
                           actual_commands_and_palm_reconstructed=True,native20um=True,gates=e['gates'],
                           task_early_stop=e['failure_reason'],final_reference_rise_m=float(rise[-1]) if len(holds) else None,
                           final_bottom_clearance_m=float(bottom[-1]),qualifying_steps=int(qualifying.sum()),
                           lift_bilateral_load_steps=int((lift&both).sum()),final_finger_loads_n=t['finger_loads_n'][-1].tolist(),
                           max_palm_position_error_m=float(position.max()),max_penetration_m=penetration))
    a=dict(schema='anydex_boya_candidate_readonly_audit_v1',complete=True,protocol_sha256=sha(out/'protocol.json'),
           source_input_job_hashes_match=True,source_loop_no_state_reset=True,results=output,
           success=sum(r['task_success'] for r in output),physical_steps=sum(r['physical_steps'] for r in output),
           limitations=['Known-object bounded multi-candidate diagnosis, no top1/generalization claim',
                        'Nonhand load and base gates bound to native reports; raw all-step contact lists are not retained',
                        'Original approximate coarse collision proxies and controlled wrist remain'])
    assert a['success']==report['success'] and a['physical_steps']==report['physical_steps']
    save(out/'INDEPENDENT_AUDIT.json',a)
    return a


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('prepare','execute','audit'),required=True)
    args=p.parse_args()
    value=globals()[args.stage](args.output.resolve())
    print(json.dumps({k:value[k] for k in ('status','success','physical_steps','planned_physics_attempts') if k in value}),flush=True)


if __name__=='__main__':
    main()
