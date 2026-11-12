"""One bounded paired diagnostic of visible-cloud Boya palm translation.

This is an opt-in known-object experiment, with unchanged scores and closure.
The source executor, old pools, observations and labels remain immutable.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'reproduction/boya'
IDS = ('egad_A0', 'egad_A3', 'egad_A6', 'egad_E0')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def source_job(oid):
    if oid == 'egad_A0':
        return ROOT/'outputs/boya_contact_proxy_alignment_20261108_v2/jobs/egad_A0_coarse.json'
    return ROOT/'outputs/egad_floating_three_20261108_v1/jobs'/(oid+'.json')


def prepare(out):
    from closure_translation_refinement import refine
    if out.exists():
        raise ValueError('New output required; do not overwrite or retune a prepared experiment')
    out.mkdir(parents=True)
    began = time.monotonic()
    sources, inputs, derived, cases, pending_jobs = {}, {}, {}, [], {}
    parents = ('egad_floating_three_20261108_v1', 'boya_contact_proxy_alignment_20261108_v2')
    for name in parents:
        path = ROOT/'outputs'/name/'protocol.json'
        p = read(path)
        sources.update(p['source_sha256'])
        inputs.update(p['input_sha256'])
        inputs[str(path)] = sha(path)
    # Preserve the original 435 records and eight old heads without retraining.
    old = ROOT/'outputs/boya_shared_score_20261111_v2/PROTOCOL.json'
    inputs.update(read(old)['input_sha256'])
    inputs[str(old)] = sha(old)
    for name in ('run_boya_closure_refinement.py', 'closure_translation_refinement.py',
                 'audit_boya_closure_refinement.py', 'render_egad_floating_three.py',
                 'run_boya_floating_diagnostic.py', 'build_trial_pool.py'):
        path = B/name
        sources[str(path)] = sha(path)
    for oid in IDS:
        original_path = source_job(oid)
        original = read(original_path)
        assert not original.get('collision_refinement')
        pool = Path(original['candidate_dir'])
        generation = read(pool/'generation.json')
        assert sha(pool/'candidates.npz') == generation['candidates_sha256']
        inputs[str(original_path)] = sha(original_path)
        inputs.update(generation['input_sha256'])
        for name in ('generation.json', 'candidates.npz', 'world_object_pose.json'):
            inputs[str(pool/name)] = sha(pool/name)
        with np.load(pool/'candidates.npz', allow_pickle=False) as z:
            original_arrays = {k: z[k].copy() for k in z.files}
        if len(original_arrays['proposal_palm']) != 1:
            raise ValueError('Expected original single-candidate pool')
        delta, report = refine(original)
        report_path = out/'static'/(oid+'.json')
        save(report_path, report)
        inputs.update(report['input_sha256'])
        derived[str(report_path)] = sha(report_path)
        for variant in ('baseline', 'refined'):
            cid = oid+'_'+variant
            shift = np.zeros(3) if variant == 'baseline' else np.asarray(delta, dtype=float)
            new_pool = out/'pools'/cid
            new_pool.mkdir(parents=True)
            arrays = {k: v.copy() for k, v in original_arrays.items()}
            arrays['proposal_palm'][0, :3, 3] += shift
            np.savez_compressed(new_pool/'candidates.npz', **arrays)
            (new_pool/'world_object_pose.json').write_bytes((pool/'world_object_pose.json').read_bytes())
            gen = copy.deepcopy(generation)
            gen['kind'] = 'anydex_boya_closure_translation_diagnostic_v1'
            gen['hand_head_loaded'] = False
            gen['original_head_ranking_reused'] = True
            gen['source_pool'] = str(pool)
            gen['candidates_sha256'] = sha(new_pool/'candidates.npz')
            gen['input_sha256'].update({str(pool/'candidates.npz'): sha(pool/'candidates.npz'),
                                       str(pool/'generation.json'): sha(pool/'generation.json'),
                                       str(report_path): sha(report_path)})
            gen['translation_refinement'] = dict(variant=variant, delta_world_m=shift.tolist(),
                static_report=str(report_path), historical_record_geometry_applies_to_original_pose=True,
                current_geometry_report=str(report_path) if variant == 'refined' else 'original zero translation')
            save(new_pool/'generation.json', gen)
            for name in ('candidates.npz', 'world_object_pose.json', 'generation.json'):
                derived[str(new_pool/name)] = sha(new_pool/name)
            case = dict(object_id=cid, source_object_id=oid, variant=variant,
                source_job=str(original_path), source_pool=str(pool), candidate_dir=str(new_pool),
                candidate_rank=original['candidate_rank'], delta_world_m=shift.tolist(),
                static_report=str(report_path))
            cases.append(case)
            job = copy.deepcopy(original)
            job.update(case, candidate_index=0, output_dir=str(out/'execution'/cid))
            job['selection_reason'] = 'Frozen original candidate, paired bounded palm translation only'
            pending_jobs[cid] = job
        print('REFINEMENT_PREPARED', oid, np.asarray(delta).tolist(), flush=True)
    for section in (sources, inputs, derived):
        for path, h in section.items():
            if sha(path) != h:
                raise ValueError('Dependency changed before freeze: '+path)
    protocol = dict(schema='anydex_boya_closure_translation_paired_v1',
        created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), cases=cases,
        purpose='Known-object falsification of one uniform visible-cloud closure translation rule',
        source_sha256=sources, input_sha256=inputs, derived_sha256=derived,
        changed='Palm world translation only; no candidate, score, orientation, hand command or object change',
        optimization_inputs='Cached single-frame sparse visible cloud, fixed open/closed hand FK and distal pads only',
        search=dict(world_xyz_offsets_m=[-.01,-.005,0.,.005,.01], count=125,
            closure_fractions=11, shared_time_for_two_sides=True, displacement_regularizer=.1),
        physics_attempts=8, maximum_physics_steps=16800, steps_per_case=2100,
        success_gate='Unchanged seven execution gates, final source bottom>=5mm, final bilateral load>1mN per side and reference rise>=50mm',
        no_outcome_dependent_retry=True, no_early_success_stop=True,
        new_CGR_forwards=0, new_training_updates=0, training_writeback=False,
        independent_generalization=False, default_promotion=False,
        decision='Promising only if baseline positive control succeeds, refined preserves it, and at least one baseline failure succeeds with refinement; otherwise stop this rule',
        limitations=['Known selected objects; one attempt per variant; no success-rate claim',
                    'Partial visible surface and approximate original convex object collision',
                    'Controlled wrist translation; no complete arm or real hardware validation'],
        preparation_wall_s=time.monotonic()-began)
    save(out/'protocol.json', protocol)
    jobs = {}
    for cid, job in pending_jobs.items():
        job.update(protocol_path=str(out/'protocol.json'), protocol_sha256=sha(out/'protocol.json'))
        path = out/'jobs'/(cid+'.json')
        save(path, job)
        jobs[str(path)] = sha(path)
    save(out/'PREPARED.json', dict(protocol_sha256=sha(out/'protocol.json'), jobs_sha256=jobs,
        physics_steps=0, planned_physics_attempts=8))
    return protocol


def verify(out):
    p, prepared = read(out/'protocol.json'), read(out/'PREPARED.json')
    assert sha(out/'protocol.json') == prepared['protocol_sha256']
    for section in (p['source_sha256'], p['input_sha256'], p['derived_sha256'], prepared['jobs_sha256']):
        for path, h in section.items():
            if sha(path) != h:
                raise ValueError('Frozen dependency changed: '+path)
    return p


def execute(out):
    p = verify(out)
    if (out/'DISPATCH.json').exists() or (out/'RESULTS.json').exists():
        raise ValueError('Already dispatched; no implicit repeat')
    save(out/'DISPATCH.json', dict(protocol_sha256=sha(out/'protocol.json'), pid=os.getpid(),
        started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), concurrency=2, waves=4))
    start = time.monotonic()
    results, errors = {}, []

    def one(slot, c):
        cid = c['object_id']
        argv = ['taskset', '-c', f'{32+8*slot}-{39+8*slot}', sys.executable, '-B',
                str(B/'run_boya_floating_diagnostic.py'), '--job', str(out/'jobs'/(cid+'.json'))]
        env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            CANONDEX_COLLISION_PARTITION_CACHE=str(ROOT/'outputs/boya_collision_partition_cache'))
        began = time.monotonic()
        with (out/(cid+'.physics.log')).open('xb') as log:
            rc = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=600).returncode
        evaluation = out/'execution'/cid/'evaluation.json'
        if rc or not evaluation.exists():
            raise RuntimeError(f'{cid}: process {rc}; original log retained')
        e = read(evaluation)
        if not e['complete']:
            raise RuntimeError('Incomplete trial: '+cid)
        return dict(**c, evaluation=str(evaluation), evaluation_sha256=sha(evaluation),
            task_success=e['task_success'], physical_steps=e['total_physics_steps'],
            task_early_stop=e['failure_reason'], wall_s=time.monotonic()-began,
            argv=argv, process_returncode=rc)

    for first in range(0, len(p['cases']), 2):
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = {workers.submit(one, i, c): c for i, c in enumerate(p['cases'][first:first+2])}
            for f in as_completed(futures):
                c = futures[f]
                try:
                    results[c['object_id']] = f.result()
                except Exception as exc:
                    errors.append(dict(object_id=c['object_id'], error=repr(exc)))
                save(out/'PROGRESS.json', dict(status='executing', results=list(results.values()), errors=errors))
                print('PAIRED_TRIAL', c['object_id'], results.get(c['object_id'], errors[-1:] or None), flush=True)
        if errors:
            break
    rows = [results[c['object_id']] for c in p['cases'] if c['object_id'] in results]
    counts = {v: sum(r['task_success'] for r in rows if r['variant']==v) for v in ('baseline','refined')}
    promising = (not errors and results['egad_A0_baseline']['task_success'] and
                 results['egad_A0_refined']['task_success'] and
                 any(not results[o+'_baseline']['task_success'] and results[o+'_refined']['task_success'] for o in IDS[1:]))
    report = dict(schema='anydex_boya_closure_translation_results_v1', status='complete' if not errors else 'infrastructure_error',
        protocol_sha256=sha(out/'protocol.json'), results=rows, errors=errors,
        actual_grasps=len(rows), success=sum(r['task_success'] for r in rows), paired_success=counts,
        physical_steps=sum(r['physical_steps'] for r in rows), decision='promising_diagnostic_only' if promising else 'stop_this_rule',
        wall_s=time.monotonic()-start, benchmark_only=True, training_writeback=False, default_promotion=False)
    verify(out)
    save(out/'RESULTS.json', report)
    save(out/'PROGRESS.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare','execute','verify'), required=True)
    a = parser.parse_args()
    r = globals()[a.stage](a.output.resolve())
    print(json.dumps({k:r[k] for k in ('status','paired_success','decision','physical_steps','physics_attempts') if k in r}), flush=True)


if __name__ == '__main__':
    main()
