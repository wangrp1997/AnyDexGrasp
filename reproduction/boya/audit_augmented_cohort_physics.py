"""Independent command, native geometry, physical gates and pickup audit."""
from pathlib import Path
import argparse
import numpy as np

from run_augmented_cohort import (ROOT, OBJECTS, RECIPES, INPUTS, baseline, eligible,
    same_pose, read, save, sha, check_hashes)
from audit_boya_fixed_pool import reconstruct_commands, PHYSICS_KEYS
from audit_boya_closure_refinement import audit_execution, arrays


def audit(out):
    protocol = read(out/'protocol.json'); result = read(out/'RESULTS.json')
    assert result['status'] == 'complete' and result['protocol_sha256'] == sha(out/'protocol.json')
    assert protocol['choices_freeze_sha256'] == sha(out/'ALL_CHOICES_FREEZE.json')
    check_hashes((protocol['input_sha256'], read(out/'PREPARED.json')['jobs_sha256']))
    assert len(protocol['conditions']) == 16
    assert len({c['condition_id'] for c in protocol['conditions']}) == 16
    assert (out/'ALL_CHOICES_FREEZE.json').stat().st_mtime_ns < (out/'protocol.json').stat().st_mtime_ns
    rows = {r['candidate_id']:r for r in result['results']}; audits = []
    for case in protocol['cases']:
        cid, oid = case['object_id'], case['source_object_id']
        row = rows[cid]; job = read(case['job_path']); reference = read(baseline(oid))
        assert eligible(job, reference) and same_pose(job['candidate_dir'], case['candidate_dir'])
        execution = Path(job['output_dir']); e = read(execution/'evaluation.json')
        old = read(Path(reference['output_dir'])/'evaluation.json')
        assert e['protocol_ref'] == dict(path=job['protocol_path'], sha256=job['protocol_sha256'])
        for key in PHYSICS_KEYS: assert e[key] == old[key], (cid,key)
        for key in ('actuation','gantry','mass_kg','native_mimics','solver_position_iteration_count'):
            assert e['audit'][key] == old['audit'][key], (cid,key)
        for body in ('robot','object'):
            for key in ('static_friction','dynamic_friction','restitution','rest_offset_m','contact_offset_m'):
                assert e['audit']['materials'][body][key] == old['audit']['materials'][body][key]
        assert arrays(execution/'commands.npz')['joint_names'].size == 23
        commands = reconstruct_commands(job)
        a = audit_execution(execution, job, job, arrays(Path(job['candidate_dir'])/'candidates.npz'), np.zeros(3), row)
        a.update(candidate_id=cid, source_object_id=oid, kind=case['kind'], independently_reconstructed_commands=commands)
        audits.append(a)
    for condition in protocol['conditions']:
        cid = condition['candidate_id']
        if cid is not None:
            case = next(c for c in protocol['cases'] if c['object_id'] == cid)
            assert case['source_object_id'] == condition['source_object_id']
            assert same_pose(case['candidate_dir'], condition['candidate_dir'])
    metrics = []
    for input_kind in INPUTS:
        for recipe in RECIPES:
            subset = [c for c in protocol['conditions'] if c['input_kind']==input_kind and c['recipe']==recipe]
            outcomes = {c['source_object_id']:bool(c['candidate_id'] and rows[c['candidate_id']]['task_success']) for c in subset}
            metrics.append(dict(input_kind=input_kind, recipe=recipe, success=sum(outcomes[o] for o in OBJECTS[:3]),
                denominator=3, outcomes=outcomes, A0_positive_control=outcomes['egad_A0']))
    assert result['new_physics_steps'] == sum(a['physical_steps'] for a in audits if a['kind']=='new')
    save(out/'INDEPENDENT_PHYSICS_AUDIT.json', dict(status='complete', cases=audits, metrics=metrics,
        source_sha256={str(Path(__file__).resolve()):sha(__file__)},
        all_condition_aliases_exact=True, commands_independently_reconstructed=True,
        all_source_vertex_bottoms_and_bilateral_loads_recomputed=True, inputs_unchanged=True,
        new_physics_steps=0, new_training_updates=0))
    print('COHORT_AUDIT', metrics, flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args().output.resolve())
