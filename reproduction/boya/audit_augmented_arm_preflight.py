"""Check frozen rankings, four-candidate planning budgets and both head recipes.

Must run before the physical output exists. No Kit, network inference or new IK.
"""
from pathlib import Path
import json

from prepare_s40_port_top1_v2 import verify_selected
from run_augmented_arm_external6 import RUN,AUDIT,IDS,RECIPES,read,save,sha,verify


def main():
    assert not (RUN/'execution').exists(),'Preflight must precede physical dispatch'
    study=read(AUDIT/'STUDY_PROTOCOL.json');verify(study['frozen_files'])
    protocol=read(RUN/'protocol.json')
    assert sha(RUN/'protocol.json')==read(RUN/'EVALUATION_FREEZE.json')['protocol_sha256']
    verify({protocol['planning_gate_ref']['path']:protocol['planning_gate_ref']['sha256']})
    branch=read(RUN/'INDEPENDENT_BRANCH_AUDIT.json')
    assert branch['status']=='complete' and len(branch['cases'])==6
    assert all(v['identity_exact'] and v['feature_pairing_exact'] and v['collision_cloud_exact'] for v in branch['cases'])
    score_refs=[]
    for oid in IDS:
        p=AUDIT/(oid+'_SCORE_AUDIT.json');audit=read(p)
        assert audit['status']=='complete' and len(audit['cases'])==4
        assert audit['all_four_rankings_frozen_before_geometry']
        assert all(v['outputs20_max_error']==0 and v['ranking_all_exact'] and v['canonical_all_exact'] for v in audit['cases'])
        score_refs.append(dict(object_id=oid,path=str(p),sha256=sha(p)))
    planning=read(RUN/'planning/RESULTS.json');jobs=read(RUN/'planning/PLAN.json')
    assert planning['status']=='complete' and not planning['errors'] and planning['physical_steps']==0
    expected={(o,r) for o in IDS for r in RECIPES}
    assert {(v['object_id'],v['recipe']) for v in planning['results']}==expected
    catalog={(v['object_id'],v['recipe']):v for v in jobs['jobs']}
    rows=[]
    for v in planning['results']:
        key=(v['object_id'],v['recipe']);item=catalog[key]
        geometry=read(RUN/key[0]/'comparison'/('augmented11__'+key[1])/'geometry.json')
        admitted=geometry['admitted'][:4]
        assert [c['row'] for c in item['candidates']]==[c['candidate'] for c in admitted]
        attempts=v['attempts']
        assert len(attempts)==v['arm_preparations']<=4
        assert [a['global_score_rank'] for a in attempts]==[c['candidate']['global_score_rank'] for c in admitted[:len(attempts)]]
        selected=v['selected']
        if selected:
            assert attempts and attempts[-1]['status']=='ready' and attempts[-1]['result']==selected
            assert all(a['status']=='no_full_arm_plan_in_declared_budget' for a in attempts[:-1])
            verify_selected(item['candidates'][len(attempts)-1],selected)
        else:
            assert len(attempts)==len(admitted)
            assert all(a['status']=='no_full_arm_plan_in_declared_budget' for a in attempts)
        rows.append(dict(object_id=key[0],recipe=key[1],status=v['status'],
            arm_preparations=len(attempts),attempted_global_ranks=[a['global_score_rank'] for a in attempts],
            selected_trial_id=selected['trial_id'] if selected else None,
            selected_binding_sha256=selected['trial_binding_sha256'] if selected else None,
            selected_commands_sha256=selected['commands_sha256'] if selected else None))
    receipt=dict(schema='boya_augmented_arm6_prephysical_audit_v1',status='complete',
        protocol_sha256=sha(RUN/'protocol.json'),planning_sha256=sha(RUN/'planning/RESULTS.json'),
        branch_audit_sha256=sha(RUN/'INDEPENDENT_BRANCH_AUDIT.json'),score_audits=score_refs,
        cases=rows,all_twelve_conditions_accounted_for=True,first_ready_and_budget_verified=True,
        original_models_and_435_unchanged=True,new_physics_steps=0,new_IK_calls=0,new_model_forwards=0,
        source_sha256={str(Path(__file__).resolve()):sha(__file__)})
    save(AUDIT/'PREPHYSICAL_AUDIT.json',receipt)
    print(json.dumps(dict(status='complete',conditions=len(rows),ready=sum(r['selected_trial_id'] is not None for r in rows))))


if __name__=='__main__':main()
