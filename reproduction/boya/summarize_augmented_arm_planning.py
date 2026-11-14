"""Read saved planner rejections without another IK, collision query or trial."""
from collections import Counter
from pathlib import Path

from run_augmented_arm_external6 import RUN,AUDIT,read,save,sha


def main():
    report=read(RUN/'planning/RESULTS.json')
    assert report['status']=='complete' and not report['errors']
    rows=[];reasons=Counter();baseline=0;recovered=0
    for path in sorted((RUN/'planning').glob('*/*/plan_000/RECOVERY_RESULTS.json')):
        record=read(path);assert record['complete'] and record['physics_steps']==0
        baseline+=record['baseline_admitted'];recovered+=record['recovered']
        for attempt in record['attempts']:
            if attempt['admitted']:continue
            reason=attempt.get('reason','unspecified');reasons[reason]+=1
            geometry=Path(attempt['plan_dir'])/'path_geometry.json'
            row=dict(trial_id=path.parents[1].name,recipe=path.parents[2].name,
                branch=attempt['label'],reason=reason,geometry_ref=None,
                violation_kinds={},object_contact_links=[],only_finger_object_violations=False)
            if geometry.exists():
                detail=read(geometry);violations=detail.get('violations',[])
                links=sorted({v['link'] for v in violations if v['kind']=='object'})
                row.update(geometry_ref=dict(path=str(geometry),sha256=sha(geometry)),
                    violation_kinds=dict(Counter(v['kind'] for v in violations)),
                    object_contact_links=links,
                    only_finger_object_violations=bool(violations) and all(v['kind']=='object' and v['link'].startswith(('th','ff','mf','rf','lf')) for v in violations),
                    minimum_reported_distance_m=min((v['distance_m'] for v in violations if 'distance_m' in v),default=None))
            rows.append(row)
    receipt=dict(schema='boya_augmented_arm6_saved_planning_rejections_v1',status='complete',
        planning_results_sha256=sha(RUN/'planning/RESULTS.json'),
        conditions=len(report['results']),ready=sum(v['selected'] is not None for v in report['results']),
        candidate_preparations=sum(v['arm_preparations'] for v in report['results']),
        baseline_admitted_candidates=baseline,recovered_candidates=recovered,
        rejected_native_branches=len(rows),reason_counts=dict(reasons),
        branches_with_only_finger_object_violations=sum(v['only_finger_object_violations'] for v in rows),
        rejections=rows,new_IK_calls=0,new_collision_queries=0,new_physics_steps=0,
        boundary='Nested branch rejections are diagnostic counts, not additional independent objects or physical failures; no unreachable-geometry proof',
        source_sha256={str(Path(__file__).resolve()):sha(__file__)})
    save(AUDIT/'PLANNING_REJECTIONS_SUMMARY.json',receipt)
    print({k:receipt[k] for k in ('conditions','ready','candidate_preparations','rejected_native_branches','branches_with_only_finger_object_violations')})


if __name__=='__main__':main()
