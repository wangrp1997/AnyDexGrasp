"""Zero-physics control: does the frozen pre-head domain contain any hand geometry?

Read previously frozen predictions, check ALL combinations in that same domain.
This is a diagnostic enumeration, not a deployed threshold change or selector.
No training labels, outcome records, new head/CGR inference, arms or physics.
"""
from pathlib import Path
import argparse
from collections import Counter
import json
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate_boya_decision import Reader
from score_s40_domain_candidates import make_domain_queries
from deploy_score_first import admit,expand_and_rank,save


def run(args):
    began=time.monotonic();reader=Reader();reader.freeze(Path(__file__).resolve())
    source=reader.read(args.diagnostic/'RESULTS.json')
    if source['schema']!='anydex_s40_domain_score_first_internal_diagnostic_v1' or source['status']!='complete':
        raise ValueError('Complete frozen domain diagnostic required')
    for path,digest in source['input_sha256'].items():reader.freeze(path,digest)
    policy=source['domain_policy']['policy'];profile=source['parameters']
    objects=reader.read(args.collection/'objects.json')['objects'];byid={o['object_id']:o for o in objects}
    if args.output.exists():raise ValueError('Fresh diagnostic control output required')
    args.output.mkdir(parents=True);rows=[]
    for previous in source['objects']:
        oid=previous['object_id'];obj=byid[oid];out=args.output/oid;out.mkdir()
        obs_path=Path(obj['observation_json']);obs=reader.read(obs_path);frame=Path(obj['representation_dir'])
        original=args.diagnostic/oid;freeze=reader.read(original/'RANKING_FREEZE.json')
        for path,digest in freeze['outputs_sha256'].items():reader.freeze(path,digest)
        with np.load(frame/'raw_camera_candidates.npz',allow_pickle=False)as z:
            raw=z['candidates'].copy();features=z['candidate_features'].copy()
        q,x,indices,domain=make_domain_queries(raw,features,obs['T_WC'],policy['config'],profile,{'original_policy':source['domain_policy']['sha256']})
        with np.load(original/'head_outputs.npz',allow_pickle=False)as z:
            for key,expected in [('source_indices',[v['source_candidate_index']for v in q]),('gesture_ids',[v['gesture_id']for v in q]),('raw_depth_indices',[v['raw_depth_index']for v in q]),('if_flip',[v['if_flip']for v in q])]:
                if not np.array_equal(z[key],np.asarray(expected)):raise ValueError('Cached query identity mismatch')
            if z['outputs20'].shape!=(len(q),20):raise ValueError('Wrong cached head shape')
            for v,scores in zip(q,z['outputs20']):v['all_head_outputs20']=scores.tolist()
        ranked,shortlist=expand_and_rank(q,profile)
        if len(shortlist)!=previous['shortlist']:raise ValueError('Original shortlist changed')
        save(out/'CONTROL_FREEZE.json',dict(scope='All frozen-domain combinations, diagnostic only; original deployment threshold unchanged',
            source_diagnostic=str(args.diagnostic),source_inputs=reader.files.copy(),combination_count=len(ranked),
            labels_read=False,new_head_inference=False,new_physics_steps=0))
        geom_args=argparse.Namespace(stage='geometry',observation=obs_path,frame_dir=frame,calibration=args.calibration,output_dir=out,max_arm_plans=4)
        g=admit(ranked,raw,obs,geom_args,reader,out/'CONTROL_FREEZE.json') if ranked else dict(geometry_checked=0,geometry_admitted=0,arm_checked=0)
        details=json.loads((out/'geometry.json').read_text()) if ranked else dict(checked=[],admitted=[])
        reasons=Counter(v['geometry']['reason']for v in details['checked']if not v['geometry']['ok'])
        row=dict(object_id=oid,domain_raw_candidates=len(indices),all_combinations=len(ranked),geometry_checked=g['geometry_checked'],
            geometry_admitted=g['geometry_admitted'],rejection_counts=dict(reasons),
            admitted_scores=[v['candidate']['learned_score']for v in details['admitted']],
            original_deployment_shortlist=previous['shortlist'],original_deployment_admitted=previous['geometry_admitted'])
        save(out/'RESULTS.json',row);rows.append(row);print(json.dumps(row),flush=True)
    reader.verify()
    result=dict(schema='anydex_s40_domain_feasibility_control_v1',status='complete',objects=rows,
        objects_with_any_geometry=sum(v['geometry_admitted']>0 for v in rows),
        checked=sum(v['geometry_checked']for v in rows),admitted=sum(v['geometry_admitted']for v in rows),
        source_diagnostic=str(args.diagnostic),domain_policy_sha256=source['domain_policy']['sha256'],
        original_threshold_changed=False,deployment_results_replaced=False,new_head_forwards=0,new_CGR_forwards=0,
        new_physics_steps=0,arm_calls=0,labels_read=False,wall_s=time.monotonic()-began,input_sha256=reader.files)
    save(args.output/'RESULTS.json',result)
    print(json.dumps({k:v for k,v in result.items()if k not in ['objects','input_sha256']},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['diagnostic','collection','calibration','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
