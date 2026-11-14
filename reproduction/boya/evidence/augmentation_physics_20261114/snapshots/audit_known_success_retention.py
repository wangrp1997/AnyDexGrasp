"""Post-outcome descriptive check only; never feeds candidate selection."""
from pathlib import Path
import sys
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');sys.path.insert(0,str(ROOT/'reproduction/boya'))
from run_augmented_cohort import read,save,INPUTS,RECIPES
import numpy as np
r=Path(__file__).resolve().parent;old=ROOT/'outputs/boya_fixed_pool_20261112_v1'
labels={x['candidate_id']:x['task_success']for x in read(old/'RESULTS.json')['results']}
successes=[c for c in read(old/'CATALOG.json')['candidates']if labels[c['candidate_id']]]
rows=[]
for oid in ('egad_A3','egad_A6','egad_E0'):
 for input_kind in INPUTS:
  for recipe in RECIPES:
   d=r/oid/'comparison'/f'{input_kind}__{recipe}';qs=read(d/'queries.json');shortlist=read(d/'shortlist.json')['candidates'];geometry=read(d/'geometry.json')['admitted']
   expanded=[(float(q['all_head_outputs20'][q['raw_depth_index']*4+e]),q['query_index'],e)for q in qs for e in q['allowed_extra_depth_indices']];expanded.sort(key=lambda v:(-v[0],v[1],v[2]))
   rankmap={(q,e):i+1 for i,(score,q,e)in enumerate(expanded)};shortkeys={(c['query_index'],c['extra_depth_index'])for c in shortlist};geokeys={(c['candidate']['query_index'],c['candidate']['extra_depth_index'])for c in geometry}
   for c in [c for c in successes if c['object_id']==oid]:
    rec=c['candidate_record'];matches=[q for q in qs if q['source_branch']==0 and q['source_branch_row']==rec['source_candidate_index'] and all(q[k]==rec[k] for k in ('gesture_id','if_flip','raw_depth_index'))]
    assert len(matches)==1
    q=matches[0];key=(q['query_index'],rec['extra_depth_index']);score=float(q['all_head_outputs20'][q['raw_depth_index']*4+rec['extra_depth_index']])
    rows.append(dict(object_id=oid,candidate_id=c['candidate_id'],input_kind=input_kind,recipe=recipe,branch0_query_retained=True,score=score,rank1=rankmap.get(key),strict_threshold_pass=score>float(np.float32(.85)),in_shortlist=key in shortkeys,in_admitted_geometry=key in geokeys))
save(r/'KNOWN_SUCCESS_RETENTION.json',dict(status='complete',scope='Post-hoc description of five known historical successes, not an executable selection rule',rows=rows,new_model_forwards=0,new_geometry_checks=0,new_physics_steps=0))
print('KNOWN_SUCCESS_RETENTION',[(x['object_id'],x['recipe'],x['candidate_id'],x['rank1'],x['in_admitted_geometry'])for x in rows if x['input_kind']=='augmented11'])
