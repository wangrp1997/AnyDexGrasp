"""Read-only old ledger/preparation rejection review; no planning or simulation."""
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
import audit_author_collection_runtime as A

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1')
paths=sorted((ROOT/'development/batch_v1/ledger').glob('*.json'))
paths+=sorted((ROOT/'development_waves/waves_v1/children').glob('*/batch/ledger/*.json'))
bindings=A.Bindings();rows=[];counts=defaultdict(Counter);types=defaultdict(Counter)
for path in paths:
    bindings.add(path);row=A.read(path)
    split,status,g=row['split'],row['status'],row['gesture_id']
    counts[split][status]+=1
    entry=dict(scene_id=row['scene_id'],split=split,gesture_id=g,status=status,label=row['label'],
        ledger=dict(path=str(path),sha256=A.sha(path)))
    if row['label'] is not None:types[split][g]+=1
    prep_binding=row['artifact_bindings'].get('preparation')
    if prep_binding:
        bindings.add(prep_binding['path'],prep_binding['sha256']);prep=A.read(prep_binding['path'])
        entry.update(preparation_stage=prep.get('stage'),collection_counts=prep.get('collection_counts'))
        home=prep.get('home_preparation',{})
        for operation in home.get('operations',[]):
            if operation.get('stage')!='rrt':continue
            source=Path(operation['receipt_path']);bindings.add(source);rrt=A.read(source)
            entry.update(home_search_reason=rrt.get('reason'),home_search_skipped=rrt.get('search_skipped'))
        native=Path(prep.get('arm_preparation',{}).get('plan_dir') or '/nonexistent')
        geometry=native/'path_geometry.json';commands=native/'commands.npz'
        if status=='null_geometry' and geometry.is_file() and commands.is_file():
            bindings.add(geometry);bindings.add(commands);report=A.read(geometry)
            with np.load(commands,allow_pickle=False) as z:phases=z['phase'].copy()
            categories=Counter()
            for violation in report.get('violations',[]):
                sample=violation['sample'];A.need(0<=sample<len(phases),'Geometry sample outside original commands')
                categories[str(phases[sample])+':'+violation['kind']]+=1
            entry.update(saved_path_violation_counts=dict(categories),saved_min_floor_m=report.get('min_floor_m'),
                classification_scope='Saved direct path only; not a new geometric query or global reachability proof')
    if status=='placement_null':
        spec=row['scene_spec'];bindings.add(spec['path'],spec['sha256']);scene=A.read(spec['path'])
        entry['placement_reported_reason']=scene.get('reason',scene.get('placement_failure'))
    rows.append(entry)
A.need(len(rows)==64 and sum(types['fit'].values())==14 and sum(types['internal'].values())==1,
       'Old fixed ledger population changed')
bindings.verify()
value=dict(status='old_ledger_readonly_review_complete',ledger_rows=len(rows),
    by_split={k:dict(v) for k,v in counts.items()},valid_records_by_split_gesture={k:dict(v) for k,v in types.items()},
    null_geometry_reported_reasons=dict(Counter(r.get('home_search_reason','unspecified') for r in rows if r['status']=='null_geometry')),
    no_candidate_collision_admitted=[r['collection_counts']['collision_admitted'] for r in rows if r['status']=='no_candidate'],
    rows=rows,input_sha256=bindings.files,script=dict(path=str(Path(__file__).resolve()),sha256=A.sha(__file__)),
    new_physics_steps=0,new_Kit_launches=0,new_IK_queries=0,new_collision_queries=0,new_CGR_forwards=0,
    new_labels=0,new_training_updates=0,
    limitations=['Saved rejection categories only; no global impossibility conclusion',
        'Nulls are not negative labels; no object/candidate replacement or relaxed gate'])
with (HERE/'NULL_REVIEW.json').open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in value.items() if k in ('status','ledger_rows','by_split',
    'valid_records_by_split_gesture','null_geometry_reported_reasons','no_candidate_collision_admitted')},indent=2))
