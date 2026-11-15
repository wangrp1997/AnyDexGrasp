"""Review only saved rejection evidence after the finite continuation ends."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import time

import numpy as np
import author_production_continuation as D

B=D.B
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
assert B.read(a.run_dir/'OUTER_WALL.json')['status']in('complete','stopped')
began=time.monotonic();p=D.verify(a.run_dir);ledger=D.ledger(a.run_dir,p)
rows=[];checked={};counts=defaultdict(Counter)
def bind(path,expected=None):
    path=Path(path);sha=B.sha(path)
    if expected is not None:assert sha==expected
    checked[str(path)]=sha
for row in ledger:
    entry=dict(queue_index=row['queue_index'],scene_id=row['scene_id'],split=row['split'],gesture_id=row['gesture_id'],
        status=row['status'],label=row['label'])
    counts[row['split']][row['status']]+=1
    preparation=row['artifact_bindings'].get('preparation')
    if preparation:
        bind(preparation['path'],preparation['sha256']);prep=B.read(preparation['path'])
        entry.update(stage=prep.get('stage'),collection_counts=prep.get('collection_counts'))
        for op in prep.get('home_preparation',{}).get('operations',[]):
            if op.get('stage')=='rrt':
                bind(op['receipt_path']);report=B.read(op['receipt_path'])
                entry.update(home_search_reason=report.get('reason'),home_search_skipped=report.get('search_skipped'))
        native=Path(prep.get('arm_preparation',{}).get('plan_dir')or'/nonexistent')
        geometry=native/'path_geometry.json';commands=native/'commands.npz'
        if row['status']=='null_geometry'and geometry.is_file()and commands.is_file():
            bind(geometry);bind(commands);report=B.read(geometry)
            with np.load(commands,allow_pickle=False)as data:phase=data['phase'].copy()
            categories=Counter()
            for violation in report.get('violations',[]):
                i=violation['sample'];assert 0<=i<len(phase)
                categories[str(phase[i])+':'+violation['kind']]+=1
            entry.update(saved_path_violation_counts=dict(categories),min_floor_m=report.get('min_floor_m'))
    if row['status']=='placement_null':
        source=row['scene_spec'];bind(source['path'],source['sha256']);scene=B.read(source['path'])
        entry['placement_object_statuses']=[o['placement_status']for o in scene['objects']]
    rows.append(entry)
B.save(a.output,dict(status='pass',protocol=B.binding(a.run_dir/D.PROTOCOL),rows=rows,
    by_split={s:dict(c)for s,c in counts.items()},
    null_geometry_home_reasons=dict(Counter(r.get('home_search_reason','unspecified')for r in rows if r['status']=='null_geometry')),
    checked_files=checked,source=B.binding(__file__),wall_s=time.monotonic()-began,
    new_Kit_launches=0,new_physics_steps=0,new_IK_queries=0,new_collision_queries=0,new_CGR_forwards=0,
    scope='Saved direct paths and original rejection receipts only; no global reachability proof or changed policy'),exclusive=True)
print({s:dict(c)for s,c in counts.items()})
