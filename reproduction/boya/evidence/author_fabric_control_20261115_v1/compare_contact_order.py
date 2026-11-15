"""Read-only exact contact-multiset comparison; never drops/rounds any point."""
from collections import Counter
import json
from pathlib import Path
import time

import audit_author_collection_runtime as A

HERE=Path(__file__).resolve().parent
root=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/fabric_control_20261115_v1')
paths=[root/arm/'worker_00/result/env_0/execution/MULTI_OBJECT_CONTACTS.json'
       for arm in ('baseline','both_disabled')]
bindings=A.Bindings()
streams=[A.iter_contact_rows(path,bindings) for path in paths]
began=time.monotonic();rows=0;ordered_exact=0;unordered_exact=0;first_difference=None
points=[0,0]
while True:
    pair=[next(stream,None) for stream in streams]
    if pair==[None,None]:break
    A.need(all(row is not None for row in pair),'Contact lengths differ')
    rows+=1
    A.need(pair[0]['step']==pair[1]['step']==rows,'Contact clock changed')
    ordered_exact+=pair[0]==pair[1]
    matched=True
    for category in ('object_contacts','robot_contacts'):
        values=[row[category] for row in pair]
        for i,value in enumerate(values):points[i]+=len(value)
        counters=[Counter(json.dumps(point,sort_keys=True,separators=(',',':'),allow_nan=False)
                          for point in value) for value in values]
        if counters[0]!=counters[1]:
            matched=False
            if first_difference is None:
                first_difference=dict(tick=rows,category=category,
                    exact_records_only_in_baseline=sum((counters[0]-counters[1]).values()),
                    exact_records_only_in_disabled=sum((counters[1]-counters[0]).values()))
    unordered_exact+=matched
A.need(rows==100,'Exactly 100 rows per arm required')
bindings.verify()
value=dict(status='exact_contact_multisets_match' if unordered_exact==rows else 'contact_multisets_differ',
    contact_rows_per_arm=rows,ordered_exact_rows=ordered_exact,unordered_exact_rows=unordered_exact,
    contact_points_per_arm=points,first_unordered_difference=first_difference,
    fields_dropped=0,points_dropped=0,zero_force_points_dropped=0,rounding_applied=False,
    scope='Sort exact serialized point records within each original category/tick; preserve every field and multiplicity',
    input_sha256=bindings.files,script=dict(path=str(Path(__file__).resolve()),sha256=A.sha(__file__)),
    wall_s=time.monotonic()-began,physics_steps=0,Kit_launches=0)
with (HERE/'CONTACT_ORDER_COMPARISON.json').open('x') as stream:json.dump(value,stream,indent=2)
print(json.dumps({k:v for k,v in value.items() if k not in ('input_sha256','script')},indent=2))
