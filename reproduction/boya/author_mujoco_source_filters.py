"""Transfer only the existing source hand's contact/exclude rules to MuJoCo.

Corrects the separate unfiltered smoke model, without editing that source/run.
No new geometric exclusions are inferred from observed success or failure.
"""
import argparse
from pathlib import Path
import threading
import xml.etree.ElementTree as ET

import author_mujoco_prefix_smoke as SMOKE

S,B=SMOKE.S,SMOKE.B
_OWNER=None


def apply_filters(spec,scene_xml):
    bodies={b.name for b in spec.bodies};pairs=[];seen=set()
    for element in ET.parse(scene_xml).getroot().findall('contact/exclude'):
        pair=(element.get('body1'),element.get('body2'))
        if any(name not in bodies for name in pair)or pair[0]==pair[1]:raise ValueError('Source exclusion body missing/identical')
        key=tuple(sorted(pair))
        if key in seen:raise ValueError('Duplicate source structural exclusion')
        seen.add(key);spec.add_exclude(bodyname1=pair[0],bodyname2=pair[1]);pairs.append(list(pair))
    if not pairs:raise ValueError('Source hand exclusions required')
    return pairs


def run(prepared,output):
    global _OWNER
    if _OWNER is not None:raise RuntimeError('One owning native-filter smoke thread required')
    prepared=Path(prepared).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    ready=B.read(prepared/'READY.json');plan=B.read(ready['plan']['path']);scene_xml=Path(plan['arguments']['hand_scene'])
    sources={str(Path(p).resolve()):B.binding(p)for p in (__file__,SMOKE.__file__,S.__file__,S.ROBOT.__file__)}
    inputs=dict(source_hand_xml=B.binding(scene_xml),source_prepared=B.binding(prepared/'READY.json'))
    B.verify(sources);B.verify(inputs)
    B.save(output/'PROTOCOL.json',dict(schema='anydex_boya_mujoco_source_filters_smoke_v1',sources=sources,inputs=inputs,
        max_steps=20,Kit_starts=0,retries=0,exclude_training=True,new_labels=0,
        policy='Apply the exact existing source contact/exclude list; no data-dependent new filters'),exclusive=True)
    original=S.add_scene;record={};_OWNER=threading.get_ident()
    def add_scene(spec,scene):
        record['source_exclusions']=apply_filters(spec,scene_xml)
        return original(spec,scene)
    try:
        S.add_scene=add_scene;result=SMOKE.run(prepared,output/'smoke')
    finally:S.add_scene=original;_OWNER=None
    B.verify(sources);B.verify(inputs)
    B.save(output/'FILTER_MAPPING.json',dict(source_hand_xml=inputs['source_hand_xml'],**record),exclusive=True)
    value=dict(status=result['status'],smoke_result=B.binding(output/'smoke/RESULTS.json'),
               source_exclusion_count=len(record.get('source_exclusions',[])),physics_steps_returned=result['physics_steps_returned'],
               Kit_starts=0,new_labels=0,training_updates=0,full_grasp_gates_validated=False)
    B.save(output/'RESULTS.json',value,exclusive=True);return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();v=run(a.prepared_dir,a.output_dir);print(v,flush=True)
    if v['status']!='prefix_smoke_complete':raise SystemExit(1)
