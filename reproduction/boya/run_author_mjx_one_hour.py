"""Freeze and run one measured wall-hour, with no extrapolated sample count."""
import argparse
import copy
from pathlib import Path
import shutil
import sys
import time

import author_collection_batch as B
import author_collection_wave_resources as R
import author_mjx_batch_collection_v9 as F
from author_mjx_full_comparison_v4 import dispatch


def freeze(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    source=B.ROOT/'outputs/boya_mjx_full_comparison_20261118_v4/env_016/PROTOCOL.json'
    old=B.read(source);ready=[r for r in old['rows']if r.get('scene')]
    if len(ready)!=7:raise ValueError('All seven source layouts must be represented')
    rows=[]
    for index in range(128):
        original=ready[index%7];row=copy.deepcopy(original);scene=B.read(row['scene']['path'])
        scene['source_layout_control_identity']=scene['scene_id'];scene['scene_id']+=f'_hour_{index:03d}'
        path=root/'queue_inputs'/f'{index:03d}.json';B.save(path,scene,exclusive=True)
        row.update(index=index,scene_id=scene['scene_id'],scene=B.binding(path),source_scene=original['scene'],
            origin='cyclic_all_seven_ready_layouts_no_outcome_selection')
        rows.append(row)
    queue=root/'QUEUE_SOURCE.json';B.save(queue,dict(rows=rows,repeated_layout_diagnostic=True,new_independent_scene_draws=0),exclusive=True)
    base=F.freeze(root/'base_freeze',environments=16,source_protocol=queue)
    directory=root/'collection';directory.mkdir();p=copy.deepcopy(base)
    p.update(schema='anydex_mjx_actual_one_hour_v1',duration_s=3600,background_closed_wave_audits=True,
        cold_start_in_window=True,repeated_layout_diagnostic=True,new_independent_scene_draws=0,
        admission_deadline_only=True,inflight_grasps_and_returns_drain=True,
        retries=0,comparison_underfilled=False,training_updates=0)
    p['env']['ANYDEX_CONTACT_SPOOL_ROOT']=str(directory/'contact_spool')
    p['inputs']['base_freeze_protocol']=B.binding(root/'base_freeze/PROTOCOL.json')
    for name in ('author_mjx_hour_collection.py','author_hour_metrics.py','author_hour_audit.py',
                 'audit_author_mjx_wave_snapshot.py',Path(__file__).name):p['sources'][name]=B.binding(B.HERE/name)
    archive=directory/'source_archive';archive.mkdir()
    for name,binding in p['sources'].items():shutil.copyfile(binding['path'],archive/name)
    B.save(directory/'PROTOCOL.json',p,exclusive=True)
    shutil.copytree(source.parent/'warp_cache',directory/'warp_cache')
    control=dict(duration_s=3600,environments=16,queued_slots=128,max_native_environment_steps=p['max_native_environment_steps'],
        collection_protocol=B.binding(directory/'PROTOCOL.json'),formal_collection=False,training_updates=0,
        primary_count='Distinct eligible labels with complete physical return and independent audit finished inside the exact 3600s wall window',
        secondary_counts=['eligible runtime labels inside window','complete execution inside window','physical return inside window','tail completions after deadline'],
        repeated_layouts=True,new_independent_scene_draws=0,
        sources={n:B.binding(B.HERE/n)for n in ('finish_author_mjx_collection.py','author_mjx_full_comparison_v4.py',Path(__file__).name)})
    B.save(root/'CONTROL_PROTOCOL.json',control,exclusive=True);return control


def run(root):
    root=Path(root).resolve();control=B.read(root/'CONTROL_PROTOCOL.json')
    B.verify(control['sources']);B.verify({'collection':control['collection_protocol']})
    directory=root/'collection'
    if(root/'PIPELINE_STARTED.json').exists():raise ValueError('No one-hour retry')
    B.save(root/'PIPELINE_STARTED.json',dict(process=R.process_identity(__import__('os').getpid()),unix_s=time.time()),exclusive=True)
    start=time.monotonic_ns();window=dict(started_monotonic_ns=start,deadline_monotonic_ns=start+3600_000_000_000,
        started_unix_s=time.time(),duration_s=3600,cold_start_included=True,deadline_exclusive=True)
    B.save(directory/'HOUR_WINDOW.json',window,exclusive=True)
    print({'hour_started':window,'root':str(directory)},flush=True)
    stages={}
    try:
        stages['collect_and_audit']=dispatch([sys.executable,'-B',str(B.HERE/'author_mjx_hour_collection.py'),'--root',str(directory)],root/'timing/collect_and_audit')
        result=B.read(directory/'RESULTS.json');counts=B.read(directory/'HOUR_COUNTERS.json')
        if result['status']!='complete'or not counts['window_elapsed']:raise RuntimeError('A complete real hour was not observed')
        stages['export']=dispatch([sys.executable,'-B',str(B.HERE/'finish_author_mjx_collection.py'),'--root',str(directory)],root/'timing/export')
        summary=dict(status='actual_hour_complete',window=window,counts=counts,stages=stages,
            total_wall_including_drain_and_export_s=(time.monotonic_ns()-start)/1e9,
            cost=result['cost'],world=result['world'],source_result=B.binding(directory/'RESULTS.json'),
            source_audit=B.binding(directory/'READ_ONLY_AUDIT.json'),source_export=B.binding(directory/'export/MANIFEST.json'),
            repeated_layouts=True,new_independent_scene_draws=0,formal_training_eligible=False,training_updates=0)
        B.verify(control['sources']);B.save(root/'ACTUAL_HOUR_RESULTS.json',summary,exclusive=True)
        print({'status':summary['status'],'actual_counts':counts},flush=True)
    except BaseException as error:
        B.save(root/'PIPELINE_FAILURE.json',dict(error=repr(error),stages=stages,elapsed_s=(time.monotonic_ns()-start)/1e9),exclusive=True)
        raise
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('freeze','run'))
    parser.add_argument('--root',type=Path,required=True);a=parser.parse_args()
    if a.action=='freeze':print(freeze(a.root))
    else:run(a.root)
