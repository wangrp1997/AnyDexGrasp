"""Fresh fixed-camera-binding revision; never resumes a failed native world."""
import argparse
import os
from pathlib import Path
import sys
import author_live_collection as ORIGINAL
import author_live_world_v2 as WORLD

B,R,C=ORIGINAL.B,ORIGINAL.R,ORIGINAL.C


def freeze(source_batch,geometry_template,root,failed_root,indices=(3,4)):
    failed_root=Path(failed_root).resolve();old=ORIGINAL.verify(failed_root)
    outcome=B.read(failed_root/'RESULTS.json');world=B.read(failed_root/'runtime/WORLD_RECEIPT.json')
    if outcome['status']!='stopped'or world['sim_steps']!=750 or outcome['jobs']!=[]or(failed_root/'scenes/00/live/EXECUTION_STARTED.json').exists():
        raise ValueError('Only the reviewed camera-metadata infrastructure failure before any grasp can start this revision')
    R.assert_quiescent(failed_root)
    p=ORIGINAL.freeze(source_batch,geometry_template,root,indices);root=Path(root).resolve()
    B.save(root/'BASE_PROTOCOL.json',p,exclusive=True)
    for name in ('author_live_world_v2.py','author_live_collection_service_v2.py','author_live_collection_v2.py'):
        p['sources'][name]=B.binding(B.HERE/name)
    p['sources']['service_entry']=p['sources']['author_live_collection_service_v2.py']
    p['inputs'].update(previous_failed_protocol=B.binding(failed_root/'PROTOCOL.json'),previous_failed_result=B.binding(failed_root/'RESULTS.json'),
        previous_failed_world=B.binding(failed_root/'runtime/WORLD_RECEIPT.json'),base_protocol=B.binding(root/'BASE_PROTOCOL.json'))
    p.update(live_world_revision=2,previous_cost=dict(Kit_starts=1,environment_steps=750,actual_grasps=0,labels=0),
        max_combined_Kit_starts=2,max_combined_environment_steps=750+p['max_environment_steps'],
        revision_change='Bind fixed camera metadata only; original physics/controls/placement/selection unchanged',
        original_failed_run_preserved=True)
    p['source_archive']=C.archive_sources(root/'revision_sources',p['sources']);B.save(root/'PROTOCOL.json',p)
    # Static check before any new native application starts.
    WORLD.capture_function(type('DeclaredWorld',(),{'p':p,'acquire':lambda *a:None})())
    return ORIGINAL.verify(root)


if __name__=='__main__':
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('stage',choices=['freeze','run']);q.add_argument('--root',type=Path,required=True)
    q.add_argument('--source-batch',type=Path);q.add_argument('--geometry-template',type=Path);q.add_argument('--failed-root',type=Path)
    q.add_argument('--queue-indices',nargs='+',type=int,default=[3,4]);a=q.parse_args()
    if a.stage=='run':
        marker=B.sha(a.root/'PROTOCOL.json')
        if os.environ.get('ANYDEX_LIVE_PROTOCOL_ENV')!=marker:
            env=B._environment(ORIGINAL.verify(a.root));env['ANYDEX_LIVE_PROTOCOL_ENV']=marker
            os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),'run','--root',str(a.root.resolve())],env)
        result=ORIGINAL.run(a.root)
    else:result=freeze(a.source_batch,a.geometry_template,a.root,a.failed_root,a.queue_indices)
    print({k:result.get(k)for k in ('status','wall_s','error','diagnostic_positive','diagnostic_negative')},flush=True)
