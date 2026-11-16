"""Complete primary import-audit revision; both earlier failures stay intact."""
import argparse
import os
from pathlib import Path
import sys
import author_live_collection as ORIGINAL
import author_live_world_v2 as CAMERA

B,R,C=ORIGINAL.B,ORIGINAL.R,ORIGINAL.C


def freeze(source_batch,geometry_template,root,failed_roots,indices=(3,4)):
    failures=[]
    for index,old_root in enumerate(failed_roots):
        old_root=Path(old_root).resolve();ORIGINAL.verify(old_root);result=B.read(old_root/'RESULTS.json');world=B.read(old_root/'runtime/WORLD_RECEIPT.json')
        if result['status']!='stopped'or world['sim_steps']!=750 or list(old_root.glob('scenes/*/live/EXECUTION_STARTED.json')):raise ValueError('Only the two reviewed pre-grasp adapter failures are covered')
        if index==0:
            if "name 'cam' is not defined"not in B.read(old_root/'runtime/SERVICE_ERROR.json')['error']:raise ValueError('Unexpected first failure')
        else:
            audit=B.read(old_root/'scenes/00/capture/INDEPENDENT_CAPTURE_AUDIT.json')
            if 'Native body shapes differ'not in audit['error']:raise ValueError('Unexpected second failure')
        R.assert_quiescent(old_root);failures.append(dict(protocol=B.binding(old_root/'PROTOCOL.json'),result=B.binding(old_root/'RESULTS.json'),world=B.binding(old_root/'runtime/WORLD_RECEIPT.json')))
    if len(failures)!=2:raise ValueError('Both preceding errors/costs must be retained')
    p=ORIGINAL.freeze(source_batch,geometry_template,root,indices);root=Path(root).resolve();B.save(root/'BASE_PROTOCOL.json',p,exclusive=True)
    for name in ('author_live_world_v2.py','author_live_world_v3.py','author_live_collection_service_v3.py','author_live_collection_v3.py'):
        p['sources'][name]=B.binding(B.HERE/name)
    p['sources']['service_entry']=p['sources']['author_live_collection_service_v3.py']
    for i,failure in enumerate(failures):
        for key,value in failure.items():p['inputs'][f'previous_failure_{i}:{key}']=value
    p['inputs']['base_protocol']=B.binding(root/'BASE_PROTOCOL.json')
    p.update(live_world_revision=3,previous_failures=failures,previous_cost=dict(Kit_starts=2,environment_steps=1500,actual_grasps=0,labels=0),
        max_combined_Kit_starts=3,max_combined_environment_steps=1500+p['max_environment_steps'],
        revision_changes=['Fixed camera metadata binding','Include primary in capture native20um audit'],original_failed_runs_preserved=True)
    p['source_archive']=C.archive_sources(root/'revision_sources',p['sources']);B.save(root/'PROTOCOL.json',p)
    CAMERA.capture_function(type('DeclaredWorld',(),{'p':p,'acquire':lambda *a:None})())
    return ORIGINAL.verify(root)


if __name__=='__main__':
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('stage',choices=['freeze','run']);q.add_argument('--root',type=Path,required=True)
    q.add_argument('--source-batch',type=Path);q.add_argument('--geometry-template',type=Path);q.add_argument('--failed-roots',nargs=2,type=Path)
    q.add_argument('--queue-indices',nargs='+',type=int,default=[3,4]);a=q.parse_args()
    if a.stage=='run':
        marker=B.sha(a.root/'PROTOCOL.json')
        if os.environ.get('ANYDEX_LIVE_PROTOCOL_ENV')!=marker:
            env=B._environment(ORIGINAL.verify(a.root));env['ANYDEX_LIVE_PROTOCOL_ENV']=marker
            os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),'run','--root',str(a.root.resolve())],env)
        result=ORIGINAL.run(a.root)
    else:result=freeze(a.source_batch,a.geometry_template,a.root,a.failed_roots,a.queue_indices)
    print({k:result.get(k)for k in ('status','wall_s','error','diagnostic_positive','diagnostic_negative')},flush=True)
