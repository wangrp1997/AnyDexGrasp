"""Reusable fixed-scene entry, without dependencies on debugging run history.

Freeze an explicit finite scene list, preload its object bank, and alternate
photograph/grasp in one native world. This development entry excludes training.
"""
import argparse
import os
from pathlib import Path
import sys
import author_live_collection as BASE
import author_live_world_v2 as CAMERA

B,C=BASE.B,BASE.C


def freeze(source_batch,geometry_template,root,indices=(3,4)):
    p=BASE.freeze(source_batch,geometry_template,root,indices);root=Path(root).resolve()
    B.save(root/'BASE_PROTOCOL.json',p,exclusive=True)
    for name in ('author_fixed_scene_collection.py','author_live_world_v2.py','author_live_world_v3.py','author_live_world_v4.py','author_live_collection_service_v4.py'):
        p['sources'][name]=B.binding(B.HERE/name)
    p['sources']['service_entry']=p['sources']['author_live_collection_service_v4.py']
    p['inputs']['base_protocol']=B.binding(root/'BASE_PROTOCOL.json')
    p.update(live_world_revision=4,entry='Fixed robot/table/camera with preloaded object bank; no history-specific preconditions',
        geometry_template_use='Native model/physics setup only; zero commands executed from this template, fresh RGBD/CGR for each round')
    p['source_archive']=C.archive_sources(root/'entry_sources',p['sources']);B.save(root/'PROTOCOL.json',p)
    CAMERA.capture_function(type('DeclaredWorld',(),{'p':p,'acquire':lambda *a:None})())
    return BASE.verify(root)


if __name__=='__main__':
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('stage',choices=['freeze','run']);q.add_argument('--root',type=Path,required=True)
    q.add_argument('--source-batch',type=Path);q.add_argument('--geometry-template',type=Path);q.add_argument('--queue-indices',nargs='+',type=int,default=[3,4]);a=q.parse_args()
    if a.stage=='run':
        marker=B.sha(a.root/'PROTOCOL.json')
        if os.environ.get('ANYDEX_LIVE_PROTOCOL_ENV')!=marker:
            env=B._environment(BASE.verify(a.root));env['ANYDEX_LIVE_PROTOCOL_ENV']=marker
            os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),'run','--root',str(a.root.resolve())],env)
        result=BASE.run(a.root)
    else:result=freeze(a.source_batch,a.geometry_template,a.root,a.queue_indices)
    print({k:result.get(k)for k in ('status','wall_s','error','diagnostic_positive','diagnostic_negative')},flush=True)
