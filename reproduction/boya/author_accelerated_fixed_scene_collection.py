"""Reusable finite fixed-world collection with full-source fast checks.

No diagnostic-history input is required. This development entry still excludes
training and freezes an explicit scene list before any simulator is started.
"""
import argparse
from importlib.metadata import version
import os
from pathlib import Path
import sys
import author_fixed_scene_collection as BASE

B,LIVE=BASE.B,BASE.BASE


def freeze(source_batch,geometry_template,root,indices=(3,4)):
    p=BASE.freeze(source_batch,geometry_template,root,indices);root=Path(root).resolve()
    B.save(root/'BASE_ENTRY_PROTOCOL.json',p,exclusive=True)
    names=('author_accelerated_fixed_scene_collection.py','author_fast_geometry_checks.py','author_fast_live_world.py',
        'author_observer_partition.py','author_fast_live_collection_service.py','audit_author_fast_live_world.py',
        'author_cpu_fabric_control.py','author_cpu_fabric.py')
    for name in names:p['sources'][name]=B.binding(B.HERE/name)
    p['sources']['service_entry']=p['sources']['author_fast_live_collection_service.py']
    p['sources']['audit_entry']=p['sources']['audit_author_fast_live_world.py']
    p['inputs']['base_entry_protocol']=B.binding(root/'BASE_ENTRY_PROTOCOL.json')
    p.update(fast_geometry_observer=True,physics_settings_unchanged=True,independent_runtime_audit_unchanged=True,
        accelerator_versions={name:version(name)for name in ('numpy','numba','scipy')},
        max_additional_native_startup_steps=4,max_combined_environment_steps_including_native_startup=p['max_environment_steps']+4,
        entry='Finite persistent CPU world with compiled full-source geometry checks; excludes training',
        CPU_Fabric_enabled=False,source_vertices_or_triangles_omitted=0)
    p['source_archive']=LIVE.C.archive_sources(root/'fast_sources',p['sources']);B.save(root/'PROTOCOL.json',p)
    return LIVE.verify(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','run']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source-batch',type=Path);p.add_argument('--geometry-template',type=Path);p.add_argument('--queue-indices',nargs='+',type=int,default=[3,4]);a=p.parse_args()
    if a.action=='freeze':result=freeze(a.source_batch,a.geometry_template,a.root,a.queue_indices)
    else:
        marker=B.sha(a.root/'PROTOCOL.json')
        if os.environ.get('ANYDEX_FAST_WORLD_ENV')!=marker:
            env=B._environment(LIVE.verify(a.root));env['ANYDEX_FAST_WORLD_ENV']=marker
            os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),'run','--root',str(a.root.resolve())],env)
        result=LIVE.run(a.root)
    print({k:result.get(k)for k in ('status','wall_s','diagnostic_positive','diagnostic_negative')},flush=True)
