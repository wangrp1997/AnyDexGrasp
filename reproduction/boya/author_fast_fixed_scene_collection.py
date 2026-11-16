"""Finite same-world control with full-source accelerated observation/audits."""
import argparse
import copy
from importlib.metadata import version
import os
from pathlib import Path
import sys
import author_fixed_scene_collection as BASE
import author_live_collection as LIVE

B=LIVE.B


def freeze(reference,root,benchmark):
    reference,root=Path(reference).resolve(),Path(root).resolve();original=LIVE.verify(reference)
    LIVE.R.assert_quiescent(reference);measured=B.read(benchmark)
    if measured['status']!='pass'or measured['physics_steps']!=0 or not measured['all_source_vertices_and_triangles_retained']:
        raise ValueError('Full saved-data equivalence before live execution required')
    if not all(r['ray_maximum_saved_difference_m']<=1e-10 and all(o['maximum_saved_all_pose_difference_m']<=1e-12 for o in r['objects'])for r in measured['rounds']):
        raise ValueError('Unchanged geometry checks required')
    p=BASE.freeze(Path(original['inputs']['source_manual_batch']['path']).parent,
        original['inputs']['geometry_template']['path'],root,(3,))
    B.save(root/'BASE_ENTRY_PROTOCOL.json',p,exclusive=True)
    # Retain the reference's entire six-body bank; only one round is dispatched.
    p['object_bank']=copy.deepcopy(original['object_bank'])
    p['inputs'].update(reference_fixed_world=B.binding(reference/'PROTOCOL.json'),geometry_equivalence_benchmark=B.binding(benchmark),
        base_entry_protocol=B.binding(root/'BASE_ENTRY_PROTOCOL.json'))
    for name in ('author_fast_fixed_scene_collection.py','author_fast_geometry_checks.py','author_fast_live_world.py',
                 'author_observer_partition.py','author_fast_live_collection_service.py','audit_author_fast_live_world.py',
                 'author_cpu_fabric_control.py','author_cpu_fabric.py'):
        p['sources'][name]=B.binding(B.HERE/name)
    p['sources']['service_entry']=p['sources']['author_fast_live_collection_service.py']
    p['sources']['audit_entry']=p['sources']['audit_author_fast_live_world.py']
    p.update(fast_geometry_observer=True,physics_settings_unchanged=True,independent_runtime_audit_unchanged=True,
        accelerator_versions={name:version(name)for name in ('numpy','numba','scipy')},
        max_additional_native_startup_steps=4,max_combined_environment_steps_including_native_startup=8754,
        fixed_control_scope='One fixed round, unchanged six-body bank, fresh RGBD/CGR/first candidate; no retry or training',
        comparison_scope='Historical full loop is not a randomized paired throughput experiment')
    p['source_archive']=LIVE.C.archive_sources(root/'fast_sources',p['sources']);B.save(root/'PROTOCOL.json',p)
    return LIVE.verify(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','run']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--benchmark',type=Path);a=p.parse_args()
    if a.action=='freeze':result=freeze(a.reference,a.root,a.benchmark)
    else:
        marker=B.sha(a.root/'PROTOCOL.json')
        if os.environ.get('ANYDEX_FAST_WORLD_ENV')!=marker:
            env=B._environment(LIVE.verify(a.root));env['ANYDEX_FAST_WORLD_ENV']=marker
            os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),'run','--root',str(a.root.resolve())],env)
        result=LIVE.run(a.root)
    print({k:result.get(k)for k in ('status','wall_s','diagnostic_positive','diagnostic_negative')},flush=True)
