"""Freeze the verified native dependency environment and probe before any Kit."""
from pathlib import Path
import subprocess

import author_collection_batch as B


def restore_and_probe(source_batch,reference_protocol,output):
    path=Path(source_batch)/'BATCH_PROTOCOL.json';base=B.read(path)
    reference=B.read(reference_protocol);B.verify(reference['inputs']);B.verify(reference['sources'])
    base['inherited_env']=dict(reference['inherited_env'])
    overlay=B.ROOT/'outputs/native_deps/clarabel_0111_overlay/clarabel'
    for dependency in sorted(overlay.iterdir()):
        if dependency.is_file():base['inputs']['clarabel_dependency:'+dependency.name]=B.binding(dependency)
    base['sources'][Path(__file__).name]=B.binding(__file__)
    B.save(path,base)
    command=[base['inputs']['python']['path'],'-B','-c',
        'import sys,clarabel,torch,numpy; sys.path.insert(0,'+repr(str(B.ROOT.parent/'CanonDex'))+
        '); import canondex.grasp.cooking_audit; print(clarabel.__file__); print(torch.__version__)']
    result=subprocess.run(command,env=B._environment(base),capture_output=True,text=True,timeout=30.)
    value=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
        reference=B.binding(reference_protocol),inherited_env=base['inherited_env'],
        source=B.binding(__file__),Kit_launches=0,physics_steps=0,model_loads=0)
    B.save(output,value,exclusive=True)
    if result.returncode!=0:raise RuntimeError('Dependency import failed before physical dispatch: '+result.stderr)
    return base
