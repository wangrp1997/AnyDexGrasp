"""Prebind native hand references before any capture imports their defaults.

The failed service remains frozen. This opt-in replacement only adds the
original bind_hand call and guards; it does not start or retry any job on import.
"""
import argparse
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import author_collection_service as ORIGINAL
import author_transport as T

B,W=ORIGINAL.B,ORIGINAL.W


def bind_references(inputs):
    B.verify(inputs);backend=B.read(inputs['backend']['path'])
    paths={k:Path(backend[k]).resolve()for k in ('hand_urdf','hand_scene','reference_root')}
    if not all(paths[k].is_file()for k in ('hand_urdf','hand_scene'))or not paths['reference_root'].is_dir():
        raise ValueError('Frozen native hand/reference assets must exist before Kit')
    T._backend_path()
    from run_template_arm_positive import bind_hand
    bind_hand(SimpleNamespace(**paths))
    for name,key in (('canondex.robots.hand_actuation','HAND_SCENE'),('canondex.robots.ur16e_boya','SCENE_XML')):
        module=sys.modules.get(name)
        if module is not None and Path(getattr(module,key)).resolve()!=paths['hand_scene']:
            raise ValueError('Stale hand defaults were imported before native reference binding: '+name)
    return dict(backend=inputs['backend'],hand_urdf=B.binding(paths['hand_urdf']),
                hand_scene=B.binding(paths['hand_scene']),reference_root=str(paths['reference_root']),
                binding_before_capture_imports=True,Kit_starts=0,physics_steps=0)


class BoundRuntime(ORIGINAL.CollectionRuntime):
    def __init__(self,root,sources,inputs):
        self.references=bind_references(inputs)
        Path(root).mkdir(parents=True,exist_ok=True)
        W.save(Path(root)/'NATIVE_REFERENCE_BINDING.json',self.references)
        super().__init__(root,sources)

    def run_job(self,job):
        B.verify({k:self.references[k]for k in ('backend','hand_urdf','hand_scene')})
        if job['role']=='execute':
            ready=B.read(Path(job['prepared_dir'])/'READY.json');args=B.read(ready['plan']['path'])['arguments']
            for key in ('hand_urdf','hand_scene','reference_root'):
                expected=self.references[key]['path']if key!='reference_root'else self.references[key]
                if Path(args[key]).resolve()!=Path(expected).resolve():
                    self.poisoned=True;raise ValueError('Prepared native reference differs from pre-Kit binding: '+key)
        return super().run_job(job)


def serve(root,protocol_path,lease_fd):
    # Pin the reviewed failed source; preserve its mailbox/order/storage logic.
    expected=B.read(protocol_path)['sources']['author_collection_service.py']
    if expected!=B.binding(ORIGINAL.__file__):raise ValueError('Original service source pin changed')
    tree=ast.parse(Path(ORIGINAL.__file__).read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='serve');count=0
    for n in ast.walk(function):
        if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='CollectionRuntime':
            n.func.id='BoundRuntime';n.args.append(ast.parse("p['inputs']",mode='eval').body);count+=1
    if count!=1:raise ValueError('One native owning-runtime construction required')
    scope=dict(vars(ORIGINAL),BoundRuntime=BoundRuntime)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),__file__+':prebound','exec'),scope)
    return scope['serve'](root,protocol_path,lease_fd)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--lease-fd',type=int,required=True)
    a=p.parse_args();serve(a.root,a.protocol,a.lease_fd)
