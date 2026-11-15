"""Original per-frame CGR body with one persistent model across streamed frames."""
import ast
from pathlib import Path
from types import SimpleNamespace
import time

import run_collection_representation as ORIGINAL

PIN='ee8b43bc334ba11e394d4030d377c0bb5ced211c9bff85ca92ac1bbc33f97093'


def frame_function():
    if ORIGINAL.sha(ORIGINAL.__file__)!=PIN:raise ValueError('Frozen original CGR runner changed')
    tree=ast.parse(Path(ORIGINAL.__file__).read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    target=ast.dump(ast.parse('j,row',mode='eval').body,include_attributes=False).replace('Load()','Store()')
    loops=[n for n in ast.walk(main)if isinstance(n,ast.For)and ast.dump(n.target,include_attributes=False)==target]
    if len(loops)!=1:raise ValueError('Original frame body changed')
    fn=ast.parse('def process(a,engine,row,j,batch,manifest_digest,start,batch_path):\n    failed=False\n    for j,row in [(j,row)]:\n        pass\n    return report').body[0]
    fn.body[1].body=loops[0].body
    scope=dict(vars(ORIGINAL));exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),__file__,'exec'),scope)
    return scope['process']


class RepresentationStream:
    def __init__(self,args,rows):
        self.args=args;self.rows=rows;self.function=frame_function();self.engine=None;self.reports=[];self.failed=False
        self.manifest_digest=ORIGINAL.sha(args.manifest);self.started=time.monotonic()
        source_paths=[Path(__file__),Path(ORIGINAL.__file__),Path(ORIGINAL.__file__).with_name('run_pilot_representation.py'),
            ORIGINAL.ROOT/'reproduction/runtime/run_official_rgbd_frame.py',ORIGINAL.ROOT/'models/minkowski_graspnet_single_point.py',
            ORIGINAL.ROOT/'robot_dh3.py',ORIGINAL.ROOT/'utils/pt_utils.py']
        self.batch=dict(args={k:str(v)if isinstance(v,Path)else v for k,v in vars(args).items()},
            source_sha256={str(p.resolve()):ORIGINAL.sha(p)for p in source_paths},objects=[dict(r,status='not_started')for r in rows],
            network_forward_count=0,timing_seconds={})

    def forward(self,index):
        if self.failed or index!=len(self.reports):raise ValueError('Original frozen frame order required; failures stop dispatch')
        if self.engine is None:self.engine=ORIGINAL.PersistentOfficialCGR(self.args)
        report=self.function(self.args,self.engine,self.rows[index],index,self.batch,self.manifest_digest,self.started,
                             self.args.run_dir/'BATCH_REPRESENTATION.json')
        self.reports.append(report)
        if report['status']!='representation_forward_complete':
            self.failed=True;self.save();raise RuntimeError('Original CGR frame failed')
        self.save();return report

    def save(self):
        value=dict(schema='anydex_boya_stream_representation_v1',model_load_count=int(self.engine is not None),
                   network_forward_count=0 if self.engine is None else self.engine.calls,hand_head_load_count=0,
                   objects=self.reports,source_sha256=self.batch['source_sha256'],new_training_steps=0,new_physics_steps=0)
        ORIGINAL.save(self.args.run_dir/'STREAM_REPRESENTATION.json',value);return value
