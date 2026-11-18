"""Paired held-out layout comparison with original native physics and audit."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import author_collection_batch as B
import author_mjx_batch_collection_v9 as F
from author_mjx_hour_collection import collect_wave
from author_expansion_progress import Progress
from author_hour_audit import AuditPool, combine
from author_mjx_policy_compare_worker import planning_batch


def run(root):
    import author_collection_wave_resources as R
    import author_resident_prepare as P
    root=Path(root).resolve();p=B.read(root/'PROTOCOL.json');B.verify(p['inputs']);B.verify(p['sources'])
    if(root/'RUN_STARTED.json').exists():raise ValueError('No dispatched-root retry')
    P.bind_backend(dict(backend=p['inputs']['backend']['path']))
    from author_compiled_kinematics import CompiledKinematics
    from author_mjx_native_batch import Batch
    from author_mujoco_representation import Stream
    template=B.read(p['inputs']['template:0']['path']);plan=B.read(template['plan']['path'])
    scenes=[B.read(s['scene']['path'])for s in p['rows']if s.get('scene')]
    with np.load(template['commands']['path'])as z:names=z['joint_names'].tolist();initial=z['initial_position'].copy()
    metrics=Progress(root,len(p['rows']));original_save=B.save
    def journaled_save(path,value,*args,**kwargs):
        answer=original_save(path,value,*args,**kwargs);path=Path(path).resolve()
        if path.parent.parent==root/'scenes':
            if path.name=='RECORD.json':
                execution=value.get('bindings',{}).get('execution')
                complete=bool(execution and B.read(execution['path'])['status']=='complete')
                metrics.record(int(path.parent.name),value,execution_complete=complete)
            elif path.name=='RETURN.json':metrics.returned(int(path.parent.name),value['pass_'])
        return answer
    batch=engine=auditor=None;waves=[];error=None;stop_reason='queue_exhausted'
    cost=dict(CGR_forwards=0,planning_jobs=0,grasp_attempts=0);began=time.monotonic()
    lock=B.ROOT/'outputs/author_collection_gpu0.lock'
    try:
        with R.lease(root/'DRIVER.lock'),R.lease(lock)as lease,R.active_run(root,lock.with_suffix('.wave_active.json'),B.binding(root/'PROTOCOL.json')):
            B.save(root/'RUN_STARTED.json',dict(process=R.process_identity(os.getpid()),started_unix_s=time.time()),exclusive=True)
            pool=R.ResourcePool(root/'resource_slots',lease,B.binding(root/'PROTOCOL.json'))
            with patch.object(B,'save',journaled_save):
                try:
                    with CompiledKinematics().installed():
                        batch=Batch(plan,names,initial,scenes,count=p['environments'],max_ticks=p['max_global_control_ticks'])
                        batch.set_cost_journal(root/'NATIVE_CALLS.jsonl')
                        (root/'MODEL.xml').write_text(batch.base.spec.to_xml())
                        B.save(root/'MODEL_RECEIPT.json',dict(model=B.binding(root/'MODEL.xml'),model_format='Original native CPU MJCF source spec',
                            GPU_CCD=batch.compatibility,GPU_graph_mode='native_single_control_graph',source_template=p['inputs']['template:0'],camera_count=1),exclusive=True)
                        args=SimpleNamespace(checkpoint=Path(p['inputs']['checkpoint']['path']),native_deps=Path(p['native_deps']),device='cuda:0',
                            seed=p['seed'],num_seed=2048,voxel_size=.005,depth_min=.2,depth_max=.65)
                        engine=Stream(args);auditor=AuditPool(p,pool,metrics)
                        for number,start in enumerate(range(0,len(p['rows']),p['environments'])):
                            auditor.raise_if_failed()
                            if batch.max_ticks-batch.ticks<2500+8000+8100:raise RuntimeError('Frozen budget cannot admit the next wave')
                            specs=p['rows'][start:start+p['environments']]
                            wave,audit_root=collect_wave(root,p,batch,engine,pool,metrics,initial,specs,number,cost)
                            waves.append(wave);B.save(root/'WAVES.json',waves)
                            auditor.submit(audit_root,number)
                except BaseException:error=traceback.format_exc();stop_reason='controller_error'
                finally:
                    if batch is not None:
                        try:batch.close()
                        except BaseException:error=(error or '')+'\nCLOSE_ERROR\n'+traceback.format_exc()
                    if auditor is not None:
                        failures=auditor.close()
                        if failures:error=(error or '')+'\nAUDIT_ERRORS\n'+json.dumps(failures)
                    for spec in p['rows']:
                        if not(root/'scenes'/f"{spec['index']:02d}"/'RECORD.json').exists():
                            F.write_record(root,spec,terminal='controller_error_or_not_executed',error=error)
                    B.verify(p['inputs']);B.verify(p['sources'])
                    records=[B.read(root/'scenes'/f"{s['index']:02d}"/'RECORD.json')for s in p['rows']]
                    result=dict(status='complete'if error is None else'failed',error=error,stop_reason=stop_reason,waves=waves,records=records,cost=cost,
                        world=None if batch is None else batch.receipt(),representation=None if engine is None else engine.receipt(),
                        parent_wall_s=time.monotonic()-began,batch_progress=metrics.summary(),labels=sum(r['eligible']for r in records),training_updates=0,
                        production_ready=False,formal_quota_frozen=False,comparison_only=True,formal_training_eligible=False,video_recording=False)
                    B.save(root/'RESULTS.json',result,exclusive=True)
                    if error is None:combine(root,p,result['world'],result)
                    B.save(root/'OWNER_SCOPE_FINISHED.json',dict(process=R.process_identity(os.getpid()),finished_unix_s=time.time(),controller_scope_exited=True),exclusive=True)
    finally:metrics.close()
    print({'status':result['status'],'stop_reason':stop_reason,'counts':metrics.summary()},flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);a=parser.parse_args()
    p=B.read(a.root/'PROTOCOL.json');marker=B.sha(a.root/'PROTOCOL.json')
    if os.environ.get('ANYDEX_MJX_POLICY_COMPARE_ENV')!=marker:
        env=B._environment(p);env.update(PYTHONPATH=os.pathsep.join((str(F.WARP),str(F.DEPS),env['PYTHONPATH'])),
            CUDA_VISIBLE_DEVICES=F.GPU,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',WARP_CACHE_PATH=str(a.root.resolve()/'warp_cache'),ANYDEX_MJX_POLICY_COMPARE_ENV=marker)
        os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(a.root.resolve())],env)
    with patch.object(F,'planning_batch',planning_batch):
        if run(a.root)['status']!='complete':raise SystemExit(1)
