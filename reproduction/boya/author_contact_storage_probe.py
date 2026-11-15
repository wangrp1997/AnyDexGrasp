"""Replay frozen real contact bytes through storage only; optional Kit, zero steps."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import author_collection_batch as B
import author_collection_execute as OLD
import author_contact_compressor as W
import author_verified_contact_journal as J


def native_state():
    import omni.timeline
    import omni.usd
    from pxr import UsdPhysics
    import isaaclab.sim as sim
    stage=omni.usd.get_context().get_stage()
    scenes=[] if stage is None else[str(p.GetPath())for p in stage.Traverse()if p.IsA(UsdPhysics.Scene)]
    context=sim.SimulationContext.instance()
    return dict(timeline_playing=omni.timeline.get_timeline_interface().is_playing(),physics_scenes=scenes,
        simulation_context_exists=context is not None,
        physics_step_count=None if context is None else context.get_physics_step_count())


def one(root,index,algorithm,protocol):
    directory=root/algorithm/f'writer_{index:02d}';directory.mkdir(parents=True,exist_ok=False)
    target=directory/'CONTACTS.jsonl.gz';started=time.monotonic();result=dict(writer=index,algorithm=algorithm,
        status='failed',Kit_starts=0,physics_steps=0,synthetic_replay_records=0,label=None,exclude_training=True)
    journal=None
    try:
        journal=(OLD.ContactJournal(target,source_bindings=protocol['source_sha256'])if algorithm=='original_gzip'else
            J.ContactJournal(target,source_bindings=protocol['source_sha256'],spool_root=protocol['spool_root']))
        with Path(protocol['fixture']['raw']['path']).open('rb')as stream:
            for line in stream:
                row=json.loads(line);journal.append(row);result['synthetic_replay_records']+=1
        pointer=journal.pointer();B.save(directory/'POINTER.json',pointer,exclusive=True)
        command=[sys.executable,'-I','-S','-B',str(Path(W.__file__).resolve()),'--inspect-existing',str(target),
            '--receipt',str(directory/'INDEPENDENT_READBACK.json'),'--sha256',protocol['fixture']['logical']['sha256'],
            '--bytes',str(protocol['fixture']['logical']['bytes']),'--records',str(protocol['fixture']['logical']['records'])]
        env={k:v for k,v in os.environ.items()if k not in('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','LD_PRELOAD')}
        with(directory/'reader.log').open('xb')as log:
            process=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300.)
        readback=B.read(directory/'INDEPENDENT_READBACK.json')
        if readback['status']!='verified'or process.returncode!=0:raise ValueError('Stored logical bytes failed independent readback: '+str(readback.get('error')))
        if pointer['record_count']!=protocol['fixture']['logical']['records']:raise ValueError('Producer record count differs')
        if algorithm=='verified_isolated':J.verify_pointer(pointer)
        result.update(status='verified',pointer=B.binding(directory/'POINTER.json'),readback=B.binding(directory/'INDEPENDENT_READBACK.json'))
    except BaseException:result['error']=traceback.format_exc()
    finally:
        if journal is not None:
            try:journal.close()
            except BaseException:result['close_error']=traceback.format_exc()
    result.update(wall_s=time.monotonic()-started)
    B.save(directory/'RESULT.json',result,exclusive=True)
    return result


def run(root,protocol_path,mode):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);protocol=B.read(protocol_path)
    B.verify(protocol['inputs']);B.verify(protocol['sources']);B.verify({'raw':protocol['fixture']['raw']})
    app=None;before=after=None;groups=[];error=None;started=time.monotonic();Kit_starts=0
    B.save(root/'STARTED.json',dict(protocol=B.binding(protocol_path),mode=mode,pid=os.getpid(),physics_steps=0),exclusive=True)
    try:
        if mode=='kit':
            _,pipeline=OLD._modules();app=OLD._launch_native_gpu0(pipeline.common);Kit_starts=1
            before=native_state()
            if before['timeline_playing']or before['physics_scenes']or before['simulation_context_exists']:
                raise ValueError('Storage-only Kit probe requires stopped empty stage without SimulationContext')
        for algorithm in('original_gzip','verified_isolated'):
            with ThreadPoolExecutor(max_workers=protocol['writers'])as executor:
                workers=list(executor.map(lambda i:one(root,i,algorithm,protocol),range(protocol['writers'])))
            groups.append(dict(algorithm=algorithm,workers=workers,all_verified=all(w['status']=='verified'for w in workers)))
        if mode=='kit':
            after=native_state()
            if after!=before:raise ValueError('Storage probe changed timeline, stage or simulation context')
    except BaseException:error=traceback.format_exc()
    finally:
        result=dict(status='verified_storage'if error is None and len(groups)==2 and groups[-1]['all_verified']else'failed',
            mode=mode,groups=groups,error=error,Kit_starts=Kit_starts,physics_steps=0,
            no_sim_step_called=True,native_before=before,native_after=after,wall_s=time.monotonic()-started,
            label=None,exclude_training=True,CGR_forwards=0,training_updates=0)
        B.verify(protocol['inputs']);B.verify(protocol['sources']);B.save(root/'RESULTS.json',result,exclusive=True)
        if app is not None:app.app.close(exit_code=0 if result['status']=='verified_storage'else 1)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--mode',choices=['standalone','kit'],required=True)
    a=p.parse_args();r=run(a.root,a.protocol,a.mode)
    print(json.dumps(dict(status=r['status'],mode=a.mode,physics_steps=0)))
    if r['status']!='verified_storage':raise SystemExit(1)
