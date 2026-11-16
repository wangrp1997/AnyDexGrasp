"""Single owning Kit worker for interleaved photograph/grasp rounds."""
import argparse
import os
from pathlib import Path
import time
import traceback
import author_collection_batch as B
import author_collection_wave_resources as R
from author_live_world import World


def serve(root,protocol,lease_fd):
    root=Path(root).resolve();p=B.read(protocol);B.verify(p['sources']);B.verify(p['inputs'])
    if p['max_Kit_starts']!=1 or p['exclude_training']is not True or p['max_execution_steps_per_job']!=8000:raise ValueError('Finite diagnostic live world required')
    roles=[j['role']for j in p['worker_jobs']]
    if roles!=['capture','execute']*len(p['rows']):raise ValueError('Interleaved observation-to-execution order required')
    lock=B.ROOT/'outputs/author_collection_gpu0.lock';R.check_inherited_lease(lease_fd,lock,[lock.stat().st_dev,lock.stat().st_ino])
    root.mkdir(parents=True,exist_ok=True);mailbox=root/'mailbox';mailbox.mkdir(exist_ok=False)
    world=World(root,p);binding=B.binding(protocol);B.save(root/'SERVICE_READY.json',dict(pid=os.getpid(),protocol=binding),exclusive=True)
    error=None;index=0
    try:
        while True:
            request=mailbox/f'{index:03d}.json';deadline=time.monotonic()+p['service_idle_timeout_s']
            while not request.exists():
                if time.monotonic()>deadline:raise TimeoutError('Finite live-world idle timeout')
                time.sleep(.02)
            value=B.read(request)
            if value['sequence']!=index or value['protocol']!=binding:raise ValueError('Unbound live-world request')
            if value['role']=='abort':error='Parent aborted';break
            if value['role']=='close':
                if index!=len(p['worker_jobs']):raise ValueError('Unexpected early close')
                break
            if index>=len(p['worker_jobs'])or value['job']!=p['worker_jobs'][index]:raise ValueError('Frozen round order/paths changed')
            B.verify(p['sources']);job=value['job'];began=time.monotonic()
            if job['role']=='capture':result=world.capture(job['round_index'],job['scene'],job['output_dir'])
            else:
                prepared=Path(job['prepared_dir'])
                if (prepared/'READY.json').exists():
                    ready=B.read(prepared/'READY.json')
                    if B.read(ready['plan']['path'])['total_command_steps']>8000:raise ValueError('Original finite command budget exceeded')
                elif not(prepared/'NOT_EXECUTED.json').exists():raise ValueError('No prepared plan or explicit planning null')
                result=world.execute(prepared)
            B.verify(p['sources']);B.save(mailbox/f'{index:03d}.RESULT.json',dict(status='complete',request=B.binding(request),job=result,wall_s=time.monotonic()-began),exclusive=True)
            index+=1
    except BaseException:error=traceback.format_exc();B.save(root/'SERVICE_ERROR.json',dict(error=error,sequence=index),exclusive=True)
    finally:
        B.save(root/'SERVICE_FINISHED.json',dict(status='complete'if error is None else'failed',error=error,jobs_completed=index),exclusive=True)
        world.close(int(error is not None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--lease-fd',type=int,required=True);a=p.parse_args();serve(a.root,a.protocol,a.lease_fd)
