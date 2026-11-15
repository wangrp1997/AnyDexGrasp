"""Parent-side finite client. Worker reservations cover the entire process life."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import time
import traceback

import author_collection_batch as B
import author_collection_wave_resources as R
import author_scaling_control as C
import author_contact_compressor as W


class ResidentClient:
    def __init__(self,root,protocol_path,pool,lease_fd):
        self.root=Path(root);self.path=Path(protocol_path);self.p=B.read(protocol_path);self.pool=pool;self.lease_fd=lease_fd
        self.sequence=0;self.closed=False;self.process=None
        self.stage=self.root/'service_stage';self.stage.mkdir(parents=True,exist_ok=False)
        self.runtime=self.root/'runtime';self.executor=ThreadPoolExecutor(max_workers=1)
        self.future=self.executor.submit(self._lifetime)
        try:self._wait_file(self.runtime/'SERVICE_READY.json',120)
        except BaseException:
            self.close(abort=True);raise

    def _lifetime(self):
        row=dict(status='failed',started_unix_s=time.time());process=None
        try:
            with self.pool.stage(('kit',),self.stage):
                with(self.stage/'process.log').open('xb')as log:
                    command=[self.p['inputs']['python']['path'],'-B',self.p['sources']['service_entry']['path'],
                             '--root',str(self.runtime),'--protocol',str(self.path),'--lease-fd',str(self.lease_fd)]
                    process=self.pool.popen(command,cwd=B.ROOT,env=B._environment(self.p),stdout=log,
                                            stderr=subprocess.STDOUT,start_new_session=True)
                    self.process=process;row['process']=R.process_identity(process.pid)
                    W.save(self.stage/'PROCESS.json',row)
                    try:code=process.wait(timeout=self.p['service_wall_timeout_s'])
                    except BaseException:
                        row['signals']=C.finish_owned(process,row['process'],self.p);raise
                    row.update(returncode=code,status='complete'if code==0 else'stopped')
        except BaseException:
            row['error']=traceback.format_exc()
            if process is not None and process.poll()is None:
                row['signals']=C.finish_owned(process,row['process'],self.p)
        finally:
            row['finished_unix_s']=time.time();W.save(self.stage/'RESULT.json',row)
        return row

    def _wait_file(self,path,timeout):
        deadline=time.monotonic()+timeout
        while not path.exists():
            if self.future.done():raise RuntimeError('Resident worker exited: '+str(self.future.result()))
            if time.monotonic()>deadline:raise TimeoutError('Resident result deadline: '+str(path))
            time.sleep(.02)
        return B.read(path)

    def submit(self,index):
        if self.closed or index!=self.sequence:raise ValueError('No duplicate/out-of-order request')
        file=self.runtime/'mailbox'/f'{index:03d}.json'
        W.save(file,dict(sequence=index,protocol=B.binding(self.path),role='job',job=self.p['worker_jobs'][index]))
        self.sequence+=1
        return index

    def wait(self,index):
        return self._wait_file(self.runtime/'mailbox'/f'{index:03d}.RESULT.json',self.p['stage_timeout_s'])

    def close(self,abort=False):
        if self.closed:return
        self.closed=True
        if not self.future.done():
            W.save(self.runtime/'mailbox'/f'{self.sequence:03d}.json',dict(sequence=self.sequence,
                protocol=B.binding(self.path),role='abort'if abort else'close'))
        try:
            try:return self.future.result(timeout=60)
            except TimeoutError:
                if self.process is not None:
                    C.finish_owned(self.process,B.read(self.stage/'PROCESS.json')['process'],self.p)
                return self.future.result(timeout=10)
        finally:self.executor.shutdown(wait=True)


class ColdClient:
    """Same ordered asynchronous jobs, one independently closed Kit per job."""
    def __init__(self,root,protocol_path,pool,lease_fd):
        self.root=Path(root);self.p=B.read(protocol_path);self.sequence=0;self.closed=False
        self.executor=ThreadPoolExecutor(max_workers=1);self.futures={}

    def submit(self,index):
        if self.closed or index!=self.sequence:raise ValueError('Frozen cold-job order required')
        self.sequence+=1;self.futures[index]=self.executor.submit(self._run,index);return index

    def _run(self,index):
        job=self.p['worker_jobs'][index];directory=self.root/'cold_jobs'/f'{index:03d}'
        py=self.p['inputs']['python']['path']
        if job['role']=='capture':
            command=[py,'-B',self.p['sources']['capture_entry']['path'],'--scene',job['scene'],'--output-dir',job['output_dir']]
            expected=[Path(job['output_dir'])/'RESULTS.json']
        else:
            output=self.root/'cold_sessions'/f'{index:03d}';bindings=self.root/f'COLD_SOURCES_{index:03d}.json'
            B.save(bindings,self.p['sources'],exclusive=True)
            command=[py,'-B',self.p['sources']['execution_entry']['path'],'execute-many','--prepared-dirs',job['prepared_dir'],
                     '--output-dir',str(output),'--source-bindings',str(bindings)]
            expected=[output/'EXECUTE_MANY_RESULTS.json',output/'SESSION_RECEIPT.json']
        result=B._stage(directory,command,expected,self.p)
        if result['status']!='complete':raise RuntimeError('Cold control job failed: '+str(result.get('error')))
        return result

    def wait(self,index):return self.futures[index].result(timeout=self.p['stage_timeout_s'])

    def close(self,abort=False):
        if self.closed:return
        self.closed=True;self.executor.shutdown(wait=True,cancel_futures=abort)
