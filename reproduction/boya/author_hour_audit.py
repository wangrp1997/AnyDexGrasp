"""Audit immutable completed waves while the same world collects later waves."""
from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import author_collection_batch as B
import author_collection_wave_resources as R


def snapshot(root,p,batch,start_tick,segment,specs,wave_index):
    directory=root/'wave_audits'/f'{wave_index:03d}'
    directory.mkdir(parents=True,exist_ok=False)
    protocol=copy.deepcopy(p);protocol['rows']=copy.deepcopy(specs)
    protocol['global_step_offset']=int(start_tick)
    protocol['inputs']['source_hour_protocol']=B.binding(root/'PROTOCOL.json')
    protocol['closed_wave_snapshot']=True
    B.save(directory/'PROTOCOL.json',protocol,exclusive=True)
    (directory/'source_archive').symlink_to(root/'source_archive',target_is_directory=True)
    (directory/'GPU_JOURNAL.jsonl').symlink_to(segment)
    (directory/'MODEL.xml').symlink_to(root/'MODEL.xml')
    B.save(directory/'MODEL_RECEIPT.json',B.read(root/'MODEL_RECEIPT.json'),exclusive=True)
    (directory/'scenes').mkdir()
    records=[]
    for spec in specs:
        name=f"{spec['index']:02d}"
        (directory/'scenes'/name).symlink_to(root/'scenes'/name,target_is_directory=True)
        records.append(B.read(root/'scenes'/name/'RECORD.json'))
    world=batch.receipt();end=batch.ticks
    world.update(cumulative_global_control_ticks=end,cumulative_native_environment_integrations=world['native_environment_integrations'],
        global_control_ticks=end-start_tick,native_environment_integrations=(end-start_tick)*batch.count*8)
    result=dict(status='complete_closed_wave',records=records,world=world,training_updates=0,
        source_hour_protocol=B.binding(root/'PROTOCOL.json'),source_segment=B.binding(segment))
    B.save(directory/'RESULTS.json',result,exclusive=True)
    return directory


class AuditPool:
    def __init__(self,p,pool,metrics):
        self.p=p;self.pool=pool;self.metrics=metrics
        self.executor=ThreadPoolExecutor(max_workers=2);self.futures=[]

    def submit(self,directory,wave_index):
        future=self.executor.submit(self._run,directory,wave_index)
        self.futures.append((directory,future));return future

    def _run(self,directory,wave_index):
        command=[sys.executable,'-B',str(B.HERE/'audit_author_mjx_wave_snapshot.py'),'--root',str(directory)]
        B.save(directory/'DISPATCH.json',dict(command=command),exclusive=True)
        began=time.monotonic()
        with self.pool.stage(('audit',),directory):
            with (directory/'process.log').open('x')as log:
                process=self.pool.popen(command,cwd=B.ROOT,env=B._environment(self.p),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                identity=R.process_identity(process.pid);B.save(directory/'PROCESS.json',dict(process=identity),exclusive=True)
                try:code=process.wait(timeout=self.p['stage_timeout_s'])
                except BaseException:
                    if R.process_identity(process.pid)==identity:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=15)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=15)
                    raise
                if code:raise RuntimeError('Closed-wave audit failed: '+str(directory))
        audit=B.read(directory/'READ_ONLY_AUDIT.json');result=B.read(directory/'RESULTS.json')
        if audit['status']!='read_only_audit_pass'or audit['original_result']!=B.binding(directory/'RESULTS.json'):
            raise ValueError('Closed-wave audit is not bound to the source wave')
        checked={r['index']:r for r in audit['rounds']};eligible=[]
        for spec,record in zip(B.read(directory/'PROTOCOL.json')['rows'],result['records']):
            if record['eligible']:
                item=checked[spec['index']]
                if not item.get('eligible')or not item.get('return_pass'):raise ValueError('Incomplete verified sample')
                eligible.append(spec['index'])
        event=self.metrics.audited(wave_index,eligible)
        B.save(directory/'FINISHED.json',dict(status='pass',wall_s=time.monotonic()-began,event=event,
            audit=B.binding(directory/'READ_ONLY_AUDIT.json'),eligible_indices=eligible),exclusive=True)
        return audit

    def raise_if_failed(self):
        for _,future in self.futures:
            if future.done():future.result()

    def close(self):
        self.executor.shutdown(wait=True)
        errors=[]
        for directory,future in self.futures:
            try:future.result()
            except BaseException as error:errors.append(dict(directory=str(directory),error=repr(error)))
        return errors


def combine(root,p,world,result):
    rounds={};proofs=[];total_ticks=total_native=total_contacts=matched=0;error=0.
    for directory in sorted((root/'wave_audits').glob('*')):
        audit=B.read(directory/'READ_ONLY_AUDIT.json')
        if audit['status']!='read_only_audit_pass':raise ValueError('Unverified wave')
        if audit['global_step_offset']!=total_ticks:raise ValueError('Missing physical journal segment')
        for row in audit['rounds']:
            if row['index']in rounds:raise ValueError('Repeated scene audit')
            rounds[row['index']]=row
        total_ticks+=audit['global_ticks'];total_native+=audit['native_environment_integrations']
        total_contacts+=audit['all_world_contacts_including_idle'];matched+=audit['matched_trial_rows']
        error=max(error,audit['max_native_palm_FK_error']);proofs.append(B.binding(directory/'READ_ONLY_AUDIT.json'))
    if total_ticks!=world['global_control_ticks']or total_native!=world['native_environment_integrations']:
        raise ValueError('Wave audit costs do not cover the complete physical run')
    for spec,record in zip(p['rows'],result['records']):
        if spec['index']not in rounds:
            if record['eligible']:raise ValueError('Eligible record was never audited')
            rounds[spec['index']]=dict(index=spec['index'],status=record['status'],label=None)
    output=dict(status='read_only_audit_pass',original_result=B.binding(root/'RESULTS.json'),
        rounds=[rounds[r['index']]for r in p['rows']],closed_wave_proofs=proofs,
        global_ticks=total_ticks,native_environment_integrations=total_native,
        all_world_contacts_including_idle=total_contacts,matched_trial_rows=matched,
        max_native_palm_FK_error=error,source_archive_and_inputs_verified=True,
        all_complete_grasps_pass=all(r.get('eligible',False)for r in rounds.values()if r.get('trace')),
        additional_physics_steps=0,additional_CGR_forwards=0,production_ready=False,
        implementation=B.binding(__file__))
    B.save(root/'READ_ONLY_AUDIT.json',output,exclusive=True);return output
