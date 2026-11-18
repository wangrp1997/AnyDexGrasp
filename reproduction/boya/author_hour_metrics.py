"""Exact wall-window accounting; no throughput extrapolation or simulator imports."""
import json
from pathlib import Path
import threading
import time


class HourMetrics:
    def __init__(self,directory,window,*,clock_ns=time.monotonic_ns):
        self.directory=Path(directory);self.window=window;self.clock_ns=clock_ns
        self.start=int(window['started_monotonic_ns']);self.deadline=int(window['deadline_monotonic_ns'])
        if self.deadline<=self.start:raise ValueError('A positive wall window is required')
        self.lock=threading.RLock();self.events=[];self.records={};self.returns={};self.verified={}
        self.stream=(self.directory/'HOUR_EVENTS.jsonl').open('x',buffering=1)

    def expired(self):return self.clock_ns()>=self.deadline

    def _event(self,kind,**values):
        now=self.clock_ns()
        if now<self.start:raise ValueError('Clock predates the frozen start')
        row=dict(kind=kind,monotonic_ns=now,elapsed_s=(now-self.start)/1e9,
                 within_window=now<self.deadline,**values)
        self.stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');self.events.append(row)
        return row

    def stage(self,kind,**values):
        with self.lock:
            row=self._event(kind,**values);self._save();return row

    def record(self,index,record,*,execution_complete=False):
        with self.lock:
            if index in self.records:raise ValueError('Duplicate scene record')
            row=self._event('record_committed',index=index,status=record['status'],
                label=record['label'],eligible=record['eligible'],execution_complete=execution_complete)
            self.records[index]=row;self._save();return row

    def returned(self,index,passed):
        with self.lock:
            if index in self.returns:raise ValueError('Duplicate physical return')
            row=self._event('physical_return_complete',index=index,passed=bool(passed))
            self.returns[index]=row;self._save();return row

    def audited(self,wave,indices):
        with self.lock:
            if any(i in self.verified for i in indices):raise ValueError('Duplicate audited label')
            for i in indices:
                if i not in self.records or not self.records[i]['eligible']:raise ValueError('Audit lacks a committed eligible label')
                if i not in self.returns or not self.returns[i]['passed']:raise ValueError('Audit lacks a complete valid return')
            row=self._event('wave_audit_pass',wave=wave,eligible_indices=list(indices))
            for i in indices:self.verified[i]=row
            self._save();return row

    def summary(self):
        with self.lock:
            valid={i:r for i,r in self.records.items()if r['eligible']and r['within_window']}
            returned={i:r for i,r in valid.items()if i in self.returns and self.returns[i]['passed']and self.returns[i]['within_window']}
            audited={i:r for i,r in returned.items()if i in self.verified and self.verified[i]['within_window']}
            all_valid={i:r for i,r in self.records.items()if r['eligible']}
            return dict(window_s=(self.deadline-self.start)/1e9,
                observed_wall_s=(self.clock_ns()-self.start)/1e9,window_elapsed=self.expired(),
                grasp_attempts_started_within_window=sum(r['kind']=='grasp_started'and r['within_window']for r in self.events),
                complete_executions_within_window=sum(r.get('execution_complete',False)and r['within_window']for r in self.records.values()),
                eligible_labels_within_window=len(valid),eligible_positive_within_window=sum(r['label']==1 for r in valid.values()),
                eligible_negative_within_window=sum(r['label']==0 for r in valid.values()),
                grasp_and_return_within_window=len(returned),audited_labels_within_window=len(audited),
                audited_positive_within_window=sum(r['label']==1 for r in audited.values()),
                audited_negative_within_window=sum(r['label']==0 for r in audited.values()),
                eligible_labels_including_drain=len(all_valid),audited_labels_including_drain=len(self.verified),
                timed_out_tail_labels=len(all_valid)-len(valid),
                eligible_indices_within_window=sorted(valid),audited_indices_within_window=sorted(audited))

    def _save(self):
        path=self.directory/'HOUR_COUNTERS.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.summary(),indent=2)+'\n');temporary.replace(path)

    def close(self):
        with self.lock:self._save();self.stream.close()
