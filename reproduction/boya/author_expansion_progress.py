"""Untimed, bounded collection progress using the proven event invariants."""
import json
from pathlib import Path
import threading
import time

from author_hour_metrics import HourMetrics


class Progress(HourMetrics):
    """Reuse record/return/audit validation, without an hourly admission limit."""
    def __init__(self, directory, scene_count):
        self.directory = Path(directory)
        self.scene_count = int(scene_count)
        self.clock_ns = time.monotonic_ns
        self.start = self.clock_ns()
        self.lock = threading.RLock()
        self.events, self.records, self.returns, self.verified = [], {}, {}, {}
        self.stream = (self.directory/'BATCH_EVENTS.jsonl').open('x', buffering=1)

    def expired(self):
        return False

    def _event(self, kind, **values):
        now = self.clock_ns()
        row = dict(kind=kind, monotonic_ns=now, elapsed_s=(now-self.start)/1e9, **values)
        self.stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
        self.events.append(row)
        return row

    def summary(self):
        with self.lock:
            valid = [r for r in self.records.values() if r['eligible']]
            return dict(scene_budget=self.scene_count, terminal_scenes=len(self.records),
                elapsed_s=(self.clock_ns()-self.start)/1e9,
                eligible_labels=len(valid), positive=sum(r['label']==1 for r in valid),
                negative=sum(r['label']==0 for r in valid), null=len(self.records)-len(valid),
                audited_labels=len(self.verified),
                successful_returns=sum(r['passed'] for r in self.returns.values()),
                grasp_attempts=sum(r['kind']=='grasp_started' for r in self.events),
                eligible_indices=sorted(i for i,r in self.records.items() if r['eligible']),
                audited_indices=sorted(self.verified), training_updates=0)

    def _save(self):
        path = self.directory/'BATCH_COUNTERS.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.summary(), indent=2)+'\n')
        temporary.replace(path)
