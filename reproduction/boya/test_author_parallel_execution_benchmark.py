"""Dispatcher checks use ordinary Python children only: no Kit or models."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import author_parallel_execution_benchmark as M


class BenchmarkTest(unittest.TestCase):
    def fixture(self, root):
        root = Path(root); sources = root/'original'; sources.mkdir()
        jobs = []
        for g in (1, 6):
            old = sources/f'g{g}'; old.mkdir()
            job = root/'jobs'/f'g{g}'; job.mkdir(parents=True)
            copies = {}
            for name in M.COPIES:
                data = ('{"source": "g%d", "name": "%s"}\n' % (g, name)).encode()
                (old/name).write_bytes(data); (job/name).write_bytes(data)
                copies[name] = M.B.binding(old/name)
            jobs.append(dict(job_id=f'g{g}', gesture_id=g, prepared_dir=str(job),
                copied_files=copies, baseline_stage_wall_s=100.))
        protocol = dict(schema=M.SCHEMA, purpose='diagnostic_only', exclude_training=True,
            job_count=2, executions_per_job=1, retry_budget=0,
            resource_limits=M.R.LIMITS, gpu=0, inputs={}, sources={},
            frozen_dependencies={}, jobs=jobs)
        M.B.save(root/'BENCHMARK_PROTOCOL.json', protocol)
        return protocol

    def test_copy_contract_detects_old_and_new_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root); M.verify(root)
            for target in (root/'jobs/g1/READY.json', root/'original/g6/LIFT_PIPELINE_PROTOCOL.json'):
                old = target.read_bytes(); target.write_bytes(old+b' ')
                with self.assertRaisesRegex(ValueError, 'bytes changed'):
                    M.verify(root)
                target.write_bytes(old)
            p = M.B.read(root/'BENCHMARK_PROTOCOL.json'); p['jobs'][0]['copied_files'].pop('READY.json')
            M.B.save(root/'BENCHMARK_PROTOCOL.json', p)
            with self.assertRaisesRegex(ValueError, 'Exactly three'):
                M.verify(root)

    def test_started_benchmark_never_relaunches_even_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            M.B.save(root/'BENCHMARK_STARTED.json', {'prior':'unknown'})
            M.B.save(root/'BENCHMARK_RESULTS.json', {'status':'infrastructure_or_audit_failure'})
            with patch.object(M.R, 'lease', side_effect=AssertionError('must not acquire/dispatch')):
                with self.assertRaises(FileExistsError): M.run(root)

    def test_speedup_is_stage_span_and_missing_audit_is_not_success(self):
        protocol={'jobs':[{'baseline_stage_wall_s':100.},{'baseline_stage_wall_s':100.}]}
        rows=[dict(execution_stage=dict(status='complete',started_unix_s=s,wall_s=100.),
                   independent_audit_status='pass',physics_steps=10,execution_steps_unknown=False)
              for s in (1.,11.)]
        result=M._summary(protocol,rows,150.)
        self.assertEqual(result['execution_stage_overlap_s'],90.)
        self.assertEqual(result['new_execution_stage_span_s'],110.)
        self.assertAlmostEqual(result['observed_execution_stage_speedup'],200/110)
        self.assertEqual(result['new_training_labels'],0)
        rows[1]['independent_audit_status']='failed'
        self.assertIsNone(M._summary(protocol,rows,150.)['observed_execution_stage_speedup'])
        self.assertIsNone(M._summary(protocol,[],150.)['execution_stage_overlap_s'])

    def test_original_auditor_accepts_copied_local_protocol_before_execution(self):
        # Run the actual auditor's pre-execution branch. READY intentionally
        # retains the absolute original protocol, while new local bytes match.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); original=root/'old.json'; local=root/'new'; local.mkdir()
            scene=root/'scene.json'; M.B.save(scene,dict(scene_id='s',source_sha256={},input_sha256={}))
            protocol=dict(task=M.B.TASK,bin=None,inputs={'scene':M.B.binding(scene)},sources={},
                original_object_ids=['a','b','c'],primary_is_candidate_target=False,candidate_budget=1,fallback_candidate=False)
            M.B.save(original,protocol); (local/'LIFT_PIPELINE_PROTOCOL.json').write_bytes(original.read_bytes())
            M.B.save(local/'READY.json',dict(protocol=M.B.binding(original)))
            M.B.save(local/'LIFT_PIPELINE_RESULTS.json',dict(status='ready'))
            with patch.object(M.A,'_identity',return_value=(['a','b','c'],['x']*3,['y']*3,'fit')):
                result=M.A.audit_trial(local)
            self.assertEqual(result['status'],'not_executed',result)
            self.assertIsNone(result['label'])

    def test_two_plain_children_inherit_real_existing_pool_locks(self):
        # An integration check of this dispatch path, using harmless scripts.
        # The child rejects a missing inherited exclusive global/slot FD.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); exe=root/'execute.py'; audit=root/'audit.py'
            exe.write_text('''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--prepared-dir');a=p.parse_args();d=Path(a.prepared_dir)
locks=[]
for fd in Path('/proc/self/fdinfo').glob('*'):
 try:
  t=fd.read_text()
  if 'FLOCK' in t and 'WRITE' in t:locks.append(str(fd))
 except FileNotFoundError:pass
assert len(locks)>=2,locks
now=time.time();time.sleep(.4)
(d/'COLLECTION_RECORD.json').write_text('{}')
(d/'LIFT_PIPELINE_EXECUTION.json').write_text(json.dumps(dict(complete=True,status='complete',label=1,physics_steps=0,started=now,ended=time.time(),locks=locks)))
''')
            audit.write_text('''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--prepared-dir');p.add_argument('--split-manifest');p.add_argument('--output');a=p.parse_args()
Path(a.output).write_text(json.dumps(dict(status='pass',label=1)))
''')
            split=root/'split.json'; split.write_text('{}')
            protocol=dict(inputs={'python':M.B.binding(sys.executable)},
                sources={'execution_entry':M.B.binding(exe),'audit_entry':M.B.binding(audit),
                         'capture_entry':M.B.binding(root/'capture.py') if (root/'capture.py').exists() else {'path':str(root/'capture.py'),'sha256':''},
                         'scene_entry':{'path':str(root/'build.py'),'sha256':''}},
                inherited_env={},env=dict(M.B.DEFAULT_ENV, PYTHONPATH=''),stage_timeout_s=5.,split_manifest=M.B.binding(split))
            # classify accesses all entry names, but only execution/audit run.
            protocol['sources'].pop('capture_entry');protocol['sources'].pop('scene_entry')
            for name in ('capture_entry','scene_entry'):
                dummy=root/(name+'.py');dummy.write_text('');protocol['sources'][name]=M.B.binding(dummy)
            jobs=[]
            for g in (1,6):
                d=root/'jobs'/f'g{g}';d.mkdir(parents=True)
                jobs.append(dict(job_id=f'g{g}',prepared_dir=str(d)))
            with M.R.lease(root/'global.lock') as fd:
                pool=M.R.ResourcePool(root/'resource_slots',fd,{'path':'unit','sha256':'unit'})
                with M.R.install_worker_adapters(pool,root/'global.lock'):
                    with M.ThreadPoolExecutor(max_workers=2) as executor:
                        futures=[executor.submit(M._job,root,protocol,row) for row in jobs]
                        rows=[f.result() for f in futures]
            receipts=[M.B.read(Path(j['prepared_dir'])/'LIFT_PIPELINE_EXECUTION.json') for j in jobs]
            self.assertLess(max(r['started'] for r in receipts),min(r['ended'] for r in receipts))
            self.assertTrue(all(r['independent_audit_status']=='pass' for r in rows))
            states=[M.B.read(p) for p in (root/'resource_slots/kit').glob('*.json')]
            self.assertEqual(len(states),2)
            self.assertTrue(all(s['state']=='released' and not M.R.group_alive(s['pgid']) for s in states))


if __name__=='__main__':unittest.main()
