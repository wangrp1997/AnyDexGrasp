"""Lossless storage, unchanged evaluator and receipt-before-close regressions."""
import gzip
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from reproduction.boya import author_collection_execute as entry
from reproduction.boya import author_lift as lift
from reproduction.boya import author_transport as common
from reproduction.boya.test_author_lift import IDS,trace


def fill(runtime,root):
    runtime.scene_path=root/'scene.json';common.save(runtime.scene_path,{'bin':None})
    runtime.scene=dict(objects=[dict(object_id=i) for i in IDS],target_object_id='primary')
    runtime.output_dir=root;runtime.rows=trace();runtime.bodies=dict.fromkeys(IDS)
    runtime.steps=len(runtime.rows);runtime.dt=.004
    runtime.max_penetration=runtime.max_illegal_object_force=0.
    runtime.collection_binding=dict(task=lift.TASK,collection_candidate_sha256='a'*64)
    runtime.commands_sha256='b'*64
    return runtime


class StreamTests(unittest.TestCase):
    def test_strict_original_launcher_keeps_cpu_and_forces_vulkan_gpu0(self):
        calls=[]
        common_stub=SimpleNamespace(__file__=str(entry.HERE/'author_pipeline.py'),os=os)
        app=SimpleNamespace(AppLauncher=lambda **kw:calls.append(kw) or SimpleNamespace(app='stub'))
        with patch.dict(sys.modules,{'isaaclab':SimpleNamespace(app=app),'isaaclab.app':app}):
            self.assertEqual(entry._launch_native_gpu0(common_stub).app,'stub')
        self.assertEqual(calls,[dict(headless=True,device='cpu',visualizer=None,kit_args=entry.GPU0_KIT_ARGS)])
        with patch.object(entry,'LAUNCHER_AST_SHA256','0'*64),self.assertRaisesRegex(ValueError,'AST changed'):
            entry._launch_native_gpu0(common_stub)

    def test_gzip_roundtrip_preserves_every_contact_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'contacts.gz';journal=entry.ContactJournal(path)
            rows=[dict(step=i+1,object_contacts=[dict(bodies=['body','ground'],normal_force_n=.001+i/17,
                separation_m=-1e-7,position_world_m=[.01,.02,.03])],robot_contacts=[]) for i in range(7)]
            for row in rows:journal.append(row)
            pointer=journal.pointer();journal.close()
            with gzip.open(path,'rt') as stream:actual=[json.loads(line) for line in stream]
            self.assertEqual(actual,rows);self.assertEqual(pointer['record_count'],7)
            self.assertEqual(pointer['data']['sha256'],entry.sha(path))
            self.assertEqual(pointer['data']['bytes'],path.stat().st_size)
            self.assertNotIn('rows',pointer);self.assertEqual(pointer['contact_records_omitted'],0)
            with self.assertRaises(FileExistsError):entry.ContactJournal(path)
            with self.assertRaises(ValueError):journal.append(dict(step=8))

    def test_malformed_or_out_of_order_row_rejected_before_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=entry.ContactJournal(Path(tmp)/'contacts.gz')
            for row in (dict(step=2),dict(step=1,x=float('nan'))):
                with self.assertRaises(ValueError):journal.append(row)
            self.assertEqual(journal.records,0);journal.close()

    def test_original_finish_results_unchanged_except_bound_contact_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'plain';b=root/'stream';a.mkdir();b.mkdir()
            original=fill(lift.MultiObjectLiftRuntime.__new__(lift.MultiObjectLiftRuntime),a)
            rows=[dict(step=i+1,object_contacts=[],robot_contacts=[]) for i in range(6)]
            original.contact_rows=rows
            gate=dict(complete=True,gates=dict.fromkeys(lift.ORIGINAL_GATES,True))
            plain=original.finish(gate)
            cls,ast_sha=entry.runtime_class(lift,{'entry':entry.sha(entry.__file__)},[])
            streamed=fill(cls.__new__(cls),b)
            streamed.contact_rows=entry.ContactJournal(b/'MULTI_OBJECT_CONTACTS.jsonl.gz')
            for row in rows:streamed.contact_rows.append(row)
            actual=streamed.finish(gate)
            # Paths, trace hashes (scene-free traces identical) and pointer hash
            # are artifact bindings; all physical and per-body fields match.
            for key in set(plain)|set(actual):
                if key in ('scene_path','contacts_sha256'):continue
                self.assertEqual(actual[key],plain[key],key)
            self.assertTrue(streamed.contact_rows.closed);self.assertEqual(len(ast_sha),64)
            pointer=common.read(b/'MULTI_OBJECT_CONTACTS.json')
            self.assertEqual(pointer['schema'],entry.POINTER_SCHEMA)
            self.assertEqual(actual['contacts_sha256'],entry.sha(b/'MULTI_OBJECT_CONTACTS.json'))
            with gzip.open(pointer['data']['path'],'rt') as stream:
                self.assertEqual([json.loads(line) for line in stream],rows)

    def test_error_during_evaluation_still_closes_complete_contact_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cls,_=entry.runtime_class(lift,{},[])
            runtime=fill(cls.__new__(cls),root)
            runtime.rows[0]['objects'].pop('third')
            runtime.contact_rows=entry.ContactJournal(root/'MULTI_OBJECT_CONTACTS.jsonl.gz')
            runtime.contact_rows.append(dict(step=1,object_contacts=[],robot_contacts=[]))
            with self.assertRaises(ValueError):runtime.finish({})
            self.assertTrue(runtime.contact_rows.closed)
            pointer=common.read(root/'MULTI_OBJECT_CONTACTS.json')
            self.assertEqual(pointer['record_count'],1)
            self.assertFalse((root/'LIFT_EVALUATION.json').exists())

    def test_finish_source_change_fails_closed(self):
        with patch.object(entry,'sha',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'source changed'):entry.runtime_class(lift,{},[])

    def test_wrapper_receipt_is_finalized_before_close_can_exit_process(self):
        script=r'''
import json,os,sys
from pathlib import Path
from types import SimpleNamespace
from reproduction.boya import author_collection_execute as e
from reproduction.boya import author_lift as lift
root=Path(sys.argv[1])
pipeline=SimpleNamespace(_backend=lambda:lift)
pipeline.common=SimpleNamespace(_launch_native=lambda:SimpleNamespace(app=SimpleNamespace(close=lambda **kw:os._exit(kw.get('exit_code',0)))))
def execute(path):
    launcher=pipeline.common._launch_native()
    e.save(root/'LIFT_PIPELINE_EXECUTION.json',{'status':'physical_failure','physics_steps':23,'label':0})
    launcher.app.close(exit_code=0)
pipeline.execute=execute
e._modules=lambda:(lift,pipeline)
e._launch_native_gpu0=lambda common:common._launch_native_original()
pipeline.common._launch_native_original=pipeline.common._launch_native
e.execute(root)
raise AssertionError('fast close unexpectedly returned')
'''
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'-B','-c',script,tmp],cwd=common.ROOT,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(common.ROOT)),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            receipt=common.read(Path(tmp)/'STREAMED_EXECUTION_ENTRY.json')
            self.assertEqual(receipt['status'],'finalized')
            self.assertEqual(receipt['pipeline_execution_receipt']['sha256'],entry.sha(Path(tmp)/'LIFT_PIPELINE_EXECUTION.json'))
            self.assertFalse(receipt['changed_sources'])


if __name__=='__main__':unittest.main()
