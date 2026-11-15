"""Lifecycle and original-body checks, including an actual non-Kit child."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from types import CodeType
import numpy as np

import author_collection_service as S
import author_collection_service_client as CLIENT
import author_collection_wave_resources as R
import author_stream_representation as C

B=S.B


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.addCleanup(self.tmp.cleanup)

    def test_capture_hook_changes_only_launcher_call(self):
        module,scope=S.capture_template();sites=0
        for n in ast.walk(module):
            if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='resident_launcher':
                sites+=1;n.func=n.args.pop(0)
        self.assertEqual(sites,1)
        expected=S.CAPTURE.capture_function().__code__
        exec(compile(ast.fix_missing_locations(module),expected.co_filename,'exec'),scope);actual=scope['capture'].__code__
        self.assertEqual(actual.co_code,expected.co_code);self.assertEqual(actual.co_names,expected.co_names)
        def constants(code):
            return tuple((v.co_code,v.co_names,constants(v))if isinstance(v,CodeType)else v for v in code.co_consts)
        self.assertEqual(constants(actual),constants(expected))

    def test_one_real_app_for_two_captures_and_two_executions(self):
        apps=[]
        class App:
            def __init__(self,**kw):apps.append(self);self.app=self;self.closed=0
            def close(self,**kw):self.closed+=1
        class Session:
            def __init__(self,path,**kw):
                Path(path).mkdir(parents=True);self.scene_rebuild_count=0;self.controller=None;self._geometry=None
            def execute(self,path):
                B.save(path/'LIFT_PIPELINE_EXECUTION.json',dict(status='complete'))
                return dict(status='complete',physics_steps=3)
            def close(self,**kw):self.launcher.app.close(**kw)
        module=ast.parse("def capture(scene,output):\n    app=resident_launcher(App).app\n    save(output/'RESULTS.json',dict(status='complete',physics_ticks=750))\n    app.close()")
        with patch.object(S.RESIDENT,'ResidentExecutionSession',Session),patch.object(S,'capture_template',return_value=(module,dict(App=App,save=B.save))):
            runtime=S.CollectionRuntime(self.root/'runtime',{})
            with patch.object(runtime,'clear_scene')as clear:
                for i in range(2):runtime.run_job(dict(role='capture',scene='unused',output_dir=str(self.root/f'c{i}')))
                for i in range(2):
                    path=self.root/f'e{i}';B.save(path/'plan.json',dict(total_command_steps=3))
                    B.save(path/'READY.json',dict(plan=B.binding(path/'plan.json')))
                    runtime.run_job(dict(role='execute',prepared_dir=str(path)))
                self.assertEqual(clear.call_count,3)
            self.assertEqual(len(apps),1);self.assertEqual(apps[0].closed,0)
            runtime.close();runtime.close();self.assertEqual(apps[0].closed,1)
            receipt=B.read(self.root/'runtime/WORKER_RECEIPT.json')
            self.assertEqual(receipt['Kit_starts'],1);self.assertEqual(len(receipt['jobs']),4)
            with self.assertRaises(RuntimeError):runtime.run_job(dict(role='capture'))

    def test_failed_job_poisoning_blocks_later_dispatch(self):
        runtime=S.CollectionRuntime(self.root/'runtime',{})
        with self.assertRaises(ValueError):runtime.run_job(dict(role='unknown'))
        self.assertTrue(runtime.poisoned)
        with self.assertRaises(RuntimeError):runtime.run_job(dict(role='capture'))
        runtime.close();self.assertEqual(B.read(self.root/'runtime/WORKER_RECEIPT.json')['status'],'closed_after_failure')

    def stream(self):
        manifest=self.root/'manifest.json';B.save(manifest,{})
        args=SimpleNamespace(manifest=manifest,run_dir=self.root/'repr',seed=7);args.run_dir.mkdir()
        rows=[]
        for i in range(2):
            path=self.root/f'obs{i}.json';B.save(path,{})
            rows.append(dict(object_id=str(i),observation_json=str(path),output_dir=str(self.root/f'frame{i}')))
        return C.RepresentationStream(args,rows)

    def test_original_frame_body_reuses_one_model_and_preserves_order(self):
        loads=[]
        class Engine:
            def __init__(self,a):
                loads.append(self);self.calls=0;self.checkpoint=self.native=self.overlay=self.parser_receipt={}
            def forward(self,points,coords,directory):
                self.calls+=1
                for name in ('raw_camera_candidates.npz','representation_outputs.npz'):np.savez(directory/name,values=points)
                return dict(timing_seconds={},raw_candidates=dict(count=1))
        def frame(row,a):
            return np.ones((1,3)),np.ones((1,3)),dict(observation_sha256=B.sha(row['observation_json'])),{},{},0.
        with patch.object(C.ORIGINAL,'PersistentOfficialCGR',Engine),patch.object(C.ORIGINAL,'read_frame',frame):
            stream=self.stream();stream.forward(0);stream.forward(1)
            self.assertEqual(len(loads),1);self.assertEqual(loads[0].calls,2)
            self.assertTrue(stream.reports[1]['persistent_model_reused'])
            self.assertEqual(stream.batch['network_forward_count'],2)
            with self.assertRaises(ValueError):stream.forward(1)

    def test_original_frame_failure_stops_without_retry(self):
        with patch.object(C.ORIGINAL,'PersistentOfficialCGR',return_value=SimpleNamespace(calls=0)),\
             patch.object(C.ORIGINAL,'read_frame',side_effect=ValueError('test failure')):
            stream=self.stream()
            with self.assertRaises(RuntimeError):stream.forward(0)
            with self.assertRaises(ValueError):stream.forward(1)
            self.assertEqual(len(stream.reports),1)

    def test_real_child_holds_slot_until_close_and_does_not_block_audit(self):
        child=self.root/'worker.py'
        child.write_text('''import argparse,json,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--protocol');p.add_argument('--lease-fd');a=p.parse_args()
r=Path(a.root);m=r/'mailbox';m.mkdir(parents=True)
(r/'SERVICE_READY.json').write_text('{}')
for i in range(5):
 f=m/f'{i:03d}.json'
 while not f.exists():time.sleep(.01)
 v=json.loads(f.read_text())
 if v['role'] in ('close','abort'):break
 (m/f'{i:03d}.RESULT.json').write_text(json.dumps(dict(status='complete')))
''')
        path=self.root/'PROTOCOL.json';B.save(path,dict(inputs=dict(python=B.binding(sys.executable)),
            sources=dict(service_entry=B.binding(child)),env={},inherited_env=dict(PYTHONPATH=''),worker_jobs=[{}]*4,
            service_wall_timeout_s=10,stage_timeout_s=5,cancel_sigint_grace_s=1,cancel_sigterm_grace_s=1,cancel_sigkill_grace_s=1))
        with R.lease(self.root/'lease.lock')as fd:
            pool=R.ResourcePool(self.root/'slots',fd,B.binding(path))
            client=CLIENT.ResidentClient(self.root/'run',path,pool,fd)
            try:
                self.assertEqual(B.read(self.root/'slots/kit/0.json')['state'],'running')
                audit=self.root/'audit';audit.mkdir()
                with pool.stage(('audit',),audit):
                    process=pool.popen([sys.executable,'-I','-S','-B','-c','pass'],start_new_session=True)
                    self.assertEqual(process.wait(timeout=5),0)
                for i in range(4):client.submit(i);self.assertEqual(client.wait(i)['status'],'complete')
                self.assertIsNone(client.process.poll())
            finally:client.close()
            self.assertEqual(B.read(self.root/'slots/kit/0.json')['state'],'released')
            self.assertFalse(R.group_alive(client.process.pid))

    def test_frozen_control_has_finite_diagnostic_budget(self):
        import author_collection_stream_control as D
        with patch.object(B.subprocess,'Popen',side_effect=AssertionError('Freeze must not dispatch')):
            p=D.freeze(self.root/'control')
        self.assertEqual(p['max_Kit_starts'],5);self.assertEqual(p['max_environment_steps'],35000)
        for group in p['groups']:
            cp=B.read(group['protocol']['path']);self.assertTrue(cp['exclude_training'])
            self.assertEqual(cp['env']['CUDA_VISIBLE_DEVICES'],D.REFERENCE.P.GPU_UUID)
            self.assertEqual([j['role']for j in cp['worker_jobs']],['capture','capture','execute','execute'])
        p['max_Kit_starts']=6;B.save(self.root/'control/CONTROL.json',p)
        with self.assertRaises(ValueError):D.verify(self.root/'control')


if __name__=='__main__':unittest.main()
