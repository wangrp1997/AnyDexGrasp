"""Storage replay orchestration and the empty-stage guard; no native App."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import author_contact_storage_probe as P
import author_contact_storage_control as C
import gzip


class ProbeTests(unittest.TestCase):
    def test_freeze_uses_window_runtime_gpu_binding_before_any_native_dispatch(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);trial=root/'trial';(trial/'execution').mkdir(parents=True)
            data=trial/'execution/data.gz'
            with gzip.open(data,'wb')as stream:
                for i in range(1,1025):stream.write((json.dumps(dict(step=i),separators=(',',':'))+'\n').encode())
            P.B.save(trial/'execution/MULTI_OBJECT_CONTACTS.json',dict(data=P.B.binding(data)))
            P.B.save(trial/'RUNTIME_AUDIT.json',dict(status='pass'))
            smi=root/'smi';smi.write_text('fixture')
            window=root/'window.json';P.B.save(window,dict(inputs={'nvidia_smi':P.B.binding(smi)},
                env=dict(CUDA_VISIBLE_DEVICES=C.P.GPU_UUID,NVIDIA_VISIBLE_DEVICES='0'),inherited_env={}))
            index=root/'index.json';P.B.save(index,dict(windows=[dict(protocol=P.B.binding(window))]))
            old=root/'old.json';P.B.save(old,dict(inputs={},sources={},window_index=P.B.binding(index),
                env=dict(CUDA_VISIBLE_DEVICES='wrong-parent-default')))
            with patch.object(C,'REFERENCE',old),patch.object(C,'GOOD_TRIAL',trial),\
                 patch.object(C.C,'gpu_identity',return_value={'uuid':C.P.GPU_UUID}),\
                 patch.object(P.B.subprocess,'Popen',side_effect=AssertionError('Freeze launched process')):
                protocol=C.freeze(root/'control')
            self.assertEqual(protocol['env']['CUDA_VISIBLE_DEVICES'],C.P.GPU_UUID)
            self.assertEqual(protocol['inputs']['nvidia_smi'],P.B.binding(smi))
            self.assertFalse((root/'control/RUN_STARTED.json').exists())

    def fixture(self,root):
        raw=root/'reference.jsonl'
        content=b''.join((json.dumps(dict(step=i,points=[{'force':0.,'pair':['a','floor']}]*3),separators=(',',':'))+'\n').encode()for i in range(1,6))
        raw.write_bytes(content)
        p=dict(inputs={},sources={},writers=4,source_sha256={},spool_root=str(root/'spool'),
            fixture=dict(raw=P.B.binding(raw),logical=dict(sha256=hashlib.sha256(content).hexdigest(),bytes=len(content),records=5)))
        path=root/'PROTOCOL.json';P.B.save(path,p)
        return path

    def test_two_algorithms_four_writers_preserve_every_original_byte_without_Kit(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);path=self.fixture(root)
            with patch.object(P.OLD,'_launch_native_gpu0',side_effect=AssertionError('CPU replay launched Kit')):
                result=P.run(root/'result',path,'standalone')
            self.assertEqual(result['status'],'verified_storage');self.assertEqual(result['Kit_starts'],0)
            self.assertEqual(result['physics_steps'],0)
            self.assertTrue(all(g['all_verified']and len(g['workers'])==4 for g in result['groups']))
            self.assertEqual(sum(w['synthetic_replay_records']for g in result['groups']for w in g['workers']),40)

    def test_existing_physics_context_rejects_probe_before_any_replay(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);path=self.fixture(root);closed=[]
            fake=SimpleNamespace(app=SimpleNamespace(close=lambda **kw:closed.append(kw)))
            with patch.object(P.OLD,'_modules',return_value=(None,SimpleNamespace(common=None))),\
                 patch.object(P.OLD,'_launch_native_gpu0',return_value=fake),\
                 patch.object(P,'native_state',return_value=dict(timeline_playing=False,physics_scenes=['/Physics'],simulation_context_exists=True)),\
                 patch.object(P,'one',side_effect=AssertionError('Replayed into an active physics context')):
                result=P.run(root/'result',path,'kit')
            self.assertEqual(result['status'],'failed');self.assertEqual(result['groups'],[])
            self.assertEqual(closed,[dict(exit_code=1)])


if __name__=='__main__':unittest.main()
