"""Fabric intervention safety and persistence checks; zero real simulation."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import author_fabric_execution as E
import author_fabric_audit as A


class Settings:
    def __init__(self):
        self.values={E.KEYS[0]:False,E.KEYS[1]:True,E.OTHER_KEYS[0]:False,
            E.OTHER_KEYS[1]:True,E.OTHER_KEYS[2]:False}
        self.writes=[]
    def get(self,key):return self.values.get(key)
    def set_bool(self,key,value):self.writes.append((key,value));self.values[key]=value


class Bindings:
    def tree(self,value):pass


def sidecar():
    before=Settings().values
    disabled=dict(before,**{k:False for k in E.KEYS})
    return dict(schema=E.SCHEMA,status='partial_fabric',mode='both_disabled',label=None,raw_label=None,
        formal_training_eligible=False,exclude_training=True,full_lift_certified=False,prefix_ticks=100,
        scaling_receipt=dict(path='/tmp/frozen/SCALING_EXECUTION.json',sha256='fixture'),
        after_app_before_scene=dict(before=before,after=disabled),only_changed_settings=list(E.KEYS),
        before_prefix=dict(physics_clock=0,cameras=[],visualizer_cfg_count=0,use_fabric=True,
            physics_device='cuda:0',settings=disabled),
        after_prefix=dict(physics_clock=100,settings=disabled),
        settings_after_ticks=[dict(tick=i,values=copy.deepcopy(disabled)) for i in range(1,101)],
        convex_compatibility=dict(status='queried',rows=[dict(path='/World/envs/env_0/mesh',gpu_compatible=True)],
            checked_convex_prims=1,gpu_compatible=1,gpu_incompatible=0,
            physics_clock_before=100,physics_clock_after=100,physics_steps_added=0,
            root_layer_sha256_before='same',root_layer_sha256_after='same',
            measured_prefix_includes_this_query=False))


class FabricTests(unittest.TestCase):
    def test_baseline_preserves_loaded_mixed_defaults_without_any_write(self):
        settings=Settings();before=copy.deepcopy(settings.values)
        result=E.configure(settings,'baseline')
        self.assertEqual(settings.values,before);self.assertEqual(settings.writes,[])
        self.assertEqual(result['before'],result['after'])

    def test_disabled_writes_both_flags_and_preserves_contacts_interop_readback(self):
        settings=Settings();other={k:settings.get(k) for k in E.OTHER_KEYS}
        E.configure(settings,'both_disabled')
        self.assertEqual(settings.writes,[(k,False) for k in E.KEYS])
        self.assertEqual({k:settings.get(k) for k in E.OTHER_KEYS},other)
        E.check_settings(settings.values,'both_disabled')

    def test_tick_guard_rejects_half_switch_contacts_off_and_baseline_drift(self):
        settings=Settings()
        with self.assertRaisesRegex(ValueError,'reenabled'):E.check_settings(settings.values,'both_disabled')
        E.configure(settings,'both_disabled');settings.values[E.OTHER_KEYS[2]]=True
        with self.assertRaisesRegex(ValueError,'contacts'):E.check_settings(settings.values,'both_disabled')
        settings=Settings();baseline=copy.deepcopy(settings.values);settings.values[E.KEYS[0]]=True
        with self.assertRaisesRegex(ValueError,'drifted'):E.check_settings(settings.values,'baseline',baseline)

    def test_output_selection_refuses_unfrozen_or_multiple_lanes(self):
        worker=dict(output_dir='/tmp/pair/baseline/worker_00/result',lanes=1)
        p=dict(groups=[dict(name='baseline',mode='gpu',count=1,workers=[worker])])
        self.assertEqual(E.arm_for_output(worker['output_dir'],p)[0],'baseline')
        with self.assertRaises(ValueError):E.arm_for_output('/tmp/unfrozen',p)
        worker['lanes']=4
        with self.assertRaises(ValueError):E.arm_for_output(worker['output_dir'],p)

    def test_independent_audit_rejects_single_bad_tick_labels_and_stage_changes(self):
        good=sidecar();path=good['scaling_receipt']['path']
        A.check_sidecar(good,'both_disabled',path,Bindings())
        mutations=[lambda v:v['settings_after_ticks'][71]['values'].__setitem__(E.KEYS[1],True),
            lambda v:v.__setitem__('label',0),
            lambda v:v['convex_compatibility'].__setitem__('physics_clock_after',101),
            lambda v:v['convex_compatibility'].__setitem__('root_layer_sha256_after','changed'),
            lambda v:v['before_prefix'].__setitem__('cameras',['/World/task_camera']),
            lambda v:v['settings_after_ticks'].pop()]
        for change in mutations:
            value=copy.deepcopy(good);change(value)
            with self.subTest(change=change),self.assertRaises(ValueError):
                A.check_sidecar(value,'both_disabled',path,Bindings())

    def test_comparison_reports_actual_difference_without_relaxing_equality(self):
        first={'q':np.asarray([[1.,2.]],dtype=np.float32),'phase':np.asarray(['settle'])}
        second={k:v.copy() for k,v in first.items()};second['q'][0,1]+=.125
        report=A.compare_arrays(first,second)
        self.assertFalse(report['all_arrays_exact'])
        self.assertEqual(report['fields']['q']['max_abs_difference'],.125)
        with self.assertRaises(ValueError):A.compare_arrays(first,{'other':np.zeros(1)})

    def test_source_pin_stops_before_any_launch(self):
        with patch.object(E.P,'sha',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'runner changed'):
                E.run('/unused','/unused',protocol_path='/unused')

    def test_sidecar_is_saved_before_simulated_fast_app_exit(self):
        script=r'''
import json,os,sys,types
from pathlib import Path
from unittest.mock import patch
import author_fabric_execution as E
out=Path(sys.argv[1]);out.mkdir()
prepared=out.parent/'prepared';prepared.mkdir()
protocol=out.parent/'protocol.json'
protocol.write_text(json.dumps(dict(schema='anydex_boya_author_fabric_control_v1',prefix_ticks=100,
 retry_budget=0,sources={},groups=[dict(name='baseline',mode='gpu',count=1,
 workers=[dict(output_dir=str(out),prepared_dir=str(prepared),lanes=1)])])))
storage=types.SimpleNamespace(save=lambda p,v:Path(p).write_text(json.dumps(v)))
def fake_run(*a,**kw):
 storage.save(out/'SCALING_EXECUTION.json',dict(status='partial_scaling',source_sha256={},
 physical_clock_steps=100,physical_steps_unknown=False))
 os._exit(0)
with patch.object(E.P,'sha',side_effect=lambda p:E.SCALING_SHA if Path(p)==Path(E.S.__file__) else 'fixture'),\
 patch.object(E.V,'verify_sources'),patch.object(E.V,'_modules',return_value=[storage]),\
 patch.object(E.V,'source_bindings',return_value={}),patch.object(E.S,'run',side_effect=fake_run):
 E.run(prepared,out,protocol_path=protocol)
'''
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'result'
            done=subprocess.run([sys.executable,'-B','-c',script,str(output)],
                env=dict(os.environ,PYTHONPATH=str(Path(E.__file__).parent)),capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            value=json.loads((output/'FABRIC_EXECUTION.json').read_text())
            self.assertEqual(value['status'],'partial_fabric')
            self.assertEqual(value['physical_cost']['physical_clock_steps'],100)
            self.assertIsNone(value['label'])


if __name__=='__main__':unittest.main()
