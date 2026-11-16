"""Publication-only overlay and finite profile admission, without Kit."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
import author_cpu_fabric as F
import author_cpu_fabric_control as C


class CpuFabricTests(unittest.TestCase):
    def factory(self):
        tree=ast.parse(F.RUNTIME.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='create_scene')
        scope=dict(Path=Path,np=np,install_first_reset_fix=Mock(),build=Mock(return_value=Path('fixture.urdf')),
            sim_utils=SimpleNamespace(SimulationContext=Mock(),SimulationCfg=Mock(side_effect=RuntimeError('configuration captured'))))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'native-test','exec'),scope)
        return scope['create_scene'],scope

    def test_only_fabric_flag_changes_all_scene_physics_arguments_preserved(self):
        original,scope=self.factory();overlay=F.scene_factory(original)
        mods={'canondex.robots.arm_actuation':SimpleNamespace(ARM_ACTUATION_MODES=['isaaclab']),
              'canondex.grasp.physics':SimpleNamespace(GraspPhysicsConfig=lambda:None)}
        cfg=object();profile=SimpleNamespace(scene_cfg=lambda:cfg)
        with patch.dict('sys.modules',mods):
            for fn in (original,overlay):
                with self.assertRaisesRegex(RuntimeError,'configuration captured'):
                    fn(Path('/tmp/fixture'),object(),'cpu',physics_profile=profile,timestep=.004)
        calls=scope['sim_utils'].SimulationCfg.call_args_list
        before,after=(dict(c.kwargs)for c in calls)
        self.assertFalse(before.pop('use_fabric'));self.assertTrue(after.pop('use_fabric'))
        self.assertEqual(before,after);self.assertIs(after['physics'],cfg)
        self.assertEqual(after['dt'],.004);self.assertEqual(after['device'],'cpu')
        scope['install_first_reset_fix'].assert_called_once_with()

    def test_overlay_is_scoped_and_restored_on_failure(self):
        original,_=self.factory();module=SimpleNamespace(create_scene=original)
        with self.assertRaisesRegex(RuntimeError,'failure'):
            with F.installed(module):
                self.assertIsNot(module.create_scene,original);raise RuntimeError('failure')
        self.assertIs(module.create_scene,original)

    def test_changed_prefix_budget_rejected_before_source_or_physics_access(self):
        fixture=dict(schema=C.SCHEMA,max_Kit_starts=2,ticks_per_job=65,warmup_ticks=16,
            max_environment_steps=128,jobs=[],retry_budget=0,exclude_training=True)
        with patch.object(C.B,'read',return_value=fixture),patch.object(C.B,'verify')as verify:
            with self.assertRaises(ValueError):C.verify(Path('/tmp/unused'))
        verify.assert_not_called()


if __name__=='__main__':unittest.main()
