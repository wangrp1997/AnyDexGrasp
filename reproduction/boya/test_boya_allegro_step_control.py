"""Meaningful source-level and whole-sequence checks; no hardware or physics."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_boya_allegro_step_control as adapter

ROOT = Path(__file__).resolve().parents[2]


def original_close_targets(opened, closed):
    """Execute only published methods against a recording fake controller."""
    source = ROOT / 'ur_toolbox/ur_toolbox/robot/Allegro/Allegro.py'
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == 'Allegro')
    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)
               and n.name in ('set_pose', 'get_current_angle', 'close_gripper')]
    cls = ast.ClassDef(name='IsolatedAllegro', bases=[], keywords=[],
                       body=methods, decorator_list=[])
    sleeps = []
    env = dict(np=np, time=SimpleNamespace(sleep=sleeps.append))
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(source), 'exec'), env)
    hand = env['IsolatedAllegro']()
    targets = []
    pose = SimpleNamespace(position=np.asarray(opened).copy(), effort=np.zeros(16))

    def record(q, wait):
        targets.append(q.copy())
        pose.position = q.copy()

    hand.allegro_controller = SimpleNamespace(current_joint_pose=pose, hand_pose=record)
    hand.close_gripper(closed, tactile=object())
    return np.asarray(targets), sleeps, methods


class Closure:
    def __init__(self, opened, closed):
        self.opened, self.closed = opened, closed

    def command(self, thumb, fingers):
        if thumb != fingers:
            raise ValueError('This control uses synchronized progress')
        return (1. - thumb) * self.opened + thumb * self.closed


class AlignmentTest(unittest.TestCase):
    def fixture(self):
        opened = np.linspace(-.4, .9, 16)
        closed = np.linspace(.2, 1.3, 16)
        palm = np.eye(4)
        palm[:3, 3] = [.5, .1, .2]
        return dict(opened=opened, closed=closed, palm=palm,
                    direction=np.array([0., 0., -1.]),
                    closure=Closure(opened, closed))

    def test_matches_executed_original_ten_positions_and_waits(self):
        inputs = self.fixture()
        targets, sleeps, _ = original_close_targets(inputs['opened'], inputs['closed'])
        result = adapter.command_sequence(inputs)
        close = result['motor_target'][result['phase'] == 'close']
        self.assertEqual(targets.shape, (10, 16))
        np.testing.assert_allclose(close, np.repeat(targets, 25, axis=0), atol=1e-15, rtol=0)
        self.assertEqual(sleeps, [.1] * 10 + [.2])
        np.testing.assert_allclose(close[-1], inputs['closed'], atol=0, rtol=0)

    def test_only_close_motor_targets_change(self):
        inputs = self.fixture()
        baseline = adapter.BASE_COMMAND_SEQUENCE(inputs)
        control = adapter.command_sequence(inputs)
        for key in baseline:
            if key != 'motor_target':
                np.testing.assert_array_equal(control[key], baseline[key])
        outside = baseline['phase'] != 'close'
        np.testing.assert_array_equal(control['motor_target'][outside], baseline['motor_target'][outside])
        self.assertEqual(np.count_nonzero(np.any(control['motor_target'] != baseline['motor_target'], axis=1)), 240)
        np.testing.assert_array_equal(baseline['motor_target'], adapter.BASE_COMMAND_SEQUENCE(inputs)['motor_target'])

    def test_no_tactile_or_torque_use_in_original_close_body(self):
        _, _, methods = original_close_targets(np.zeros(16), np.ones(16))
        close = next(n for n in methods if n.name == 'close_gripper')
        loaded = {n.id for n in ast.walk(close) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        called_attrs = {n.func.attr for n in ast.walk(close) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertNotIn('tactile', loaded)
        self.assertNotIn('set_torque', called_attrs)
        self.assertNotIn('get_angle_contact', called_attrs)

    def test_targets_are_ten_levels_and_never_exceed_endpoints(self):
        inputs = self.fixture()
        result = adapter.command_sequence(inputs)
        close = result['motor_target'][result['phase'] == 'close']
        self.assertEqual(len(np.unique(close, axis=0)), 10)
        self.assertTrue(np.all(close >= np.minimum(inputs['opened'], inputs['closed']) - 1e-15))
        self.assertTrue(np.all(close <= np.maximum(inputs['opened'], inputs['closed']) + 1e-15))


if __name__ == '__main__':
    unittest.main()
