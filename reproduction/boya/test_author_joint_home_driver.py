"""Exercise dispatch/provenance using fake children, never real route search."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import author_joint_home_driver as driver


class HomeDriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.seed = self.root/'seed'; self.seed.mkdir()
        self.pool = self.root/'pool'; self.pool.mkdir()
        self.out = self.root/'out'
        self.scene = self.root/'scene.json'; self.scene.write_text('{}')
        self.config = self.root/'config.json'; self.config.write_text('{}')
        for name in ('commands.npz', 'path_geometry.json', 'lift_arm_geometry.json'):
            (self.seed/name).write_text(name)
        (self.pool/'candidates.npz').write_bytes(b'fixed candidate')
        driver.transport.save(self.pool/'generation.json', dict(input_sha256={}))
        driver.transport.save(self.seed/'plan.json', dict(arguments=dict(candidates=str(self.pool)),
            source_sha256={}, commands_sha256=driver.transport.sha(self.seed/'commands.npz'),
            path_geometry_sha256=driver.transport.sha(self.seed/'path_geometry.json'),
            lift_arm_geometry_sha256=driver.transport.sha(self.seed/'lift_arm_geometry.json'),
            generation_sha256=driver.transport.sha(self.pool/'generation.json'),
            candidates_sha256=driver.transport.sha(self.pool/'candidates.npz')))
        self.calls = []
        self.validator = patch.object(driver.transport, 'validate_scene', return_value={'input_sha256':{}})
        self.validator.start(); self.addCleanup(self.validator.stop)

    def child(self, directory, status, *, reason=None, geometry_ok=True, derivative=None,
              connected=True, wall=3.):
        directory.mkdir()
        protocol = directory/'HOME_ROUTE_PROTOCOL.json'
        driver.transport.save(protocol, dict(input_sha256={}, source_sha256={}))
        receipt = dict(status=status, complete=status!='infrastructure_failure', admitted=status=='ready',
            plan_dir=str(directory/'native'), protocol_sha256=driver.transport.sha(protocol), wall_s=wall)
        if status != 'infrastructure_failure':
            receipt.update(search=dict(connected=connected, iterations_executed=28), search_wall_s=.6)
        if reason is not None: receipt['reason'] = reason
        if reason == driver.RETIME_REASON or status == 'ready':
            geometry = directory/'full_scene_geometry.json'
            driver.transport.save(geometry, dict(ok=geometry_ok, violations=[] if geometry_ok else [{'distance_m':-.001}]))
            (directory/'connected_path.npz').write_bytes(b'identical frozen path')
            (directory/'candidate_commands.npz').write_bytes(b'fixed candidate commands')
            receipt.update(geometry_sha256=driver.transport.sha(geometry),
                connected_path_sha256=driver.transport.sha(directory/'connected_path.npz'),
                candidate_commands_sha256=driver.transport.sha(directory/'candidate_commands.npz'),
                derivative_audit=derivative or dict(ok=False, joint_limit_excess_rad=0.,
                    velocity_ratio=.2, acceleration_ratio=.8, jerk_ratio=1.07))
        if status == 'ready':
            native = directory/'native'; native.mkdir()
            (native/'commands.npz').write_bytes(b'admitted commands')
            shutil.copyfile(directory/'full_scene_geometry.json', native/'path_geometry.json')
            plan = dict(admitted=True, complete=True,
                author_home_route=dict(original_seed_plan_sha256=driver.transport.sha(self.seed/'plan.json')),
                commands_sha256=driver.transport.sha(native/'commands.npz'),
                path_geometry_sha256=driver.transport.sha(native/'path_geometry.json'), source_sha256={})
            driver.transport.save(native/'plan.json', plan)
            receipt['plan_sha256'] = driver.transport.sha(native/'plan.json')
        driver.transport.save(directory/'HOME_ROUTE_PREPARATION.json', receipt)
        return receipt

    def rrt(self, status='null_geometry', **kwargs):
        def callback(seed, scene, output, *, config_path, archive_sources):
            self.calls.append(('rrt', Path(output)))
            protocol = driver.transport.read(self.out/'HOME_ROUTE_PROTOCOL.json')
            self.assertEqual(protocol['policy']['max_rrt_preparations'], 1)
            self.assertEqual(protocol['policy']['rrt']['max_iterations'], 1200)
            self.assertEqual(protocol['policy']['rrt']['random_seed'], 20261114)
            self.assertEqual(protocol['policy']['retime']['max_iterations_including_original'], 8)
            self.assertEqual(protocol['policy']['ik_calls'], 0)
            self.assertEqual(protocol['input_sha256'][str(self.seed/'plan.json')], driver.transport.sha(self.seed/'plan.json'))
            self.assertEqual(protocol['source_sha256'][str(Path(driver.joint.__file__).resolve())],
                             driver.CORE_PINS[str(Path(driver.joint.__file__).resolve())])
            return self.child(Path(output), status, **kwargs)
        return callback

    def retiming(self, status='ready', **kwargs):
        def callback(prepared, seed, scene, output, *, config_path, archive_sources):
            self.calls.append(('retime', Path(output)))
            self.assertEqual(Path(prepared), self.out/'rrt')
            self.assertEqual(Path(seed), self.seed)
            self.assertEqual(Path(scene), self.scene)
            return self.child(Path(output), status, **kwargs)
        return callback

    def run_driver(self, first, second=None):
        with patch.object(driver.joint, 'prepare_home_route', side_effect=first) as search:
            with patch.object(driver.retime, 'retime_home_route', side_effect=second) as retime:
                result = driver.prepare_home_route(self.seed, self.scene, self.out, config_path=self.config)
                return result, search.call_count, retime.call_count

    def test_ready_has_one_search_no_retime_and_protocol_precedes_call(self):
        result, searches, retimes = self.run_driver(self.rrt('ready'))
        self.assertEqual((result['status'], searches, retimes), ('ready', 1, 0))
        self.assertEqual(result['plan_dir'], str(self.out/'rrt'/'native'))
        self.assertEqual(result['cost']['actual_search_runs'], 1)
        self.assertEqual(result['cost']['reported_child_wall_s'], 3.)

    def test_timing_only_failure_retimes_exactly_once_on_identical_path(self):
        result, searches, retimes = self.run_driver(self.rrt(reason=driver.RETIME_REASON), self.retiming(wall=5.))
        self.assertEqual((result['status'], searches, retimes), ('ready', 1, 1))
        self.assertEqual(result['plan_dir'], str(self.out/'retime'/'native'))
        self.assertEqual(result['cost']['reported_child_wall_s'], 8.)
        self.assertEqual(result['cost']['actual_search_runs'], 1)
        self.assertEqual(driver.transport.sha(self.out/'rrt'/'connected_path.npz'),
                         driver.transport.sha(self.out/'retime'/'connected_path.npz'))

    def test_collision_unknown_null_budget_and_infrastructure_never_retry(self):
        fixtures = [dict(reason=driver.RETIME_REASON, geometry_ok=False),
            dict(reason='frozen_endpoint_or_non_home_collision'),
            dict(reason='frozen_rrt_budget_exhausted', connected=False),
            dict(reason='original_time_scaling_rejected'),
            dict(reason='future_unknown_reason'), dict(status='infrastructure_failure')]
        for index, fixture in enumerate(fixtures):
            with self.subTest(fixture=fixture):
                self.out = self.root/('case'+str(index))
                fixture = dict(fixture); status = fixture.pop('status', 'null_geometry')
                result, searches, retimes = self.run_driver(self.rrt(status, **fixture))
                self.assertEqual((result['status'], searches, retimes), (status, 1, 0))
                self.assertIsNone(result['plan_dir'])

    def test_position_limit_or_passing_derivative_does_not_enter_retime(self):
        audits = [dict(ok=False, joint_limit_excess_rad=.001, velocity_ratio=.2, acceleration_ratio=.8, jerk_ratio=1.07),
                  dict(ok=True, joint_limit_excess_rad=0., velocity_ratio=.2, acceleration_ratio=.8, jerk_ratio=.9)]
        for index, audit in enumerate(audits):
            with self.subTest(audit=audit):
                self.out = self.root/('limit'+str(index))
                result, searches, retimes = self.run_driver(self.rrt(reason=driver.RETIME_REASON, derivative=audit))
                self.assertEqual((result['status'], searches, retimes), ('null_geometry', 1, 0))

    def test_tampered_geometry_is_infrastructure_failure_without_retime(self):
        first = self.rrt(reason=driver.RETIME_REASON)
        def tamper(*args, **kwargs):
            result = first(*args, **kwargs)
            driver.transport.save(self.out/'rrt'/'full_scene_geometry.json', dict(ok=True, violations=[], injected=True))
            return result
        result, searches, retimes = self.run_driver(tamper)
        self.assertEqual((result['status'], searches, retimes), ('infrastructure_failure', 1, 0))

    def test_retime_failure_is_final_without_another_search(self):
        for index, status in enumerate(('null_geometry', 'infrastructure_failure')):
            with self.subTest(status=status):
                self.out = self.root/('retimefailure'+str(index))
                result, searches, retimes = self.run_driver(self.rrt(reason=driver.RETIME_REASON),
                    self.retiming(status, reason='retimed_full_scene_tick_collision'))
                self.assertEqual((result['status'], searches, retimes), (status, 1, 1))
                self.assertIsNone(result['plan_dir'])

    def test_exception_records_one_attempt_and_unknown_search_cost_without_retry(self):
        def broken(*args, **kwargs): raise OSError('native preparation interrupted')
        result, searches, retimes = self.run_driver(broken)
        self.assertEqual((result['status'], searches, retimes), ('infrastructure_failure', 1, 0))
        self.assertEqual(result['cost']['rrt_preparations'], 1)
        self.assertIsNone(result['cost']['actual_search_runs'])
        self.assertGreaterEqual(result['cost']['measured_child_calls_wall_s'], 0.)

    def test_changed_seed_fails_before_any_child(self):
        (self.seed/'commands.npz').write_bytes(b'changed')
        result, searches, retimes = self.run_driver(self.rrt('ready'))
        self.assertEqual((result['status'], searches, retimes), ('infrastructure_failure', 0, 0))

    def test_path_change_after_retime_is_infrastructure_failure_not_ready(self):
        second = self.retiming()
        def changed(*args, **kwargs):
            result = second(*args, **kwargs)
            path = self.out/'retime'/'connected_path.npz'
            path.write_bytes(b'a different route')
            result['connected_path_sha256'] = driver.transport.sha(path)
            driver.transport.save(self.out/'retime'/'HOME_ROUTE_PREPARATION.json', result)
            return result
        result, searches, retimes = self.run_driver(self.rrt(reason=driver.RETIME_REASON), changed)
        self.assertEqual((result['status'], searches, retimes), ('infrastructure_failure', 1, 1))
        self.assertIsNone(result['plan_dir'])

    def test_input_change_during_child_cannot_be_admitted(self):
        first = self.rrt('ready')
        def changed(*args, **kwargs):
            result = first(*args, **kwargs)
            self.config.write_text('{"changed":true}')
            return result
        result, searches, retimes = self.run_driver(changed)
        self.assertEqual((result['status'], searches, retimes), ('infrastructure_failure', 1, 0))


if __name__ == '__main__': unittest.main()
