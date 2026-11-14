"""Pipeline state/binding tests: no Kit, inference, collection or optimization."""
import json
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from reproduction.boya import author_pipeline as pipeline

T = pipeline.transport


def candidate(name='first'):
    value = dict(schema=pipeline.collector.CANDIDATE_SCHEMA, candidate_id=name, slot_id='slot',
        actual_row15_camera=[0.]*15, world_T_camera=np.eye(4).tolist(), extra_depth_m=0.,
        task='declared_boya_clutter_transport_release_v1')
    value['candidate_sha256'] = pipeline.collector._candidate_id(value)
    return value


def fixture(root):
    source = root/'asset'; (source/'mesh').mkdir(parents=True)
    mesh = source/'mesh/simplified.obj'; mesh.write_text('source binding')
    physics = root/'physics.json'; T.save(physics, {})
    obj = dict(object_id='primary', object_root=str(source), scale=1., mass_kg=.1, friction=.6,
               world_T_object=np.eye(4).tolist(), source_mesh=str(mesh), source_mesh_sha256=T.sha(mesh))
    scene_path = root/'scene.json'
    T.save(scene_path, dict(schema='anydex_boya_author_scene_v1', scene_id='development',
        target_object_id='primary', objects=[obj, dict(obj, object_id='second')], physics_config=str(physics),
        bin=dict(world_T_bin=np.eye(4).tolist(), inner_half_extents_m=[.15, .15, .1],
                 floor_thickness_m=.01, wall_thickness_m=.01, friction=.6)))
    obs = root/'observation.json'
    T.save(obs, dict(scene_json=str(scene_path), scene_sha256=T.sha(scene_path), object_pose_as_network_input=False))
    rep = root/'representation'; rep.mkdir()
    for name in ('raw_camera_candidates.npz', 'representation_outputs.npz', 'RESULTS.json'):
        (rep/name).write_text('cache binding')
    cal, cloud, backend = [root/name for name in ('calibration.json', 'cloud.json', 'backend.json')]
    T.save(cal, {}); T.save(cloud, {}); T.save(backend, dict(physics_config=str(physics)))
    return (scene_path, rep, obs, 6, 2, 20261114, cal, cloud, backend)


def ready_fixture(root):
    args = fixture(root)
    scene = args[0]
    chosen = candidate()
    raw = root/'raw'; raw.mkdir()
    names = ['joint_'+str(i) for i in range(24)]
    arrays = dict(joint_names=np.array(names), joint_target=np.zeros((3, 24)),
                  initial_position=np.zeros(24), phase=np.array(['approach', 'close', 'release']),
                  desired_palm=np.repeat(np.eye(4)[None], 3, axis=0), time_s=np.array([.004, .008, .012]))
    np.savez_compressed(raw/'commands.npz', **arrays)
    T.save(raw/'plan.json', dict(arguments={}, author_collection_binding=dict(
        collection_candidate_sha256=chosen['candidate_sha256'], task=chosen['task'])))
    contract = SimpleNamespace(validate_command_arrays=lambda *a, **kw: dict(validation='stub; no physical claim'))
    with patch.object(pipeline, 'ActionContract', return_value=contract):
        binding = pipeline._bind_transport_plan(raw, root/'transport_bound', chosen, args[6], scene)
    T.save(root/'candidate.json', chosen)
    protocol = dict(inputs={'scene': pipeline._binding(scene)}, sources={})
    T.save(root/'PIPELINE_PROTOCOL.json', protocol)
    ready = dict(status='ready', protocol=pipeline._binding(root/'PIPELINE_PROTOCOL.json'),
        scene=pipeline._binding(scene), candidate=pipeline._binding(root/'candidate.json'),
        plan=pipeline._binding(root/'transport_bound/plan.json'),
        commands=pipeline._binding(root/'transport_bound/commands.npz'),
        command_binding=pipeline._binding(root/'transport_bound/COMMAND_BINDING.json'),
        collection_candidate_sha256=chosen['candidate_sha256'])
    T.save(root/'READY.json', ready)
    T.save(root/'PIPELINE_RESULTS.json', dict(status='ready', ready_sha256=T.sha(root/'READY.json')))
    return chosen, binding, arrays


class PipelineTests(unittest.TestCase):
    def test_no_candidate_cannot_start_arm_or_try_other_choices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = fixture(root)
            with patch.object(pipeline.collector, 'run_protocol', return_value=dict(candidates=[], counts={'top10': 0})), \
                    patch.object(pipeline.bridge, 'bridge_candidate') as bridge, \
                    patch.object(pipeline, 'prepare_approach') as arm:
                result = pipeline.prepare(*args, root/'out')
            self.assertEqual(result['status'], 'no_candidate'); bridge.assert_not_called(); arm.assert_not_called()
            self.assertEqual(result['new_labels'], 0)
            self.assertTrue((root/'out/PIPELINE_PROTOCOL.json').exists())
            self.assertEqual(T.read(root/'out/BACKEND.json')['pregrasp_distance_m'], .04)
            protocol = T.read(root/'out/PIPELINE_PROTOCOL.json')
            self.assertEqual(protocol['collection_motion_policy']['approach_distance_m'], .04)
            self.assertEqual(protocol['inputs']['collection_motion_source']['sha256'], pipeline.COLLECTION_SOURCE_SHA256)
            with self.assertRaises(FileExistsError): pipeline.prepare(*args, root/'out')

    def test_fixed_first_candidate_and_arm_null_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = fixture(root)
            choices = [candidate('first'), candidate('second')]
            with patch.object(pipeline.collector, 'run_protocol', return_value=dict(candidates=choices, counts={'top10': 2})), \
                    patch.object(pipeline.bridge, 'bridge_candidate') as bridge, \
                    patch.object(pipeline, 'prepare_approach', return_value=dict(status='no_plan')) as arm, \
                    patch.object(pipeline.transport, 'prepare_transport') as transport:
                result = pipeline.prepare(*args, root/'out')
            self.assertEqual(result['status'], 'no_plan')
            self.assertEqual(bridge.call_args.args[0], choices[0]); self.assertEqual(arm.call_count, 1)
            transport.assert_not_called(); self.assertFalse((root/'out/READY.json').exists())

    def test_transport_null_and_preparation_error_are_durable(self):
        for failure in (False, True):
            with self.subTest(infrastructure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); args = fixture(root)
                with patch.object(pipeline.collector, 'run_protocol', return_value=dict(candidates=[candidate()], counts={})), \
                        patch.object(pipeline.bridge, 'bridge_candidate'), \
                        patch.object(pipeline, 'prepare_approach', return_value=dict(status='ready', plan_dir='unused')), \
                        patch.object(pipeline.transport, 'prepare_transport',
                            side_effect=RuntimeError('native infrastructure') if failure else None,
                            return_value=dict(admitted=False, status='null_geometry')):
                    if failure:
                        with self.assertRaises(RuntimeError): pipeline.prepare(*args, root/'out')
                    else:
                        self.assertEqual(pipeline.prepare(*args, root/'out')['status'], 'null_geometry')
                receipt = T.read(root/'out/PIPELINE_RESULTS.json')
                self.assertEqual(receipt['status'], 'infrastructure_failure' if failure else 'null_geometry')
                self.assertEqual(receipt['physics_steps'], 0)

    def test_binding_adds_metadata_without_changing_native_24_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); chosen, binding, arrays = ready_fixture(root)
            with np.load(binding['commands_path'], allow_pickle=False) as bound:
                self.assertEqual(bound['joint_target'].shape, (3, 24))
                for name, value in arrays.items(): np.testing.assert_array_equal(bound[name], value)
                self.assertEqual(str(bound['collection_candidate_sha256']), chosen['candidate_sha256'])

    def test_execution_refuses_tampering_before_launch_and_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); ready_fixture(root)
            with (root/'transport_bound/commands.npz').open('ab') as stream: stream.write(b'changed')
            with patch.object(pipeline, '_execute_native') as launch:
                with self.assertRaises(ValueError): pipeline.execute(root)
            launch.assert_not_called(); self.assertFalse((root/'EXECUTION_STARTED.json').exists())

    def test_failed_native_start_is_not_implicitly_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); ready_fixture(root)
            with patch.object(pipeline.transport, 'args_for_plan', return_value=SimpleNamespace()), \
                    patch.object(pipeline, '_launch_native', return_value=SimpleNamespace(app=SimpleNamespace(close=lambda **kw: None))), \
                    patch.object(pipeline, '_execute_native', side_effect=RuntimeError('native startup')) as launch:
                with self.assertRaises(RuntimeError): pipeline.execute(root)
                with self.assertRaises(FileExistsError): pipeline.execute(root)
            self.assertEqual(launch.call_count, 1)
            receipt = T.read(root/'PIPELINE_EXECUTION.json')
            self.assertEqual(receipt['status'], 'infrastructure_failure')
            self.assertEqual(receipt['physics_steps'], 0)

    def test_full_failure_can_be_negative_but_wrong_task_cannot(self):
        for correct_task in (True, False):
            with self.subTest(correct_task=correct_task), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); chosen, binding, _ = ready_fixture(root)
                def native(*unused):
                    result = dict(status='physical_failure', complete=True, label_eligible=True, total_physics_steps=20,
                        completed_phases=['approach', 'close', 'retreat', 'transport', 'release', 'post_release'],
                        task=chosen['task'] if correct_task else 'old_vertical_lift', task_success=False,
                        collection_candidate_sha256=chosen['candidate_sha256'], commands_sha256=binding['commands_sha256'])
                    T.save(root/'execution/TRANSPORT_EVALUATION.json', result)
                    return result
                with patch.object(pipeline.transport, 'args_for_plan', return_value=SimpleNamespace()), \
                        patch.object(pipeline, '_launch_native', return_value=SimpleNamespace(app=SimpleNamespace(close=lambda **kw: None))), \
                        patch.object(pipeline, '_execute_native', side_effect=native):
                    result = pipeline.execute(root)
                self.assertEqual(result['label'], 0 if correct_task else None)
                self.assertEqual(result['new_labels'], int(correct_task))

    def test_final_receipts_precede_process_terminating_simulator_close(self):
        for failing in (False, True):
            with self.subTest(run_failure=failing), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); ready_fixture(root)
                script = '''
import os, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from reproduction.boya import author_pipeline as p
root=Path(sys.argv[1]); failing=sys.argv[2]=='True'
def native(*args):
    if failing: raise RuntimeError('simulated native failure before close')
    return dict(status='physical_failure', complete=True, total_physics_steps=23)
record=dict(label=0,label_state='physical_negative')
launcher=SimpleNamespace(app=SimpleNamespace(close=lambda **kw: os._exit(kw['exit_code'])))
with patch.object(p,'_launch_native',return_value=launcher), patch.object(p,'_execute_native',side_effect=native), patch.object(p.transport,'args_for_plan',return_value=SimpleNamespace()), patch.object(p.collector,'collect_execution',return_value=record):
    p.execute(root)
'''
                result = subprocess.run([sys.executable, '-B', '-c', script, str(root), str(failing)],
                    cwd=T.ROOT, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, int(failing), result.stderr)
                receipt = T.read(root/'PIPELINE_EXECUTION.json')
                self.assertEqual(receipt['status'], 'infrastructure_failure' if failing else 'physical_failure')
                self.assertEqual((root/'COLLECTION_RECORD.json').exists(), not failing)
                if not failing:
                    self.assertEqual(receipt['label'], 0); self.assertEqual(receipt['physics_steps'], 23)

    def test_approach_seed_never_requests_native_lift_and_keeps_local_null(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); pool = root/'pool'; pool.mkdir()
            chosen = candidate(); T.save(pool/'selected_author_candidate.json', chosen)
            (pool/'candidates.npz').write_bytes(b'bound pool')
            T.save(pool/'generation.json', dict(complete=True, selected_indices=[0],
                selected_candidate_sha256=T.sha(pool/'selected_author_candidate.json'),
                candidates_sha256=T.sha(pool/'candidates.npz'), input_sha256={},
                compatibility_object_root='explicit_compatibility_body'))
            config = {key: 'bound_'+key for key in ('robot_urdf', 'reference_root', 'hand_urdf',
                'hand_scene', 'package', 'physics_config', 'diagnostic_config')}
            config.update(pregrasp_distance_m=.1, restore_object_initial_state=True)
            T.save(root/'backend.json', config)
            def rejected(command, **kwargs):
                self.assertIn('--approach-only', command)
                self.assertIn('--restore-object-initial-state', command)
                self.assertEqual(float(command[command.index('--pregrasp-distance-m')+1]), .04)
                native = Path(command[command.index('--output-dir')+1]); native.mkdir()
                T.save(native/'admission_failure.json', dict(complete=True, kind='arm_plan_rejected', physical_executed=False))
                return SimpleNamespace(returncode=1)
            with patch.object(pipeline.subprocess, 'run', side_effect=rejected):
                result = pipeline.prepare_approach(pool, root/'arm', backend_config=root/'backend.json')
            self.assertEqual(result['status'], 'no_plan')
            self.assertFalse(result['unused_world_z_lift_required'])
            self.assertEqual(result['backend_requested_pregrasp_distance_m'], .1)
            self.assertEqual(result['actual_pregrasp_distance_m'], .04)
            self.assertEqual(result['physics_steps'], 0)

    def test_source_ast_pins_shared_collection_collision_and_execution_distance(self):
        policy = pipeline.source_collection_motion_policy()
        self.assertEqual(policy['approach_distance_m'], .04)
        self.assertEqual(policy['assignment_line'], 469)
        self.assertEqual(policy['call_lines'], {'mfcdetector.detect': 472, 'robot.grasp_and_throw': 548})
        self.assertFalse(policy['backend_distance_is_authoritative'])
        original = pipeline.COLLECTION_SOURCE.read_text()
        for mutation in ('hash', 'constant', 'execution_argument'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary)/'source.py'
                changed = original
                if mutation == 'hash': changed += '\n# source changed\n'
                if mutation == 'constant': changed = changed.replace('approach_distance = 0.04', 'approach_distance = 0.10')
                if mutation == 'execution_argument':
                    before, after = changed.rsplit('approach_dist=approach_distance', 1)
                    changed = before+'approach_dist=0.10'+after
                source.write_text(changed)
                digest = pipeline.COLLECTION_SOURCE_SHA256 if mutation == 'hash' else T.sha(source)
                with patch.object(pipeline, 'COLLECTION_SOURCE', source), \
                        patch.object(pipeline, 'COLLECTION_SOURCE_SHA256', digest):
                    with self.assertRaises(ValueError): pipeline.source_collection_motion_policy()

    def test_source_four_cm_distance_is_the_transport_pregrasp_retreat(self):
        policy = pipeline.source_collection_motion_policy()
        palm = np.eye(4); palm[:3, 3] = [-.75, .1, .2]
        direction = np.array([0., 0., -1.])
        bin_spec = dict(world_T_bin=np.eye(4).tolist(), inner_half_extents_m=[.15, .15, .1])
        goals = T._waypoint_targets(palm, direction, policy['approach_distance_m'], bin_spec, np.eye(4))
        np.testing.assert_allclose(goals[1][1][:3, 3]-palm[:3, 3], [0., 0., .04], atol=1e-15)
        self.assertEqual(goals[1][0], 'retreat_pregrasp')


if __name__ == '__main__':
    unittest.main()
