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
    T.save(cal, {}); T.save(cloud, {})
    robot = root/'robot.urdf'; robot.write_text('<robot name="fixture"/>')
    refinement = root/'arm_refinement.json'
    T.save(refinement, dict(schema_version=1, combined_urdf_sha256=T.sha(robot),
        physics_config_sha256=T.sha(physics), links={'wrist_1_link': {'partitions': []}}))
    T.save(backend, dict(physics_config=str(physics), robot_urdf=str(robot),
                        arm_collision_refinement=str(refinement)))
    release_pose = np.eye(4); release_pose[:3, 3] = [0., 0., .5]
    T.save(root/'release.json', dict(schema='anydex_boya_author_fixed_release_v1',
        scene=pipeline._binding(scene_path), bin_sha256=T.json_sha(T.read(scene_path)['bin']),
        world_T_release_palm=release_pose.tolist(), frame='world', link='palm',
        pose_rule='fixed_world_se3', depends_on_candidate_pose=False, depends_on_object_pose=False,
        source_sha256={str(Path(__file__).resolve()): T.sha(__file__)},
        source_policy={'fixed_throw': 'test source binding'}, adaptation={'declared': 'fixed simulator pose'}))
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
        refinement = pipeline.stage_arm_collision_refinement(root/'backend.json', root)
        binding = pipeline._bind_transport_plan(raw, root/'transport_bound', chosen, args[6], scene,
            root/'release.json', arm_collision_refinement=refinement)
    T.save(root/'candidate.json', chosen)
    protocol = dict(inputs={'scene': pipeline._binding(scene), 'fixed_release': pipeline._binding(root/'release.json')}, sources={})
    T.save(root/'PIPELINE_PROTOCOL.json', protocol)
    ready = dict(status='ready', protocol=pipeline._binding(root/'PIPELINE_PROTOCOL.json'),
        scene=pipeline._binding(scene), fixed_release=pipeline._binding(root/'release.json'), candidate=pipeline._binding(root/'candidate.json'),
        plan=pipeline._binding(root/'transport_bound/plan.json'),
        commands=pipeline._binding(root/'transport_bound/commands.npz'),
        command_binding=pipeline._binding(root/'transport_bound/COMMAND_BINDING.json'),
        collection_candidate_sha256=chosen['candidate_sha256'])
    T.save(root/'READY.json', ready)
    T.save(root/'PIPELINE_RESULTS.json', dict(status='ready', ready_sha256=T.sha(root/'READY.json')))
    return chosen, binding, arrays


def continuation_fixture(root):
    args = fixture(root)
    previous = root/'previous'; previous.mkdir()
    T.save(previous/'BACKEND.json', T.read(root/'backend.json'))
    chosen = candidate(); pool = previous/'pool'; pool.mkdir()
    T.save(pool/'selected_author_candidate.json', chosen)
    (pool/'candidates.npz').write_bytes(b'immutable candidate pool')
    generation = dict(complete=True, selected_indices=[0], selected_candidate_sha256=T.sha(pool/'selected_author_candidate.json'),
                      candidates_sha256=T.sha(pool/'candidates.npz'))
    T.save(pool/'generation.json', generation)
    T.save(previous/'PIPELINE_PROTOCOL.json', dict(schema=pipeline.SCHEMA,
        inputs=dict(scene=pipeline._binding(args[0]), calibration=pipeline._binding(args[6])),
        derived_backend=pipeline._binding(previous/'BACKEND.json'),
        collection_motion_policy=dict(approach_distance_m=.04), candidate_budget=1, fallback_candidate=False))
    seed = previous/'arm/native'; seed.mkdir(parents=True)
    arrays = dict(joint_names=np.array(['motor']), joint_target=np.zeros((3, 1)),
        initial_position=np.zeros(1), phase=np.array(['settle', 'home_to_pregrasp', 'approach']),
        desired_palm=np.repeat(np.eye(4)[None], 3, axis=0), time_s=np.arange(1, 4)*.004)
    np.savez_compressed(seed/'commands.npz', **arrays)
    plan = dict(complete=True, approach_only=True, admitted=False,
        arguments=dict(pregrasp_distance_m=.04), commands_sha256=T.sha(seed/'commands.npz'),
        candidates_sha256=generation['candidates_sha256'], generation_sha256=T.sha(pool/'generation.json'),
        source_sha256={}, extra_input_sha256={})
    T.save(seed/'plan.json', plan)
    T.save(previous/'PIPELINE_RESULTS.json', dict(complete=True, physics_steps=0,
        protocol_sha256=T.sha(previous/'PIPELINE_PROTOCOL.json'),
        arm_preparation=dict(plan_sha256=T.sha(seed/'plan.json')),
        candidate_sha256=chosen['candidate_sha256'], status='no_plan'))
    routed = root/'route/native'; routed.mkdir(parents=True)
    np.savez_compressed(routed/'commands.npz', **arrays)
    T.save(routed/'path_geometry.json', dict(ok=True))
    T.save(root/'route/HOME_ROUTE_PROTOCOL.json', dict(scope='fixture full scene'))
    plan.update(admitted=True, path_geometry_sha256=T.sha(routed/'path_geometry.json'),
        commands_sha256=T.sha(routed/'commands.npz'), author_home_route=dict(
            original_seed_plan_sha256=T.sha(seed/'plan.json'),
            protocol_path=str(root/'route/HOME_ROUTE_PROTOCOL.json'),
            protocol_sha256=T.sha(root/'route/HOME_ROUTE_PROTOCOL.json')))
    T.save(routed/'plan.json', plan)
    contract = SimpleNamespace(names=['motor'], validate_command_arrays=lambda *a, **k: {},
                               mapped_palm=lambda *a, **k: np.eye(4))
    return previous, routed, contract


class PipelineTests(unittest.TestCase):
    def test_native_arm_refinement_copy_binds_robot_physics_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root)
            copied = root/'staged'; copied.mkdir()
            receipt = pipeline.stage_arm_collision_refinement(root/'backend.json', copied)
            self.assertEqual(Path(receipt['staged']['path']).read_bytes(), (root/'arm_refinement.json').read_bytes())
            self.assertEqual(receipt['source']['sha256'], receipt['staged']['sha256'])
            self.assertEqual(pipeline.stage_arm_collision_refinement(root/'backend.json', copied), receipt)
            for mutation in ('source_sha', 'robot_sha', 'physics_sha', 'missing_config', 'staged_copy'):
                with self.subTest(mutation=mutation):
                    cfg = T.read(root/'backend.json'); original = T.read(root/'arm_refinement.json')
                    changed = dict(original)
                    if mutation == 'source_sha': cfg['arm_collision_refinement_sha256'] = '0'*64
                    if mutation == 'robot_sha': changed['combined_urdf_sha256'] = '0'*64
                    if mutation == 'physics_sha': changed['physics_config_sha256'] = '0'*64
                    if mutation == 'missing_config': cfg.pop('arm_collision_refinement')
                    T.save(root/'mutated_backend.json', cfg); T.save(root/'arm_refinement.json', changed)
                    if mutation == 'staged_copy': (copied/'arm_collision_refinements.json').write_text('different physics input')
                    with self.assertRaises(ValueError):
                        pipeline.stage_arm_collision_refinement(root/'mutated_backend.json', copied)
                    T.save(root/'arm_refinement.json', original)

    def test_native_arm_refinement_tampering_refuses_before_physical_launcher(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); ready_fixture(root)
            with (root/'arm_collision_refinements.json').open('ab') as f: f.write(b' ')
            with patch.object(pipeline, '_launch_native') as launch:
                with self.assertRaises(ValueError): pipeline.execute(root)
            launch.assert_not_called(); self.assertFalse((root/'EXECUTION_STARTED.json').exists())

    def test_post_execution_dependency_drift_cannot_create_an_eligible_training_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); ready_fixture(root)
            def native(*unused):
                with (root/'arm_collision_refinements.json').open('ab') as f: f.write(b' ')
                evaluation = dict(status='physical_failure', complete=True, total_physics_steps=23)
                T.save(root/'execution/TRANSPORT_EVALUATION.json', evaluation)
                return evaluation
            with patch.object(pipeline, '_launch_native', return_value=SimpleNamespace(app=SimpleNamespace(close=lambda **k: None))), \
                    patch.object(pipeline.transport, 'args_for_plan', return_value=SimpleNamespace()), \
                    patch.object(pipeline, '_execute_native', side_effect=native), \
                    patch.object(pipeline.collector, 'collect_execution') as label:
                with self.assertRaises(ValueError): pipeline.execute(root)
            label.assert_not_called(); self.assertFalse((root/'COLLECTION_RECORD.json').exists())
            receipt = T.read(root/'PIPELINE_EXECUTION.json')
            self.assertEqual(receipt['status'], 'infrastructure_failure')
            self.assertEqual(receipt['physics_steps'], 23); self.assertEqual(receipt['new_labels'], 0)

    def test_robot_meshes_bind_original_relative_and_file_paths_and_reject_missing_or_package_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root)
            (root/'meshes').mkdir(); a, b = root/'meshes/a.obj', root/'meshes/b.obj'
            a.write_text('first source surface'); b.write_text('second source surface')
            robot = root/'robot.urdf'; repair = root/'arm_refinement.json'
            def set_meshes(filenames):
                robot.write_text('<robot>'+''.join('<link><visual><geometry><mesh filename="'+name+
                    '"/></geometry></visual></link>' for name in filenames)+'</robot>')
                spec = T.read(repair); spec['combined_urdf_sha256'] = T.sha(robot); T.save(repair, spec)
            set_meshes(['meshes/a.obj', 'file://'+str(b), str(a)])
            staged = root/'staged'; staged.mkdir()
            receipt = pipeline.stage_arm_collision_refinement(root/'backend.json', staged)
            self.assertEqual(set(receipt['robot_meshes']), {str(a), str(b)})
            bindings = pipeline._arm_refinement_inputs(receipt)
            self.assertEqual(bindings['native_robot_mesh:'+str(a)], pipeline._binding(a))
            a.write_text('modified surface after preparation')
            with self.assertRaises(ValueError): pipeline._verify(bindings)
            for filename, error in (('meshes/absent.obj', FileNotFoundError),
                                    ('package://unknown/mesh.obj', ValueError)):
                with self.subTest(filename=filename):
                    set_meshes([filename])
                    with self.assertRaises(error):
                        pipeline.stage_arm_collision_refinement(root/'backend.json', staged)

    def test_component_continuation_reuses_candidate_and_never_repeats_collection_or_HOME_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); previous, routed, contract = continuation_fixture(root)
            with patch.object(pipeline, 'ActionContract', return_value=contract), \
                    patch.object(pipeline.collector, 'expanded_planned_hand', side_effect=lambda a, c: (a, {})), \
                    patch.object(pipeline.collector, 'run_protocol') as collect, \
                    patch.object(pipeline, 'prepare_approach') as approach, \
                    patch.object(pipeline, '_prepare_home_route') as route, \
                    patch.object(pipeline.transport, 'prepare_transport',
                        return_value=dict(status='null_geometry', admitted=False)) as moved:
                result = pipeline.prepare_from_frozen_candidate_plan(previous, routed, root/'continued',
                    release_spec_path=root/'release.json')
            self.assertEqual(result['status'], 'null_geometry')
            collect.assert_not_called(); approach.assert_not_called(); route.assert_not_called()
            self.assertEqual(moved.call_args.kwargs['release_spec_path'], root/'release.json')
            self.assertFalse((root/'continued/PIPELINE_RESULTS.json').exists())
            self.assertTrue((root/'continued/COMPONENT_INTEGRATION_PROTOCOL.json').exists())
            self.assertEqual(result['new_collection_collision_queries'], 0)
            self.assertEqual(result['new_native_approach_IK'], 0)

    def test_component_continuation_refuses_changed_candidate_endpoint_and_route_dependency(self):
        for mutation in ('candidate', 'pregrasp_endpoint', 'route_dependency'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); previous, routed, contract = continuation_fixture(root)
                if mutation == 'candidate':
                    with (previous/'pool/selected_author_candidate.json').open('ab') as f: f.write(b' ')
                if mutation == 'pregrasp_endpoint':
                    with np.load(routed/'commands.npz', allow_pickle=False) as z:
                        arrays = {key: z[key].copy() for key in z.files}
                    arrays['desired_palm'][1, 0, 3] += .01
                    np.savez_compressed(routed/'commands.npz', **arrays)
                    plan = T.read(routed/'plan.json'); plan['commands_sha256'] = T.sha(routed/'commands.npz')
                    T.save(routed/'plan.json', plan)
                if mutation == 'route_dependency':
                    plan = T.read(routed/'plan.json')
                    plan['extra_input_sha256'][str(root/'scene.json')] = '0'*64
                    T.save(routed/'plan.json', plan)
                with patch.object(pipeline, 'ActionContract', return_value=contract), \
                        patch.object(pipeline.transport, 'prepare_transport') as moved:
                    with self.assertRaises(ValueError):
                        pipeline.prepare_from_frozen_candidate_plan(previous, routed, root/'continued',
                            release_spec_path=root/'release.json')
                moved.assert_not_called()
                self.assertEqual(T.read(root/'continued/COMPONENT_INTEGRATION_RESULTS.json')['physics_steps'], 0)

    def test_component_ready_is_bound_to_new_receipt_and_can_use_explicit_once_executor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); previous, routed, contract = continuation_fixture(root)
            def moved(seed, scene, output, **kwargs):
                output.mkdir()
                with np.load(seed/'commands.npz', allow_pickle=False) as z:
                    arrays = {name: z[name].copy() for name in z.files}
                np.savez_compressed(output/'commands.npz', **arrays)
                plan = T.read(seed/'plan.json'); plan['commands_sha256'] = T.sha(output/'commands.npz')
                T.save(output/'plan.json', plan)
                return dict(status='ready', admitted=True)
            with patch.object(pipeline, 'ActionContract', return_value=contract), \
                    patch.object(pipeline.collector, 'expanded_planned_hand', side_effect=lambda a, c: (a, {})), \
                    patch.object(pipeline.transport, 'prepare_transport', side_effect=moved):
                result = pipeline.prepare_from_frozen_candidate_plan(previous, routed, root/'continued',
                    release_spec_path=root/'release.json')
            self.assertEqual(result['status'], 'ready')
            ready = T.read(root/'continued/READY.json')
            self.assertEqual(ready['preparation_receipt'], 'COMPONENT_INTEGRATION_RESULTS.json')
            self.assertFalse((root/'continued/PIPELINE_RESULTS.json').exists())
            with patch.object(pipeline, '_launch_native', return_value=SimpleNamespace(app=SimpleNamespace(close=lambda **k: None))), \
                    patch.object(pipeline.transport, 'args_for_plan', return_value=SimpleNamespace()), \
                    patch.object(pipeline, '_execute_native', return_value=dict(status='physical_failure', complete=True,
                        total_physics_steps=23)), \
                    patch.object(pipeline.collector, 'collect_execution', return_value=dict(label=None, label_state='test_mock')):
                executed = pipeline.execute(root/'continued')
            self.assertEqual(executed['physics_steps'], 23)
            self.assertEqual(executed['new_labels'], 0)

    def test_route_strategies_select_distinct_modules_without_implicit_fallback(self):
        for strategy, module in pipeline.HOME_ROUTE_MODULES.items():
            fake = SimpleNamespace(prepare_home_route=lambda *a, **k: dict(strategy=strategy))
            with self.subTest(strategy=strategy), patch('importlib.import_module', return_value=fake) as imported:
                result = pipeline._prepare_home_route('seed', 'scene', 'out', config_path='config', strategy=strategy)
            imported.assert_called_once_with(module[:-3]); self.assertEqual(result['strategy'], strategy)
        with self.assertRaises(ValueError):
            pipeline._prepare_home_route('seed', 'scene', 'out', config_path='config', strategy='unbounded')

    def test_release_is_explicit_and_bound_before_any_candidate_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = fixture(root)
            with patch.object(pipeline.collector, 'run_protocol') as collect:
                with self.assertRaisesRegex(ValueError, 'Explicit fixed release'):
                    pipeline.prepare(*args, root/'missing_release')
            collect.assert_not_called()
            self.assertEqual(T.read(root/'missing_release/PIPELINE_RESULTS.json')['physics_steps'], 0)
            with patch.object(pipeline.collector, 'run_protocol', return_value=dict(candidates=[], counts={})):
                pipeline.prepare(*args, root/'bound_release', release_spec_path=root/'release.json')
            protocol = T.read(root/'bound_release/PIPELINE_PROTOCOL.json')
            self.assertEqual(protocol['inputs']['fixed_release'], pipeline._binding(root/'release.json'))
            self.assertEqual(protocol['release_pose_rule'], 'fixed_world_se3')

    def test_no_candidate_cannot_start_arm_or_try_other_choices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = fixture(root)
            with patch.object(pipeline.collector, 'run_protocol', return_value=dict(candidates=[], counts={'top10': 0})), \
                    patch.object(pipeline.bridge, 'bridge_candidate') as bridge, \
                    patch.object(pipeline, 'prepare_approach') as arm:
                result = pipeline.prepare(*args, root/'out', release_spec_path=root/'release.json')
            self.assertEqual(result['status'], 'no_candidate'); bridge.assert_not_called(); arm.assert_not_called()
            self.assertEqual(result['new_labels'], 0)
            self.assertTrue((root/'out/PIPELINE_PROTOCOL.json').exists())
            self.assertEqual(T.read(root/'out/BACKEND.json')['pregrasp_distance_m'], .04)
            protocol = T.read(root/'out/PIPELINE_PROTOCOL.json')
            self.assertEqual(protocol['collection_motion_policy']['approach_distance_m'], .04)
            self.assertEqual(protocol['inputs']['collection_motion_source']['sha256'], pipeline.COLLECTION_SOURCE_SHA256)
            with self.assertRaises(FileExistsError): pipeline.prepare(*args, root/'out', release_spec_path=root/'release.json')

    def test_fixed_first_candidate_and_arm_null_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = fixture(root)
            choices = [candidate('first'), candidate('second')]
            with patch.object(pipeline.collector, 'run_protocol', return_value=dict(candidates=choices, counts={'top10': 2})), \
                    patch.object(pipeline.bridge, 'bridge_candidate') as bridge, \
                    patch.object(pipeline, 'prepare_approach', return_value=dict(status='no_plan')) as arm, \
                    patch.object(pipeline.transport, 'prepare_transport') as transport:
                result = pipeline.prepare(*args, root/'out', release_spec_path=root/'release.json')
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
                        with self.assertRaises(RuntimeError): pipeline.prepare(*args, root/'out', release_spec_path=root/'release.json')
                    else:
                        self.assertEqual(pipeline.prepare(*args, root/'out', release_spec_path=root/'release.json')['status'], 'null_geometry')
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

    def test_binding_cannot_overwrite_conflicting_existing_candidate_or_task(self):
        for key, changed in (('task', 'old_vertical_lift'), ('collection_candidate_sha256', '0'*64),
                             ('candidate', candidate('another'))):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); chosen, _, _ = ready_fixture(root)
                plan = T.read(root/'raw/plan.json'); plan['author_collection_binding'][key] = changed
                T.save(root/'raw/plan.json', plan)
                contract = SimpleNamespace(validate_command_arrays=lambda *a, **kw: {})
                with patch.object(pipeline, 'ActionContract', return_value=contract):
                    with self.assertRaises(ValueError):
                        pipeline._bind_transport_plan(root/'raw', root/'conflicting', chosen,
                            root/'calibration.json', root/'scene.json', root/'release.json',
                            arm_collision_refinement=pipeline.stage_arm_collision_refinement(root/'backend.json', root))

    def test_execution_refuses_tampering_before_launch_and_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); ready_fixture(root)
            with (root/'transport_bound/commands.npz').open('ab') as stream: stream.write(b'changed')
            with patch.object(pipeline, '_execute_native') as launch:
                with self.assertRaises(ValueError): pipeline.execute(root)
            launch.assert_not_called(); self.assertFalse((root/'EXECUTION_STARTED.json').exists())

    def test_fixed_release_tampering_cannot_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); ready_fixture(root)
            release = T.read(root/'release.json')
            release['world_T_release_palm'][0][3] += .01
            T.save(root/'release.json', release)
            with patch.object(pipeline, '_launch_native') as launch:
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

    def test_opt_in_home_router_can_replace_complete_rejected_native_home(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); pool = root/'pool'; pool.mkdir()
            chosen = candidate(); T.save(pool/'selected_author_candidate.json', chosen)
            (pool/'candidates.npz').write_bytes(b'bound pool')
            T.save(pool/'generation.json', dict(complete=True, selected_indices=[0],
                selected_candidate_sha256=T.sha(pool/'selected_author_candidate.json'),
                candidates_sha256=T.sha(pool/'candidates.npz'), input_sha256={},
                calibration='stub_calibration', compatibility_object_root='compatibility_body'))
            config = {key: 'bound_'+key for key in ('robot_urdf', 'reference_root', 'hand_urdf',
                'hand_scene', 'package', 'physics_config', 'diagnostic_config')}
            config['pregrasp_distance_m'] = .1
            T.save(root/'backend.json', config); T.save(root/'scene.json', {})
            T.save(root/'route.json', dict(policy='fixed budget, full scene'))
            arrays = dict(joint_names=np.array(['motor']), joint_target=np.zeros((3, 1)),
                initial_position=np.zeros(1), phase=np.array(['settle', 'home_to_pregrasp', 'approach']),
                desired_palm=np.repeat(np.eye(4)[None], 3, axis=0), time_s=np.array([.004, .008, .012]))
            def rejected(command, **unused):
                native = Path(command[command.index('--output-dir')+1]); native.mkdir()
                np.savez_compressed(native/'commands.npz', **arrays)
                T.save(native/'plan.json', dict(complete=True, admitted=False, approach_only=True,
                    commands_sha256=T.sha(native/'commands.npz'), arguments=dict(pregrasp_distance_m=.04)))
                T.save(native/'admission_failure.json', dict(complete=True, kind='arm_plan_rejected', physical_executed=False))
                return SimpleNamespace(returncode=1)
            def routed(seed, scene, output, **kwargs):
                self.assertFalse(T.read(seed/'plan.json')['admitted'])
                self.assertEqual(Path(kwargs['config_path']), root/'route.json')
                native = output/'native'; native.mkdir(parents=True)
                np.savez_compressed(native/'commands.npz', **arrays)
                plan = T.read(seed/'plan.json'); plan.update(admitted=True, commands_sha256=T.sha(native/'commands.npz'))
                T.save(native/'plan.json', plan)
                return dict(status='ready', admitted=True, plan_dir=str(native), physics_steps=0)
            contract = SimpleNamespace(names=['motor'], validate_command_arrays=lambda *a, **k: {},
                                       mapped_palm=lambda *a, **k: np.eye(4))
            with patch.object(pipeline.subprocess, 'run', side_effect=rejected), \
                    patch.object(pipeline, '_prepare_home_route', side_effect=routed) as route, \
                    patch.object(pipeline, 'ActionContract', return_value=contract), \
                    patch.object(pipeline.collector, 'expanded_planned_hand', side_effect=lambda a, c: (a, {})):
                receipt = pipeline.prepare_approach(pool, root/'arm', backend_config=root/'backend.json',
                    scene_path=root/'scene.json', home_route_config=root/'route.json')
            self.assertEqual(receipt['status'], 'ready'); self.assertEqual(route.call_count, 1)
            self.assertEqual(receipt['native_direct_status'], 'no_plan')
            self.assertEqual(Path(receipt['plan_dir']), root/'arm/home_route/native')
            self.assertEqual(receipt['returncode'], 1)
            self.assertEqual(receipt['physics_steps'], 0)

    def test_opt_in_home_router_never_converts_missing_native_plan_into_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); pool = root/'pool'; pool.mkdir()
            chosen = candidate(); T.save(pool/'selected_author_candidate.json', chosen)
            (pool/'candidates.npz').write_bytes(b'bound pool')
            T.save(pool/'generation.json', dict(complete=True, selected_indices=[0],
                selected_candidate_sha256=T.sha(pool/'selected_author_candidate.json'),
                candidates_sha256=T.sha(pool/'candidates.npz'), input_sha256={},
                compatibility_object_root='compatibility_body'))
            config = {key: 'bound_'+key for key in ('robot_urdf', 'reference_root', 'hand_urdf',
                'hand_scene', 'package', 'physics_config', 'diagnostic_config')}
            config['pregrasp_distance_m'] = .1
            for name in ('backend', 'scene', 'route'): T.save(root/(name+'.json'), config if name == 'backend' else {})
            with patch.object(pipeline.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), \
                    patch.object(pipeline, '_prepare_home_route') as route:
                receipt = pipeline.prepare_approach(pool, root/'arm', backend_config=root/'backend.json',
                    scene_path=root/'scene.json', home_route_config=root/'route.json')
            self.assertEqual(receipt['status'], 'infrastructure_failure'); route.assert_not_called()

    def test_home_replacement_independently_rejects_changes_to_fixed_motion_or_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source, routed = root/'source', root/'routed'
            source.mkdir(); routed.mkdir()
            old = dict(joint_names=np.array(['arm', 'motor']), initial_position=np.zeros(2),
                joint_target=np.zeros((3, 2)), phase=np.array(['settle', 'home_to_pregrasp', 'approach']),
                desired_palm=np.repeat(np.eye(4)[None], 3, axis=0), time_s=np.arange(1, 4)*.004)
            new = dict(joint_names=old['joint_names'].copy(), initial_position=old['initial_position'].copy(),
                joint_target=np.zeros((4, 2)),
                phase=np.array(['settle', 'home_to_pregrasp', 'home_to_pregrasp', 'approach']),
                desired_palm=np.repeat(np.eye(4)[None], 4, axis=0), time_s=np.arange(1, 5)*.004)
            plan = dict(index=0, gesture_id=6, palm=np.eye(4).tolist(), dt_s=.004)
            T.save(source/'plan.json', plan); T.save(routed/'plan.json', plan)
            np.savez_compressed(source/'commands.npz', **old)
            np.savez_compressed(routed/'commands.npz', **new)
            proof = pipeline._verify_home_preservation(source, routed)
            self.assertEqual(proof['source_home_rows'], 1)
            self.assertEqual(proof['routed_home_rows'], 2)
            for mutation in ('approach_arm', 'settle_palm', 'clock', 'initial', 'candidate'):
                with self.subTest(mutation=mutation):
                    altered = {key: value.copy() for key, value in new.items()}
                    changed_plan = dict(plan)
                    if mutation == 'approach_arm': altered['joint_target'][-1, 0] += .01
                    if mutation == 'settle_palm': altered['desired_palm'][0, 0, 3] += .01
                    if mutation == 'clock': altered['time_s'][-1] += .004
                    if mutation == 'initial': altered['initial_position'][0] += .01
                    if mutation == 'candidate': changed_plan['gesture_id'] = 7
                    np.savez_compressed(routed/'commands.npz', **altered)
                    T.save(routed/'plan.json', changed_plan)
                    with self.assertRaises(ValueError): pipeline._verify_home_preservation(source, routed)

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
        release_pose = np.eye(4); release_pose[:3, 3] = [0., 0., .5]
        goals = T._waypoint_targets(palm, direction, policy['approach_distance_m'], release_pose)
        np.testing.assert_allclose(goals[1][1][:3, 3]-palm[:3, 3], [0., 0., .04], atol=1e-15)
        self.assertEqual(goals[1][0], 'retreat_pregrasp')


if __name__ == '__main__':
    unittest.main()
