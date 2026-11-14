"""Explicit bin-free multi-object lift through the unchanged native executor.

Source geometry measures every original body; it never associates candidates
with a ground-truth object. The caller owns SimulationApp and persists its
collection receipt before closing the app. Importing this module starts no Kit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from . import author_transport as common
except ImportError:
    import author_transport as common

TASK = 'declared_boya_clutter_lift_v1'
SCENE_SCHEMA = 'anydex_boya_author_lift_scene_v1'
POLICY = dict(schema='anydex_boya_author_lift_evaluation_policy_v1', task=TASK,
    dt_s=.004, source_bottom_clearance_m=.005, observed_pickup_reference_rise_m=.005,
    final_reference_rise_m=.05, bilateral_force_threshold_n=.001,
    max_illegal_robot_support_force_n=.001, max_penetration_m=.001,
    early_stop_penetration_m=.005, contact_capacity=262144,
    required_phases=['settle', 'home_to_pregrasp', 'approach', 'close', 'hold', 'lift'],
    additional_final_hold_steps=0, minimum_continuous_hold_s=None,
    duration_is_diagnostic_only=True, object_selection='any original tabletop body')
ORIGINAL_GATES = ('joint_limits', 'mimics', 'penetration', 'palm_tracking',
                  'no_arm_object_support', 'fixed_base', 'complete_execution')


def validate_scene(path, archive_sources=None):
    scene = common.validate_scene(path, archive_sources)
    if (scene.get('schema') != SCENE_SCHEMA or scene.get('task') != TASK
            or scene.get('bin', 'missing') is not None):
        raise ValueError('Explicit bin-free lift scene/task required')
    expected = [row['object_id'] for row in scene['objects']]
    if scene.get('original_object_ids', expected) != expected:
        raise ValueError('Original lift body identities must match the whole scene')
    if scene.get('dt_s') != POLICY['dt_s']:
        raise ValueError('Frozen native lift timestep required')
    return scene


def source_metrics(vertices, pose_xyzw):
    """Full source visual vertices in native body coordinates, without a bin."""
    from scipy.spatial.transform import Rotation
    vertices, pose = np.asarray(vertices, float), np.asarray(pose_xyzw, float)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or pose.shape != (7,) or not np.isfinite(vertices).all()
            or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1) > 1e-4):
        raise ValueError('Finite source vertices and native XYZW pose required')
    world = vertices@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]
    return dict(source_bottom_m=float(world[:, 2].min()), source_top_m=float(world[:, 2].max()))


def evaluate_lift(rows, original_object_ids, dt):
    """Original pickup/final-rise rule, independently evaluated for every body.

    The original final rule requires 5cm reference rise; the 5mm pickup state
    is also retained for timing diagnostics. No minimum hold duration is added.
    Physical validity gates are applied by finish(), separately from observation.
    """
    ids = list(original_object_ids)
    if (not ids or len(set(ids)) != len(ids) or any(not isinstance(i, str) for i in ids)
            or not np.isfinite(dt) or dt != POLICY['dt_s']):
        raise ValueError('Unique original body identities and native timestep required')
    required = POLICY['required_phases']
    blocks = [r['phase'] for i, r in enumerate(rows) if i == 0 or rows[i-1]['phase'] != r['phase']]
    phase_order = blocks == required
    for i, row in enumerate(rows):
        if (set(row['objects']) != set(ids) or row.get('step') != i+1
                or not np.isfinite(row.get('time_s', np.nan))
                or not np.isclose(row['time_s'], (i+1)*dt, atol=1e-12, rtol=0)):
            raise ValueError('Complete aligned all-body native trace required')
        for state in row['objects'].values():
            pose, fingers = np.asarray(state['pose_xyzw'], float), np.asarray(state['finger_loads_n'], float)
            if (pose.shape != (7,) or fingers.shape != (5,)
                    or not np.isfinite(pose).all() or not np.isfinite(fingers).all()
                    or np.any(fingers < 0) or not np.isfinite(state['source_bottom_m'])):
                raise ValueError('Finite native pose, source bottom and five finger loads required')
    holds = [r for r in rows if r['phase'] == 'hold']
    outcomes, objects = [], {}
    for key in ids:
        reference = float(holds[-1]['objects'][key]['pose_xyzw'][2]) if holds else None
        flags, lift_states, rises = [], [], []
        for row in rows:
            state = row['objects'][key]
            rise = float(state['pose_xyzw'][2]-reference) if reference is not None else None
            fingers = np.asarray(state['finger_loads_n'], float)
            bearing = bool(fingers[0] > .001 and fingers[1:].sum() > .001)
            flags.append(bool(row['phase'] == 'lift' and rise is not None
                and state['source_bottom_m'] >= .005 and rise >= .005 and bearing))
            if row['phase'] == 'lift':
                lift_states.append(state); rises.append(rise)
        longest = run = 0
        for flag in flags:
            run = run+1 if flag else 0
            longest = max(longest, run)
        final = rows[-1]['objects'][key] if rows else None
        final_rise = float(final['pose_xyzw'][2]-reference) if final is not None and reference is not None else None
        raw = bool(phase_order and flags and flags[-1] and final_rise >= .05)
        result = dict(object_id=key, task_success=raw, observed_final_lift_success=raw,
            observed_pickup=any(flags), qualifying_steps=sum(flags),
            longest_observed_pickup_run_s=float(longest*dt),
            trailing_observed_pickup_run_s=float(run*dt),
            prelift_reference_z_m=reference, final_reference_rise_m=final_rise,
            final_source_bottom_m=final['source_bottom_m'] if final else None,
            final_state_still_lifted_and_bearing=bool(flags and flags[-1]),
            first_pickup_time_s=next((row['time_s'] for row, flag in zip(rows, flags) if flag), None),
            peak_held_reference_rise_m=max((rise for rise, state in zip(rises, lift_states)
                if rise is not None and state['source_bottom_m'] >= .005
                and state['finger_loads_n'][0] > .001 and sum(state['finger_loads_n'][1:]) > .001), default=None),
            duration_is_diagnostic_only=True, final=final)
        outcomes.append(result); objects[key] = result
    return dict(original_object_ids=ids, object_outcomes=outcomes, objects=objects,
        task_success=any(r['task_success'] for r in outcomes), phase_order_valid=phase_order,
        actual_phase_blocks=blocks, required_phases=list(required))


def args_for_plan(plan_dir, output_dir):
    return common.args_for_plan(plan_dir, output_dir)


class MultiObjectLiftRuntime(common.MultiObjectRuntime):
    """Same native all-body spawning/control/contact loop, with zero bin shapes."""
    def __init__(self, scene_path, output_dir, *, archive_sources=None):
        validate_scene(scene_path, archive_sources)
        super().__init__(scene_path, output_dir, archive_sources=archive_sources)

    # ready() is copied below from the audited transport implementation with
    # only its final full-source measurement made independent of containment.

    def ready(self):
        import torch
        import warp as wp
        from pxr import Usd, UsdPhysics
        from template_gripper_native_audit import audit_scene
        from canondex.scene.execute_motion import RobotContactReports
        from canondex.scene.runtime import forward_scene
        from canondex.planning.model import PlanningModel
        from canondex.grasp.template_gripper_adapter import TemplateFamily

        # Original executor already applies target weight and material readback.
        # Every additional body gets exactly the same native mass/friction and
        # persistent world force before the first physical tick.
        material, entry = {}, {}
        for key, body in self.bodies.items():
            body.update(self.dt)
            actual_pose = body.data.root_link_pose_w.torch[0].cpu().numpy().copy()
            actual_velocity = body.data.root_link_vel_w.torch[0].cpu().numpy().copy()
            desired = self.assets[key].pose_xyzw
            error = min(float(abs(actual_pose-desired).max()),
                        float(abs(actual_pose-np.r_[desired[:3], -desired[3:]]).max()))
            if error > 2e-5 or float(abs(actual_velocity).max()) > 2e-5:
                raise RuntimeError('Declared cold scene entry does not match actual native state: '+key)
            entry[key] = dict(pose_xyzw=actual_pose.tolist(), velocity=actual_velocity.tolist(), max_pose_error=error)
            if float(body.root_view.get_masses().numpy().ravel()[0]) != float(np.float32(.1)):
                raise RuntimeError('Actual dynamic object mass changed: '+key)
            if key != self.target_id:
                properties = body.root_view.get_material_properties().numpy().copy()
                properties[..., :2] = .6; properties[..., 2] = 0.
                ids = wp.array(np.arange(len(properties), dtype=np.int32), dtype=wp.int32, device='cpu')
                body.root_view.set_material_properties(wp.array(properties, dtype=wp.float32, device='cpu'), ids)
                actual = body.root_view.get_material_properties().numpy()
                if not np.allclose(actual, properties, atol=1e-7, rtol=0):
                    raise RuntimeError('Native distractor friction readback changed')
                body.permanent_wrench_composer.reset()
                body.permanent_wrench_composer.add_forces_and_torques_index(
                    forces=torch.tensor([[[0., 0., -.98]]], dtype=torch.float32, device=body.device), is_global=True)
                body.write_data_to_sim()
                material[key] = dict(mass_kg=.1, friction=.6, native_readback_ok=True, world_weight_n=[0., 0., -.98])
        forward_scene(self.sim)
        for body in self.bodies.values():
            body.update(self.dt)
        audit_dir = self.output_dir/'multi_object_native'
        audit_dir.mkdir(parents=True, exist_ok=False)
        native = audit_scene(self.sim, audit_dir, ['all', 'object'], roots=tuple(self.paths.values()), fast=True)
        if not native['within_20um']:
            raise RuntimeError('Original native20um gate rejected additional body geometry')
        own = RobotContactReports(self.sim, capacity=POLICY['contact_capacity'], compact=True)
        self.robot_bodies = own.bodies
        plan = common.read(self.args.plan_dir/'plan.json')
        model = PlanningModel(self.args.robot_urdf)
        family = TemplateFamily.load(self.args.package, self.args.hand_urdf, self.args.hand_scene,
                                     plan['gesture_id'])
        self.legal_hand_owners = {model.fk.fixed_attachment(n, np.eye(4))[0] for n in family.hand.links}
        self.contact_bodies = list(self.paths.values())
        self.contact_targets = self.robot_bodies+list(self.paths.values())+own.targets[len(own.bodies):]
        self.contacts = self.sim.physics_sim_view.create_rigid_contact_view(self.contact_bodies,
            filter_patterns=[self.contact_targets]*len(self.contact_bodies),
            max_contact_data_count=POLICY['contact_capacity'])
        # No bin collision shapes are spawned; the unmodified reporter retains
        # the full robot/self/floor audit and all dynamic-body contacts.
        self.robot_contacts = own
        self.setup_audit = dict(native=native, additional_materials=material, entry=entry,
            additional_initial_cold_restores=self.initial_restores,
            legal_hand_rigid_owners=sorted(self.legal_hand_owners),
            body_paths=self.paths, contact_targets=self.contact_targets,
            global_gravity_m_s2=[0., 0., 0.], every_dynamic_body_weight_n=[0., 0., -.98],
            physical_state_writes_during_execution=0,
            entry_semantics='Declared independent cold recreation of measured multi-object snapshot; no claim of continuous observation process')
        for key, body in self.bodies.items():
            self.initial_bottom[key] = source_metrics(self.assets[key].mesh.vertices,
                body.data.root_link_pose_w.torch[0].cpu().numpy())['source_bottom_m']
        common.save(self.output_dir/'MULTI_OBJECT_SETUP.json', self.setup_audit)

    def after_step(self, phase):
        from canondex.scene.contact_readback import contact_data_to_host
        for key, body in self.bodies.items():
            if key != self.target_id:
                body.update(self.dt)
        contacts = common.decode_contacts(self.contact_bodies, self.contact_targets,
            contact_data_to_host(self.contacts, self.dt, POLICY['contact_capacity'], compact=True),
            POLICY['contact_capacity'])
        robot_contacts = self.robot_contacts.read()
        states = {}
        for key, body in self.bodies.items():
            pose = body.data.root_link_pose_w.torch[0].cpu().numpy().copy()
            velocity = body.data.root_link_vel_w.torch[0].cpu().numpy().copy()
            if not np.isfinite(pose).all() or not np.isfinite(velocity).all():
                raise FloatingPointError('Nonfinite dynamic source state: '+key)
            loads = common.object_robot_loads(self.paths[key], contacts, self.robot_bodies, self.legal_hand_owners)
            self.max_illegal_object_force = max(self.max_illegal_object_force, loads['illegal_robot_support_force_n'])
            states[key] = dict(pose_xyzw=pose.tolist(), velocity=velocity.tolist(), **loads,
                              **source_metrics(self.assets[key].mesh.vertices, pose))
        if phase == 'settle':
            self.settled_xy = {key: np.asarray(state['pose_xyzw'][:2]) for key, state in states.items()}
        depth = max([0.]+[-c['separation_m'] for c in contacts+robot_contacts])
        self.max_penetration = max(self.max_penetration, depth)
        self.rows.append(dict(step=self.steps, time_s=self.steps*self.dt, phase=phase, objects=states))
        self.contact_rows.append(dict(step=self.steps, object_contacts=contacts, robot_contacts=robot_contacts))
        if depth > POLICY['early_stop_penetration_m']:
            raise common.PhysicalStop('multi_object_deep_collision_stop')
        pushed = common.preclosure_pushes(states, self.settled_xy, phase)
        if pushed:
            raise common.PhysicalStop('source_object_pushed_before_closure:'+','.join(pushed))

    def finish(self, original, error=None, physical_stop=None):
        ids = [r['object_id'] for r in self.scene['objects']]
        result = evaluate_lift(self.rows, ids, self.dt)
        native = original.get('gates', {}) if original else {}
        gates = {name: bool(native.get(name, False)) for name in ORIGINAL_GATES}
        gates.update(all_dynamic_contacts=self.max_penetration <= POLICY['max_penetration_m'],
            all_dynamic_objects_no_arm_support=self.max_illegal_object_force <= .001,
            author_phase_order=result['phase_order_valid'], no_runtime_error=error is None,
            no_early_physical_stop=physical_stop is None)
        valid = bool(original and original.get('complete') and all(gates.values()) and self.steps == len(self.rows))
        binding = getattr(self, 'collection_binding', {})
        for outcome in result['object_outcomes']:
            outcome['task_success'] = bool(valid and outcome['observed_final_lift_success'])
        success = any(r['task_success'] for r in result['object_outcomes'])
        eligible = bool(valid and binding.get('collection_candidate_sha256') and binding.get('task') == TASK
                        and getattr(self, 'commands_sha256', None))
        result.update(schema='anydex_boya_author_lift_results_v1', task=TASK,
            complete=bool(original and original.get('complete')),
            status='infrastructure_error' if error else ('success' if success else 'physical_failure'),
            task_success=success, label_eligible=eligible, gates=gates, policy=POLICY,
            scene_path=str(self.scene_path), scene_sha256=common.sha(self.scene_path),
            total_physics_steps=self.steps, physics_steps=self.steps, dynamic_object_steps=self.steps*len(self.bodies),
            completed_phases=result['actual_phase_blocks'],
            collection_candidate_sha256=binding.get('collection_candidate_sha256'),
            commands_sha256=getattr(self, 'commands_sha256', None),
            error=error, physical_stop=physical_stop, max_penetration_m=self.max_penetration,
            max_all_object_illegal_support_force_n=self.max_illegal_object_force,
            legacy_target_id=self.scene['target_object_id'], legacy_primary_task_success_used=False,
            state_writes_during_execution=0, bin_geometry_count=0,
            coarse_convex_collision_is_exact_source_mesh=False,
            label_scope='Any original tabletop body: final source bottom >=5mm, final reference rise from last hold >=5cm, concurrent thumb and opposing load each >1mN; complete lift and original seven plus all-body validity gates',
            duration_scope='Pickup run durations are diagnostics; no extra hold window is a task gate',
            support_scope='Original nonhand robot support limit applies to every body; no new external-support-chain rule',
            preclosure_guard_scope='Original >3cm XY push stop applied to every source body, with no GT target association')
        common.save(self.output_dir/'MULTI_OBJECT_TRACE.json', dict(rows=self.rows))
        common.save(self.output_dir/'MULTI_OBJECT_CONTACTS.json', dict(rows=self.contact_rows))
        result['trace_sha256'] = common.sha(self.output_dir/'MULTI_OBJECT_TRACE.json')
        result['contacts_sha256'] = common.sha(self.output_dir/'MULTI_OBJECT_CONTACTS.json')
        common.save(self.output_dir/'LIFT_EVALUATION.json', result)
        return result


def execution_steps(args, scene_path):
    """Caller advances exactly one physics tick for each yielded step event."""
    common._backend_path()
    import execute_template_arm_positive as executor
    from run_template_arm_positive import bind_hand
    bind_hand(args)
    plan = common.read(args.plan_dir/'plan.json')
    binding = plan.get('author_lift')
    if (not binding or binding.get('task') != TASK or binding.get('scene_sha256') != common.sha(scene_path)
            or plan.get('approach_only') is not False or plan.get('author_transport')):
        raise ValueError('Prepared unchanged explicit multi-object lift plan required')
    if getattr(args, 'tactile_feedback', False):
        raise ValueError('Feedback changes the frozen lift trajectory')
    runtime = MultiObjectLiftRuntime(scene_path, args.output_dir,
                                    archive_sources=binding.get('scene_archive_sources'))
    runtime.collection_binding = plan.get('author_collection_binding', {})
    runtime.commands_sha256 = plan['commands_sha256']
    if runtime.collection_binding.get('task') != TASK:
        raise ValueError('Lift candidate binding must declare the lift task')
    with np.load(args.plan_dir/'commands.npz', allow_pickle=False) as data:
        phases = data['phase'].tolist()
    if list(dict.fromkeys(phases)) != POLICY['required_phases']:
        raise ValueError('Complete original lift phase sequence required')
    generator = executor.execution_steps(args, runtime_factory=runtime.factory)
    error = physical_stop = None
    try:
        event = next(generator)
        while event is not None:
            if event['kind'] == 'ready':
                runtime.ready()
                yield dict(event, multi_object_runtime=runtime)
                event = next(generator)
            elif event['kind'] == 'step':
                runtime.before_step()
                before = event['sim'].get_physics_step_count()
                yield event
                if event['sim'].get_physics_step_count() != before+1:
                    raise RuntimeError('Caller must advance the shared physics clock exactly once')
                runtime.steps += 1
                try:
                    following = next(generator)
                except StopIteration:
                    following = None
                except BaseException:
                    runtime.after_step(phases[event['step']-1])
                    raise
                runtime.after_step(phases[event['step']-1])
                event = following
            else:
                raise RuntimeError('Unexpected native execution event')
    except common.PhysicalStop as exception:
        physical_stop = str(exception)
    except BaseException as exception:
        error = repr(exception)
        raise
    finally:
        generator.close()
        original_path = args.output_dir/'evaluation.json'
        original = common.read(original_path) if original_path.exists() else {}
        runtime.finish(original, error, physical_stop)


def run(args, scene_path):
    """Run one prepared plan; caller must finalize collection before app.close."""
    for event in execution_steps(args, scene_path):
        if event['kind'] == 'step':
            event['sim'].step(render=False)
    return common.read(args.output_dir/'LIFT_EVALUATION.json')
