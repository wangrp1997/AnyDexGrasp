"""Opt-in three-body lift reuse through existing native resident hooks.

One SimulationApp belongs to the session. The original collection execute entry
still owns each once-only marker, full commands, journal, validity and record.
Only job boundaries reset state; distinct asset groups rebuild an official
SimulationContext without restarting the application. Import starts no Kit.
"""
from __future__ import annotations

import importlib
import argparse
import json
import os
from pathlib import Path
import threading
import time
import traceback
from types import SimpleNamespace

import numpy as np

import author_collection_batch as B
import author_collection_execute as E

SCHEMA = 'anydex_boya_resident_execution_v1'
_ACTIVE = None
NATIVE_RESIDENT = Path('/home/wangrenpeng/CanonDex/canondex/grasp/arm_resident_scene.py')
OFFICIAL_CONTEXT = Path('/home/wangrenpeng/tendon-spin/.venv/lib/python3.12/site-packages/isaaclab/source/isaaclab/isaaclab/sim/simulation_context.py')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def geometry_binding(scene, plan):
    """Bind all three asset roles; poses and gesture commands are job state."""
    _need(scene.get('task') == B.TASK and scene.get('bin', 'missing') is None
          and len(scene['objects']) == 3, 'Exactly three original bin-free lift bodies required')
    arguments = plan['arguments']
    roles = {name: B.binding(arguments[name])['sha256'] for name in
        ('robot_urdf', 'hand_urdf', 'hand_scene', 'package', 'physics_config', 'diagnostic_config')}
    roots = [str(Path(row['object_root']).resolve())+os.sep for row in scene['objects']]
    repairs = [str(Path(row['collision_refinement']).resolve().parent)+os.sep
               for row in scene['objects'] if row.get('collision_refinement')]
    mesh_inputs = {str(Path(path).resolve()): digest for path, digest in scene.get('input_sha256', {}).items()
                   if any(str(Path(path).resolve()).startswith(root) for root in roots+repairs)}
    B.verify(B.normalized_bindings(mesh_inputs))
    objects = []
    for row in scene['objects']:
        _need(row.get('mass_kg') == .1 and row.get('friction') == .6, 'Frozen .1kg/.6 body material required')
        source = B.binding(row['source_mesh'])
        _need(source['sha256'] == row['source_mesh_sha256'], 'Actual body source mesh changed')
        objects.append(dict(object_id=row['object_id'], object_root=str(Path(row['object_root']).resolve()),
            source_mesh_sha256=source['sha256'], scale=row['scale'], mass_kg=row['mass_kg'], friction=row['friction'],
            collision_refinement_sha256=(B.sha(row['collision_refinement']) if row.get('collision_refinement') else None),
            native_object_urdf_sha256=B.sha(Path(row['object_root'])/'urdf/coacd.urdf')))
    value = dict(objects=objects, target_object_id=scene['target_object_id'], native_roles=roles,
        mesh_inputs=mesh_inputs, dt_s=scene['dt_s'], bin=None)
    _need(value['dt_s'] == .004 and len({r['object_id'] for r in objects}) == 3, 'Frozen timestep and distinct original bodies required')
    return dict(sha256=_digest(value), inputs=value)


def _native_key(binding):
    value = dict(binding['inputs'])
    # Staged arm refinements live in different gesture directories. Content
    # aliases may differ; full body/robot roles above still bind their meaning.
    value['files'] = sorted(set(value['files'].values()))
    value.pop('world_T_object')
    return _digest(value)


def _scene_type():
    """Import the existing plain-Python resident controller after backend setup."""
    from canondex.grasp.arm_resident_scene import ResidentArmScene, restore_declared_object_state

    class BoyaResidentScene(ResidentArmScene):
        def __init__(self, session):
            super().__init__(); self.session = session; self.cached = None; self.current_runtime = None
            self.last_resets = {}; self.reset_physics_steps = 0

        def check(self, binding, *, allow_object_reposition=False):
            if self.binding is not None and _native_key(binding) != _native_key(self.binding):
                raise ValueError('Resident native asset content/physics/reference changed')
            return False

        def accept_placement(self, binding):
            self.check(binding)
            if self.binding != binding:
                self.placement_updates.append(dict(previous_sha256=self.binding['sha256'],
                    current_sha256=binding['sha256'], after_native_audit_count=self.native_audit_count,
                    all_three_declared_poses=[r['world_T_object'] for r in self.session._scene['objects']]))
                self.binding = binding

        def bind(self, *args):
            super().bind(*args)
            self.session.asset_setup_count += 1

        def remember(self, runtime):
            self.cached = dict(bodies=dict(runtime.bodies), assets=dict(runtime.assets), paths=dict(runtime.paths))
            self.current_runtime = runtime

        def reset(self, initial, object_pose):
            sim = self.runtime[0]; before = sim.get_physics_step_count()
            super().reset(initial, object_pose)
            from run_template_arm_positive import object_asset
            from canondex.scene.runtime import forward_scene
            self.last_resets = {}
            for row in self.session._scene['objects']:
                key = row['object_id']
                asset = object_asset(SimpleNamespace(object_root=Path(row['object_root'])),
                    np.asarray(row['world_T_object']), row['scale'])
                self.cached['assets'][key] = asset
                if key != self.session._scene['target_object_id']:
                    self.last_resets[key] = restore_declared_object_state(sim, self.cached['bodies'][key], asset.pose_xyzw)
            # Native executor re-applies the target's original -.98N; the
            # original lift.ready re-applies each other body's original force.
            # No hidden settling, additional sim.reset, or integration is used.
            forward_scene(sim)
            for body in self.cached['bodies'].values():
                body.update(sim.get_physics_dt())
            _need(sim.get_physics_step_count() == before, 'Three-body boundary reset integrated physics')
            self.session.state_reset_count += 1

        def attach(self, runtime, args):
            _need(self.cached is not None, 'Resident all-body handles were not captured')
            runtime.args = args
            runtime.sim, runtime.robot = self.runtime[:2]
            runtime.bodies = dict(self.cached['bodies']); runtime.assets = dict(self.cached['assets'])
            runtime.paths = dict(self.cached['paths']); runtime.dt = runtime.sim.get_physics_dt()
            runtime.target_id = self.session._scene['target_object_id']
            runtime.initial_restores = dict(self.last_resets)
            self.current_runtime = runtime

    return BoyaResidentScene


class ResidentExecutionSession:
    """Serial jobs in one App; caller owns global GPU0 lease and total budget.

    Finish caller ledgers/results before close(): native fast shutdown may exit
    the process. This class neither selects candidates nor retries any job.
    """
    def __init__(self, output_dir, *, source_bindings=None):
        self.output_dir = Path(output_dir).resolve(); self.output_dir.mkdir(parents=True, exist_ok=False)
        self.sources = B.normalized_bindings(source_bindings or {})
        B.verify(self.sources)
        self.sources.update(adapter=B.binding(__file__), streamed_entry=B.binding(E.__file__), batch_helpers=B.binding(B.__file__),
            native_resident=B.binding(NATIVE_RESIDENT), official_context=B.binding(OFFICIAL_CONTEXT))
        B.verify(self.sources)
        self.launcher = self.controller = self._geometry = None
        self.asset_setup_count = self.state_reset_count = self.scene_rebuild_count = 0
        self.kit_start_count = self.deferred_close_count = 0
        self.jobs = []; self.closed = self.poisoned = False
        self.owner_thread = threading.get_ident(); self.started_unix_s = time.time()
        self.kit_launch_wall_s = 0.
        self._scene = self._args = None
        B.save(self.output_dir/'SESSION_STARTED.json', dict(schema=SCHEMA, source_bindings=self.sources,
            pid=os.getpid(), started_unix_s=self.started_unix_s, policy='One App, sequential once-only frozen jobs; no automatic retry'), exclusive=True)

    def _launch(self, common, original_launch):
        if self.launcher is None:
            start = time.monotonic()
            self.launcher = original_launch(common); self.kit_start_count += 1
            self.kit_launch_wall_s += time.monotonic()-start
        common.transport._backend_path()
        if self.controller is None:
            self.controller = _scene_type()(self)
        class DeferredApp:
            def close(_, *args, **kwargs):
                self.deferred_close_count += 1
        return SimpleNamespace(app=DeferredApp())

    def _rebuild_if_needed(self, binding):
        if self._geometry is not None and binding != self._geometry:
            if self.controller is not None and self.controller.runtime is not None:
                import isaaclab.sim as sim_utils
                sim = self.controller.runtime[0]
                sim.stop()
                self.controller.contacts = None; self.controller.cached = None
                self.controller.current_runtime = None; self.controller.runtime = None
                # Installed official API tears down PhysX/render/stage resources
                # and singleton; the application itself remains alive.
                sim_utils.SimulationContext.clear_instance()
                sim_utils.create_new_stage()
                self.controller = None; self.scene_rebuild_count += 1
        self._geometry = binding

    def execute(self, prepared_dir):
        global _ACTIVE
        _need(not self.closed and not self.poisoned, 'Closed or failed resident session cannot dispatch')
        _need(threading.get_ident() == self.owner_thread and _ACTIVE is None,
              'Resident temporary adapters require one owning thread and no nested execution')
        B.verify(self.sources)
        prepared = Path(prepared_dir).resolve()
        for marker in ('EXECUTION_STARTED.json', 'STREAMED_EXECUTION_ENTRY.json', 'RESIDENT_EXECUTION_ENTRY.json'):
            _need(not (prepared/marker).exists(), 'A started resident/native job is never restarted: '+str(prepared/marker))
        lift, pipeline = E._modules()
        ready = B.read(prepared/'READY.json'); plan = B.read(ready['plan']['path'])
        self._scene = pipeline._scene(ready['scene']['path'], plan['author_lift'].get('scene_archive_sources'))
        geometry = geometry_binding(self._scene, plan)
        entry = dict(schema=SCHEMA, status='starting', prepared_dir=str(prepared), ready=B.binding(prepared/'READY.json'),
            scene_id=self._scene['scene_id'], source_bindings=self.sources, geometry_binding=geometry,
            started_unix_s=time.time(), state_writes_scope='All reset writes occur at job boundaries before original initial/native gates')
        B.save(prepared/'RESIDENT_EXECUTION_ENTRY.json', entry, exclusive=True)
        original_launch, original_runtime_class = E._launch_native_gpu0, E.runtime_class
        native = original_native = None
        launched_before = self.kit_start_count; builds_before = self.asset_setup_count; resets_before = self.state_reset_count
        before = time.monotonic(); _ACTIVE = self
        try:
            self._rebuild_if_needed(geometry)
            def runtime_class(module, bindings, journals):
                bindings.update({v['path']:v['sha256'] for v in self.sources.values()})
                streamed, ast_sha = original_runtime_class(module, bindings, journals)
                session = self
                class Runtime(streamed):
                    def factory(inner, *args, **kwargs):
                        start = time.monotonic()
                        result = super().factory(*args, **kwargs)
                        session.controller.remember(inner)
                        entry['native_scene_factory_wall_s'] = time.monotonic()-start
                        entry['physics_clock_after_scene_factory'] = inner.sim.get_physics_step_count()
                        return result
                    def ready(inner):
                        if not inner.bodies:
                            session.controller.attach(inner, session._args)
                        start = time.monotonic()
                        result = super().ready()
                        entry['original_all_body_ready_wall_s'] = time.monotonic()-start
                        entry['physics_clock_before_first_command'] = inner.sim.get_physics_step_count()
                        entry['native_and_all_body_initial_gates_completed'] = True
                        return result
                return Runtime, ast_sha
            def launch(common):
                nonlocal native, original_native
                launcher = self._launch(common, original_launch)
                common.transport._backend_path()
                native = importlib.import_module('execute_template_arm_positive')
                original_native = native.execution_steps
                def execution_steps(args, *, resident_scene=None, runtime_factory=None):
                    _need(resident_scene is None, 'Nested resident scene override forbidden')
                    self._args = args
                    return original_native(args, resident_scene=self.controller, runtime_factory=runtime_factory)
                native.execution_steps = execution_steps
                return launcher
            E.runtime_class, E._launch_native_gpu0 = runtime_class, launch
            result = E.execute(prepared)
            B.verify(self.sources)
            entry.update(status='complete', pipeline_execution=B.binding(prepared/'LIFT_PIPELINE_EXECUTION.json'),
                label=result.get('label'), label_state=result.get('label_state'), physics_steps=result.get('physics_steps'),
                complete=result.get('complete'), original_validity_and_label_logic_unchanged=True)
            return result
        except BaseException:
            self.poisoned = True
            receipt = prepared/'LIFT_PIPELINE_EXECUTION.json'
            persisted = B.read(receipt) if receipt.exists() else {}
            entry.update(status='infrastructure_failure', error=traceback.format_exc(),
                physics_steps=persisted.get('physics_steps'),
                execution_steps_unknown=not persisted or persisted.get('status') == 'starting'
                    or not (prepared/'execution/LIFT_EVALUATION.json').exists())
            raise
        finally:
            if native is not None and original_native is not None:
                native.execution_steps = original_native
            E.runtime_class, E._launch_native_gpu0 = original_runtime_class, original_launch
            _ACTIVE = None
            entry.update(finished_unix_s=time.time(), wall_s=time.monotonic()-before,
                Kit_starts_added=self.kit_start_count-launched_before,
                asset_setups_added=self.asset_setup_count-builds_before,
                boundary_resets_added=self.state_reset_count-resets_before,
                resident_scene=self.controller.receipt() if self.controller is not None else None)
            B.save(prepared/'RESIDENT_EXECUTION_ENTRY.json', entry)
            self.jobs.append(entry)
            self._save('open' if not self.poisoned else 'failed_no_more_dispatch')

    def _save(self, status):
        value = dict(schema=SCHEMA, status=status, source_bindings=self.sources,
            started_unix_s=self.started_unix_s, updated_unix_s=time.time(), Kit_starts=self.kit_start_count,
            asset_setup_count=self.asset_setup_count, state_reset_count=self.state_reset_count,
            scene_rebuild_count=self.scene_rebuild_count, deferred_per_job_close_calls=self.deferred_close_count,
            Kit_launch_wall_s=self.kit_launch_wall_s,
            jobs=self.jobs, known_execution_physics_steps=sum(j.get('physics_steps') or 0 for j in self.jobs),
            execution_starts_with_unknown_steps=sum(bool(j.get('execution_steps_unknown')) for j in self.jobs),
            added_reset_physics_steps=0, candidate_fallback=False, retry_count=0,
            cold_initialization_physics_steps_scope='Original native scene factory unchanged; not included in known command-step sum. Per-job initial clock is recorded when ready.',
            label_scope='Original per-job collection record and independent runtime audit remain authoritative',
            reuse_scope='Same asset group reuses all three native bodies and robot; changed geometry rebuilds a context/stage in the same application',
            physics_device='cpu', renderer_active_gpu=0, renderer_multi_gpu_enabled=False)
        B.save(self.output_dir/'SESSION_RECEIPT.json', value)
        return value

    def close(self, *, exit_code=None):
        if self.closed:
            return B.read(self.output_dir/'SESSION_RECEIPT.json')
        self.closed = True
        code = int(self.poisoned) if exit_code is None else exit_code
        changed = [b['path'] for b in self.sources.values()
                   if not Path(b['path']).is_file() or B.sha(b['path']) != b['sha256']]
        if changed:
            self.poisoned = True; code = 1
        receipt = self._save('closed' if not self.poisoned else 'closed_after_failure')
        receipt.update(changed_sources=changed, exit_code=code)
        B.save(self.output_dir/'SESSION_RECEIPT.json', receipt)
        if self.launcher is not None:
            self.launcher.app.close(exit_code=code)
        return receipt

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.poisoned |= exc_type is not None
        self.close()


def execute_many(prepared_dirs, output_dir, *, source_bindings=None):
    """Thin once-only group entry; persist group results before real App close.

    The caller supplies the original fixed order and inherited resource locks.
    A failed/unknown attempt stops the group; no job is retried or relabelled.
    """
    paths = [str(Path(path).resolve()) for path in prepared_dirs]
    _need(paths and len(set(paths)) == len(paths), 'Distinct ordered prepared directories required')
    session = ResidentExecutionSession(output_dir, source_bindings=source_bindings)
    root = session.output_dir
    result = dict(schema=SCHEMA, status='starting', started_unix_s=time.time(),
        source_bindings=session.sources, ordered_prepared_dirs=paths,
        retry_budget=0, fail_closed=True, jobs=[], execution_steps_unknown=False)
    B.save(root/'EXECUTE_MANY_STARTED.json', result, exclusive=True)
    code = 1; start = time.monotonic()
    try:
        for index, path in enumerate(paths):
            row = dict(prepared_dir=path, status='dispatching', started=None)
            result['jobs'].append(row)
            try:
                receipt = session.execute(path)
                row.update(status='complete', started=True,
                    execution_receipt=B.binding(Path(path)/'LIFT_PIPELINE_EXECUTION.json'),
                    label=receipt.get('label'), label_state=receipt.get('label_state'),
                    physics_steps=receipt.get('physics_steps'), execution_steps_unknown=False)
            except BaseException:
                markers = [name for name in ('RESIDENT_EXECUTION_ENTRY.json', 'STREAMED_EXECUTION_ENTRY.json', 'EXECUTION_STARTED.json')
                           if (Path(path)/name).exists()]
                receipt_path = Path(path)/'LIFT_PIPELINE_EXECUTION.json'
                receipt = B.read(receipt_path) if receipt_path.exists() else {}
                row.update(status='infrastructure_failure', started=bool(markers),
                    existing_start_markers=markers, error=traceback.format_exc(),
                    execution_receipt=B.binding(receipt_path) if receipt_path.exists() else None,
                    physics_steps=receipt.get('physics_steps'),
                    execution_steps_unknown=bool(markers) and (not receipt or receipt.get('status') == 'starting'
                        or not (Path(path)/'execution/LIFT_EVALUATION.json').exists()))
                session.poisoned = True
                result['jobs'].extend(dict(prepared_dir=p, status='not_started', started=False,
                    reason='Prior resident group failure; no retry or fallback', physics_steps=0,
                    execution_steps_unknown=False) for p in paths[index+1:])
                result['status'] = 'infrastructure_failure'
                break
        else:
            result['status'] = 'complete'; code = 0
        result.update(finished_unix_s=time.time(), wall_s=time.monotonic()-start,
            known_execution_physics_steps=sum(r.get('physics_steps') or 0 for r in result['jobs']),
            execution_steps_unknown=any(r.get('execution_steps_unknown', False) for r in result['jobs']))
        B.save(root/'EXECUTE_MANY_RESULTS.json', result)
        return result
    finally:
        # SimulationApp.close may call os._exit: both outer and session receipts
        # are durable before the only real application shutdown.
        session.close(exit_code=code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    run = sub.add_parser('execute-many')
    run.add_argument('--prepared-dirs', type=Path, nargs='+', required=True)
    run.add_argument('--output-dir', type=Path, required=True)
    run.add_argument('--source-bindings', type=Path)
    args = parser.parse_args()
    sources = B.read(args.source_bindings) if args.source_bindings else None
    result = execute_many(args.prepared_dirs, args.output_dir, source_bindings=sources)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
