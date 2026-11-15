"""Opt-in one/two-environment GPU PhysX diagnostic for frozen Boya lift plans.

The existing native generator controls each lane. Official IsaacLab USD cloning
and the existing advance_together barrier provide one Kit/one physical clock.
No simulator import occurs until run_group is explicitly invoked. Old prepared
trials remain read-only; all new artifacts are diagnostic and excluded from fit.
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
CANON = Path('/home/wangrenpeng/CanonDex')
SITE = Path('/home/wangrenpeng/tendon-spin/.venv/lib/python3.12/site-packages')
SCHEMA = 'anydex_boya_author_gpu_vector_execution_v1'
LIFT_SHA256 = '40e519d035c45bd25289fe5dcb154808baef1153e82c62e0cf69b5110b47b20d'
KIT_ARGS = '--/log/file=/dev/null --/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false'


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _modules():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import author_collection_execute as storage
    import author_lift as lift
    import author_lift_pipeline as pipeline
    import author_resident_execution as resident
    return storage, lift, pipeline, resident


def validate_layout(env_roots, origins_m, count):
    roots = list(env_roots)
    origins = np.asarray(origins_m, dtype=float)
    _need(type(count) is int and count in (1, 2), 'Only explicit one/two-environment diagnostics are supported')
    _need(len(roots) == count and len(set(roots)) == count, 'One distinct namespace per environment required')
    _need(all(re.fullmatch(r'/World/(?:[A-Za-z_][A-Za-z_0-9]*/)+env_'+str(i), root)
              for i, root in enumerate(roots)), 'Explicit sibling USD namespaces ending env_0/env_1 required')
    parent = str(Path(roots[0]).parent)
    _need(all(str(Path(r).parent) == parent for r in roots), 'Environment namespaces must be siblings')
    _need(origins.shape == (count, 3) and np.isfinite(origins).all()
          and np.array_equal(origins[0], [0., 0., 0.]) and np.all(origins[:, 2] == 0.),
          'Finite XY translations with prototype at original origin required')
    _need(count == 1 or np.linalg.norm(origins[1, :2]) >= 4., 'Frozen four-metre environment separation required')
    return roots, origins


def load_jobs(prepared_dirs):
    """Perform original ready/input/geometry/candidate checks without a launch."""
    storage, lift, pipeline, resident = _modules()
    T = lift.common
    jobs = []
    for source in prepared_dirs:
        source = Path(source).resolve()
        ready = T.read(source/'READY.json')
        prep = T.read(source/'LIFT_PIPELINE_RESULTS.json')
        _need(ready.get('schema') == pipeline.SCHEMA and ready.get('task') == lift.TASK
              and ready.get('status') == 'ready' and prep.get('status') == 'ready'
              and prep.get('ready_sha256') == T.sha(source/'READY.json'), 'Unchanged ready lift preparation required')
        protocol = T.read(ready['protocol']['path'])
        _need(protocol.get('schema') == pipeline.SCHEMA and protocol.get('task') == lift.TASK
              and protocol.get('bin', 'missing') is None, 'Explicit original bin-free lift protocol required')
        pipeline._verify_protocol(protocol)
        bindings = {key: ready[key] for key in ('protocol','scene','candidate','plan','commands','command_binding')}
        bindings['ready'] = dict(path=str(source/'READY.json'), sha256=T.sha(source/'READY.json'))
        bindings['preparation'] = dict(path=str(source/'LIFT_PIPELINE_RESULTS.json'), sha256=T.sha(source/'LIFT_PIPELINE_RESULTS.json'))
        pipeline.common._verify(bindings)
        scene = pipeline._scene(ready['scene']['path'], protocol.get('scene_archive_sources'))
        plan = T.read(ready['plan']['path']); candidate = T.read(ready['candidate']['path'])
        command_binding = T.read(ready['command_binding']['path'])
        pipeline._candidate(candidate)
        pipeline.common._verify_plan_dependencies(plan)
        pipeline._verify_plan_artifacts(Path(ready['plan']['path']).parent, plan)
        ids = [r['object_id'] for r in scene['objects']]
        _need(len(ids) == 3 and plan.get('task') == lift.TASK and plan.get('admitted')
              and plan.get('approach_only') is False and not plan.get('author_transport')
              and candidate['candidate_sha256'] == ready['collection_candidate_sha256']
              == plan['author_collection_binding']['collection_candidate_sha256']
              and plan['commands_sha256'] == ready['commands']['sha256'] == command_binding['commands_sha256']
              and command_binding['execution_spec'].get('task') == lift.TASK
              and command_binding['execution_spec'].get('original_object_ids') == ids,
              'Frozen candidate/commands/all-three-body binding differs')
        arrays = pipeline._arrays(Path(ready['plan']['path']).parent)
        _need(list(dict.fromkeys(arrays['phase'].tolist())) == lift.POLICY['required_phases'], 'Complete frozen lift phases required')
        jobs.append(dict(source_prepared=str(source), ready=ready, bindings=bindings, protocol=protocol,
            scene=scene, plan=plan, candidate=candidate, command_binding=command_binding,
            command_steps=len(arrays['phase']), geometry=resident.geometry_binding(scene, plan)))
    _need(len(jobs) in (1, 2) and len({j['source_prepared'] for j in jobs}) == len(jobs), 'One/two distinct frozen source preparations required')
    _need(all(j['geometry'] == jobs[0]['geometry'] and j['ready']['scene']['sha256'] == jobs[0]['ready']['scene']['sha256']
              for j in jobs), 'One identical three-body measured scene/physical asset cohort required')
    return jobs


def verify_jobs(jobs):
    _, lift, pipeline, _ = _modules()
    for job in jobs:
        pipeline._verify_protocol(job['protocol']); pipeline.common._verify(job['bindings'])
        pipeline.common._verify_plan_dependencies(job['plan'])
        pipeline._verify_plan_artifacts(Path(job['ready']['plan']['path']).parent, job['plan'])


def bind_reference_assets(jobs):
    """Use the native binding before physics imports capture module defaults."""
    storage, lift, _, _ = _modules()
    lift.common._backend_path()
    keys = ('hand_urdf', 'hand_scene', 'reference_root')
    paths = {key: Path(jobs[0]['plan']['arguments'][key]).resolve() for key in keys}
    _need(all(Path(job['plan']['arguments'][key]).resolve() == paths[key]
              for job in jobs for key in keys), 'Shared native reference paths must agree')
    _need(paths['hand_urdf'].is_file() and paths['hand_scene'].is_file()
          and paths['reference_root'].is_dir(), 'Bound hand URDF/scene and reference directory must exist before Kit')
    from run_template_arm_positive import bind_hand
    bind_hand(SimpleNamespace(**paths))
    # These modules use import-time constants. A caller with stale imports must
    # start a fresh process; silently editing captured defaults is unnecessary.
    for name, attribute in (('canondex.robots.hand_actuation', 'HAND_SCENE'),
                            ('canondex.robots.ur16e_boya', 'SCENE_XML')):
        module = sys.modules.get(name)
        if module is not None:
            _need(Path(getattr(module, attribute)).resolve() == paths['hand_scene'],
                  'Stale native hand reference was imported before binding: '+name)
    return dict(hand_urdf=dict(path=str(paths['hand_urdf']), sha256=storage.sha(paths['hand_urdf'])),
                hand_scene=dict(path=str(paths['hand_scene']), sha256=storage.sha(paths['hand_scene'])),
                reference_root=str(paths['reference_root']))


def source_bindings():
    storage, lift, pipeline, resident = _modules()
    lift.common._backend_path()
    paths = [Path(__file__).resolve(), Path(storage.__file__), Path(lift.__file__), Path(pipeline.__file__),
        Path(resident.__file__), CANON/'canondex/grasp/arm_parallel.py',
        CANON/'canondex/grasp/template_gripper_batch.py', CANON/'canondex/grasp/arm_resident_scene.py',
        CANON/'canondex/scene/runtime.py', CANON/'canondex/scene/contact_readback.py',
        CANON/'canondex/scene/execute_motion.py', CANON/'canondex/scene/object_contacts.py',
        CANON/'scripts/research/execute_template_arm_positive.py',
        SITE/'isaaclab/source/isaaclab/isaaclab/cloner/usd.py',
        SITE/'isaaclab/source/isaaclab/isaaclab/cloner/collision_filter.py',
        SITE/'isaaclab_physx/physics/physx_manager.py', SITE/'isaaclab_physx/physics/physx_manager_cfg.py',
        HERE/'author_gpu_capability_probe.py']
    return {str(p.resolve()): storage.sha(p) for p in paths}


def verify_sources(bindings):
    storage, _, _, _ = _modules()
    for name, digest in bindings.items():
        _need(Path(name).is_file() and storage.sha(name) == digest, 'GPU diagnostic dependency changed: '+name)


def shared_clock(streams, step, *, on_step=None):
    """Reuse the original barrier; only ready metadata carries the common bound.

    No generator receives a new command or extra iteration. A completed lane's
    world continues integrating until the longest lane ends; that tail is cost,
    not another command, label, freeze, reset, or grasp trial.
    """
    from canondex.grasp.arm_parallel import advance_together
    ready = [next(stream) for stream in streams]
    _need(all(e['kind'] == 'ready' for e in ready), 'Every lane must pass native/all-body ready before integration')
    lengths = [int(e['steps']) for e in ready]
    bound = max(lengths)
    def replay(event, stream):
        yield dict(event, steps=bound)
        yield from stream
    wrapped = [replay(event, stream) for event, stream in zip(ready, streams)]
    result = advance_together(wrapped, step, on_step=on_step)
    result.update(original_command_steps=lengths, scheduler_barrier_upper_bound=bound,
        barrier_metadata_only=True, commands_added=0,
        completed_lane_tail_integrated_steps=[result['physics_ticks']-n for n in result['lane_steps']],
        actual_integrated_environment_steps=result['physics_ticks']*len(streams))
    return result


def adapted_methods(lift):
    """Two coordinate/name substitutions, with every original gate/loop intact."""
    import hashlib
    _need(hashlib.sha256(Path(lift.__file__).read_bytes()).hexdigest() == LIFT_SHA256, 'Frozen lift source changed')
    tree = ast.parse(Path(lift.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiObjectLiftRuntime')
    methods = {}; edits = dict(robot_namespace=0, desired_origin=0, observed_origin=0)
    for name in ('ready', 'after_step'):
        fn = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))
        for node in ast.walk(fn):
            if (name == 'ready' and isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'RobotContactReports'):
                _need(not any(k.arg == 'robot_paths' for k in node.keywords), 'Unexpected existing namespace override')
                node.keywords.append(ast.keyword(arg='robot_paths', value=ast.parse('[self.robot_path]', mode='eval').body))
                edits['robot_namespace'] += 1
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
                if name == 'ready' and target == 'desired':
                    node.value = ast.Call(func=ast.parse('self.physical_pose', mode='eval').body, args=[node.value], keywords=[])
                    edits['desired_origin'] += 1
                elif name == 'after_step' and target == 'pose':
                    node.value = ast.Call(func=ast.parse('self.local_pose', mode='eval').body, args=[node.value], keywords=[])
                    edits['observed_origin'] += 1
        namespace = dict(vars(lift))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(lift.__file__), 'exec'), namespace)
        methods[name] = namespace[name]
    _need(edits == dict(robot_namespace=1, desired_origin=1, observed_origin=1), 'Unexpected frozen lift AST structure')
    return methods, edits


def _normalize_relationship(path, root):
    value = str(path)
    return '{ENV}'+value[len(root):] if value == root or value.startswith(root+'/') else value


class TailRecorder:
    """Observe continued integration without changing the saved trial endpoint."""
    def __init__(self, runtime, source_bindings):
        storage, _, _, _ = _modules()
        self.runtime = copy.copy(runtime)
        self.runtime.rows = []; self.runtime.steps = 0
        self.runtime.max_penetration = self.runtime.max_illegal_object_force = 0.
        self.runtime.contact_rows = storage.ContactJournal(runtime.output_dir/'TAIL_CONTACTS.jsonl.gz', source_bindings=source_bindings)
        self.physical_stop_diagnostics = []; self.first_shared_tick = None

    def before_step(self):
        # Reapply cached PD/velocity targets and the same permanent weight.
        # No set_position/set_velocity or state-write API is called here.
        self.runtime.robot.write_data_to_sim()
        for body in self.runtime.bodies.values(): body.write_data_to_sim()

    def after_step(self, shared_tick):
        _, lift, _, _ = _modules()
        runtime = self.runtime
        runtime.robot.update(runtime.dt); runtime.bodies[runtime.target_id].update(runtime.dt)
        runtime.steps += 1
        if self.first_shared_tick is None: self.first_shared_tick = shared_tick
        try:runtime.after_step('tail_after_original_completion')
        except lift.common.PhysicalStop as error:
            self.physical_stop_diagnostics.append(dict(shared_tick=shared_tick, reason=str(error)))
        runtime.rows[-1]['shared_physics_tick'] = shared_tick

    def finish(self):
        storage, _, _, _ = _modules()
        self.runtime.contact_rows.close()
        trace = self.runtime.output_dir/'TAIL_TRACE.json'; contacts = self.runtime.output_dir/'TAIL_CONTACTS.json'
        storage.save(trace, dict(rows=self.runtime.rows, label_scope='No original endpoint or label is changed'))
        storage.save(contacts, self.runtime.contact_rows.pointer())
        return dict(steps=self.runtime.steps, first_shared_tick=self.first_shared_tick,
            body_weight_write_count=self.runtime.steps*len(self.runtime.bodies),
            new_commands=0, state_writes=0, original_completion_evaluation_not_modified=True,
            max_penetration_m=self.runtime.max_penetration,
            max_illegal_support_force_n=self.runtime.max_illegal_object_force,
            physical_stop_diagnostics=self.physical_stop_diagnostics,
            trace=dict(path=str(trace), sha256=storage.sha(trace)),
            contacts=dict(path=str(contacts), sha256=storage.sha(contacts)))


class VectorScene:
    """One prepared prototype, official USD clones, and lane-local native views."""
    def __init__(self, roots, origins, jobs, output_dir):
        self.roots, self.origins, self.jobs, self.output_dir = roots, origins, jobs, Path(output_dir)
        self.lanes = []; self.sim = None; self.cross = []
        self.audit = dict(asset_setup_count=0, physics_scene_count=1, cross_environment_contacts=0,
                          actual_gpu_validated=False, clone_api='isaaclab.cloner.usd_replicate/filter_collisions')

    def _clone(self, runtime, sim, robot, objects):
        import torch
        from pxr import Usd, UsdGeom, UsdPhysics
        from isaaclab.assets import Articulation, RigidObject
        from isaaclab.cloner import usd_replicate, filter_collisions
        stage = sim.stage; first = self.roots[0]
        UsdGeom.Xform.Define(stage, str(Path(first).parent)); UsdGeom.Xform.Define(stage, first)
        for name in ('Robot', 'Object', 'AuthorScene'):
            source = '/World/'+name
            _need(bool(stage.GetPrimAtPath(source)), 'Missing source clone role: '+source)
            editor = Usd.NamespaceEditor(stage); editor.MovePrimAtPath(source, first+'/'+name)
            _need(bool(editor.CanApplyEdits()) and editor.ApplyEdits() is not False, 'Cannot namespace original native asset: '+source)
        expected = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(first), Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
                targets = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
                _need(all(bool(stage.GetPrimAtPath(t)) and
                          (str(t).startswith(first+'/') or str(t).startswith('/World/ground/')) for t in targets),
                      'Namespaced structural exclusion is dangling or leaves its physical environment')
                expected.extend((_normalize_relationship(prim.GetPath(), first), _normalize_relationship(t, first)) for t in targets)
        usd_replicate(stage, [first], [str(Path(first).parent)+'/env_{}'],
            torch.arange(len(self.roots)), positions=torch.as_tensor(self.origins, dtype=torch.float32))
        scenes = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        _need(len(scenes) == 1, 'Exactly one physical scene required')
        filter_collisions(stage, scenes[0], '/World/GPUVectorCollisionGroups', self.roots, global_paths=['/World/ground'])
        prototype = dict(runtime.bodies); prototype[runtime.target_id] = objects['object']
        paths = {key: first+('/Object' if key == runtime.target_id else '/AuthorScene/'+key) for key in prototype}
        robot.cfg.prim_path = first+'/Robot'
        for key, body in prototype.items(): body.cfg.prim_path = paths[key]
        for i, root in enumerate(self.roots):
            actual = []
            for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
                if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
                    targets = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
                    _need(all(bool(stage.GetPrimAtPath(t)) and
                              (str(t).startswith(root+'/') or str(t).startswith('/World/ground/')) for t in targets),
                          'Cloned structural exclusion is dangling or crosses environments')
                    actual.extend((_normalize_relationship(prim.GetPath(), root), _normalize_relationship(t, root)) for t in targets)
            _need(sorted(actual) == sorted(expected), 'Clone changed native structural collision exclusions')
            if i == 0:
                lane_robot, bodies = robot, dict(prototype)
            else:
                cfg = robot.cfg.copy(); cfg.prim_path = root+'/Robot'; cfg.spawn = None
                lane_robot = Articulation(cfg); bodies = {}
                for key, body in prototype.items():
                    cfg = body.cfg.copy(); cfg.prim_path = paths[key].replace(first, root, 1); cfg.spawn = None
                    bodies[key] = RigidObject(cfg)
            local = dict(objects, object=bodies[runtime.target_id], robot_path=root+'/Robot',
                         object_path=root+'/Object', environment_origin_m=self.origins[i].tolist())
            self.lanes.append(dict(robot=lane_robot, bodies=bodies, assets=dict(runtime.assets),
                paths={key:path.replace(first, root, 1) for key,path in paths.items()}, objects=local))
        self.audit.update(asset_setup_count=1, environment_roots=self.roots,
            origins_m=self.origins.tolist(), structural_exclusions_per_env=len(expected))

    def factory(self, runtime, args, initial, asset, build, *, prototype_factory):
        from canondex.scene.runtime import snapshot, forward_scene
        from canondex.grasp.arm_resident_scene import ResidentArmScene, restore_declared_object_state
        import torch
        index = runtime.env_index
        if self.sim is None:
            _need(index == 0, 'Prototype lane must initialize first')
            def clone_build(before_reset):
                def setup(sim, robot, objects):
                    before_reset(sim, robot, objects); self._clone(runtime, sim, robot, objects)
                return build(setup)
            result = prototype_factory(runtime, args, initial, asset, clone_build)
            self.sim, _, self.urdf, _ = result
            self.root_pose = snapshot(self.lanes[0]['robot'])['body_pose'][0]
            self.lanes[0]['assets'] = dict(runtime.assets)
            for lane in self.lanes[1:]: lane['assets'] = dict(runtime.assets)
        lane = self.lanes[index]
        runtime.args = args; runtime.sim = self.sim; runtime.robot = lane['robot']
        runtime.target_id = runtime.scene['target_object_id']; runtime.dt = self.sim.get_physics_dt()
        runtime.bodies, runtime.assets, runtime.paths = dict(lane['bodies']), dict(lane['assets']), dict(lane['paths'])
        runtime.assets[runtime.target_id] = asset; runtime.robot_path = self.roots[index]+'/Robot'
        runtime.origin = self.origins[index].copy(); runtime.initial_restores = {}
        runtime.assets[runtime.target_id] = asset
        if index:
            target = args.output_dir/'ur16e_boya.urdf'; shutil.copyfile(self.urdf, target)
            controller = ResidentArmScene()
            root_pose = self.root_pose.copy(); root_pose[:3] += runtime.origin
            controller.bind({}, self.sim, runtime.robot, target, lane['objects'], root_pose)
            physical = np.asarray(asset.pose_xyzw).copy(); physical[:3] += runtime.origin
            before = self.sim.get_physics_step_count(); controller.reset(initial, physical)
            for key, body in runtime.bodies.items():
                if key != runtime.target_id:
                    pose = np.asarray(runtime.assets[key].pose_xyzw).copy(); pose[:3] += runtime.origin
                    runtime.initial_restores[key] = restore_declared_object_state(self.sim, body, pose)
            _need(self.sim.get_physics_step_count() == before, 'Environment initialization integrated physics')
        else:
            target = self.urdf
        for body in runtime.bodies.values(): body.update(runtime.dt)
        _need(runtime.robot.num_instances == 1 and runtime.robot.is_fixed_base
              and all(b.num_instances == 1 for b in runtime.bodies.values()), 'Lane native asset views are not independent single instances')
        return self.sim, runtime.robot, target, dict(lane['objects'])

    def prepare_guard(self):
        from pxr import Usd, UsdPhysics
        scene = self.sim.stage.GetPrimAtPath(self.sim.cfg.physics_prim_path)
        attrs = {key:scene.GetAttribute(key).Get() for key in ('physxScene:enableGPUDynamics','physxScene:broadphaseType','physxScene:enableCCD','physxScene:timeStepsPerSecond')}
        _need(attrs['physxScene:enableGPUDynamics'] is True and str(attrs['physxScene:broadphaseType']) == 'GPU'
              and attrs['physxScene:enableCCD'] is False and attrs['physxScene:timeStepsPerSecond'] == 250,
              'Requested unchanged-timestep GPU PhysX backend is not active')
        bodies = [[str(p.GetPath()) for p in Usd.PrimRange(self.sim.stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies())
                   if p.HasAPI(UsdPhysics.RigidBodyAPI)] for root in self.roots]
        if len(bodies) == 2:
            self.cross = [self.sim.physics_sim_view.create_rigid_contact_view(bodies[0],
                filter_patterns=[bodies[1]]*len(bodies[0]), max_contact_data_count=262144)]
        self.audit.update(native_scene=attrs, rigid_bodies_per_env=list(map(len, bodies)), actual_gpu_validated=True)

    def check_cross(self):
        from canondex.scene.contact_readback import contact_data_to_host
        for view in self.cross:
            data = contact_data_to_host(view, .004, 262144, compact=True)
            count = sum(map(int, np.asarray(data[4]).flat))
            self.audit['cross_environment_contacts'] = max(self.audit['cross_environment_contacts'], count)
            _need(count == 0, 'GPU environment collision isolation failed')


def run_group(prepared_dirs, output_dir, *, env_roots, origins_m, device='cuda:0', protocol_path=None):
    """Explicit bounded diagnostic launch; never writes a source prepared trial."""
    _need(device == 'cuda:0', 'This diagnostic is explicitly restricted to GPU0')
    jobs = load_jobs(prepared_dirs)
    roots, origins = validate_layout(env_roots, origins_m, len(jobs))
    storage, lift, pipeline, _ = _modules()
    sources = source_bindings(); verify_sources(sources)
    out = Path(output_dir).resolve()
    _need(not any(out == Path(j['source_prepared']) or Path(j['source_prepared']) in out.parents for j in jobs),
          'New GPU output must be outside every read-only source preparation')
    out.mkdir(parents=True, exist_ok=False)
    protocol_binding = None if protocol_path is None else dict(path=str(Path(protocol_path).resolve()), sha256=storage.sha(protocol_path))
    receipt = dict(schema=SCHEMA, status='starting', complete=False, device=device, exclude_training=True,
        formal_training_eligible=False, label=None, automatic_training=False, retry_count=0,
        source_sha256=sources, protocol=protocol_binding, environment_roots=roots, origins_m=origins.tolist(),
        jobs=[], physical_clock_steps=0, recorded_command_environment_steps=0, actual_integrated_environment_steps=0,
        labels_scope='Diagnostic raw labels at each original command completion, never admitted to training',
        physics_scope='Explicit CPU-to-GPU PhysX backend change; preserved dt, source assets, commands, force/control and original validity/label logic; numerical equivalence is not assumed')
    storage.save(out/'GPU_VECTOR_STARTED.json', receipt)
    journals=[]; streams=[]; tails={}; launcher=None; begin=time.monotonic(); exit_code=1
    scene = VectorScene(roots, origins, jobs, out)
    try:
        receipt['reference_assets'] = bind_reference_assets(jobs)
        from isaaclab.app import AppLauncher
        mark=time.monotonic(); launcher=AppLauncher(headless=True,device=device,visualizer=None,kit_args=KIT_ARGS)
        receipt['Kit_launch_wall_s']=time.monotonic()-mark
        from canondex.grasp.physics import load_physics_config
        from author_gpu_capability_probe import probe_live
        methods, edits = adapted_methods(lift)
        base, finish_ast = storage.runtime_class(lift, sources, journals)
        class Runtime(base):
            ready = methods['ready']; after_step = methods['after_step']
            def factory(self, *args): return scene.factory(self, *args, prototype_factory=base.factory)
            def finish(self, *args, **kwargs):
                try:return super().finish(*args, **kwargs)
                finally:self.execution_finished=True
            def physical_pose(self, value):
                result=np.asarray(value).copy();result[:3]+=self.origin;return result
            def local_pose(self, value):
                result=np.asarray(value).copy();result[:3]-=self.origin;return result
        receipt.update(ast_adaptations=edits, finish_ast_sha256=finish_ast)
        for index, job in enumerate(jobs):
            lane = out/f'env_{index}'; lane.mkdir(); execution=lane/'execution';execution.mkdir()
            args=lift.args_for_plan(Path(job['ready']['plan']['path']).parent, execution);args.physics_device=device
            _need(load_physics_config(args.physics_config).scene_cfg().enable_ccd is False,
                  'Frozen CPU configuration requests CCD unsupported by GPU; no silent physics change permitted')
            runtime=Runtime(job['ready']['scene']['path'], execution,
                archive_sources=job['plan']['author_lift'].get('scene_archive_sources'))
            runtime.env_index=index;runtime.collection_binding=job['plan']['author_collection_binding'];runtime.commands_sha256=job['plan']['commands_sha256']
            runtime.execution_finished=False
            # The frozen wrapper loop is compiled with only its runtime constructor
            # dependency supplied, preserving command/observation/finish ordering.
            fn=next(n for n in ast.parse(Path(lift.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='execution_steps')
            namespace=dict(vars(lift));namespace['MultiObjectLiftRuntime']=lambda *a,_r=runtime,**kw:_r
            exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(lift.__file__),'exec'),namespace)
            streams.append(namespace['execution_steps'](args, job['ready']['scene']['path']))
            job.update(output=str(lane),runtime=runtime)
        ready_events=[]
        for stream in streams:ready_events.append(next(stream))
        scene.prepare_guard()
        views={f'env_{i}_objects':(j['runtime'].contacts,'filtered') for i,j in enumerate(jobs)}
        views.update({f'env_{i}_robot':(j['runtime'].robot_contacts.view,'filtered') for i,j in enumerate(jobs)})
        receipt['initial_gpu_probe']=probe_live(scene.sim,out/'initial_gpu_probe',contact_views=views,
            rigid_views={f'env_{i}_{k}':b.root_view for i,j in enumerate(jobs) for k,b in j['runtime'].bodies.items()},
            articulation_views={f'env_{i}':j['runtime'].robot.root_view for i,j in enumerate(jobs)})
        _need(receipt['initial_gpu_probe']['status'] == 'live_api_snapshot_pass', 'GPU full contact/state readback probe failed; no aggregate or CPU fallback')
        def replay(event, stream):
            yield event
            yield from stream
        def step(sim):
            for i,job in enumerate(jobs):
                if job['runtime'].execution_finished:
                    if i not in tails:tails[i]=TailRecorder(job['runtime'], sources)
                    tails[i].before_step()
            sim.step(render=False)
            receipt['physical_clock_steps']+=1
            receipt['actual_integrated_environment_steps']+=len(jobs)
        def after_tick(tick, counts):
            receipt['lane_recorded_physical_steps']=counts
            receipt['recorded_command_environment_steps']=sum(counts)
            for tail in tails.values():tail.after_step(tick)
            scene.check_cross()
            if tick == 1:
                receipt['first_tick_gpu_probe']=probe_live(scene.sim,out/'first_tick_gpu_probe',contact_views=views)
                _need(receipt['first_tick_gpu_probe']['status'] == 'live_api_snapshot_pass', 'Post-step GPU full contact readback failed')
            if tick%100==0:
                storage.save(out/'GPU_VECTOR_PROGRESS.json',dict(physics_ticks=tick,lane_recorded_steps=counts,wall_s=time.monotonic()-begin))
        clock=shared_clock([replay(e,s) for e,s in zip(ready_events,streams)],step,on_step=after_tick)
        verify_jobs(jobs);verify_sources(sources)
        if protocol_binding:_need(storage.sha(protocol_binding['path'])==protocol_binding['sha256'],'Top GPU protocol changed')
        receipt.update(clock=clock, recorded_command_environment_steps=clock['recorded_environment_steps'],
            status='complete', complete=clock['all_streams_complete'], native_scene=scene.audit)
        for index,job in enumerate(jobs):
            evaluation_path=Path(job['output'])/'execution/LIFT_EVALUATION.json'
            record=pipeline.collector.collect_execution(job['candidate'],job['command_binding'],evaluation_path)
            evaluation=lift.common.read(evaluation_path)
            diagnostic=dict(record, diagnostic_raw_label=record['label'], label=None, formal_training_eligible=False,
                            exclude_training=True, original_collection_record_not_replaced=True)
            storage.save(Path(job['output'])/'DIAGNOSTIC_RECORD.json', diagnostic)
            receipt['jobs'].append(dict(environment=index,environment_root=roots[index],origin_m=origins[index].tolist(),
                source_prepared=job['source_prepared'], input_bindings=job['bindings'], command_steps=job['command_steps'],
                completion_tick=evaluation['total_physics_steps'],tail_integrated_steps=clock['completed_lane_tail_integrated_steps'][index],
                evaluation=dict(path=str(evaluation_path),sha256=storage.sha(evaluation_path)),
                diagnostic_record=dict(path=str(Path(job['output'])/'DIAGNOSTIC_RECORD.json'),sha256=storage.sha(Path(job['output'])/'DIAGNOSTIC_RECORD.json')),
                complete=evaluation['complete'],raw_label=record['label'],label=None,label_state=record['label_state'],
                gates=evaluation['gates'],formal_training_eligible=False,exclude_training=True,
                trace_frame='Original plan world frame', contact_position_frame='Physical translated environment world',
                source_pose_translation_removed_m=origins[index].tolist()))
        exit_code=0
        return receipt
    except BaseException:
        receipt.update(status='infrastructure_failure',error=traceback.format_exc(),native_scene=scene.audit)
        raise
    finally:
        for stream in streams:
            try:stream.close()
            except BaseException:receipt.setdefault('stream_finalize_errors',[]).append(traceback.format_exc())
        for journal in journals:
            try:journal.close()
            except BaseException:receipt.setdefault('journal_finalize_errors',[]).append(traceback.format_exc());exit_code=1
        receipt['tail_observations']={}
        for i,tail in tails.items():
            try:receipt['tail_observations'][str(i)]=tail.finish()
            except BaseException:receipt.setdefault('tail_finalize_errors',[]).append(traceback.format_exc());exit_code=1
        if not receipt['jobs']:
            for i,job in enumerate(jobs):
                runtime=job.get('runtime');evaluation=Path(job.get('output',out/f'env_{i}'))/'execution/LIFT_EVALUATION.json'
                receipt['jobs'].append(dict(environment=i,source_prepared=job['source_prepared'],
                    input_bindings=job['bindings'],command_steps=job['command_steps'],
                    recorded_physical_steps=(receipt.get('lane_recorded_physical_steps') or [0]*len(jobs))[i],
                    observations_completed_steps=runtime.steps if runtime is not None else 0,
                    evaluation=(dict(path=str(evaluation),sha256=storage.sha(evaluation)) if evaluation.exists() else None),
                    label=None,raw_label=None,formal_training_eligible=False,exclude_training=True))
        if any(receipt.get(k) for k in ('stream_finalize_errors','journal_finalize_errors','tail_finalize_errors')):
            receipt.update(status='infrastructure_failure',complete=False)
            exit_code=1
        receipt.update(wall_before_close_s=time.monotonic()-begin, contact_journals=[j.pointer() for j in journals if j.closed])
        storage.save(out/'GPU_VECTOR_EXECUTION.json',receipt)
        if launcher is not None:launcher.app.close(exit_code=exit_code)


def main():
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='stage',required=True)
    run=subs.add_parser('run');run.add_argument('--prepared-dirs',type=Path,nargs='+',required=True)
    run.add_argument('--output-dir',type=Path,required=True);run.add_argument('--device',choices=['cuda:0'],required=True)
    run.add_argument('--env-roots',nargs='+',required=True);run.add_argument('--origins-json',required=True)
    run.add_argument('--protocol',type=Path)
    a=parser.parse_args()
    run_group(a.prepared_dirs,a.output_dir,env_roots=a.env_roots,origins_m=json.loads(a.origins_json),device=a.device,protocol_path=a.protocol)


if __name__=='__main__':main()
