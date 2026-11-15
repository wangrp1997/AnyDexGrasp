"""Opt-in GPU 1/4/8-lane and CPU single-lane frozen-template prefix benchmark.

Reuses the frozen vector runner, official cloning, original control/barrier and
full pointwise journals. This entry launches only when explicitly invoked.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import time

import numpy as np

import author_gpu_vector_profile as P

V = P.V
SCHEMA = 'anydex_boya_author_scaling_execution_v1'
PROFILE_SHA256 = 'ea3c7ef466ac1978d49a63a7035e9ba483cdb9f24693f06b5b278c249c6b4f1b'
PROBE_SHA256 = 'b8aca3d387b8589aefe6001511b5a01b7fb4d799cd70f2b5ba531f7b1f42d67f'


def layout(lanes):
    V._need(type(lanes) is int and lanes in (1, 4, 8), 'Explicit lane count must be 1, 4 or 8')
    return ['/World/envs/env_'+str(i) for i in range(lanes)], [[4.*i, 0., 0.] for i in range(lanes)]


def validate_layout(roots, origins_m, count):
    expected_roots, expected_origins = layout(count)
    origins = np.asarray(origins_m, dtype=float)
    V._need(list(roots) == expected_roots and origins.shape == (count, 3)
            and np.array_equal(origins, expected_origins), 'Frozen distinct namespaces and four-metre XY layout required')
    return list(roots), origins


def backend_attributes(attrs, ordinal, device):
    """Require the requested actual PhysX backend, never a CPU/GPU fallback."""
    gpu = device == 'cuda:0'
    V._need(device in ('cuda:0', 'cpu')
            and attrs['physxScene:enableGPUDynamics'] is gpu
            and str(attrs['physxScene:broadphaseType']) == ('GPU' if gpu else 'MBP')
            and attrs['physxScene:enableCCD'] is False
            and attrs['physxScene:timeStepsPerSecond'] == 250
            and ((ordinal == 0) if gpu else (ordinal == -1)),
            'Requested 250Hz PhysX backend/device attributes are not active')


def repeated_jobs(prepared_dirs, lanes, original_loader):
    """Validate the real template once; all lane manifests remain independent."""
    paths = list(prepared_dirs)
    V._need(len(paths) == lanes and len({str(Path(p).resolve()) for p in paths}) == 1,
            'All benchmark lanes must explicitly use one identical read-only template')
    validated = original_loader(paths[:1])
    V._need(len(validated) == 1, 'One completely validated source preparation required')
    return [copy.deepcopy(validated[0]) for _ in range(lanes)]


def compile_entries():
    """Two source-pinned adaptations: admit CPU; forward the chosen device."""
    V._need(P.sha(V.__file__) == P.VECTOR_SHA256 and P.sha(P.__file__) == PROFILE_SHA256,
            'Frozen vector/profile source changed')
    tree = ast.parse(Path(V.__file__).read_text())
    vector = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_group'))
    first = vector.body[1]  # docstring, then the explicit device check
    V._need(isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
            and ast.unparse(first.value.args[0]) == "device == 'cuda:0'", 'Unexpected frozen GPU device guard')
    first.value.args[0] = ast.parse("device in ('cpu', 'cuda:0')", mode='eval').body
    namespace = dict(vars(V))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[vector],type_ignores=[])),str(V.__file__),'exec'),namespace)
    vector_run = namespace['run_group']
    # The profile installs process-local wrappers after compilation. Resolve
    # those dependencies at call time rather than capturing stale functions.
    for name in ('load_jobs','validate_layout','source_bindings','VectorScene','adapted_methods','shared_clock'):
        namespace[name] = lambda *a, _name=name, **kw: getattr(V,_name)(*a,**kw)

    tree = ast.parse(Path(P.__file__).read_text())
    profile = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run'))
    profile.args.kwonlyargs.append(ast.arg(arg='device'))
    profile.args.kw_defaults.append(ast.Constant(value='cuda:0'))
    edits = 0
    for node in ast.walk(profile):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'V.run_group':
            keyword = next(k for k in node.keywords if k.arg == 'device')
            V._need(isinstance(keyword.value,ast.Constant) and keyword.value.value == 'cuda:0', 'Unexpected frozen profile device')
            keyword.value = ast.Name(id='device',ctx=ast.Load())
            edits += 1
    V._need(edits == 1, 'Exactly one device forwarding adaptation required')
    namespace = dict(vars(P))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[profile],type_ignores=[])),str(P.__file__),'exec'),namespace)
    namespace['prefix_clock'] = lambda *a, **kw: P.prefix_clock(*a,**kw)
    return vector_run, namespace['run']


def cpu_probe(probe):
    """Preserve every probe read; replace only the GPU-specific admission rule."""
    V._need(P.sha(probe.__file__) == PROBE_SHA256, 'Frozen capability probe changed')
    tree = ast.parse(Path(probe.__file__).read_text())
    fn = copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'probe_live'))
    hits = [n for n in ast.walk(fn) if isinstance(n,ast.If)
            and "Live GPU dynamics/broadphase/view ordinal not established" in ast.unparse(n)]
    V._need(len(hits) == 1, 'Exactly one frozen GPU probe guard required')
    hits[0].test = ast.parse("attrs['enableGPUDynamics'] is not False or attrs['broadphaseType'] != 'MBP' or report['simulation_view_device_ordinal'] != -1",mode='eval').body
    hits[0].body[0].value.args[0] = ast.Constant(value='Live CPU dynamics/broadphase/view ordinal not established')
    def capture(view, dt, *, mode='filtered'):
        metadata, host = probe.capture_contact_view(view,dt,mode=mode,require_cuda=False)
        V._need(all(str(item['device']).lower() == 'cpu' for item in metadata['tensors']),
                'CPU contact payload must actually be on CPU')
        return metadata, host
    namespace = dict(vars(probe),capture_contact_view=capture)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(probe.__file__),'exec'),namespace)
    original = namespace['probe_live']
    def read(*args, **kwargs):
        report = original(*args, **kwargs)
        for kind in ('rigid_bodies', 'articulations'):
            for name, getters in report[kind].items():
                for method, item in getters.items():
                    if item.get('status') != 'pass' or str(item.get('device')).lower() != 'cpu':
                        report['errors'].append('CPU state tensor mismatch: '+name+'/'+method)
        report.update(requested_physics_device='cpu',status='failed' if report['errors'] else 'live_api_snapshot_pass',
                      backend_adaptation='Explicit CPU admission; every original native contact/state getter retained')
        out = Path(args[1] if len(args)>1 else kwargs['output_dir'])
        (out/'PROBE.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return report
    return read


def scene_class(device, capture=None):
    class ScalingScene(V.VectorScene):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if capture is not None:
                capture(self)

        def prepare_guard(self):
            from pxr import Usd, UsdPhysics
            scene = self.sim.stage.GetPrimAtPath(self.sim.cfg.physics_prim_path)
            attrs = {key:scene.GetAttribute(key).Get() for key in
                ('physxScene:enableGPUDynamics','physxScene:broadphaseType','physxScene:enableCCD','physxScene:timeStepsPerSecond')}
            backend_attributes(attrs,int(self.sim.physics_sim_view.device_ordinal),device)
            bodies = [[str(p.GetPath()) for p in Usd.PrimRange(self.sim.stage.GetPrimAtPath(root),Usd.TraverseInstanceProxies())
                       if p.HasAPI(UsdPhysics.RigidBodyAPI)] for root in self.roots]
            V._need(all(bodies), 'Every lane must contain the full original native bodies')
            # Each directed lane-to-other-lanes view is audited every tick. Keep
            # the original per-view capacity and fail closed on any saturation.
            self.cross = [self.sim.physics_sim_view.create_rigid_contact_view(sensor,
                filter_patterns=[[p for j,other in enumerate(bodies) if j != i for p in other]]*len(sensor),
                max_contact_data_count=262144) for i,sensor in enumerate(bodies) if len(bodies)>1]
            self.audit.update(native_scene=attrs,rigid_bodies_per_env=list(map(len,bodies)),
                requested_physics_device=device,actual_gpu_validated=device=='cuda:0',
                actual_cpu_validated=device=='cpu',cross_environment_views=len(self.cross))
    return ScalingScene


def physical_cost(group, scene, lanes):
    """A returned step is confirmed; a pre-fetch native counter is a bound."""
    returned = group.get('physical_clock_steps')
    V._need(type(returned) is int and returned >= 0, 'Nonnegative returned physical step counter required')
    observed = [job.get('observations_completed_steps',job.get('completion_tick')) for job in group.get('jobs',[])]
    counter = None
    read_error = None
    if scene is not None and scene.sim is not None:
        try:
            counter = int(scene.sim.get_physics_step_count())
            V._need(counter >= returned, 'Native counter is below completed step returns')
        except Exception as error:
            counter = None
            read_error = repr(error)
    no_launch = group.get('Kit_launch_wall_s') is None
    if counter is None and no_launch and returned == 0:
        counter = 0
    unknown = counter is None or counter != returned
    return dict(returned_physical_steps=returned,observations_completed_steps_per_lane=observed,
        actual_simulation_step_counter=counter,physical_steps_unknown=unknown,
        inflight_step_unconfirmed=(counter > returned if counter is not None else unknown),
        physical_steps_lower_bound=returned,physical_steps_upper_bound=counter,
        integrated_environment_steps_lower_bound=returned*lanes,
        integrated_environment_steps_upper_bound=None if counter is None else counter*lanes,
        native_counter_read_error=read_error,
        counter_semantics='Native clock increments before simulate/fetch; an excess is an attempted step upper bound, not certified completed integration')


def run(prepared_dir, output_dir, *, mode, lanes, prefix_ticks=100, protocol_path=None):
    V._need(mode in ('gpu','cpu') and (lanes in (1,4,8) if mode=='gpu' else lanes==1),
            'GPU supports 1/4/8 lanes; each CPU worker must use exactly one lane')
    V._need(type(prefix_ticks) is int and 1 <= prefix_ticks <= P.MAX_PREFIX_TICKS,
            'Only explicit incomplete prefixes of 1..200 ticks are enabled')
    vector_run, profile_run = compile_entries()
    import author_gpu_capability_probe as probe
    V._need(P.sha(probe.__file__) == PROBE_SHA256, 'Frozen capability probe changed')
    prepared, out = Path(prepared_dir).resolve(), Path(output_dir).resolve()
    V._need(not out.exists(), 'Fresh scaling output required')
    roots, origins = layout(lanes)
    device = 'cuda:0' if mode == 'gpu' else 'cpu'
    patches = P.Patches()
    old_loader, old_sources, old_save, old_clock = V.load_jobs,V.source_bindings,V._modules()[0].save,P.prefix_clock
    bindings = old_sources()
    bindings.update({str(Path(__file__).resolve()):P.sha(__file__),str(Path(P.__file__).resolve()):P.sha(P.__file__)})
    started = time.time()
    interval = {}
    live_scene = {}
    storage = V._modules()[0]

    def timed_clock(*args, **kwargs):
        interval['execution_prefix_started_unix_s'] = time.time()
        interval['execution_prefix_started_monotonic_s'] = time.monotonic()
        try:
            return old_clock(*args,**kwargs)
        finally:
            interval['execution_prefix_finished_monotonic_s'] = time.monotonic()
            interval['execution_prefix_finished_unix_s'] = time.time()

    def save(path, value):
        path = Path(path)
        if path == out/'GPU_VECTOR_EXECUTION.json':
            value = dict(value,physics_scope='Explicit '+device+' PhysX scaling diagnostic; original assets, dt, controls, full contacts and gates retained',
                repeated_template=True,scaling_mode=mode,scaling_lanes=lanes)
        result = old_save(path,value)
        if path == out/'GPU_VECTOR_PROFILE.json':
            group_path = out/'GPU_VECTOR_EXECUTION.json'
            group = json.loads(group_path.read_text())
            cost = physical_cost(group,live_scene.get('scene'),lanes)
            complete_prefix = (value.get('status')=='partial_profile' and group.get('status')=='partial_profile'
                and group.get('physical_clock_steps')==prefix_ticks and len(group.get('jobs',[]))==lanes
                and group.get('clock',{}).get('lane_steps')==[prefix_ticks]*lanes
                and all(j.get('label') is None and j.get('raw_label') is None for j in group['jobs'])
                and cost['physical_steps_unknown'] is False
                and cost['observations_completed_steps_per_lane']==[prefix_ticks]*lanes)
            ready_path = prepared/'READY.json'
            ready = json.loads(ready_path.read_text())
            execution_wall = (interval.get('execution_prefix_finished_monotonic_s',0.)
                              - interval.get('execution_prefix_started_monotonic_s',0.))
            record = dict(schema=SCHEMA,status='partial_scaling' if complete_prefix else 'infrastructure_failure',
                complete=False,label=None,raw_label=None,formal_training_eligible=False,exclude_training=True,
                automatic_training=False,mode=mode,device=device,lanes=lanes,prefix_ticks=prefix_ticks,
                physical_clock_steps=group.get('physical_clock_steps',0),
                actual_integrated_environment_steps=group.get('actual_integrated_environment_steps',0),
                recorded_command_environment_steps=group.get('recorded_command_environment_steps',0),
                environment_roots=roots,origins_m=origins,
                worker_pid=os.getpid(),worker_started_unix_s=started,worker_finished_before_close_unix_s=time.time(),
                setup_wall_s=interval.get('execution_prefix_started_unix_s',time.time())-started,
                execution_wall_s=execution_wall,execution_interval_scope='Original prefix barrier including generator finish serialization; simulate/fetch separately timed',
                source_sha256=bindings,prepared_template=dict(path=str(prepared),ready_sha256=P.sha(ready_path),
                    commands_sha256=ready['commands']['sha256'],scene_sha256=ready['scene']['sha256'],
                    ready=dict(path=str(ready_path),sha256=P.sha(ready_path)),
                    commands=ready['commands'],scene=ready['scene']),
                group_receipt=dict(path=str(group_path),sha256=P.sha(group_path)),
                profile=dict(path=str(path),sha256=P.sha(path)),timing=value.get('timing'),
                no_new_commands=True,identical_template_repetition=True,labels_scope='All repeated-template prefixes are diagnostic and excluded from training',**interval,**cost)
            old_save(out/'SCALING_EXECUTION.json',record)
        return result

    patches.set(V,'load_jobs',lambda paths:repeated_jobs(paths,lanes,old_loader))
    patches.set(V,'validate_layout',validate_layout)
    patches.set(V,'VectorScene',scene_class(device,lambda scene:live_scene.update(scene=scene)))
    patches.set(V,'run_group',vector_run)
    patches.set(V,'source_bindings',lambda:dict(bindings))
    patches.set(P,'prefix_clock',timed_clock)
    patches.set(storage,'save',save)
    if mode == 'cpu':
        patches.set(probe,'probe_live',cpu_probe(probe))
    try:
        return profile_run([prepared]*lanes,out,env_roots=roots,origins_m=origins,
            prefix_ticks=prefix_ticks,protocol_path=protocol_path,device=device)
    finally:
        patches.restore()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--mode',choices=['gpu','cpu'],required=True)
    parser.add_argument('--lanes',type=int,choices=[1,4,8],required=True)
    parser.add_argument('--prefix-ticks',type=int,default=100)
    parser.add_argument('--protocol',type=Path)
    args = parser.parse_args()
    run(args.prepared_dir,args.output_dir,mode=args.mode,lanes=args.lanes,
        prefix_ticks=args.prefix_ticks,protocol_path=args.protocol)


if __name__ == '__main__':
    main()
