"""Two bounded CPU publication profiles; no capture, CGR, grasp or labels."""
import argparse
import copy
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import author_live_collection as LIVE
import author_cpu_fabric as FABRIC

B, R, C = LIVE.B, LIVE.R, LIVE.C
SCHEMA = 'anydex_boya_cpu_fabric_prefix_v1'
SITE = Path('/home/wangrenpeng/tendon-spin/.venv/lib/python3.12/site-packages')
UPDATES = ('updateToUsd', 'updateParticlesToUsd', 'updateVelocitiesToUsd',
           'updateForceSensorsToUsd', 'updateResidualsToUsd')


@contextmanager
def startup_steps(physx, events):
    """Account for native warmup outside SimulationContext's tick counter."""
    original={key:getattr(physx,key) for key in ('get_physx_interface','get_physx_simulation_interface')}
    class Proxy:
        def __init__(self,value):self.value=value
        def __getattr__(self,name):
            method=getattr(self.value,name)
            if name not in ('update_simulation','simulate'):return method
            def advance(dt,*args,**kwargs):
                if len(events)>=4:raise ValueError('Frozen startup integration ceiling exceeded')
                event=dict(api=name,dt_s=float(dt),returned=False);events.append(event)
                value=method(dt,*args,**kwargs);event['returned']=True;return value
            return advance
    for key,value in original.items():setattr(physx,key,lambda value=value:Proxy(value()))
    try:yield
    finally:
        for key,value in original.items():setattr(physx,key,value)


def freeze(reference, root):
    reference, root = Path(reference).resolve(), Path(root).resolve()
    original = LIVE.verify(reference)
    R.assert_quiescent(reference)
    p = copy.deepcopy(original)
    p['inputs']['reference_live_protocol'] = B.binding(reference/'PROTOCOL.json')
    for path in (Path(__file__), Path(FABRIC.__file__),
                 SITE/'isaaclab_physx/physics/physx_manager.py',
                 B.ROOT.parent/'CanonDex/canondex/scene/runtime.py',
                 B.ROOT.parent/'CanonDex/canondex/scene/physx_compat.py'):
        p['sources'][path.name + ':cpu_fabric_profile'] = B.binding(path)
    p.update(schema=SCHEMA, purpose='Original CPU USD publication versus CPU Fabric, first fixed scene only',
        jobs=[dict(mode='usd', use_fabric=False), dict(mode='fabric', use_fabric=True)],
        ticks_per_job=64, warmup_ticks=16, measured_ticks=48, max_Kit_starts=2,
        max_environment_steps=136, max_commanded_steps=128, max_native_startup_steps_per_job=4,
        label=None, raw_label=None, exclude_training=True,
        stage_timeout_s=600., retry_budget=0, full_grasp_validation=False,
        geometry_changed=False, solver_iterations_changed=False, timestep_changed=False,
        collection_or_training_started=False,
        sources_web=[
            'https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/guides/physics-performance.html',
            'https://isaac-sim.github.io/IsaacLab/main/source/concepts/physics-backends/physx.html'],
        created_unix_s=time.time())
    B.verify(p['inputs']); B.verify(p['sources'])
    root.mkdir(parents=True, exist_ok=False)
    p['source_archive'] = C.archive_sources(root, p['sources'])
    B.save(root/'PROTOCOL.json', p, exclusive=True)
    return p


def verify(root):
    p = B.read(Path(root)/'PROTOCOL.json')
    if (p['schema'] != SCHEMA or p['max_Kit_starts'] != 2 or p['ticks_per_job'] != 64
            or p['warmup_ticks'] != 16 or p['max_environment_steps'] != 136
            or p['max_commanded_steps'] != 128 or p['max_native_startup_steps_per_job'] != 4
            or p['jobs'] != [dict(mode='usd', use_fabric=False), dict(mode='fabric', use_fabric=True)]
            or p['retry_budget'] != 0 or not p['exclude_training']):
        raise ValueError('Frozen two-prefix budget changed')
    B.verify(p['inputs']); B.verify(p['sources']); B.verify({'archive': p['source_archive']})
    return p


def profile_worker(root, mode):
    root = Path(root).resolve(); p = verify(root)
    from author_live_world_v4 import World
    world = World(root/'jobs'/mode/'world', p)
    import author_transport as T
    T._backend_path()
    # AppLauncher precedes all simulator imports; geometry binding preceded it.
    from isaaclab.app import AppLauncher
    world.launcher = AppLauncher(headless=True, enable_cameras=True,
        device='cpu', kit_args='--/log/file=/dev/null --/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false')
    error = None; result = None; began = time.monotonic(); count = 0; startup=[]
    directory = root/'jobs'/mode
    try:
        import canondex.scene.runtime as runtime_module
        import omni.physx as physx_module
        with startup_steps(physx_module,startup), (FABRIC.installed(runtime_module) if mode == 'fabric' else nullcontext()):
            world.bootstrap()
        sim = world.sim
        scene = B.read(p['rows'][0]['built_scene']['path'])
        # Original material, weight, collision import audit and full contact view.
        _, _, bodies, roots, _, _, _, _, native_audit, _, targets, _, contacts = world.capture_handles(scene, directory)
        from carb.settings import get_settings
        settings = get_settings()
        actual = dict(use_fabric=bool(sim.cfg.use_fabric), device=str(sim.cfg.device),
            publication={key:sim.get_setting('/physics/'+key) for key in UPDATES},
            num_threads=settings.get('/persistent/physics/numThreads'),
            scene_properties={str(a.GetName()):a.Get() for a in sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path).GetAttributes()
                if str(a.GetName()) in ('physxScene:solverType','physxScene:enableGPUDynamics','physxScene:broadphaseType',
                    'physxScene:timeStepsPerSecond','physxScene:minPositionIterationCount','physxScene:maxPositionIterationCount')})
        if actual['device'] != 'cpu' or actual['use_fabric'] != (mode == 'fabric'):
            raise ValueError('CPU publication mode mismatch')
        if any(bool(v) != (mode == 'usd') for v in actual['publication'].values()):
            raise ValueError('USD publication settings were not applied')
        from canondex.scene.runtime import snapshot
        from canondex.scene.contact_readback import contact_data_to_host
        before = world.state()
        sim_rows=[]; joint_rows=[]; object_rows=[]
        timing=[]; contact_data=[]
        physx = __import__('omni.physx', fromlist=['get_physx_simulation_interface'])
        interface = physx.get_physx_simulation_interface()
        class Timed:
            def __getattr__(self, name):return getattr(interface, name)
            def simulate(self, *a, **kw):
                start=time.perf_counter(); value=interface.simulate(*a, **kw)
                current['simulate_s'] += time.perf_counter()-start; return value
            def fetch_results(self, *a, **kw):
                start=time.perf_counter(); value=interface.fetch_results(*a, **kw)
                current['fetch_s'] += time.perf_counter()-start; return value
        original_interface = physx.get_physx_simulation_interface
        physx.get_physx_simulation_interface = lambda: Timed()
        try:
            for i in range(p['ticks_per_job']):
                current=dict(simulate_s=0., fetch_s=0.)
                start=time.perf_counter(); world.capture_before_step()
                for body in bodies:body.write_data_to_sim()
                current['write_s']=time.perf_counter()-start
                start=time.perf_counter(); sim.step(render=False); count += 1
                current['step_s']=time.perf_counter()-start
                start=time.perf_counter(); world.robot.update(.004)
                for body in bodies:body.update(.004)
                current['update_s']=time.perf_counter()-start
                start=time.perf_counter(); state=snapshot(world.robot)
                poses=np.stack([body.data.root_link_pose_w.torch[0].cpu().numpy().copy() for body in bodies])
                vel=np.stack([body.data.root_link_vel_w.torch[0].cpu().numpy().copy() for body in bodies])
                current['snapshot_s']=time.perf_counter()-start
                start=time.perf_counter()
                data=[x.copy() for x in contact_data_to_host(contacts,.004,262144,compact=True)]
                current['contact_read_s']=time.perf_counter()-start
                if not all(np.isfinite(x).all() for x in [*state.values(),poses,vel]):
                    raise ValueError('Nonfinite complete observation')
                # Validate every referenced point; unused gaps are not contacts.
                start=time.perf_counter()
                decoded=T.decode_contacts(roots,targets,data,262144)
                current['contact_points']=len(decoded)
                current['contact_decode_s']=time.perf_counter()-start
                joint_rows.append(state['joint_pos']); object_rows.append(np.c_[poses,vel]); sim_rows.append(state['body_pose'])
                contact_data.append(data); timing.append(current)
                B.save(directory/'PROGRESS.json',dict(steps=count,phase='bounded_profile',label=None,wall_s=time.monotonic()-began))
        finally:
            physx.get_physx_simulation_interface = original_interface
        after=world.state()
        if after['sim_step']-before['sim_step'] != 64:
            raise ValueError('Unexpected integration count')
        arrays=dict(joint_pos=np.asarray(joint_rows),object_pose_velocity=np.asarray(object_rows),robot_body_pose=np.asarray(sim_rows))
        np.savez_compressed(directory/'trace.npz',**arrays)
        # Complete native variable-length contact buffers retained, no contact thinning.
        import gzip,pickle
        with gzip.open(directory/'full_contacts.pkl.gz','wb') as stream:pickle.dump(contact_data,stream,protocol=5)
        measured=timing[p['warmup_ticks']:]
        result=dict(schema=SCHEMA,status='partial_profile',mode=mode,complete=False,label=None,raw_label=None,
            exclude_training=True,actual_steps=count,warmup_ticks=16,measured_ticks=48,dt_s=.004,
            native_startup_integrations=startup,actual_integrated_steps=count+len(startup),
            native=native_audit,readback=actual,initial=before,final=after,per_tick_timing=timing,
            measured_totals={key:sum(t[key]for t in measured) for key in measured[0]},
            trace=B.binding(directory/'trace.npz'),contacts=B.binding(directory/'full_contacts.pkl.gz'),
            contact_roots=roots,contact_targets=targets,
            original_physics_unchanged=True,full_lift_validation=False,wall_s=time.monotonic()-began)
        B.verify(p['inputs']); B.verify(p['sources'])
        B.save(directory/'PROFILE.json',result,exclusive=True)
    except BaseException:
        error=traceback.format_exc()
        B.save(directory/'ERROR.json',dict(error=error,returned_steps=count,
            native_startup_integrations=startup,sim_clock=world.sim.get_physics_step_count()if world.sim is not None else None,label=None),exclusive=True)
    finally:
        world.close(int(error is not None))


def run(root):
    root=Path(root).resolve();p=verify(root);began=time.monotonic();stages=[]
    if C.gpu_identity('/usr/bin/nvidia-smi',p['gpu_uuid']) != p['gpu_identity']:
        raise ValueError('Bound GPU0 identity changed')
    with R.lease(root/'DRIVER.lock'),R.lease(LIVE.OLD.LEGACY_LOCK)as fd,R.active_run(root,LIVE.OLD.REGISTRY,B.binding(root/'PROTOCOL.json')):
        B.save(root/'RUN_STARTED.json',dict(process=R.process_identity(os.getpid()),unix_s=time.time()),exclusive=True)
        pool=R.ResourcePool(root/'resource_slots',fd,B.binding(root/'PROTOCOL.json'))
        for job in p['jobs']:
            directory=root/'jobs'/job['mode']; directory.mkdir(parents=True,exist_ok=False)
            old=B.subprocess
            with pool.stage(('kit',),directory/'stage'):
                B.subprocess=R.SubprocessProxy(pool)
                try:
                    stage=B._stage(directory/'stage',[p['inputs']['python']['path'],'-B',str(Path(__file__).resolve()),
                        'worker','--root',str(root),'--mode',job['mode']],[directory/'PROFILE.json'],p)
                finally:B.subprocess=old
            stages.append(dict(mode=job['mode'],stage=B.binding(directory/'stage/STAGE.json')))
            if not stage['complete']:break
        verify(root)
        result=dict(status='profiles_complete'if len(stages)==2 and all(B.read(s['stage']['path'])['complete']for s in stages)else'stopped',
            stages=stages,wall_s=time.monotonic()-began,label=None,exclude_training=True,full_lift_validation=False)
        B.save(root/'RESULTS.json',result,exclusive=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','run','worker']);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference',type=Path);parser.add_argument('--mode',choices=['usd','fabric'])
    a=parser.parse_args()
    if a.action=='freeze':result=freeze(a.reference,a.root)
    elif a.action=='worker':profile_worker(a.root,a.mode);sys.exit(0)
    else:
        marker=B.sha(a.root/'PROTOCOL.json')
        if os.environ.get('ANYDEX_CPU_FABRIC_ENV')!=marker:
            env=B._environment(verify(a.root));env['ANYDEX_CPU_FABRIC_ENV']=marker
            os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),'run','--root',str(a.root.resolve())],env)
        result=run(a.root)
    print({k:result.get(k)for k in ('status','wall_s','label')},flush=True)
