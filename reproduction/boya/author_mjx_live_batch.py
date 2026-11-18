"""Persistent full arm/hand worlds, native MJX-Warp contacts, one renderer.

CPU MjData instances are read-only kinematic/render mirrors. They never step or
solve contacts. An independent GPU forward observation matches the historical
post-step observer without replacing the live state or its warm start.
"""
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np


CONTACT_FIELDS=('dim','dist','efc_address','elem','flex','frame','friction','geom',
    'geomcollisionid','includemargin','pos','solimp','solref','solreffriction','type','vert','worldid')


def clock_table(max_ticks):
    return np.r_[np.float32(0),np.cumsum(np.full(max_ticks*8,np.float32(.0005)),dtype=np.float32)][::8]


def decode_contacts(packet,model,mapping,count):
    n=int(np.asarray(packet['nacon']).reshape(-1)[0]);capacity=len(packet['contact_dist'])
    if not 0<=n<=capacity or np.any(packet['overflow']):raise RuntimeError('Native GPU contact/constraint overflow')
    result=[[]for _ in range(count)];forces=packet['efc_force']
    for i in range(n):
        raw={key:np.asarray(packet['contact_'+key][i]).tolist() for key in CONTACT_FIELDS}
        world=int(raw['worldid']);geoms=list(map(int,raw['geom']));dim=int(raw['dim'])
        if not 0<=world<count or min(geoms)<0 or max(geoms)>=model.ngeom:raise ValueError('Native contact identity out of range')
        address=np.asarray(raw['efc_address']).reshape(-1);start=int(address[0]);wrench=np.zeros(6)
        if start>=0:
            if start+dim>forces.shape[-1]:raise RuntimeError('Native constraint address exceeds allocation')
            wrench[:dim]=forces[world,start:start+dim]
        bodies=[int(model.geom_bodyid[g])for g in geoms]
        result[world].append(dict(native_index=i,geom_ids=geoms,body_ids=bodies,bodies=[mapping[b]for b in bodies],
            separation_m=float(raw['dist']),position_world_m=raw['pos'],contact_frame_world=np.asarray(raw['frame']).ravel().tolist(),
            wrench_contact_frame=wrench.tolist(),normal_force_n=float(wrench[0]),constraint_dimension=dim,
            constraint_address=start,friction=raw['friction'],solref=raw['solref'],solimp=raw['solimp'],
            inclusion_margin_m=float(raw['includemargin']),backend='MJX-Warp',native_fields=raw,
            unavailable_CPU_fields=['exclude'],native_contact_fields_dropped=False))
    return result


class Batch:
    def __init__(self,plan,names,initial,scenes,*,count,max_ticks):
        import jax
        import jax.numpy as jp
        import mujoco
        from mujoco import mjx
        from author_mujoco_persistent_world import World
        from author_mjx_compatibility import apply_gpu_ccd_tolerance
        self.jax=jax;self.jp=jp;self.count=count;self.ticks=0;self.max_ticks=max_ticks
        self.journal=None
        self.expected_clock=clock_table(max_ticks);self.events=[];self.step_s=0.;self.observer_s=0.
        began=time.monotonic()
        self.base=World(plan,names,initial,scenes,max_steps=max_ticks,camera=True,
            source_options=True,rigid_joint_limits=True,object_dynamics=True)
        self.model=self.base.model;self.gpu_model=copy.copy(self.model);self.gpu_model.opt.noslip_iterations=0
        self.compatibility=apply_gpu_ccd_tolerance(self.gpu_model)
        self.mjx_model=mjx.put_model(self.gpu_model,impl='warp')
        self.capacity=512*min(count,8)+32*max(count-8,0)
        self.gpu=jax.vmap(lambda _:mjx.make_data(self.gpu_model,impl='warp',naconmax=self.capacity,njmax=8192))(jp.arange(count))
        self.views=[]
        for i in range(count):
            w=copy.copy(self.base);w.data=mujoco.MjData(self.model);w.observed=w.data
            w.lane=i;w.steps=0;w.phase='boundary';w.active=[];w.exchanges=0;w.events=[]
            self.views.append(w)
        q=np.broadcast_to(self.base.data.qpos,(count,self.model.nq)).copy()
        v=np.zeros((count,self.model.nv));f=np.zeros((count,self.model.nbody,6))
        for i,(key,(a,b)) in enumerate(self.base.addresses.items()):q[:,a:a+7]=[10.+i,10.,5.,1.,0.,0.,0.]
        self.targets=np.broadcast_to(initial,(count,len(names))).copy();self.speeds=np.zeros_like(self.targets)
        self.gpu=self.gpu.replace(qpos=jp.array(q),qvel=jp.array(v),xfrc_applied=jp.array(f))
        self.contacts=[[]for _ in range(count)];self.packet=None
        def packet(d):
            value={name:getattr(d,name) for name in ('qpos','qvel','time','ctrl','xpos','xmat')}
            value.update(nacon=d._impl.nacon,overflow=d._impl.overflow,efc_force=d._impl.efc__force,
                nefc=d._impl.nefc,live_warmstart=d.qacc_warmstart)
            value.update({'contact_'+name:getattr(d._impl,'contact__'+name)for name in CONTACT_FIELDS})
            return value
        self._packet=packet
        @jax.jit
        def advance(d,ctrl):
            d=d.replace(ctrl=ctrl)
            d=jax.lax.fori_loop(0,8,lambda _,state:jax.vmap(lambda one:mjx.step(self.mjx_model,one))(state),d)
            observed=jax.vmap(lambda one:mjx.forward(self.mjx_model,one))(d)
            value=packet(observed);value['live_warmstart']=d.qacc_warmstart
            return d,value
        self._advance=advance
        self.sync_kinematics(np.asarray(self.gpu.qpos),np.asarray(self.gpu.qvel),np.asarray(self.gpu.time))
        self.setup_wall_s=time.monotonic()-began

    def sync_kinematics(self,q,v,clock):
        import mujoco
        for i,w in enumerate(self.views):
            w.data.qpos[:]=q[i];w.data.qvel[:]=v[i];w.data.time=float(clock[i]);w.steps=self.ticks
            mujoco.mj_kinematics(self.model,w.data);mujoco.mj_comPos(self.model,w.data)
            mujoco.mj_comVel(self.model,w.data);mujoco.mj_camlight(self.model,w.data)

    def exchange(self,lane,scene):
        from scipy.spatial.transform import Rotation
        w=self.views[lane]
        if w.phase!='boundary':raise ValueError('Objects may change only at this lane round boundary')
        q=np.array(self.gpu.qpos);v=np.array(self.gpu.qvel);f=np.array(self.gpu.xfrc_applied)
        before=(q[:,self.base.qindices].copy(),v[:,self.base.vindices].copy(),np.array(self.gpu.time))
        oldq=q.copy();oldv=v.copy();active={r['object_id']:r for r in scene['objects']}
        if not set(active)<=set(self.base.bodyids):raise ValueError('Object outside preloaded bank')
        for i,(key,(a,b)) in enumerate(self.base.addresses.items()):
            body=self.base.bodyids[key];v[lane,b:b+6]=0.;f[lane,body]=0.
            if key in active:
                row=active[key]
                if {n:row[n]for n in ('object_root','scale','source_mesh')}!=self.base.bank_identity[key]:raise ValueError('Bank identity changed')
                pose=np.asarray(row['world_T_object']);q[lane,a:a+7]=np.r_[pose[:3,3],Rotation.from_matrix(pose[:3,:3]).as_quat(scalar_first=True)]
                f[lane,body,2]=-.98
            else:q[lane,a:a+7]=[10.+i,10.,5.,1.,0.,0.,0.]
        self.gpu=self.gpu.replace(qpos=self.jp.array(q),qvel=self.jp.array(v),xfrc_applied=self.jp.array(f))
        np.testing.assert_array_equal(q[:,self.base.qindices],before[0]);np.testing.assert_array_equal(v[:,self.base.vindices],before[1])
        np.testing.assert_array_equal(np.asarray(self.gpu.time),before[2])
        other=np.arange(self.count)!=lane
        np.testing.assert_array_equal(q[other],oldq[other]);np.testing.assert_array_equal(v[other],oldv[other])
        w.active=list(active);w.scene=copy.deepcopy(scene);w.exchanges+=1
        event=dict(lane=lane,scene_id=scene['scene_id'],tick=self.ticks,robot_and_clock_exact=True,other_lanes_exact=True)
        self.events.append(event);w.events.append(event)
        self.sync_kinematics(np.asarray(self.gpu.qpos),np.asarray(self.gpu.qvel),np.asarray(self.gpu.time))

    def step(self,targets,speeds):
        if self.ticks>=self.max_ticks:raise RuntimeError('Frozen GPU native-step budget exhausted')
        targets=np.asarray(targets);speeds=np.asarray(speeds)
        if targets.shape!=self.targets.shape or speeds.shape!=targets.shape:raise ValueError('Complete batch controls required')
        if not np.isfinite(targets).all() or not np.isfinite(speeds).all():raise ValueError('Nonfinite controls')
        ctrl=np.zeros((self.count,self.model.nu),dtype=np.float32)
        ctrl[:,self.base.position_ids]=targets;ctrl[:,self.base.velocity_ids]=speeds[:,self.base.arm_columns]
        t=time.perf_counter();self.gpu,packet=self._advance(self.gpu,self.jp.array(ctrl));packet=self.jax.device_get(packet)
        self.step_s+=time.perf_counter()-t;self.ticks+=1;self.packet=packet
        if not np.isfinite(packet['qpos']).all() or not np.isfinite(packet['qvel']).all():raise RuntimeError('GPU nonfinite state')
        np.testing.assert_array_equal(packet['time'],np.full(self.count,self.expected_clock[self.ticks],dtype=np.float32))
        np.testing.assert_array_equal(packet['ctrl'],ctrl)
        if np.any(packet['overflow']) or np.any(packet['nefc']>8192):raise RuntimeError('GPU overflow/constraint capacity exceeded')
        t=time.perf_counter();self.contacts=decode_contacts(packet,self.model,self.base.mapping,self.count)
        self.sync_kinematics(packet['qpos'],packet['qvel'],packet['time']);self.observer_s+=time.perf_counter()-t
        self.targets=targets.copy();self.speeds=speeds.copy()
        if self.journal is not None:
            self.journal.write(json.dumps(dict(tick=self.ticks,native_time_s=packet['time'].tolist(),
                canonical_time_s=self.ticks*.004,qpos=packet['qpos'].tolist(),qvel=packet['qvel'].tolist(),
                ctrl=packet['ctrl'].tolist(),native_xpos=packet['xpos'].tolist(),native_xmat=packet['xmat'].tolist(),
                contacts=self.contacts,nefc=packet['nefc'].tolist(),overflow=packet['overflow'].tolist()),
                separators=(',',':'),allow_nan=False)+'\n')

    def observe(self,lane,phase,index):
        import mujoco
        import author_mujoco_control as C
        import author_fast_geometry_checks as F
        w=self.views[lane];contacts=self.contacts[lane];states={}
        views=C.object_views(contacts,w.active);robot_paths=sorted({p for p in w.mapping.values()if p.startswith('/World/Robot/')})
        for key in w.active:
            body=w.bodyids[key];quat=np.zeros(4);mujoco.mju_mat2Quat(quat,w.data.xmat[body])
            pose=np.r_[w.data.xpos[body],quat[1:],quat[0]];velocity=np.zeros(6)
            mujoco.mj_objectVelocity(self.model,w.data,mujoco.mjtObj.mjOBJ_BODY,body,velocity,0)
            states[key]=dict(pose_xyzw=pose.tolist(),velocity=np.r_[velocity[3:],velocity[:3]].tolist(),
                **C.T.object_robot_loads('/World/'+key,views,robot_paths,w.legal_hand),**F.source_metrics(w.assets[key].mesh.vertices,pose))
        return dict(step=index,time_s=index*.004,native_time_s=float(w.data.time),world_step=self.ticks,
            phase=phase,objects=states,contacts=contacts,joint_position=w.data.qpos[w.qindices].tolist(),
            joint_velocity=w.data.qvel[w.vindices].tolist(),base_transform=C.pose_matrix(self.model,w.data,
                mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,w.planning.fk.root)).tolist(),
            palm_transform=C.pose_matrix(self.model,w.data,mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'palm')).tolist(),
            native_post_step_palm_transform=self.native_transform(lane,'palm'),backend='MJX-Warp',
            GPU_overflow=False,all_GPU_contacts_retained=True,CPU_contact_solver_used=False)

    def native_transform(self,lane,body_name):
        import mujoco
        body=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,body_name);x=np.eye(4)
        x[:3,:3]=self.packet['xmat'][lane,body].reshape(3,3);x[:3,3]=self.packet['xpos'][lane,body]
        return x.tolist()

    def snapshot(self):
        return {k:np.array(getattr(self.gpu,k)) for k in ('qpos','qvel','time','ctrl','qacc_warmstart')}

    def assert_snapshot(self,value):
        for k,v in value.items():np.testing.assert_array_equal(np.asarray(getattr(self.gpu,k)),v)

    def receipt(self):
        return dict(backend='MJX-Warp',worlds=self.count,world_model_builds=1,GPU_model_uploads=1,
            robot_initializations_per_environment=1,robot_resets_after_initialization=0,
            global_control_ticks=self.ticks,native_steps_per_control=8,
            native_environment_integrations=self.ticks*8*self.count,max_native_environment_integrations=self.max_ticks*8*self.count,
            object_bank_size=len(self.base.bodyids),object_exchanges=self.events,renderer_builds=1,
            camera_frames=self.base.camera.frames,source_CPU_model_never_integrated=True,CPU_contact_solver_used=False,
            observation='Independent post-step GPU forward; CPU kinematics/render mirrors only',
            GPU_noslip_iterations=0,GPU_CCD=self.compatibility,step_and_GPU_observer_s=self.step_s,
            contact_decode_and_kinematics_s=self.observer_s,setup_wall_s=self.setup_wall_s,
            native_clock_float32=True,clock_verified_against_every_float32_native_addition=True,
            production_ready=False,training_updates=0)

    def close(self):
        if self.journal is not None:self.journal.close()
        self.base.camera.close()


def photograph(batch,lane,scene,output,row,stability,ticks):
    import scipy.io
    import trimesh
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from render_s40_observations import ray_diagnostic
    from author_live_world_contract import SCHEMA
    import author_collection_batch as B
    w=batch.views[lane];output=Path(output);began=time.monotonic();observed=copy.deepcopy(scene);meshes=[]
    for i,obj in enumerate(observed['objects']):
        state=row['objects'][obj['object_id']];pose=np.asarray(state['pose_xyzw']);matrix=np.eye(4)
        matrix[:3,:3]=Rotation.from_quat(pose[3:]).as_matrix();matrix[:3,3]=pose[:3]
        obj.update(world_T_object=matrix.tolist(),observed_pose_xyzw=pose.tolist(),observed_twist=state['velocity'],initialization_stability=stability[i])
        mesh=w.assets[obj['object_id']].mesh.copy();mesh.apply_transform(matrix);meshes.append(mesh)
    observed.update(initial_state_boundary='Measured persistent GPU world; no reset after photograph',simulator='MJX-Warp',
        physics_backend_changed=True,source_geometry_native_20um_validated=False)
    B.save(output/'observed_scene.json',observed,exclusive=True)
    live=dict(schema=SCHEMA,scene_id=scene['scene_id'],sim_step=batch.ticks,joint_names=list(w.planning.joints),
        joint_pos=w.data.qpos[w.qindices].tolist(),joint_velocity=w.data.qvel[w.vindices].tolist(),objects=row['objects'])
    B.save(output/'LIVE_STATE.json',live,exclusive=True)
    snapshot=batch.snapshot();camera=batch.base.camera;old=camera.world;camera.world=w
    try:rgb,depth,K,T=camera.render()
    finally:camera.world=old
    batch.assert_snapshot(snapshot)
    ray=ray_diagnostic(trimesh.util.concatenate(meshes),depth,K,T);B.save(output/'RAY_CHECK.json',ray,exclusive=True)
    if not ray['diagnostic_pass']:raise ValueError('Original full source-triangle camera ray check failed')
    c=scene['camera'];valid=np.isfinite(depth)&(depth>=c['depth_range_m'][0])&(depth<=c['depth_range_m'][1])
    Image.fromarray(rgb).save(output/'color.png');Image.fromarray(np.where(valid,np.rint(depth*1000),0).astype(np.uint16)).save(output/'depth.png')
    scipy.io.savemat(output/'meta.mat',dict(intrinsic_matrix=K,factor_depth=np.array([[1000.]])))
    np.savez_compressed(output/'native_rgbd.npz',rgb=rgb,depth_m=depth)
    observation=dict(schema='anydex_boya_author_scene_observation_v1',status='rendered',simulator='MJX-Warp',renderer='MuJoCo OpenGL EGL',
        T_WC=T.tolist(),intrinsic_matrix=K.tolist(),depth_factor=1000.,object_pose_as_network_input=False,
        scene_json=str((output/'observed_scene.json').resolve()),scene_sha256=B.sha(output/'observed_scene.json'),scene_id=scene['scene_id'],
        object_ids=w.active.copy(),files={n:B.binding(output/n)for n in ('color.png','depth.png','meta.mat')},valid_depth_pixels=int(valid.sum()),
        grasp_attempts=0,render_physics_steps=0,initialization_physics_ticks=ticks,world_step=batch.ticks,lane=lane,
        new_backend_domain=True,exclude_training=True)
    B.save(output/'observation.json',observation,exclusive=True)
    result=dict(status='pass',wall_s=time.monotonic()-began,physics_steps=ticks,render_physics_steps=0,
        actor_state_writes_after_photo=0,observation=B.binding(output/'observation.json'),live_state=B.binding(output/'LIVE_STATE.json'))
    B.save(output/'RESULTS.json',result,exclusive=True);return result
