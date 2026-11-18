"""One calibrated MuJoCo camera; fresh RGB-D without integration or resets."""
import copy
from pathlib import Path
import time

import numpy as np
import author_mujoco_control as C
from render_s40_observations import look_at_ros


def configure(spec, camera, assets):
    import mujoco
    # URDF import defaults to discarding non-collision geoms. Retain the new
    # visual-only source triangles while preserving collision masks and mass.
    spec.compiler.discardvisual=False
    pose=look_at_ros(camera['eye_world_m'],camera['target_world_m'],camera['up_world'])
    rotation=pose[:3,:3]@np.diag([1.,-1.,-1.]);quat=np.zeros(4)
    mujoco.mju_mat2Quat(quat,rotation.ravel())
    if camera['fx']!=camera['fy']:raise ValueError('This calibrated pinhole requires equal fx/fy')
    fovy=np.degrees(2*np.arctan(camera['height']/(2*camera['fy'])))
    spec.worldbody.add_camera(name='collection',pos=pose[:3,3],quat=quat,fovy=float(fovy))
    spec.visual.global_.offwidth=camera['width'];spec.visual.global_.offheight=camera['height']
    # Render source triangles separately from the unchanged collision pieces.
    bodies={b.name:b for b in spec.bodies}
    for geom in spec.geoms:
        if geom.parent.name in assets:geom.group=3
    for key,asset in assets.items():
        mesh=asset.mesh;name=key+'_source_visual'
        spec.add_mesh(name=name,uservert=np.asarray(mesh.vertices).ravel(),userface=np.asarray(mesh.faces).ravel())
        bodies[key].add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=name,
            mass=0.,contype=0,conaffinity=0,group=2,rgba=[.65,.65,.7,1.])


class Camera:
    def __init__(self, world, calibration):
        import mujoco
        self.world=world;self.calibration=copy.deepcopy(calibration)
        world.model.vis.map.znear=calibration['clipping_range_m'][0]/world.model.stat.extent
        world.model.vis.map.zfar=calibration['clipping_range_m'][1]/world.model.stat.extent
        self.renderer=mujoco.Renderer(world.model,height=calibration['height'],width=calibration['width'])
        self.option=mujoco.MjvOption();self.option.geomgroup[3]=0
        self.frames=0;self.closed=False

    def render(self):
        w=self.world
        before=(w.data.qpos.copy(),w.data.qvel.copy(),float(w.data.time),w.steps)
        self.renderer.update_scene(w.data,camera='collection',scene_option=self.option)
        self.renderer.disable_depth_rendering();rgb=self.renderer.render().copy()
        self.renderer.enable_depth_rendering();depth=self.renderer.render().copy()
        self.renderer.disable_depth_rendering();self.frames+=1
        np.testing.assert_array_equal(w.data.qpos,before[0]);np.testing.assert_array_equal(w.data.qvel,before[1])
        if (float(w.data.time),w.steps)!=before[2:]:raise ValueError('Camera advanced physics')
        import mujoco
        cid=mujoco.mj_name2id(w.model,mujoco.mjtObj.mjOBJ_CAMERA,'collection')
        pose=np.eye(4);pose[:3,:3]=w.data.cam_xmat[cid].reshape(3,3)@np.diag([1.,-1.,-1.]);pose[:3,3]=w.data.cam_xpos[cid]
        c=self.calibration;K=np.array([[c['fx'],0,c['width']/2],[0,c['fy'],c['height']/2],[0,0,1.]])
        wanted=look_at_ros(c['eye_world_m'],c['target_world_m'],c['up_world'])
        if np.max(abs(wanted-pose))>1e-12:raise ValueError('Native camera pose differs from calibration')
        if rgb.shape!=(c['height'],c['width'],3) or depth.shape!=(c['height'],c['width']):raise ValueError('RGB-D shape changed')
        return rgb,depth,K,pose

    def close(self):
        if not self.closed:self.renderer.close();self.closed=True


class InitializationUnstable(RuntimeError):
    """Finite settling exhausted; this is not an executed grasp outcome."""


def capture(world, scene, output, target, *, min_steps=750, max_steps=750, settle_only=False):
    """Settle in the same world, preserving the original 250-tick window gate."""
    import json
    import scipy.io
    import trimesh
    from PIL import Image
    from settle_s40_scenes import window_metrics
    from render_s40_observations import ray_diagnostic
    from replay_author_persistent_world import observe
    from author_live_world_contract import SCHEMA
    from collections import deque
    if not 750<=min_steps<=max_steps<=5000 or min_steps%250 or max_steps%250:
        raise ValueError('Explicit finite 250-tick settling chunks, 750..5000 required')
    B=C.B;output=Path(output);output.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    exchange_begin=time.monotonic();world.exchange(scene);exchange_s=time.monotonic()-exchange_begin
    legal=world.legal_hand
    poses=deque(maxlen=250);velocities=deque(maxlen=250);support_window=deque(maxlen=250)
    checks=[]
    step_s=observe_s=0.;start_step=world.steps
    with (output/'initialization.jsonl').open('x') as log:
        for i in range(max_steps):
            t=time.perf_counter();world.step(target,np.zeros(len(target)));step_s+=time.perf_counter()-t
            t=time.perf_counter();row=observe(world,'capture_settle',i+1,start_step*.004,legal);observe_s+=time.perf_counter()-t
            poses.append([row['objects'][k]['pose_xyzw'] for k in world.active]);velocities.append([row['objects'][k]['velocity'] for k in world.active])
            support_window.append({key:any('/World/'+key in c['bodies'] and '/World/ground' in c['bodies'] and c['normal_force_n']>.001 for c in row['contacts']) for key in world.active})
            log.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
            if i+1>=min_steps and (i+1)%250==0:
                stability=[window_metrics(np.asarray(poses)[:,j],np.asarray(velocities)[:,j]) for j in range(len(world.active))]
                supports={key:sum(s[key] for s in support_window) for key in world.active}
                stable=all(s['stable'] for s in stability) and all(supports.values())
                checks.append(dict(steps=i+1,stability=stability,supported_ticks=supports,pass_=stable))
                if stable:break
    ticks=i+1
    B.save(output/'SETTLING.json',dict(steps=ticks,min_steps=min_steps,max_steps=max_steps,checks=checks,start_world_step=start_step,end_world_step=world.steps,
        stability=stability,supported_ticks=supports,exchange_s=exchange_s,step_s=step_s,observe_s=observe_s),exclusive=True)
    if not stable:raise InitializationUnstable('Original settling/window/support gate exhausted without reset')
    if settle_only:
        receipt=dict(status='pass',scope='Initialization only; no photograph, inference, grasp or training label',
            wall_s=time.monotonic()-began,physics_steps=ticks,settling=B.binding(output/'SETTLING.json'))
        B.save(output/'RESULTS.json',receipt,exclusive=True);return receipt
    observed=copy.deepcopy(scene);objects=row['objects'];meshes=[]
    from scipy.spatial.transform import Rotation
    for i,obj in enumerate(observed['objects']):
        state=objects[obj['object_id']];pose=np.asarray(state['pose_xyzw']);matrix=np.eye(4)
        matrix[:3,:3]=Rotation.from_quat(pose[3:]).as_matrix();matrix[:3,3]=pose[:3]
        obj.update(world_T_object=matrix.tolist(),observed_pose_xyzw=pose.tolist(),observed_twist=state['velocity'],initialization_stability=stability[i])
        mesh=world.assets[obj['object_id']].mesh.copy();mesh.apply_transform(matrix);meshes.append(mesh)
    observed.update(initial_state_boundary='Measured same MuJoCo world; no reset after this photograph',
        simulator='MuJoCo',physics_backend_changed=True,source_geometry_native_20um_validated=False)
    B.save(output/'observed_scene.json',observed,exclusive=True)
    live=dict(schema=SCHEMA,scene_id=scene['scene_id'],sim_step=world.steps,
        joint_names=list(world.planning.joints),joint_pos=world.data.qpos[world.qindices].tolist(),
        joint_velocity=world.data.qvel[world.vindices].tolist(),objects=objects)
    B.save(output/'LIVE_STATE.json',live,exclusive=True)
    t=time.monotonic();rgb,depth,K,T=world.camera.render();render_s=time.monotonic()-t
    ray=ray_diagnostic(trimesh.util.concatenate(meshes),depth,K,T)
    ray['renderer']='MuJoCo OpenGL';B.save(output/'RAY_CHECK.json',ray,exclusive=True)
    if not ray['diagnostic_pass']:raise ValueError('Original source-triangle depth ray check failed')
    c=scene['camera'];valid=np.isfinite(depth)&(depth>=c['depth_range_m'][0])&(depth<=c['depth_range_m'][1])
    Image.fromarray(rgb).save(output/'color.png');Image.fromarray(np.where(valid,np.rint(depth*1000),0).astype(np.uint16)).save(output/'depth.png')
    scipy.io.savemat(output/'meta.mat',dict(intrinsic_matrix=K,factor_depth=np.array([[1000.]])))
    np.savez_compressed(output/'native_rgbd.npz',rgb=rgb,depth_m=depth)
    observation=dict(schema='anydex_boya_author_scene_observation_v1',status='rendered',simulator='MuJoCo',renderer='MuJoCo OpenGL EGL',
        T_WC=T.tolist(),intrinsic_matrix=K.tolist(),depth_factor=1000.,object_pose_as_network_input=False,
        scene_json=str((output/'observed_scene.json').resolve()),scene_sha256=B.sha(output/'observed_scene.json'),
        scene_id=scene['scene_id'],object_ids=world.active.copy(),
        files={n:B.binding(output/n) for n in ('color.png','depth.png','meta.mat')},
        valid_depth_pixels=int(valid.sum()),grasp_attempts=0,render_physics_steps=0,initialization_physics_ticks=ticks,
        world_step=world.steps,model_id=id(world.model),data_id=id(world.data),camera_id=id(world.camera),
        new_backend_domain=True,exclude_training=True)
    B.save(output/'observation.json',observation,exclusive=True)
    receipt=dict(status='pass',wall_s=time.monotonic()-began,render_s=render_s,physics_steps=ticks,
        render_physics_steps=0,actor_state_writes_after_photo=0,observation=B.binding(output/'observation.json'),
        live_state=B.binding(output/'LIVE_STATE.json'),ray_check=B.binding(output/'RAY_CHECK.json'))
    B.save(output/'RESULTS.json',receipt,exclusive=True)
    return receipt
