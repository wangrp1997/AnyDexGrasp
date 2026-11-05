"""One Isaac RTX process renders frozen public DGN meshes for the Boya pilot.

This creates isolated mesh/ground observations, no robot or dynamic objects.
Only color.png, depth.png and meta.mat are network inputs; identities, source
poses and meshes are traceability/execution metadata, never inference inputs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def look_at_ros(eye,target,up):
    import numpy as np
    eye,target,up=map(lambda x:np.asarray(x,dtype=float),(eye,target,up))
    z=target-eye;z/=np.linalg.norm(z)
    x=np.cross(z,up);x/=np.linalg.norm(x)
    y=np.cross(z,x)
    T=np.eye(4);T[:3,:3]=np.column_stack([x,y,z]);T[:3,3]=eye
    if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-12)or np.linalg.det(T[:3,:3])<.999999999:
        raise ValueError('Invalid look-at camera')
    return T


def ray_diagnostic(mesh_world,depth,K,T,seed=20261105):
    """Independent source-triangle/ground first-hit comparison, no simulation."""
    import numpy as np
    from trimesh.ray.ray_triangle import RayMeshIntersector
    vertices=(mesh_world.vertices-T[:3,3])@T[:3,:3]
    if np.any(vertices[:,2]<=0):raise ValueError('Object behind camera')
    uv=vertices[:,:2]/vertices[:,2:3]@K[:2,:2].T+K[:2,2]
    lo=np.maximum(np.floor(uv.min(0)).astype(int),[0,0]);hi=np.minimum(np.ceil(uv.max(0)).astype(int),[depth.shape[1]-1,depth.shape[0]-1])
    if (hi<=lo).any():raise ValueError('Object projection outside camera')
    rng=np.random.default_rng(seed)
    pixels=np.column_stack([rng.integers(lo[0],hi[0]+1,96),rng.integers(lo[1],hi[1]+1,96)])
    # RTX rays are through pixel centers; record this convention explicitly.
    xy=(pixels+.5-K[:2,2])/np.diag(K)[:2]
    rays=np.column_stack([xy,np.ones(len(xy))])@T[:3,:3].T
    origin=np.repeat(T[:3,3][None],len(rays),axis=0)
    ground=np.divide(-T[2,3],rays[:,2],out=np.full(len(rays),np.inf),where=rays[:,2]<-1e-9)
    expected=ground.copy();is_object=np.zeros(len(rays),bool)
    hits,indices,_=RayMeshIntersector(mesh_world).intersects_location(origin,rays,multiple_hits=False)
    if len(hits):
        dz=(hits-T[:3,3])@T[:3,2]
        for index,d in zip(indices,dz):
            if d>0 and d<expected[index]:expected[index]=d;is_object[index]=True
    observed=depth[pixels[:,1],pixels[:,0]]
    valid=np.isfinite(expected)&np.isfinite(observed)&(expected>.1)&(expected<1.)&(observed>.1)&(observed<1.)
    diff=np.abs(expected[valid]-observed[valid])
    if not len(diff):raise ValueError('No valid diagnostic rays')
    report=dict(ray_count=len(rays),valid_rays=int(valid.sum()),source_object_first_hits=int(is_object.sum()),
        pixel_center_offset=.5,median_absolute_depth_error_m=float(np.median(diff)),
        p90_absolute_depth_error_m=float(np.quantile(diff,.9)),max_absolute_depth_error_m=float(diff.max()),
        pixels_xy=pixels.tolist(),expected_source_first_depth_m=[float(x)if np.isfinite(x)else None for x in expected],
        observed_rtx_depth_m=[float(x)if np.isfinite(x)else None for x in observed],source_first_hit_is_object=is_object.tolist(),
        boundary='Source triangles plus analytic z=0 plane, using declared/native-verified camera. Maximum errors may include raster silhouettes; no source pose is inferred from a grasp outcome.')
    report['diagnostic_pass']=bool(is_object.sum()>=2 and report['median_absolute_depth_error_m']<.002 and report['p90_absolute_depth_error_m']<.004)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--objects-json',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.objects_json=a.objects_json.resolve();a.output_dir=a.output_dir.resolve()
    frozen=json.loads(a.objects_json.read_text());rows=frozen['objects']
    if len(rows)!=8 or len({r['id']for r in rows})!=8:raise ValueError('Exactly eight unique frozen inputs required')
    if frozen['active_object_count']!=4330 or frozen['scale']!=.1:raise ValueError('Unexpected active dataset/scale')
    for r in rows:
        for path_key,hash_key in [('mesh_path','mesh_sha256'),('urdf_path','urdf_sha256'),('pose_path','pose_sha256')]:
            if sha(r[path_key])!=r[hash_key]:raise ValueError('Frozen source changed: '+r[path_key])
    a.output_dir.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(a.objects_json,a.output_dir/'objects.json')
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    started=time.monotonic();progress=dict(schema='boya-pilot8-isaac-rgbd-v1',status='starting_isaac',
        source_objects_sha256=sha(a.objects_json),renderer_source_sha256=sha(__file__),observations=[],
        explicit_physics_steps=0,control_steps=0,grasp_attempts=0,robot_present=False)
    save(a.output_dir/'render_progress.json',progress)
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(headless=True,enable_cameras=True,device='cpu',kit_args='--/log/file=/dev/null')
    current=None
    try:
        import numpy as np
        import torch
        from PIL import Image
        import scipy.io
        from scipy.spatial.transform import Rotation
        import trimesh
        import isaaclab.sim as sim_utils
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab.sensors.camera import Camera,CameraCfg
        from pxr import Usd,UsdGeom,Gf
        camera_spec=frozen['camera'];T=look_at_ros(camera_spec['eye_world_m'],camera_spec['target_world_m'],camera_spec['up_world'])
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.004,device='cpu',physics=PhysxCfg()))
        ground=sim_utils.GroundPlaneCfg(color=(.65,.68,.72));ground.func('/World/ground',ground)
        light=sim_utils.DomeLightCfg(intensity=2500.,color=(1.,1.,1.));light.func('/World/light',light)
        material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.85,.60,.25),roughness=.5)
        material.func('/World/ObservationMaterial',material)
        visuals={};world_meshes={};mesh_readback={}
        for row in rows:
            m=trimesh.load_mesh(row['mesh_path'],process=False);m.apply_scale(row['scale']);m.apply_transform(np.asarray(row['world_T_object']))
            path='/World/ObservedObjects/'+row['id'];mesh=UsdGeom.Mesh.Define(sim.stage,path)
            vertices=np.asarray(m.vertices,dtype=np.float32);faces=np.asarray(m.faces,dtype=np.int32)
            mesh.CreatePointsAttr([Gf.Vec3f(*map(float,v))for v in vertices]);mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none);mesh.CreateDoubleSidedAttr(True)
            sim_utils.bind_visual_material(path,'/World/ObservationMaterial')
            imageable=UsdGeom.Imageable(mesh.GetPrim());imageable.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            back=np.array(mesh.GetPointsAttr().Get(),dtype=float);max_error=float(np.abs(back-m.vertices).max())
            if max_error>1e-7:raise ValueError('USD world mesh points differ from source')
            mesh_readback[row['id']]=dict(vertices=len(vertices),triangles=len(faces),max_world_vertex_roundtrip_error_m=max_error,
                world_AABB_m=[back.min(0).tolist(),back.max(0).tolist()],native_dynamic_body=False,render_geometry='original source triangle mesh, scale .1, frozen supported rotation')
            visuals[row['id']]=imageable;world_meshes[row['id']]=m
        camera=Camera(CameraCfg(prim_path='/World/observation_camera',width=camera_spec['width'],height=camera_spec['height'],
            update_period=0,data_types=['rgb','distance_to_image_plane'],update_latest_camera_pose=True,
            spawn=sim_utils.PinholeCameraCfg(focal_length=camera_spec['focal_length'],horizontal_aperture=camera_spec['horizontal_aperture'],clipping_range=tuple(camera_spec['clipping_range_m']))))
        sim.reset()
        qxyzw=Rotation.from_matrix(T[:3,:3]).as_quat()
        camera.set_world_poses(torch.tensor([T[:3,3].tolist()],dtype=torch.float32),torch.tensor([qxyzw.tolist()],dtype=torch.float32),convention='ros')
        def cpu(value):
            t=value.torch if hasattr(value,'torch')else value
            return t.detach().cpu().numpy()
        for row in rows:
            current=row['id'];out=a.output_dir/current;out.mkdir()
            for oid,v in visuals.items():v.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if oid==current else UsdGeom.Tokens.invisible)
            for _ in range(16):sim.render();camera.update(.004,force_recompute=True)
            rgb=cpu(camera.data.output['rgb'])[0,:,:,:3];depth=cpu(camera.data.output['distance_to_image_plane'])[0].squeeze();K=cpu(camera.data.intrinsic_matrices)[0]
            if rgb.shape!=(720,1280,3)or depth.shape!=(720,1280):raise ValueError('Unexpected native image shape')
            if not np.allclose(K,[[640.,0.,640.],[0.,640.,360.],[0.,0.,1.]],atol=.002,rtol=0):raise ValueError('Native camera intrinsics changed')
            # USD has optical -Z/+Y; ROS optical has +Z/+Y down.
            prim=sim.stage.GetPrimAtPath('/World/observation_camera')
            usd=np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),dtype=float).T
            usd_ros=usd@np.diag([1.,-1.,-1.,1.])
            native_position=cpu(camera.data.pos_w)[0];native_quat_xyzw=cpu(camera.data.quat_w_ros)[0]
            actual=np.eye(4);actual[:3,:3]=Rotation.from_quat(native_quat_xyzw).as_matrix();actual[:3,3]=native_position
            pose_error=float(np.abs(actual-T).max())
            save(out/'camera_transform_check.json',dict(quaternion_order='xyzw per installed Camera.set_world_poses',authority='Camera.data.pos_w/quat_w_ros updated from native Fabric PrimView; ordinary USD Xform not runtime authority',declared_T_WC=T.tolist(),native_T_WC=actual.tolist(),unsynchronized_usd_T_WC=usd_ros.tolist(),native_position=native_position.tolist(),native_quaternion_xyzw=native_quat_xyzw.tolist(),max_error=pose_error))
            if pose_error>2e-6:raise ValueError('Native camera pose differs from declared look-at')
            if sim.stage.GetPrimAtPath('/World/Robot').IsValid():raise ValueError('Robot unexpectedly present in observation')
            ray=ray_diagnostic(world_meshes[current],depth,K,actual);save(out/'render_geometry_check.json',dict(camera_pose_max_error=pose_error,source_mesh_readback=mesh_readback[current],rays=ray))
            if not ray['diagnostic_pass']:raise ValueError('RTX/source first-hit geometry check failed: '+str({k:v for k,v in ray.items()if not isinstance(v,list)}))
            near,far=camera_spec['depth_range_m'];valid=np.isfinite(depth)&(depth>=near)&(depth<=far)
            depth_mm=np.where(valid,np.rint(depth*1000),0).astype(np.uint16)
            if np.count_nonzero(depth_mm)<100:raise ValueError('No valid RGBD pixels')
            Image.fromarray(rgb.astype(np.uint8)).save(out/'color.png');Image.fromarray(depth_mm).save(out/'depth.png')
            scipy.io.savemat(out/'meta.mat',dict(intrinsic_matrix=K,factor_depth=np.array([[1000.]])))
            report=dict(schema='anydex_boya_isaac_observation_v1',status='rendered',object_id=row['object_id'],pilot_id=current,
                simulator='Isaac Sim / PhysX',renderer='Isaac native RTX camera',robot_visible=False,robot_present=False,
                observation_protocol='Isolated original DGN triangle mesh/ground view before arm approach; one shared oblique camera for all eight objects',
                T_WC=actual.tolist(),declared_T_WC=T.tolist(),camera_pose_max_error=pose_error,intrinsic_matrix=K.tolist(),depth_factor=1000.,
                depth_range_m=[near,far],valid_depth_pixels=int(valid.sum()),camera_source=camera_spec,
                files={name:dict(path=str(out/name),sha256=sha(out/name))for name in ['color.png','depth.png','meta.mat']},
                network_input_files=['color.png','depth.png','meta.mat'],object_root=row['object_root'],world_T_object=row['world_T_object'],object_scale=row['scale'],
                object_geometry=dict(type='mesh',source_mesh=row['mesh_path'],source_mesh_sha256=row['mesh_sha256'],source_urdf=row['urdf_path'],source_urdf_sha256=row['urdf_sha256']),
                source_pose_index=row['selected_source_pose_index'],source_pose_sha256=row['pose_sha256'],source_active_list_sha256=frozen['source_active_list_sha256'],
                source_object_selection_sha256=sha(a.objects_json),source_renderer_sha256=sha(__file__),object_mass_kg=row['object_mass_kg'],object_friction=row['object_friction'],
                visual_material='Shared constant (.85,.60,.25) synthetic material for all objects; no source texture claim',
                control_steps=0,explicit_physics_steps=0,grasp_attempts=0,pybullet_imported='pybullet'in sys.modules,object_pose_as_network_input=False,
                mesh_readback=mesh_readback[current],geometry_check_sha256=sha(out/'render_geometry_check.json'),
                render_frames=16,elapsed_wall_s=time.monotonic()-started)
            save(out/'observation.json',report)
            progress['observations'].append(dict(id=current,object_id=row['object_id'],observation_json=str(out/'observation.json'),sha256=sha(out/'observation.json'),status='rendered'))
            progress['status']='rendering';save(a.output_dir/'render_progress.json',progress)
            print('BOYA_PILOT_OBSERVATION_READY '+json.dumps(progress['observations'][-1]),flush=True)
        progress.update(status='complete',elapsed_wall_s=time.monotonic()-started,isaac_processes=1)
        save(a.output_dir/'render_progress.json',progress);save(a.output_dir/'OBSERVATIONS.json',progress)
        print('BOYA_PILOT8_RENDER_COMPLETE '+json.dumps({k:v for k,v in progress.items()if k!='observations'}),flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        progress.update(status='failed',failed_object=current,error=traceback.format_exc(),elapsed_wall_s=time.monotonic()-started)
        save(a.output_dir/'FAILED_RENDER.json',progress);save(a.output_dir/'render_progress.json',progress)
        raise
    finally:
        launcher.app.close()


if __name__=='__main__':main()
