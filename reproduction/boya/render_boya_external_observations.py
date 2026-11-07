"""External scene adapter preserving frozen native RTX optics and first-hit checks.

Only scene identity/provenance validation is generalized; no grasp candidate,
score, collision, arm or controller is changed. Source helpers remain pinned.
"""
from pathlib import Path
import argparse,json,os,sys,time,hashlib
B=Path('/home/wangrenpeng/AnyDexGrasp/reproduction/boya');sys.path.insert(0,str(B))
from render_s40_observations import sha,save,look_at_ros,ray_diagnostic,prepare_output,verify_frozen_inputs,validate_camera,_rigid
PILOT_RENDERER_SHA256='7aabcaeb603e5b86226493ddef463be55cc152b441aa978374352a0b0b44852a'
READONLY_PILOT_SOURCE_SHA256='e541b3fa753954f19abf74c631cc6f3532b604f7e786f06e2d9d9b8e3dbd7307'
PROVENANCE_BOUNDARY='Unseen-to-Boya EGAD source geometry with uniform .001 source-unit-to-metre simulation scale, synthetic RGBD at measured settled pose, .1kg/.6friction; not original AnyDex printed size, material, hardware or clutter protocol.'

def load_manifest(path,requested_ids=None):
 import numpy as np
 path=Path(path).resolve();f=json.loads(path.read_text());digest=sha(path);rows=f['objects'];ids=[r['object_id']for r in rows]
 if f.get('schema')!='anydex_boya_external_scene_manifest_v1' or len(ids)!=12 or len(set(ids))!=12 or f.get('approved_object_ids')!=ids:raise ValueError('Frozen external12 manifest required')
 if requested_ids is not None:raise ValueError('This protocol renders every declared external object')
 if not f.get('settling_provenance',{}).get('complete') or not all(r.get('settling_ready')for r in rows):raise ValueError('Measured stable initialization required')
 validate_camera(f['camera']);inputs={str(B/'render_s40_observations.py'):PILOT_RENDERER_SHA256}
 for pk,hk in [('source_inventory','source_inventory_sha256')]:inputs[f[pk]]=f[hk]
 ref=f['external_protocol_ref'];inputs[ref['path']]=ref['sha256'];p=json.loads(Path(ref['path']).read_text())
 if [r['object_id']for r in p['objects']]!=ids:raise ValueError('External protocol object order mismatch')
 inputs.update(p['source_sha256']);inputs.update(p['input_sha256'])
 for r in rows:
  if r['id']!=r['object_id'] or r['split']!='external_validation' or r['scale']!=1:raise ValueError('External identity/units changed')
  r['world_T_object']=_rigid(r['world_T_object'],r['object_id'])
  if not np.isfinite(np.asarray(r['world_T_object'])).all():raise ValueError('Nonfinite object transform')
  for pk,hk in [('mesh_path','mesh_sha256'),('urdf_path','urdf_sha256'),('pose_path','pose_sha256')]:inputs[r[pk]]=r[hk]
  for part in r['convex_parts']:inputs[part['path']]=part['sha256']
  for x,h in r['source_input_sha256'].items():inputs[x]=h
  s=r['settling'];inputs[s['trace']]=s['trace_sha256']
 verify_frozen_inputs(path,digest,inputs)
 return f,rows,digest,inputs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--objects-json',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--object-ids', nargs='+', help='Explicit subset of the full frozen S40 manifest; IDs or comma-separated IDs')
    p.add_argument('--validate-only', action='store_true', help='Verify full manifest/inputs without starting Isaac or writing outputs')
    a=p.parse_args();a.objects_json=a.objects_json.resolve();a.output_dir=a.output_dir.resolve()
    frozen,rows,manifest_digest,input_digests=load_manifest(a.objects_json,a.object_ids)
    if a.output_dir.exists():raise FileExistsError('Refusing to overwrite render output: '+str(a.output_dir))
    if a.validate_only:
        print('EXTERNAL_RENDER_INPUTS_VALID '+json.dumps(dict(manifest_sha256=manifest_digest,active_object_count=len(rows),
            selected_ids=[r['id']for r in rows],frozen_file_count=len(input_digests),physics_steps=0,isaac_started=False)),flush=True)
        return
    prepare_output(a.objects_json,a.output_dir,manifest_digest)
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    started=time.monotonic();progress=dict(schema='anydex-boya-external-isaac-rgbd-v1',status='starting_isaac',
        source_objects_sha256=manifest_digest,renderer_source_sha256=sha(__file__),observations=[],
        frozen_input_sha256=input_digests,source_active_object_count=len(rows),selected_object_ids=[r['id']for r in rows],
        readonly_pilot_source_sha256=READONLY_PILOT_SOURCE_SHA256,readonly_camera_geometry_helper_sha256=PILOT_RENDERER_SHA256,
        source_inventory=frozen['source_inventory'],source_inventory_sha256=frozen['source_inventory_sha256'],
        source_provenance_boundary=PROVENANCE_BOUNDARY,
        original_AnyDex_CAD_identity_verified=False,paper_real_scene_protocol_aligned=False,
        explicit_physics_steps=0,control_steps=0,grasp_attempts=0,robot_present=False)
    save(a.output_dir/'render_progress.json',progress)
    launcher=None;current=None
    try:
        from isaaclab.app import AppLauncher
        launcher=AppLauncher(headless=True,enable_cameras=True,device='cpu',kit_args='--/log/file=/dev/null')
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
            path='/World/ObservedObjects/object_'+row['id'];mesh=UsdGeom.Mesh.Define(sim.stage,path)
            vertices=np.asarray(m.vertices,dtype=np.float32);faces=np.asarray(m.faces,dtype=np.int32)
            mesh.CreatePointsAttr([Gf.Vec3f(*map(float,v))for v in vertices]);mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none);mesh.CreateDoubleSidedAttr(True)
            sim_utils.bind_visual_material(path,'/World/ObservationMaterial')
            imageable=UsdGeom.Imageable(mesh.GetPrim());imageable.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            back=np.array(mesh.GetPointsAttr().Get(),dtype=float);max_error=float(np.abs(back-m.vertices).max())
            if max_error>1e-7:raise ValueError('USD world mesh points differ from source')
            mesh_readback[row['id']]=dict(vertices=len(vertices),triangles=len(faces),max_world_vertex_roundtrip_error_m=max_error,
                world_AABB_m=[back.min(0).tolist(),back.max(0).tolist()],native_dynamic_body=False,render_geometry='Official EGAD source triangles uniformly converted to metres, frozen settled rotation')
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
            verify_frozen_inputs(a.objects_json,manifest_digest,input_digests)
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
                observation_protocol='Isolated external EGAD source mesh/ground view at measured settled pose; same frozen Boya camera',
                source_provenance_boundary=PROVENANCE_BOUNDARY,source_external_object_id=row['object_id'],
                original_AnyDex_CAD_identity_verified=False,paper_real_scene_protocol_aligned=False,paper_physical_properties_aligned=False,
                T_WC=actual.tolist(),declared_T_WC=T.tolist(),camera_pose_max_error=pose_error,intrinsic_matrix=K.tolist(),depth_factor=1000.,
                depth_range_m=[near,far],valid_depth_pixels=int(valid.sum()),camera_source=camera_spec,
                files={name:dict(path=str(out/name),sha256=sha(out/name))for name in ['color.png','depth.png','meta.mat']},
                network_input_files=['color.png','depth.png','meta.mat'],object_root=row['object_root'],world_T_object=row['world_T_object'],object_scale=row['scale'],
                object_geometry=dict(type='mesh',source_mesh=row['mesh_path'],source_mesh_sha256=row['mesh_sha256'],source_urdf=row['urdf_path'],source_urdf_sha256=row['urdf_sha256']),
                source_pose_index=row['selected_source_pose_index'],source_pose_sha256=row['pose_sha256'],source_active_list_sha256=frozen['source_active_list_sha256'],
                source_inventory=frozen['source_inventory'],source_inventory_sha256=frozen['source_inventory_sha256'],
                readonly_camera_geometry_helper_sha256=PILOT_RENDERER_SHA256,
                source_object_selection_sha256=manifest_digest,source_renderer_sha256=sha(__file__),object_mass_kg=row['object_mass_kg'],object_friction=row['object_friction'],
                visual_material='Shared constant (.85,.60,.25) synthetic material for all objects; no source texture claim',
                control_steps=0,explicit_physics_steps=0,grasp_attempts=0,pybullet_imported='pybullet'in sys.modules,object_pose_as_network_input=False,
                mesh_readback=mesh_readback[current],geometry_check_sha256=sha(out/'render_geometry_check.json'),
                render_frames=16,elapsed_wall_s=time.monotonic()-started)
            save(out/'observation.json',report)
            progress['observations'].append(dict(id=current,object_id=row['object_id'],observation_json=str(out/'observation.json'),sha256=sha(out/'observation.json'),status='rendered'))
            progress['status']='rendering';save(a.output_dir/'render_progress.json',progress)
            print('BOYA_EXTERNAL_OBSERVATION_READY '+json.dumps(progress['observations'][-1]),flush=True)
        verify_frozen_inputs(a.objects_json,manifest_digest,input_digests)
        progress.update(status='complete',elapsed_wall_s=time.monotonic()-started,isaac_processes=1)
        save(a.output_dir/'render_progress.json',progress);save(a.output_dir/'OBSERVATIONS.json',progress)
        print('BOYA_EXTERNAL_RENDER_COMPLETE '+json.dumps({k:v for k,v in progress.items()if k!='observations'}),flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        progress.update(status='failed',failed_object=current,error=traceback.format_exc(),elapsed_wall_s=time.monotonic()-started)
        save(a.output_dir/'FAILED_RENDER.json',progress);save(a.output_dir/'render_progress.json',progress)
        raise
    finally:
        if launcher is not None:launcher.app.close()


if __name__=='__main__':main()
