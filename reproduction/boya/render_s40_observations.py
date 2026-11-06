"""Render a frozen GraspNet-derived S40 manifest through native Isaac RTX.

This independent entry does not mutate the frozen pilot/collection renderer.
It preserves the pilot optics, native camera readback, source first-hit checks,
and color.png/depth.png/meta.mat network interface. These are synthetic views
of GraspNet-derived assets, not the paper's real RGB-D scenes or verified CAD.
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


MANIFEST_SCHEMA = 'anydex_boya_s40_scene_manifest_v1'
S40_IDS = frozenset((0, 2, 5, 7, 8, 9, 11, 14, 15, 17,
    18, 20, 21, 22, 26, 27, 29, 30, 34, 36,
    37, 38, 40, 41, 43, 44, 46, 48, 51, 52,
    56, 57, 58, 60, 61, 62, 63, 66, 69, 70))
READONLY_PILOT_SOURCE_SHA256 = 'e541b3fa753954f19abf74c631cc6f3532b604f7e786f06e2d9d9b8e3dbd7307'
PROVENANCE_BOUNDARY = (
    'GraspNet-derived S40 assets rendered in synthetic isolated tabletop scenes; '
    'source CAD bytes not matched to the AnyDex author CAD, real RGB-D scene '
    'distribution and physical object parameters not aligned to the paper. '
    'No DGN_5k scale or thickness/width exclusion policy is applied.')


def _digest(value, label):
    import re
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Expected lowercase SHA256: ' + label)
    return value


def _number(value, label, positive=False):
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected finite number: ' + label)
    if positive and value <= 0:
        raise ValueError('Expected positive number: ' + label)
    return float(value)


def _rigid(value, label):
    import numpy as np
    T = np.asarray(value, dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-7, rtol=0)
            or abs(np.linalg.det(T[:3, :3])-1) > 1e-7):
        raise ValueError('Expected proper rigid transform: ' + label)
    return T.tolist()


def _source_identity(row):
    import re
    # This is source-ID checking, not a semantic/grasp classification rule.
    # Builders may supply an explicit integer or use one of these path-safe IDs.
    value = row.get('source_graspnet_object_id', row.get('graspnet_object_id'))
    if value is None:
        value = str(row['object_id'])
        match = re.fullmatch(r'(?:(?:graspnet|s40)_)?(\d{3})', value, flags=re.IGNORECASE)
        if match is None:
            raise ValueError('Missing explicit GraspNet source identity: ' + value)
        value = int(match.group(1))
    elif isinstance(value, str) and value.isdigit():
        value = int(value)
    if type(value) is not int or value not in S40_IDS:
        raise ValueError('Object is not in the frozen source S40 union: ' + str(value))
    return value


def _physical_alias(row, primary, alternate, positive):
    value = row.get(primary, row.get(alternate))
    value = _number(value, primary, positive=positive)
    if value < 0:
        raise ValueError('Negative physical parameter: ' + primary)
    if primary in row and alternate in row and value != _number(row[alternate], alternate):
        raise ValueError('Conflicting physical parameter aliases: ' + primary)
    return value


def validate_camera(camera):
    import numpy as np
    # Keep the original 1280x720 640px focal protocol. Changing optics needs a
    # new declared protocol rather than silently bypassing native K checks.
    if camera.get('width') != 1280 or camera.get('height') != 720:
        raise ValueError('S40 renderer preserves the original 1280x720 optics')
    for key in ('fx', 'fy'):
        if key in camera and _number(camera[key], key, True) != 640.:
            raise ValueError('S40 renderer preserves 640px focal length')
    focal = _number(camera.get('focal_length'), 'focal_length', True)
    aperture = _number(camera.get('horizontal_aperture'), 'horizontal_aperture', True)
    if abs(1280*focal/aperture-640.) > 1e-9:
        raise ValueError('Focal/aperture ratio differs from frozen native intrinsics')
    for key in ('eye_world_m', 'target_world_m', 'up_world'):
        value = np.asarray(camera.get(key), dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError('Invalid camera vector: ' + key)
    look_at_ros(camera['eye_world_m'], camera['target_world_m'], camera['up_world'])
    ranges = {}
    for key in ('depth_range_m', 'clipping_range_m'):
        value = camera.get(key)
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError('Expected camera depth pair: ' + key)
        ranges[key] = [_number(v, key, True) for v in value]
        if not ranges[key][0] < ranges[key][1]:
            raise ValueError('Invalid camera depth interval: ' + key)
    if (ranges['depth_range_m'][0] < ranges['clipping_range_m'][0]
            or ranges['depth_range_m'][1] > ranges['clipping_range_m'][1]):
        raise ValueError('Network depth interval exceeds native clipping range')


def verify_frozen_inputs(manifest_path, manifest_digest, input_digests):
    if sha(manifest_path) != manifest_digest:
        raise ValueError('Frozen scene manifest changed: ' + str(manifest_path))
    for path, expected in input_digests.items():
        if sha(path) != expected:
            raise ValueError('Frozen source changed: ' + str(path))


def load_manifest(manifest_path, requested_ids=None):
    """Validate all 40 inputs before applying any explicit rendering subset."""
    import re
    manifest_path = Path(manifest_path).resolve()
    payload = manifest_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    frozen = json.loads(payload)
    if frozen.get('schema') != MANIFEST_SCHEMA:
        raise ValueError('Expected independent S40 scene manifest schema')
    if (type(frozen.get('active_object_count')) is not int or frozen['active_object_count'] != 40
            or _number(frozen.get('scale'), 'manifest scale') != 1):
        raise ValueError('S40 active count must be 40 and source scale must be 1')
    _digest(frozen.get('source_active_list_sha256'), 'source_active_list_sha256')
    rows = frozen.get('objects')
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError('Full frozen S40 scene manifest required, then use --object-ids')
    validate_camera(frozen.get('camera', {}))
    validated = []
    input_digests = {}
    for original in rows:
        row = dict(original)
        for key in ('id', 'object_id'):
            if not isinstance(row.get(key), str) or re.fullmatch('[A-Za-z_][A-Za-z0-9_]*|[0-9]{3}', row[key]) is None:
                raise ValueError('Expected safe object identity: ' + key)
        row['source_graspnet_object_id'] = _source_identity(row)
        if _number(row.get('scale'), 'object scale') != 1:
            raise ValueError('Per-object S40 scale must remain 1')
        if type(row.get('selected_source_pose_index')) is not int or row['selected_source_pose_index'] < 0:
            raise ValueError('Expected nonnegative selected source pose index')
        row['world_T_object'] = _rigid(row.get('world_T_object'), row['id'])
        root = Path(row['object_root'])
        row['object_root'] = str((root if root.is_absolute() else manifest_path.parent/root).resolve())
        if Path(row['object_root']).name != row['object_id']:
            raise ValueError('Object root basename must equal object_id for execution compatibility')
        row['object_mass_kg'] = _physical_alias(row, 'object_mass_kg', 'mass', True)
        row['object_friction'] = _physical_alias(row, 'object_friction', 'friction', False)
        for path_key, hash_key in [('mesh_path', 'mesh_sha256'), ('urdf_path', 'urdf_sha256'), ('pose_path', 'pose_sha256')]:
            path = Path(row[path_key])
            path = (path if path.is_absolute() else manifest_path.parent/path).resolve()
            expected = _digest(row[hash_key], hash_key)
            if not path.is_file() or sha(path) != expected:
                raise ValueError('Frozen source changed: ' + str(path))
            if str(path) in input_digests and input_digests[str(path)] != expected:
                raise ValueError('Conflicting hashes for shared source: ' + str(path))
            input_digests[str(path)] = expected
            row[path_key] = str(path)
        validated.append(row)
    for key in ('id', 'object_id', 'source_graspnet_object_id'):
        if len({row[key] for row in validated}) != 40:
            raise ValueError('Exactly forty unique S40 identities required: ' + key)
    if {row['source_graspnet_object_id'] for row in validated} != S40_IDS:
        raise ValueError('Scene manifest does not contain the original S40 identity union')
    # Optional source inventory is a frozen dependency if a path is supplied.
    source_list = frozen.get('source_active_list')
    if source_list is not None:
        path = Path(source_list)
        path = (path if path.is_absolute() else manifest_path.parent/path).resolve()
        if not path.is_file() or sha(path) != frozen['source_active_list_sha256']:
            raise ValueError('Source active-list inventory changed')
        input_digests[str(path)] = frozen['source_active_list_sha256']
    tokens = []
    for arg in requested_ids or []:
        tokens.extend(arg.split(','))
    if requested_ids is not None and not tokens:
        raise ValueError('Explicit object filter must be nonempty')
    if tokens:
        selected = set()
        for token in tokens:
            matches = [row['id'] for row in validated if token in
                (row['id'], row['object_id'], f"{row['source_graspnet_object_id']:03d}")]
            if len(matches) != 1:
                raise ValueError('Unknown/ambiguous requested object identity: ' + token)
            if matches[0] in selected:
                raise ValueError('Duplicate requested object identity: ' + token)
            selected.add(matches[0])
        chosen = [row for row in validated if row['id'] in selected]
    else:
        chosen = validated
    verify_frozen_inputs(manifest_path, digest, input_digests)
    return frozen, chosen, digest, input_digests


def prepare_output(manifest_path, output_dir, manifest_digest):
    """Never overwrite an existing directory, including a partial prior run."""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError('Refusing to overwrite render output: ' + str(output_dir))
    if sha(manifest_path) != manifest_digest:
        raise ValueError('Scene manifest changed before output creation')
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_path, output_dir/'objects.json')
    if sha(output_dir/'objects.json') != manifest_digest:
        raise ValueError('Scene manifest changed while freezing output')
    return output_dir


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
        print('S40_RENDER_INPUTS_VALID '+json.dumps(dict(manifest_sha256=manifest_digest,active_object_count=40,
            selected_ids=[r['id']for r in rows],frozen_file_count=len(input_digests),physics_steps=0,isaac_started=False)),flush=True)
        return
    prepare_output(a.objects_json,a.output_dir,manifest_digest)
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    started=time.monotonic();progress=dict(schema='anydex-boya-s40-isaac-rgbd-v1',status='starting_isaac',
        source_objects_sha256=manifest_digest,renderer_source_sha256=sha(__file__),observations=[],
        frozen_input_sha256=input_digests,source_active_object_count=40,selected_object_ids=[r['id']for r in rows],
        readonly_pilot_source_sha256=READONLY_PILOT_SOURCE_SHA256,source_provenance_boundary=PROVENANCE_BOUNDARY,
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
                world_AABB_m=[back.min(0).tolist(),back.max(0).tolist()],native_dynamic_body=False,render_geometry='GraspNet-derived source triangle mesh, scale 1, frozen supported rotation')
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
                observation_protocol='Isolated GraspNet-derived source mesh/ground view before arm approach; one declared shared oblique camera for selected S40 objects',
                source_provenance_boundary=PROVENANCE_BOUNDARY,source_graspnet_object_id=row['source_graspnet_object_id'],
                original_AnyDex_CAD_identity_verified=False,paper_real_scene_protocol_aligned=False,paper_physical_properties_aligned=False,
                T_WC=actual.tolist(),declared_T_WC=T.tolist(),camera_pose_max_error=pose_error,intrinsic_matrix=K.tolist(),depth_factor=1000.,
                depth_range_m=[near,far],valid_depth_pixels=int(valid.sum()),camera_source=camera_spec,
                files={name:dict(path=str(out/name),sha256=sha(out/name))for name in ['color.png','depth.png','meta.mat']},
                network_input_files=['color.png','depth.png','meta.mat'],object_root=row['object_root'],world_T_object=row['world_T_object'],object_scale=row['scale'],
                object_geometry=dict(type='mesh',source_mesh=row['mesh_path'],source_mesh_sha256=row['mesh_sha256'],source_urdf=row['urdf_path'],source_urdf_sha256=row['urdf_sha256']),
                source_pose_index=row['selected_source_pose_index'],source_pose_sha256=row['pose_sha256'],source_active_list_sha256=frozen['source_active_list_sha256'],
                source_object_selection_sha256=manifest_digest,source_renderer_sha256=sha(__file__),object_mass_kg=row['object_mass_kg'],object_friction=row['object_friction'],
                visual_material='Shared constant (.85,.60,.25) synthetic material for all objects; no source texture claim',
                control_steps=0,explicit_physics_steps=0,grasp_attempts=0,pybullet_imported='pybullet'in sys.modules,object_pose_as_network_input=False,
                mesh_readback=mesh_readback[current],geometry_check_sha256=sha(out/'render_geometry_check.json'),
                render_frames=16,elapsed_wall_s=time.monotonic()-started)
            save(out/'observation.json',report)
            progress['observations'].append(dict(id=current,object_id=row['object_id'],observation_json=str(out/'observation.json'),sha256=sha(out/'observation.json'),status='rendered'))
            progress['status']='rendering';save(a.output_dir/'render_progress.json',progress)
            print('BOYA_S40_OBSERVATION_READY '+json.dumps(progress['observations'][-1]),flush=True)
        verify_frozen_inputs(a.objects_json,manifest_digest,input_digests)
        progress.update(status='complete',elapsed_wall_s=time.monotonic()-started,isaac_processes=1)
        save(a.output_dir/'render_progress.json',progress);save(a.output_dir/'OBSERVATIONS.json',progress)
        print('BOYA_S40_RENDER_COMPLETE '+json.dumps({k:v for k,v in progress.items()if k!='observations'}),flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        progress.update(status='failed',failed_object=current,error=traceback.format_exc(),elapsed_wall_s=time.monotonic()-started)
        save(a.output_dir/'FAILED_RENDER.json',progress);save(a.output_dir/'render_progress.json',progress)
        raise
    finally:
        if launcher is not None:launcher.app.close()


if __name__=='__main__':main()
