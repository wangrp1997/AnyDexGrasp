"""Frozen single-cube scene and real partial RGB-D for independent AnyDex smoke.

This module owns no controller. create_scene/render_scene accept the caller's
PyBullet client, so inference and later control can share one actual scene.
The command-line renderer forbids stepSimulation and integrates zero steps.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image
import scipy.io
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = ROOT/'outputs/native_deps/python_overlay'
SPEC = {
    'schema':'AnyDex-independent-synthetic-cube-v1',
    'object': {'mesh':'cube.obj','mesh_scale':[.05,.05,.05],
               'collision_box_size_m':[.05,.05,.05],
               'position_world_m':[0,.08,.025],'orientation_world_xyzw':[0,0,0,1],
               'mass_kg':.1,'inertia_diagonal_kg_m2':[.1*.05**2/6]*3,
               'lateral_friction':1.,'rolling_friction':0.,'spinning_friction':0.,
               'restitution':0.,'linear_damping':.04,'angular_damping':.04,
               'collision_margin_m':.001,'rgba':[.85,.60,.25,1.]},
    'table': {'size_m':[1.,1.,.05],'position_world_m':[0,0,-.025],
              'mass_kg':0.,'lateral_friction':1.,'restitution':0.,
              'collision_margin_m':.001,'rgba':[.65,.68,.72,1.]},
    'physics': {'time_step_s':1/240,'num_sub_steps':1,'solver_iterations':100,
                'gravity_m_s2':[0,0,-9.81],'deterministic_overlapping_pairs':1,
                'real_time_simulation':False},
    'camera': {'width':1280,'height':720,'fx':640.,'fy':640.,'cx':640.,'cy':360.,
               'near_m':.1,'far_m':1.,'depth_factor':1000.,
               'position_world_m':[0,0,.5],
               'R_WC':[[1,0,0],[0,-1,0],[0,0,-1]],
               'renderer':'ER_TINY_RENDERER','shadow':0,'light_direction':[1,1,3]},
    'boundary':'One new synthetic cube/pose, declared simulator assumptions; not the authors physical benchmark'
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _state(p,client_id,scene):
    result={}
    for name in ['object','table']:
        body=scene[name+'_id']
        position,orientation=p.getBasePositionAndOrientation(body,physicsClientId=client_id)
        linear,angular=p.getBaseVelocity(body,physicsClientId=client_id)
        dynamics=p.getDynamicsInfo(body,-1,physicsClientId=client_id)
        aabb=p.getAABB(body,-1,physicsClientId=client_id)
        result[name]={'position_world_m':list(position),'orientation_world_xyzw':list(orientation),
                      'linear_velocity_m_s':list(linear),'angular_velocity_rad_s':list(angular),
                      'mass_kg':float(dynamics[0]),'lateral_friction':float(dynamics[1]),
                      'local_inertia_diagonal_kg_m2':list(dynamics[2]),
                      'restitution':float(dynamics[5]),'rolling_friction':float(dynamics[6]),
                      'spinning_friction':float(dynamics[7]),'collision_margin_m':float(dynamics[11]),
                      'aabb_world_m':[list(x) for x in aabb]}
    return result


def create_scene(p,client_id,asset_dir=None):
    """Populate an empty caller-owned client; never reset or integrate it."""
    if p.getNumBodies(physicsClientId=client_id)!=0:
        raise ValueError('create_scene requires an empty client; it never resets an existing world')
    asset_dir=Path(asset_dir) if asset_dir else OVERLAY/'pybullet_data'
    cube=asset_dir/'cube.obj'
    if not cube.is_file():raise FileNotFoundError(cube)
    spec=copy.deepcopy(SPEC);physics=spec['physics'];obj=spec['object'];table=spec['table']
    p.setRealTimeSimulation(0,physicsClientId=client_id)
    p.setTimeStep(physics['time_step_s'],physicsClientId=client_id)
    p.setPhysicsEngineParameter(numSubSteps=physics['num_sub_steps'],
                               numSolverIterations=physics['solver_iterations'],
                               deterministicOverlappingPairs=1,physicsClientId=client_id)
    p.setGravity(*physics['gravity_m_s2'],physicsClientId=client_id)
    table_collision=p.createCollisionShape(p.GEOM_BOX,halfExtents=(np.array(table['size_m'])/2).tolist(),physicsClientId=client_id)
    table_visual=p.createVisualShape(p.GEOM_BOX,halfExtents=(np.array(table['size_m'])/2).tolist(),rgbaColor=table['rgba'],physicsClientId=client_id)
    table_id=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=table_collision,
                               baseVisualShapeIndex=table_visual,basePosition=table['position_world_m'],
                               physicsClientId=client_id)
    p.changeDynamics(table_id,-1,lateralFriction=table['lateral_friction'],restitution=0,
                     collisionMargin=table['collision_margin_m'],physicsClientId=client_id)
    collision=p.createCollisionShape(p.GEOM_BOX,halfExtents=(np.array(obj['collision_box_size_m'])/2).tolist(),physicsClientId=client_id)
    visual=p.createVisualShape(p.GEOM_MESH,fileName=str(cube),meshScale=obj['mesh_scale'],
                               rgbaColor=obj['rgba'],physicsClientId=client_id)
    object_id=p.createMultiBody(baseMass=obj['mass_kg'],baseCollisionShapeIndex=collision,
                                baseVisualShapeIndex=visual,basePosition=obj['position_world_m'],
                                baseOrientation=obj['orientation_world_xyzw'],physicsClientId=client_id)
    p.changeDynamics(object_id,-1,localInertiaDiagonal=obj['inertia_diagonal_kg_m2'],
                     lateralFriction=obj['lateral_friction'],rollingFriction=0,spinningFriction=0,
                     restitution=0,linearDamping=obj['linear_damping'],angularDamping=obj['angular_damping'],
                     collisionMargin=obj['collision_margin_m'],physicsClientId=client_id)
    files=[cube,asset_dir/'cube_small.urdf']
    sources=[dict(path=str(x.resolve()),bytes=x.stat().st_size,sha256=sha(x)) for x in files]
    scene={'object_id':object_id,'cube_id':object_id,'table_id':table_id,'client_id':client_id,
           'object_half_extents_m':(np.array(obj['collision_box_size_m'])/2).tolist(),
           'object_top_z_world_m':.05,'table_top_z_world_m':0.,
           'spec':spec,'source_assets':sources,
           'scene_signature_sha256':canonical_sha(dict(spec=spec,source_assets=sources)),
           'cube_small_urdf_used_only_as_geometry_provenance_not_loaded_with_legacy_inertia':True}
    initial=_state(p,client_id,scene)
    assert np.allclose(initial['object']['local_inertia_diagonal_kg_m2'],obj['inertia_diagonal_kg_m2'],rtol=0,atol=1e-14)
    assert initial['object']['mass_kg']==.1 and initial['object']['lateral_friction']==1.
    scene['initial_state']=initial
    return scene


def assert_scene_matches_manifest(p,client_id,scene,manifest):
    """Verify declared geometry/dynamics/actual state, without writing the scene."""
    manifest=json.loads(Path(manifest).read_text()) if isinstance(manifest,(str,Path)) else manifest
    if scene['scene_signature_sha256']!=manifest['scene_signature_sha256']:
        raise ValueError('Scene signature differs from rendered observation')
    current=_state(p,client_id,scene)
    for name in ['object','table']:
        for key in ['position_world_m','orientation_world_xyzw','linear_velocity_m_s',
                    'angular_velocity_rad_s','local_inertia_diagonal_kg_m2','mass_kg','lateral_friction']:
            if not np.allclose(current[name][key],manifest['actual_state'][name][key],rtol=0,atol=1e-12):
                raise ValueError('Scene observation binding differs: '+name+'/'+key)
    return True


def camera_matrices(p,spec):
    cam=spec['camera'];position=np.array(cam['position_world_m'],dtype=float)
    rotation=np.array(cam['R_WC'],dtype=float)
    target=position+rotation[:,2]
    view=p.computeViewMatrix(position.tolist(),target.tolist(),(-rotation[:,1]).tolist())
    fov=math.degrees(2*math.atan(cam['height']/(2*cam['fy'])))
    projection=p.computeProjectionMatrixFOV(fov,cam['width']/cam['height'],cam['near_m'],cam['far_m'])
    view_matrix=np.array(view).reshape(4,4,order='F')
    T_CW=np.diag([1.,-1.,-1.,1.])@view_matrix
    T_WC=np.linalg.inv(T_CW)
    expected=np.eye(4);expected[:3,:3]=rotation;expected[:3,3]=position
    assert np.allclose(T_WC,expected,rtol=0,atol=1e-12)
    K=np.array([[cam['fx'],0,cam['cx']],[0,cam['fy'],cam['cy']],[0,0,1]],dtype=float)
    return view,projection,K,T_WC,T_CW


def render_scene(p,client_id,scene,output_dir):
    """Render and validate the actual client without stepping or changing it."""
    output_dir=Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Use a new render output directory; prior observations are immutable')
    output_dir.mkdir(parents=True,exist_ok=True)
    before=_state(p,client_id,scene);cam=scene['spec']['camera']
    view,projection,K,T_WC,T_CW=camera_matrices(p,scene['spec'])
    result=p.getCameraImage(cam['width'],cam['height'],viewMatrix=view,projectionMatrix=projection,
                            renderer=p.ER_TINY_RENDERER,flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                            shadow=cam['shadow'],lightDirection=cam['light_direction'],physicsClientId=client_id)
    rgb=np.asarray(result[2],dtype=np.uint8).reshape(cam['height'],cam['width'],4)[...,:3]
    buffer=np.asarray(result[3],dtype=np.float64).reshape(cam['height'],cam['width'])
    segmentation=np.asarray(result[4],dtype=np.int32).reshape(cam['height'],cam['width'])
    valid=(buffer<1)&(segmentation>=0)
    near,far=cam['near_m'],cam['far_m']
    metric=far*near/(far-(far-near)*buffer)
    metric[~valid]=0
    depth=np.rint(metric*cam['depth_factor']).astype(np.uint16)
    u,v=np.meshgrid(np.arange(cam['width']),np.arange(cam['height']))
    z=depth.astype(np.float64)/cam['depth_factor']
    points=np.stack([(u-K[0,2])/K[0,0]*z,(v-K[1,2])/K[1,1]*z,z],axis=-1)
    world=points@T_WC[:3,:3].T+T_WC[:3,3]
    ids=segmentation&((1<<24)-1)
    cube_mask=valid&(ids==scene['object_id']);table_mask=valid&(ids==scene['table_id'])
    if cube_mask.sum()<100 or table_mask.sum()<100:
        raise ValueError('Frozen object/table not visible in rendered image')
    local=world[cube_mask]-np.array(before['object']['position_world_m'])
    half=np.array(scene['spec']['object']['collision_box_size_m'])/2
    surface_error=np.abs(np.max(np.abs(local)-half,axis=1))
    table_error=np.abs(world[table_mask,2])
    # Independent native rays on flat visible faces avoid ill-conditioned steep
    # side-depth slopes at the renderer's subpixel raster convention.
    expected_top_camera_z=.5-(before['object']['position_world_m'][2]+half[2])
    flat_cube=binary_erosion(cube_mask,iterations=2)&(np.abs(metric-expected_top_camera_z)<1e-5)
    flat_table=binary_erosion(table_mask,iterations=3)
    chosen=[]
    for mask in [flat_cube,flat_table]:
        indices=np.flatnonzero(mask)
        if not len(indices):raise ValueError('No flat validation surface')
        chosen.extend(indices[np.linspace(0,len(indices)-1,min(32,len(indices)),dtype=int)].tolist())
    flat_points=points.reshape(-1,3)[chosen]
    rays=flat_points/flat_points[:,2:3]*far
    ray_to=rays@T_WC[:3,:3].T+T_WC[:3,3]
    ray_from=np.repeat(T_WC[:3,3][None],len(chosen),axis=0)
    hits=p.rayTestBatch(ray_from.tolist(),ray_to.tolist(),physicsClientId=client_id)
    ray_world=np.array([hit[3] for hit in hits])
    observed_world=world.reshape(-1,3)[chosen]
    ray_error=np.linalg.norm(ray_world-observed_world,axis=1)
    expected_ids=ids.reshape(-1)[chosen]
    assert np.array_equal(np.array([hit[0] for hit in hits]),expected_ids)
    vertices=[]
    mesh=Path(scene['source_assets'][0]['path'])
    for line in mesh.read_text().splitlines():
        if line.startswith('v '):vertices.append([float(x) for x in line.split()[1:4]])
    vertices=np.array(vertices)*np.array(scene['spec']['object']['mesh_scale'])
    assert np.allclose(vertices.min(0),-half,atol=1e-12) and np.allclose(vertices.max(0),half,atol=1e-12)
    cube_world=vertices+np.array(before['object']['position_world_m'])
    cube_camera=cube_world@T_CW[:3,:3].T+T_CW[:3,3]
    projected=cube_camera[:,:2]/cube_camera[:,2:3]*np.array([K[0,0],K[1,1]])+K[:2,2]
    pixels=np.c_[u[cube_mask],v[cube_mask]]
    bbox_difference=max(np.max(np.abs(pixels.min(0)-projected.min(0))),np.max(np.abs(pixels.max(0)-projected.max(0))))
    # A millimetre depth PNG and finite pixel raster have a declared 1.5 mm
    # geometry tolerance; this is an input/render check, not a collision waiver.
    assert float(surface_error.max())<.0015
    assert float(table_error.max())<.00051
    assert float(ray_error.max())<.0015
    assert float(bbox_difference)<2.1
    after=_state(p,client_id,scene)
    assert before==after,'Rendering or validation changed physical state'
    Image.fromarray(rgb).save(output_dir/'color.png')
    Image.fromarray(depth).save(output_dir/'depth.png')
    scipy.io.savemat(output_dir/'meta.mat',{'intrinsic_matrix':K,'factor_depth':np.array([[cam['depth_factor']]])})
    np.save(output_dir/'depth_m.npy',metric.astype(np.float32))
    np.save(output_dir/'segmentation_for_validation_only.npy',segmentation)
    np.save(output_dir/'camera_points_m.npy',points[valid].astype(np.float32))
    validation=dict(actual_renderer='PyBullet DIRECT-compatible ER_TINY_RENDERER',
                    render_state_unchanged=True,integrated_physics_steps=0,
                    visible_cube_pixels=int(cube_mask.sum()),visible_table_pixels=int(table_mask.sum()),
                    complete_mesh_points_fed_as_perception=False,segmentation_fed_to_original_network=False,
                    cube_surface_max_error_m=float(surface_error.max()),
                    table_surface_max_error_m=float(table_error.max()),
                    native_flat_surface_ray_tests=len(chosen),native_ray_max_error_m=float(ray_error.max()),
                    projected_object_bbox_max_error_pixels=float(bbox_difference),
                    depth_factor=cam['depth_factor'],depth_conversion='far*near/(far-(far-near)*OpenGL_buffer)',
                    depth_PNG='round(metric_depth*1000), uint16; zero only for background',
                    native_ray_domain='flat cube top/table interiors; full visible cube geometry checked separately',
                    geometry_tolerance_m=.0015,not_a_collision_or_dynamic_gate=True,
                    meta_contains_no_fake_GraspNet_class_ids=True)
    files=[]
    for name in ['color.png','depth.png','meta.mat','depth_m.npy','segmentation_for_validation_only.npy','camera_points_m.npy']:
        path=output_dir/name;files.append(dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path)))
    manifest=dict(scene_signature_sha256=scene['scene_signature_sha256'],spec=scene['spec'],
                  source_assets=scene['source_assets'],actual_state=after,
                  physics_engine_parameters=p.getPhysicsEngineParameters(physicsClientId=client_id),
                  pybullet_API_version=p.getAPIVersion(),body_ids=dict(object=scene['object_id'],table=scene['table_id']),
                  camera=dict(K=K.tolist(),T_WC=T_WC.tolist(),T_CW=T_CW.tolist(),
                              view_matrix_column_major=view,projection_matrix_column_major=projection),
                  validation=validation,files=files,
                  boundary='New synthetic smoke input only; zero physics steps, no learned inference or grasp result')
    (output_dir/'scene_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    (output_dir/'RESULTS.json').write_text(json.dumps(dict(status='frozen_synthetic_scene_render_and_validation_passed',
                                                        physical_steps=0,validation=validation,
                                                        scene_manifest_sha256=sha(output_dir/'scene_manifest.json')),indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/synthetic_cube_scene_20261103_v1')
    parser.add_argument('--overlay',type=Path,default=OVERLAY)
    args=parser.parse_args();sys.path.insert(0,str(args.overlay))
    import pybullet as p
    client=p.connect(p.DIRECT)
    original_step=p.stepSimulation
    step_calls=[]
    def forbidden(*unused,**kwargs):
        step_calls.append(1)
        raise RuntimeError('The input renderer is zero-step; stepSimulation is forbidden')
    p.stepSimulation=forbidden
    try:
        scene=create_scene(p,client,args.overlay/'pybullet_data')
        manifest=render_scene(p,client,scene,args.output_dir)
        assert not step_calls
        assert_scene_matches_manifest(p,client,scene,manifest)
        print(json.dumps(dict(output_dir=str(args.output_dir),scene_signature_sha256=scene['scene_signature_sha256'],
                              physical_steps=0,validation=manifest['validation']),indent=2),flush=True)
    except Exception as exc:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        failure=dict(status='render_or_input_validation_failed',physical_steps=0,
                     error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        (args.output_dir/'FAILED_RENDER.json').write_text(json.dumps(failure,indent=2)+'\n')
        raise
    finally:
        p.stepSimulation=original_step
        p.disconnect(client)


if __name__=='__main__':main()
