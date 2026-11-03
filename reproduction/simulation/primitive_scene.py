"""Three frozen native-shape scenes and real partial RGB-D, with zero steps.

Public API: frozen_specs(), create_scene(p, client_id, spec_or_manifest),
render_scene(p, client_id, scene, output_dir), assert_scene_matches_manifest(...).
The caller owns its Bullet client. Scene creation never resets an existing world;
rendering never integrates physics. No object pose/segmentation is network input.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image
import scipy.io
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation

from synthetic_scene import SPEC as CUBE_SPEC, camera_matrices, _state

ROOT=Path(__file__).resolve().parents[2]
OVERLAY=ROOT/'outputs/native_deps/python_overlay'
DEFAULT_OUTPUT=ROOT/'outputs/dh3_primitive_generalization_20261103_v1/scenes'
_HASH_CACHE={}


def sha(path):
    path=Path(path);stat=path.stat();key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        _HASH_CACHE[key]=h.hexdigest()
    return _HASH_CACHE[key]


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def analytic_inertia(geometry,mass):
    kind=geometry['type']
    if kind=='box':
        x,y,z=geometry['size_m']
        return (mass/12*np.array([y*y+z*z,x*x+z*z,x*x+y*y])).tolist()
    r=geometry['radius_m']
    if kind=='sphere':return [2/5*mass*r*r]*3
    if kind=='cylinder':
        h=geometry['height_m'];side=mass*(3*r*r+h*h)/12
        return [side,side,mass*r*r/2]
    raise ValueError('Unsupported native primitive: '+kind)


def geometry_size(geometry):
    if geometry['type']=='box':return np.array(geometry['size_m'],dtype=float)
    radius=geometry['radius_m']
    return np.array([2*radius,2*radius,2*radius if geometry['type']=='sphere' else geometry['height_m']])


def frozen_specs():
    """Predeclared three shapes; all share the original cube camera/table/XY."""
    definitions={
        'cylinder':dict(type='cylinder',radius_m=.025,height_m=.09),
        'sphere':dict(type='sphere',radius_m=.03),
        'long_box':dict(type='box',size_m=[.1,.045,.035]),
    }
    specs={}
    for name,geometry in definitions.items():
        size=geometry_size(geometry);spec=copy.deepcopy(CUBE_SPEC)
        spec['schema']='AnyDex-independent-native-primitive-v1';spec['scene_id']=name
        spec['object']={
            'geometry':geometry,'geometry_backend':'PyBullet native dynamic primitive',
            'visual_geometry':(dict(type='procedural_cylinder_triangle_mesh',radial_segments=128,
                reason='Avoid TinyRenderer native-cylinder 32-point hull with undocumented 1mm visual inflation',
                analytic_radial_chord_error_bound_m=float(geometry['radius_m']*(1-np.cos(np.pi/128))))
                if geometry['type']=='cylinder' else
                dict(type='procedural_sphere_triangle_mesh',longitude_segments=64,latitude_segments=32,
                    reason='Avoid low-resolution native TinyRenderer sphere visual; native collision remains analytic',
                    analytic_triangle_radial_error_bound_m=float(geometry['radius_m']*(1-np.cos(np.sqrt((np.pi/32)**2+(2*np.pi/64)**2)))))
                if geometry['type']=='sphere' else dict(type='native_'+geometry['type'])),
            'collision_box_size_m':size.tolist(),
            'collision_box_size_interpretation':'Analytic bounding dimensions; actual collision shape is geometry.type',
            'position_world_m':[0,.08,float(size[2]/2)],'orientation_world_xyzw':[0,0,0,1],
            'mass_kg':.1,'inertia_diagonal_kg_m2':analytic_inertia(geometry,.1),
            'lateral_friction':1.,'rolling_friction':0.,'spinning_friction':0.,'restitution':0.,
            'linear_damping':.04,'angular_damping':.04,
            'collision_margin_m':geometry['radius_m'] if geometry['type']=='sphere' else .001,
            'collision_margin_interpretation':('Native sphere reports its radius in the margin slot; no additional radius padding'
                if geometry['type']=='sphere' else 'Native 1mm convex-shape margin; measured AABB equals declared dimensions'),
            'rgba':[.85,.60,.25,1.],
        }
        spec['boundary']='One predeclared synthetic native primitive/pose; simulator assumptions, not author physical benchmark'
        specs[name]=spec
    return specs


def _source_assets(p):
    paths=[Path(__file__),Path(__file__).with_name('synthetic_scene.py'),Path(p.__file__)]
    return [dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path)) for path in paths]


def _serial(value):
    if isinstance(value,bytes):return value.decode('utf-8',errors='replace')
    if isinstance(value,(list,tuple)):return [_serial(item) for item in value]
    return value


def _read_spec(value):
    if isinstance(value,(str,Path)):
        if str(value) in frozen_specs():value=frozen_specs()[str(value)]
        else:value=json.loads(Path(value).read_text())
    if not isinstance(value,dict):raise TypeError('Pass a frozen spec, scene manifest, manifest path, or scene id')
    expected=value if 'spec' in value else None
    spec=copy.deepcopy(value['spec'] if expected else value)
    obj=spec['object'];geometry=obj['geometry'];size=geometry_size(geometry)
    if not np.isfinite(size).all() or not (size>0).all():raise ValueError('Invalid primitive size')
    if obj['mass_kg']<=0:raise ValueError('Object must have positive dynamic mass')
    if not np.allclose(obj['inertia_diagonal_kg_m2'],analytic_inertia(geometry,obj['mass_kg']),rtol=0,atol=1e-15):
        raise ValueError('Inertia differs from the declared solid native geometry')
    if not np.allclose(size,obj['collision_box_size_m'],rtol=0,atol=1e-15):raise ValueError('Geometry dimensions differ')
    return spec,expected


def cylinder_visual_mesh(radius,height,segments):
    """Exact cap planes and inscribed side triangles; collision stays native."""
    angles=np.arange(segments)*2*np.pi/segments
    circle=np.column_stack([radius*np.cos(angles),radius*np.sin(angles)])
    vertices=np.concatenate([np.column_stack([circle,np.full(segments,-height/2)]),
        np.column_stack([circle,np.full(segments,height/2)]),[[0,0,-height/2],[0,0,height/2]]])
    indices=[]
    for i in range(segments):
        j=(i+1)%segments
        indices.extend([i,j,segments+j,i,segments+j,segments+i])
        indices.extend([2*segments,j,i,2*segments+1,segments+i,segments+j])
    return vertices.tolist(),indices


def sphere_visual_mesh(radius,longitudes,latitudes):
    """Inscribed UV triangles on declared analytic sphere; collision stays native."""
    theta=np.arange(longitudes)*2*np.pi/longitudes
    vertices=[[0,0,radius]]
    for i in range(1,latitudes):
        phi=np.pi*i/latitudes
        vertices.extend(np.column_stack([radius*np.sin(phi)*np.cos(theta),
            radius*np.sin(phi)*np.sin(theta),np.full(longitudes,radius*np.cos(phi))]).tolist())
    bottom=len(vertices);vertices.append([0,0,-radius]);indices=[]
    for j in range(longitudes):
        k=(j+1)%longitudes;indices.extend([0,1+j,1+k])
        for i in range(latitudes-2):
            a=1+i*longitudes+j;d=1+i*longitudes+k;b=a+longitudes;c=d+longitudes
            indices.extend([a,b,c,a,c,d])
        indices.extend([bottom,1+(latitudes-2)*longitudes+k,1+(latitudes-2)*longitudes+j])
    return vertices,indices


def create_scene(p,client_id,spec_or_manifest=None,*,spec=None):
    """Create native object/table in an EMPTY client, with no reset or step."""
    if p.getNumBodies(physicsClientId=client_id)!=0:raise ValueError('An empty caller-owned Bullet client is required')
    if spec is not None:
        if spec_or_manifest is not None:raise TypeError('Pass either spec_or_manifest or spec, not both')
        spec_or_manifest=spec
    spec,expected=_read_spec(spec_or_manifest);obj=spec['object'];table=spec['table'];physics=spec['physics'];g=obj['geometry']
    p.setRealTimeSimulation(0,physicsClientId=client_id)
    p.setTimeStep(physics['time_step_s'],physicsClientId=client_id)
    p.setPhysicsEngineParameter(numSubSteps=physics['num_sub_steps'],numSolverIterations=physics['solver_iterations'],
        deterministicOverlappingPairs=physics['deterministic_overlapping_pairs'],physicsClientId=client_id)
    p.setGravity(*physics['gravity_m_s2'],physicsClientId=client_id)
    half=(np.asarray(table['size_m'])/2).tolist()
    table_collision=p.createCollisionShape(p.GEOM_BOX,halfExtents=half,physicsClientId=client_id)
    table_visual=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=table['rgba'],physicsClientId=client_id)
    table_id=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=table_collision,baseVisualShapeIndex=table_visual,
        basePosition=table['position_world_m'],physicsClientId=client_id)
    p.changeDynamics(table_id,-1,lateralFriction=table['lateral_friction'],restitution=table['restitution'],
        collisionMargin=table['collision_margin_m'],physicsClientId=client_id)
    if g['type']=='box':
        kind=p.GEOM_BOX;collision=dict(halfExtents=(geometry_size(g)/2).tolist());visual=dict(collision)
    elif g['type']=='sphere':
        kind=p.GEOM_SPHERE;collision=dict(radius=g['radius_m']);visual=dict(collision)
    else:
        kind=p.GEOM_CYLINDER;collision=dict(radius=g['radius_m'],height=g['height_m'])
        visual=dict(radius=g['radius_m'],length=g['height_m'])
    object_collision=p.createCollisionShape(kind,**collision,physicsClientId=client_id)
    if g['type']=='cylinder':
        vertices,indices=cylinder_visual_mesh(g['radius_m'],g['height_m'],obj['visual_geometry']['radial_segments'])
        object_visual=p.createVisualShape(p.GEOM_MESH,vertices=vertices,indices=indices,rgbaColor=obj['rgba'],physicsClientId=client_id)
    elif g['type']=='sphere':
        vg=obj['visual_geometry'];vertices,indices=sphere_visual_mesh(g['radius_m'],vg['longitude_segments'],vg['latitude_segments'])
        object_visual=p.createVisualShape(p.GEOM_MESH,vertices=vertices,indices=indices,rgbaColor=obj['rgba'],physicsClientId=client_id)
    else:
        object_visual=p.createVisualShape(kind,**visual,rgbaColor=obj['rgba'],physicsClientId=client_id)
    object_id=p.createMultiBody(baseMass=obj['mass_kg'],baseCollisionShapeIndex=object_collision,
        baseVisualShapeIndex=object_visual,basePosition=obj['position_world_m'],
        baseOrientation=obj['orientation_world_xyzw'],physicsClientId=client_id)
    dynamics=dict(localInertiaDiagonal=obj['inertia_diagonal_kg_m2'],lateralFriction=obj['lateral_friction'],
        rollingFriction=obj['rolling_friction'],spinningFriction=obj['spinning_friction'],restitution=obj['restitution'],
        linearDamping=obj['linear_damping'],angularDamping=obj['angular_damping'])
    if g['type']!='sphere':dynamics['collisionMargin']=obj['collision_margin_m']
    p.changeDynamics(object_id,-1,**dynamics,physicsClientId=client_id)
    sources=_source_assets(p);size=geometry_size(g)
    scene=dict(scene_module='primitive_scene',object_id=object_id,cube_id=object_id,table_id=table_id,client_id=client_id,
        object_half_extents_m=(size/2).tolist(),table_top_z_world_m=table['position_world_m'][2]+table['size_m'][2]/2,
        spec=spec,source_assets=sources,scene_signature_sha256=canonical_sha(dict(spec=spec,source_assets=sources)),
        native_object_collision_shape_data=_serial(p.getCollisionShapeData(object_id,-1,physicsClientId=client_id)))
    assert scene['native_object_collision_shape_data'][0][2]==kind,'Native collision type differs from declared geometry'
    state=_state(p,client_id,scene);scene['initial_state']=state
    scene['object_top_z_world_m']=state['object']['aabb_world_m'][1][2]
    assert np.allclose(state['object']['local_inertia_diagonal_kg_m2'],obj['inertia_diagonal_kg_m2'],rtol=0,atol=1e-14)
    assert state['object']['mass_kg']==obj['mass_kg'] and state['object']['lateral_friction']==obj['lateral_friction']
    assert state['object']['collision_margin_m']==obj['collision_margin_m']
    rotation=Rotation.from_quat(obj['orientation_world_xyzw']).as_matrix()
    if np.array_equal(rotation,np.eye(3)):
        assert np.allclose(np.diff(np.array(state['object']['aabb_world_m']),axis=0)[0],size,rtol=0,atol=1e-12)
        assert abs(state['object']['aabb_world_m'][0][2]-scene['table_top_z_world_m'])<1e-12
    if expected:assert_scene_matches_manifest(p,client_id,scene,expected)
    return scene


def assert_scene_matches_manifest(p,client_id,scene,manifest):
    """Verify spec/source hashes and actual native pose/dynamics without writes."""
    manifest=json.loads(Path(manifest).read_text()) if isinstance(manifest,(str,Path)) else manifest
    if scene['scene_signature_sha256']!=manifest['scene_signature_sha256']:raise ValueError('Scene signature changed')
    if scene['spec']!=manifest['spec'] or scene['source_assets']!=manifest['source_assets']:raise ValueError('Scene provenance changed')
    for record in scene['source_assets']:
        if sha(record['path'])!=record['sha256']:raise ValueError('Source bytes changed: '+record['path'])
    actual=_state(p,client_id,scene)
    for name in ('object','table'):
        for key,value in manifest['actual_state'][name].items():
            if not np.allclose(actual[name][key],value,rtol=0,atol=1e-12):raise ValueError('Native scene state differs: '+name+'/'+key)
    return True


def _surface_and_normals(local,geometry):
    kind=geometry['type']
    if kind=='sphere':
        radial=np.linalg.norm(local,axis=1)
        return np.abs(radial-geometry['radius_m']),local/np.maximum(radial[:,None],1e-15)
    if kind=='box':
        delta=np.abs(local)-geometry_size(geometry)/2;axis=delta.argmax(axis=1)
        normals=np.zeros_like(local);normals[np.arange(len(local)),axis]=np.sign(local[np.arange(len(local)),axis])
        return np.abs(delta.max(axis=1)),normals
    radius=np.linalg.norm(local[:,:2],axis=1);side=radius-geometry['radius_m'];cap=np.abs(local[:,2])-geometry['height_m']/2
    delta=np.column_stack([side,cap]);sdf=np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(delta.max(axis=1),0)
    normal=np.zeros_like(local);is_cap=np.abs(cap)<np.abs(side)
    normal[is_cap,2]=np.sign(local[is_cap,2]);normal[~is_cap,:2]=local[~is_cap,:2]/np.maximum(radius[~is_cap,None],1e-15)
    return np.abs(sdf),normal


def render_scene(p,client_id,scene,output_dir):
    """Actual TinyRenderer RGB-D, analytic surface/native rays, zero steps."""
    output_dir=Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):raise ValueError('Use a fresh render directory; observations are immutable')
    output_dir.mkdir(parents=True,exist_ok=True)
    before=_state(p,client_id,scene);cam=scene['spec']['camera'];obj=scene['spec']['object']
    view,projection,K,T_WC,T_CW=camera_matrices(p,scene['spec'])
    rendered=p.getCameraImage(cam['width'],cam['height'],viewMatrix=view,projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        shadow=cam['shadow'],lightDirection=cam['light_direction'],physicsClientId=client_id)
    rgb=np.asarray(rendered[2],dtype=np.uint8).reshape(cam['height'],cam['width'],4)[...,:3]
    buffer=np.asarray(rendered[3],dtype=np.float64).reshape(cam['height'],cam['width'])
    segmentation=np.asarray(rendered[4],dtype=np.int32).reshape(cam['height'],cam['width'])
    valid=(buffer<1)&(segmentation>=0);metric=cam['far_m']*cam['near_m']/(cam['far_m']-(cam['far_m']-cam['near_m'])*buffer)
    metric[~valid]=0;depth=np.rint(metric*cam['depth_factor']).astype(np.uint16)
    u,v=np.meshgrid(np.arange(cam['width']),np.arange(cam['height']));z=depth.astype(float)/cam['depth_factor']
    points=np.stack([(u-K[0,2])/K[0,0]*z,(v-K[1,2])/K[1,1]*z,z],axis=-1)
    world=points@T_WC[:3,:3].T+T_WC[:3,3];ids=segmentation&((1<<24)-1)
    object_mask=valid&(ids==scene['object_id']);table_mask=valid&(ids==scene['table_id'])
    if min(object_mask.sum(),table_mask.sum())<100:raise ValueError('Frozen object/table is not visible')
    rotation=Rotation.from_quat(obj['orientation_world_xyzw']).as_matrix();position=np.asarray(obj['position_world_m'])
    local=(world[object_mask]-position)@rotation
    error,normals_local=_surface_and_normals(local,obj['geometry'])
    table_error=np.abs(world[table_mask,2]-scene['table_top_z_world_m'])
    normal_world=normals_local@rotation.T;view_dir=T_WC[:3,3]-world[object_mask]
    view_dir/=np.linalg.norm(view_dir,axis=1,keepdims=True)
    facing=np.zeros_like(object_mask);facing[object_mask]=(normal_world*view_dir).sum(axis=1)>.85
    safe_object=binary_erosion(object_mask,iterations=3)&facing;safe_table=binary_erosion(table_mask,iterations=3)
    chosen=[]
    for mask in (safe_object,safe_table):
        indices=np.flatnonzero(mask)
        if len(indices)<16:raise ValueError('Too few interior validation-ray pixels')
        chosen.extend(indices[np.linspace(0,len(indices)-1,32,dtype=int)].tolist())
    chosen=np.array(chosen);ray_points=points.reshape(-1,3)[chosen]
    ray_to=(ray_points/ray_points[:,2:3]*cam['far_m'])@T_WC[:3,:3].T+T_WC[:3,3]
    ray_from=np.repeat(T_WC[:3,3][None],len(chosen),axis=0)
    hits=p.rayTestBatch(ray_from.tolist(),ray_to.tolist(),physicsClientId=client_id)
    assert np.array_equal(np.array([row[0] for row in hits]),ids.reshape(-1)[chosen])
    ray_error=np.linalg.norm(np.array([row[3] for row in hits])-world.reshape(-1,3)[chosen],axis=1)
    recovered=world[valid]@T_CW[:3,:3].T+T_CW[:3,3]
    pixel=np.column_stack([recovered[:,0]/recovered[:,2]*K[0,0]+K[0,2],recovered[:,1]/recovered[:,2]*K[1,1]+K[1,2]])
    pixel_error=float(abs(pixel-np.column_stack([u[valid],v[valid]])).max())
    assert float(error.max())<.0015 and float(table_error.max())<.00051
    assert pixel_error<1e-9
    ray_local=(np.array([row[3] for row in hits[:32]])-position)@rotation
    native_ray_surface_error,_=_surface_and_normals(ray_local,obj['geometry'])
    after=_state(p,client_id,scene);assert before==after,'Render/validation changed physical state'
    Image.fromarray(rgb).save(output_dir/'color.png');Image.fromarray(depth).save(output_dir/'depth.png')
    scipy.io.savemat(output_dir/'meta.mat',dict(intrinsic_matrix=K,factor_depth=np.array([[cam['depth_factor']]])))
    np.save(output_dir/'depth_m.npy',metric.astype(np.float32));np.save(output_dir/'segmentation_for_validation_only.npy',segmentation)
    np.save(output_dir/'camera_points_m.npy',points[valid].astype(np.float32))
    validation=dict(actual_renderer='PyBullet DIRECT ER_TINY_RENDERER',integrated_physics_steps=0,render_state_unchanged=True,
        native_dynamic_shape=obj['geometry']['type'],visible_object_pixels=int(object_mask.sum()),visible_table_pixels=int(table_mask.sum()),
        analytic_object_surface_max_error_m=float(error.max()),analytic_object_surface_p99_error_m=float(np.quantile(error,.99)),
        table_surface_max_error_m=float(table_error.max()),native_ray_tests=len(chosen),native_ray_max_error_m=float(ray_error.max()),
        native_ray_point_distance_is_diagnostic_not_gate=True,
        native_object_ray_surface_max_error_m=float(native_ray_surface_error.max()),
        native_ray_algorithm_note='Bullet default Subsimplex convex raycast is approximate; point-distance disagreement is retained, not an input or physical admission gate',
        native_object_ray_tests=32,native_table_ray_tests=32,native_ray_domain='eroded interiors; object surface normal incidence cosine >0.85',
        depth_to_XYZ_to_pixel_max_error_pixels=pixel_error,depth_factor=cam['depth_factor'],
        depth_conversion='far*near/(far-(far-near)*OpenGL_buffer)',depth_PNG='round(metric_depth*1000), uint16; background zero',
        geometry_tolerance_m=.0015,not_a_collision_or_dynamic_gate=True,complete_geometry_fed_to_network=False,
        segmentation_fed_to_original_network=False,meta_contains_no_object_pose_or_class_labels=True)
    names=['color.png','depth.png','meta.mat','depth_m.npy','segmentation_for_validation_only.npy','camera_points_m.npy']
    files=[dict(path=str((output_dir/name).resolve()),bytes=(output_dir/name).stat().st_size,sha256=sha(output_dir/name)) for name in names]
    manifest=dict(scene_module='primitive_scene',scene_signature_sha256=scene['scene_signature_sha256'],spec=scene['spec'],source_assets=scene['source_assets'],
        actual_state=after,physics_engine_parameters=p.getPhysicsEngineParameters(physicsClientId=client_id),
        native_object_collision_shape_data=scene['native_object_collision_shape_data'],pybullet_API_version=p.getAPIVersion(),
        body_ids=dict(object=scene['object_id'],table=scene['table_id']),camera=dict(K=K.tolist(),T_WC=T_WC.tolist(),T_CW=T_CW.tolist(),
            view_matrix_column_major=view,projection_matrix_column_major=projection),validation=validation,files=files,
        boundary='Predeclared synthetic primitive observation only; zero physics/inference/grasp execution')
    (output_dir/'scene_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    (output_dir/'RESULTS.json').write_text(json.dumps(dict(status='frozen_primitive_render_and_validation_passed',physical_steps=0,
        validation=validation,scene_manifest_sha256=sha(output_dir/'scene_manifest.json')),indent=2)+'\n')
    assert_scene_matches_manifest(p,client_id,scene,manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--overlay',type=Path,default=OVERLAY)
    args=parser.parse_args();sys.path.insert(0,str(args.overlay));import pybullet as p
    if args.output_root.exists() and any(args.output_root.iterdir()):parser.error('Use a new campaign scene directory')
    args.output_root.mkdir(parents=True,exist_ok=True);specs=frozen_specs();sources=_source_assets(p)
    index=dict(schema='AnyDex-predeclared-native-primitive-scenes-v1',frozen_before_first_render=True,physical_steps=0,
        source_assets=sources,scene_module='primitive_scene',API=dict(create_scene='create_scene(p, client_id, spec_or_manifest=None, *, spec=None)',
            render_scene='render_scene(p, client_id, scene, output_dir)',
            assert_scene_matches_manifest='assert_scene_matches_manifest(p, client_id, scene, manifest)'),
        scenes=[dict(scene_id=name,directory=str((args.output_root/name).resolve()),spec=spec,
            scene_manifest=str((args.output_root/name/'scene_manifest.json').resolve()),
            scene_signature_sha256=canonical_sha(dict(spec=spec,source_assets=sources))) for name,spec in specs.items()])
    (args.output_root/'SCENES.json').write_text(json.dumps(index,indent=2)+'\n')
    original=p.stepSimulation;step_calls=[]
    def forbidden(*a,**kw):step_calls.append(1);raise RuntimeError('Primitive rendering forbids all stepSimulation calls')
    p.stepSimulation=forbidden;reports=[]
    try:
        for name,spec in specs.items():
            client=p.connect(p.DIRECT);directory=args.output_root/name
            try:
                scene=create_scene(p,client,spec);manifest=render_scene(p,client,scene,directory)
                assert not step_calls
                reports.append(dict(scene_id=name,directory=str(directory.resolve()),scene_manifest_sha256=sha(directory/'scene_manifest.json'),
                    scene_signature_sha256=scene['scene_signature_sha256'],validation=manifest['validation']))
                print(json.dumps(reports[-1],indent=2),flush=True)
            except Exception as exc:
                directory.mkdir(parents=True,exist_ok=True)
                (directory/'FAILED_RENDER.json').write_text(json.dumps(dict(status='primitive_render_validation_failed',physical_steps=0,
                    error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()),indent=2)+'\n')
                raise
            finally:p.disconnect(client)
        (args.output_root/'RESULTS.json').write_text(json.dumps(dict(status='three_frozen_native_primitive_scenes_ready',
            physical_steps=0,new_network_forwards=0,index_sha256=sha(args.output_root/'SCENES.json'),scenes=reports),indent=2)+'\n')
    finally:p.stepSimulation=original


if __name__=='__main__':main()
