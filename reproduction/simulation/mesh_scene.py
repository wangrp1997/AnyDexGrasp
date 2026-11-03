"""Real triangle-mesh RGB-D scenes with compound convex collisions; zero steps.

prepare_asset consumes explicit source mesh scales/transforms and decomposition
parts. It never replaces a non-convex whole cup/bowl by one convex hull. Visual
triangles are retained; inertia is a declared uniform-solid geometric estimate,
not identified object dynamics. The two frozen poses differ only by world yaw.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import scipy.io
from scipy.spatial.transform import Rotation

ROOT=Path(os.environ.get('ANYDEX_REPO_ROOT','/home/wangrenpeng/AnyDexGrasp')).resolve()
sys.path[:0]=[str(ROOT/'reproduction/simulation'),str(ROOT/'outputs/native_deps/python_overlay')]
from synthetic_scene import SPEC as TABLE_SPEC,camera_matrices,_state
from primitive_scene import sha,canonical_sha,_serial


def _record(value,default_scale=1.):
    r=dict(value) if isinstance(value,dict) else {'path':str(value)}
    r['path']=str(Path(r['path']).resolve())
    digest=sha(r['path'])
    if r.get('sha256',digest)!=digest:raise ValueError('Source SHA changed: '+r['path'])
    r['sha256']=digest;r['bytes']=Path(r['path']).stat().st_size
    scale=np.asarray(r.get('scale',default_scale),float)
    if scale.ndim==0:scale=np.repeat(scale,3)
    if scale.shape!=(3,) or not np.isfinite(scale).all() or not (scale>0).all():raise ValueError('Explicit positive metre scale required')
    r['scale']=scale.tolist()
    T=np.asarray(r.get('T_asset_mesh',np.eye(4)),float)
    if T.shape!=(4,4) or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-9) or np.linalg.det(T[:3,:3])<0:raise ValueError('Invalid rigid T_asset_mesh; translation is in metres after scaling')
    r['T_asset_mesh']=T.tolist();return r


def _load(record):
    import trimesh
    mesh=trimesh.load(record['path'],force='mesh',process=False)
    if not isinstance(mesh,trimesh.Trimesh):raise TypeError('Source must resolve to triangle geometry')
    mesh=mesh.copy();T=np.asarray(record['T_asset_mesh']);mesh.vertices=(np.asarray(mesh.vertices)*record['scale'])@T[:3,:3].T+T[:3,3]
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():raise ValueError('Empty/nonfinite source mesh')
    return mesh


def _obj_groups(record):
    """Preserve separate OBJ convex objects rather than convexifying their union."""
    import trimesh
    path=Path(record['path'])
    if path.suffix.lower()!='.obj':return []
    vertices=[];groups={};name='default'
    for line in path.read_text(errors='replace').splitlines():
        words=line.split()
        if not words:continue
        if words[0]=='v':vertices.append([float(x) for x in words[1:4]])
        elif words[0] in ('o','g'):name=' '.join(words[1:]) or 'default'
        elif words[0]=='f':
            ids=[int(x.split('/')[0]) for x in words[1:]];ids=[x-1 if x>0 else len(vertices)+x for x in ids]
            for i in range(1,len(ids)-1):groups.setdefault(name,[]).append([ids[0],ids[i],ids[i+1]])
    if len(groups)<2:return []
    v=np.asarray(vertices,float);T=np.asarray(record['T_asset_mesh']);v=(v*record['scale'])@T[:3,:3].T+T[:3,3]
    return [trimesh.Trimesh(vertices=v.copy(),faces=faces,process=True) for faces in groups.values() if faces]


def _convex_components(record):
    mesh=_load(record);mesh.process(validate=False)
    if mesh.is_convex:return [mesh.convex_hull]
    candidates=_obj_groups(record)
    if not candidates:candidates=list(mesh.split(only_watertight=False))
    result=[]
    for part in candidates:
        part.process(validate=False)
        if not part.is_convex:
            raise ValueError('Source collision part is non-convex; refusing whole-mesh convexification: '+record['path'])
        result.append(part.convex_hull)
    if not result:raise ValueError('No valid convex decomposition parts')
    return result


def _compound_mass_properties(parts,mass):
    volumes=np.asarray([abs(float(x.volume)) for x in parts])
    if not np.isfinite(volumes).all() or (volumes<=1e-15).any():raise ValueError('Degenerate convex-part volume')
    weights=mass*volumes/volumes.sum();centres=np.asarray([x.center_mass for x in parts]);centre=np.sum(weights[:,None]*centres,axis=0)/mass
    inertia=np.zeros((3,3))
    for part,weight,c in zip(parts,weights,centres):
        d=c-centre;inertia+=np.asarray(part.moment_inertia)*(weight/abs(float(part.volume)))+weight*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
    return centre,inertia


def prepare_asset(asset,output_dir):
    """Normalize real geometry to metres / COM / principal inertial body frame.

    asset: {id,visual_mesh:{path,sha256,scale,T_asset_mesh},convex_parts:[...],
    scale:optional shared metre scale,upright_orientation_xyzw:optional}.
    Existing decomposed meshes/OBJ convex object groups are preferred. A missing
    decomposition is an error rather than a silent whole-object convex hull.
    """
    import trimesh
    output_dir=Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    visual_record=_record(asset['visual_mesh'],asset.get('scale',1.));visual=_load(visual_record)
    part_records=[_record(r,asset.get('scale',1.)) for r in asset['convex_parts']]
    if not part_records:raise ValueError('Explicit convex decomposition required; whole-mesh hull is forbidden')
    parts=[part for r in part_records for part in _convex_components(r)]
    if asset.get('requires_cavity',False) and len(parts)<2:raise ValueError('A single convex part cannot preserve this declared cavity')
    mass=.1;inertia_mesh=visual.copy();inertia_mesh.process(validate=False)
    if inertia_mesh.is_watertight and abs(float(inertia_mesh.volume))>1e-15:
        if inertia_mesh.volume<0:inertia_mesh.invert()
        centre=np.asarray(inertia_mesh.center_mass);inertia=np.asarray(inertia_mesh.moment_inertia)*mass/float(inertia_mesh.volume)
        inertia_model='Uniform-density watertight visual triangle solid, normalized to0.1kg; geometric estimate, not measured'
    else:
        centre,inertia=_compound_mass_properties(parts,mass)
        inertia_model='Volume-weighted uniform convex-part mixture, normalized to0.1kg; overlaps count per part; geometric approximation, not measured'
    inertia=(inertia+inertia.T)/2;eigenvalues,axes=np.linalg.eigh(inertia)
    if not np.isfinite(eigenvalues).all() or (eigenvalues<=0).any():raise ValueError('Geometric inertia is not positive definite')
    if np.linalg.det(axes)<0:axes[:,0]*=-1
    # v_body = R_asset_body.T (v_asset-COM); the world body orientation later
    # includes R_asset_body, so the original source shape/up direction is kept.
    visual.vertices=(np.asarray(visual.vertices)-centre)@axes
    visual_path=output_dir/'visual_metres.obj';visual.export(visual_path)
    part_paths=[]
    for i,part in enumerate(parts):
        part=part.copy();part.vertices=(np.asarray(part.vertices)-centre)@axes
        path=output_dir/('convex_%04d.obj'%i);part.export(path);part_paths.append(path)
    visual_bounds=np.asarray(visual.bounds);collision_bounds=np.array([np.min([x.bounds[0] for x in parts],axis=0),np.max([x.bounds[1] for x in parts],axis=0)])
    # Compare in the common source asset frame, before principal-axis rotation.
    original_visual=_load(visual_record);aabb_delta=float(abs(collision_bounds-np.asarray(original_visual.bounds)).max())
    dimensions=np.diff(np.asarray(original_visual.bounds),axis=0)[0]
    if (dimensions<=0).any() or not np.isfinite(dimensions).all():raise ValueError('Invalid physical mesh dimensions')
    tolerance=float(asset.get('aabb_tolerance_m',max(.005,.05*float(dimensions.max()))))
    if aabb_delta>tolerance:raise ValueError('Visual/decomposition AABB disagrees by %.6gm; check source units/frames'%aabb_delta)
    robot=ET.Element('robot',name='AnyDex_independent_real_mesh');link=ET.SubElement(robot,'link',name='object_link')
    inertial=ET.SubElement(link,'inertial');ET.SubElement(inertial,'origin',xyz='0 0 0',rpy='0 0 0');ET.SubElement(inertial,'mass',value=repr(mass))
    ET.SubElement(inertial,'inertia',ixx=repr(float(eigenvalues[0])),iyy=repr(float(eigenvalues[1])),izz=repr(float(eigenvalues[2])),ixy='0',ixz='0',iyz='0')
    visual_xml=ET.SubElement(link,'visual');ET.SubElement(ET.SubElement(visual_xml,'geometry'),'mesh',filename=str(visual_path))
    ET.SubElement(ET.SubElement(visual_xml,'material',name='plain_visual_color'),'color',rgba='0.85 0.60 0.25 1')
    for path in part_paths:ET.SubElement(ET.SubElement(ET.SubElement(link,'collision'),'geometry'),'mesh',filename=str(path))
    urdf=output_dir/'object.urdf';ET.indent(robot);ET.ElementTree(robot).write(urdf,encoding='unicode',xml_declaration=True)
    products=[visual_path,*part_paths,urdf]
    prepared=dict(schema='AnyDex-real-mesh-prepared-v1',id=asset['id'],mass_kg=mass,inertia_diagonal_kg_m2=eigenvalues.tolist(),inertia_model=inertia_model,
        source_COM_asset_m=centre.tolist(),R_asset_body_principal=axes.tolist(),inertia_full_asset_kg_m2=inertia.tolist(),
        inertia_reconstruction_max_error=float(abs(axes@np.diag(eigenvalues)@axes.T-inertia).max()),
        upright_orientation_xyzw=asset.get('upright_orientation_xyzw',[0,0,0,1]),
        visual_mesh=str(visual_path),convex_parts=[str(x) for x in part_paths],urdf=str(urdf),
        source_records=[visual_record,*part_records],source_asset_metadata=asset,
        product_records=[dict(path=str(x),bytes=x.stat().st_size,sha256=sha(x)) for x in products],
        visual_triangle_faces=int(len(visual.faces)),convex_part_count=len(parts),
        visual_bounds_body_m=visual_bounds.tolist(),source_visual_dimensions_m=dimensions.tolist(),
        visual_collision_source_AABB_max_disagreement_m=aabb_delta,AABB_tolerance_m=tolerance,
        cavity_boundary='Each supplied convex component stays a separate collision shape; no union-to-single-hull operation. Fidelity of supplied convex decomposition is a source approximation, not exact triangle contacts.',
        visual_boundary='Source triangle topology retained after metre/rigid coordinate normalization; constant RGBA, original textures not used',physical_steps=0)
    (output_dir/'PREPARED.json').write_text(json.dumps(prepared,indent=2,allow_nan=False)+'\n');return prepared


def frozen_specs(prepared_assets,yaws=(0,90)):
    specs={}
    for value in prepared_assets:
        prepared=json.loads(Path(value).read_text()) if isinstance(value,(str,Path)) else copy.deepcopy(value)
        body_to_asset=Rotation.from_matrix(prepared['R_asset_body_principal']);upright=Rotation.from_quat(prepared['upright_orientation_xyzw'])
        for yaw in yaws:
            name=prepared['id']+'_yaw%03d'%yaw;spec=copy.deepcopy(TABLE_SPEC)
            spec.update(schema='AnyDex-independent-real-mesh-v1',scene_id=name)
            spec['object']=dict(prepared=prepared,position_world_xy_m=[0,.08],yaw_degrees=float(yaw),
                orientation_world_xyzw=(Rotation.from_euler('z',yaw,degrees=True)*upright*body_to_asset).as_quat().tolist(),
                placement='Initial COM XY fixed; shift COM Z once before all integration until native compound-vs-table closest distance is zero. Mesh broadphase AABB is conservative and may extend below the support plane',
                mass_kg=.1,inertia_diagonal_kg_m2=prepared['inertia_diagonal_kg_m2'],lateral_friction=1.,rolling_friction=0.,spinning_friction=0.,restitution=0.,linear_damping=.04,angular_damping=.04,
                collision_margin_m=.001,collision_margin_boundary='Bullet convex compound import/default child margins; actual AABB and shape inventory recorded')
            spec['boundary']='Real source triangle visual and compound convex dynamic port; no semantic labels, object pose, full mesh or segmentation supplied to original network'
            specs[name]=spec
    return specs


def _source_assets(p,spec):
    prepared=spec['object']['prepared'];paths=[Path(__file__).resolve(),ROOT/'reproduction/simulation/synthetic_scene.py',ROOT/'reproduction/simulation/primitive_scene.py',Path(p.__file__)]
    records=[dict(path=str(x),bytes=x.stat().st_size,sha256=sha(x)) for x in paths]
    for record in [*prepared['source_records'],*prepared['product_records']]:
        if sha(record['path'])!=record['sha256']:raise ValueError('Frozen geometry changed: '+record['path'])
        records.append({key:record[key] for key in ['path','bytes','sha256']})
    return records


def create_scene(p,client_id,spec_or_manifest=None,*,spec=None):
    if spec is not None:
        if spec_or_manifest is not None:raise TypeError('Pass one frozen spec')
        spec_or_manifest=spec
    value=json.loads(Path(spec_or_manifest).read_text()) if isinstance(spec_or_manifest,(str,Path)) else spec_or_manifest
    expected=value if 'spec' in value else None;spec=copy.deepcopy(value['spec'] if expected else value)
    if p.getNumBodies(physicsClientId=client_id):raise ValueError('Requires an empty caller-owned client, never resets an existing world')
    obj=spec['object'];table=spec['table'];physics=spec['physics'];prepared=obj['prepared']
    sources=_source_assets(p,spec)
    p.setRealTimeSimulation(0,physicsClientId=client_id);p.setTimeStep(physics['time_step_s'],physicsClientId=client_id)
    p.setPhysicsEngineParameter(numSubSteps=1,numSolverIterations=physics['solver_iterations'],deterministicOverlappingPairs=1,physicsClientId=client_id)
    p.setGravity(*physics['gravity_m_s2'],physicsClientId=client_id)
    half=(np.asarray(table['size_m'])/2).tolist();collision=p.createCollisionShape(p.GEOM_BOX,halfExtents=half,physicsClientId=client_id);visual=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=table['rgba'],physicsClientId=client_id)
    table_id=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=collision,baseVisualShapeIndex=visual,basePosition=table['position_world_m'],physicsClientId=client_id)
    p.changeDynamics(table_id,-1,lateralFriction=1,restitution=0,collisionMargin=.001,physicsClientId=client_id)
    safe_z=.1+2*float(np.linalg.norm(prepared['visual_bounds_body_m']))
    body=p.loadURDF(prepared['urdf'],basePosition=[*obj['position_world_xy_m'],safe_z],baseOrientation=obj['orientation_world_xyzw'],useFixedBase=False,flags=p.URDF_USE_INERTIA_FROM_FILE,physicsClientId=client_id)
    p.changeDynamics(body,-1,lateralFriction=1,rollingFriction=0,spinningFriction=0,restitution=0,linearDamping=.04,angularDamping=.04,physicsClientId=client_id)
    top=float(table['position_world_m'][2]+table['size_m'][2]/2)
    closest=p.getClosestPoints(body,table_id,2.,physicsClientId=client_id)
    if not closest:raise ValueError('No compound/table support point within2m')
    support_distance=min(float(row[8]) for row in closest)
    position,orientation=p.getBasePositionAndOrientation(body,physicsClientId=client_id)
    position=list(position);position[2]-=support_distance
    p.resetBasePositionAndOrientation(body,position,orientation,physicsClientId=client_id)
    support_distance_after=min(float(row[8]) for row in p.getClosestPoints(body,table_id,.02,physicsClientId=client_id))
    if abs(support_distance_after)>1e-9:raise ValueError('Initial native table support distance is not zero')
    shape_data=_serial(p.getCollisionShapeData(body,-1,physicsClientId=client_id))
    if len(shape_data)!=prepared['convex_part_count'] or any(x[2]!=p.GEOM_MESH for x in shape_data):raise ValueError('Compound shape inventory changed / decomposition collapsed')
    dynamics=p.getDynamicsInfo(body,-1,physicsClientId=client_id)
    if not np.allclose(dynamics[3],[0,0,0],atol=1e-14) or not np.allclose(dynamics[4],[0,0,0,1],atol=1e-14):raise ValueError('Object COM/principal frame does not match normalized body frame')
    scene=dict(scene_module='mesh_scene',object_id=body,cube_id=body,table_id=table_id,client_id=client_id,spec=spec,source_assets=sources,
        table_top_z_world_m=top,scene_signature_sha256=canonical_sha(dict(spec=spec,source_assets=sources)),
        native_object_collision_shape_data=shape_data,initial_placement_resets_before_physics=1,
        base_link_frame_equals_COM_principal_frame=True,initial_native_support_distance_m=support_distance_after)
    state=_state(p,client_id,scene);scene['initial_state']=state;scene['object_top_z_world_m']=state['object']['aabb_world_m'][1][2]
    assert np.allclose(state['object']['position_world_m'][:2],[0,.08],rtol=0,atol=1e-12)
    assert state['object']['mass_kg']==.1 and state['object']['lateral_friction']==1
    assert np.allclose(state['object']['local_inertia_diagonal_kg_m2'],obj['inertia_diagonal_kg_m2'],rtol=0,atol=1e-14)
    if expected:assert_scene_matches_manifest(p,client_id,scene,expected)
    return scene


def assert_scene_matches_manifest(p,client_id,scene,manifest):
    manifest=json.loads(Path(manifest).read_text()) if isinstance(manifest,(str,Path)) else manifest
    if scene['scene_signature_sha256']!=manifest['scene_signature_sha256'] or scene['spec']!=manifest['spec'] or scene['source_assets']!=manifest['source_assets']:raise ValueError('Scene/source binding differs')
    for record in scene['source_assets']:
        if sha(record['path'])!=record['sha256']:raise ValueError('Source bytes changed')
    actual=_state(p,client_id,scene)
    for name in ['object','table']:
        for key,value in manifest['actual_state'][name].items():
            if not np.allclose(actual[name][key],value,rtol=0,atol=1e-12):raise ValueError('Actual scene differs: '+name+'/'+key)
    return True


def render_scene(p,client_id,scene,output_dir):
    output_dir=Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True);before=_state(p,client_id,scene);cam=scene['spec']['camera']
    view,projection,K,T_WC,T_CW=camera_matrices(p,scene['spec'])
    rendered=p.getCameraImage(cam['width'],cam['height'],viewMatrix=view,projectionMatrix=projection,renderer=p.ER_TINY_RENDERER,flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,shadow=0,lightDirection=[1,1,3],physicsClientId=client_id)
    rgb=np.asarray(rendered[2],dtype=np.uint8).reshape(cam['height'],cam['width'],4)[...,:3];buffer=np.asarray(rendered[3],dtype=float).reshape(cam['height'],cam['width']);segmentation=np.asarray(rendered[4],dtype=np.int32).reshape(cam['height'],cam['width'])
    valid=(buffer<1)&(segmentation>=0);metric=cam['far_m']*cam['near_m']/(cam['far_m']-(cam['far_m']-cam['near_m'])*buffer);metric[~valid]=0
    depth=np.rint(metric*cam['depth_factor']).astype(np.uint16);u,v=np.meshgrid(np.arange(cam['width']),np.arange(cam['height']));z=depth.astype(float)/cam['depth_factor']
    points=np.stack([(u-K[0,2])/K[0,0]*z,(v-K[1,2])/K[1,1]*z,z],axis=-1);world=points@T_WC[:3,:3].T+T_WC[:3,3];ids=segmentation&((1<<24)-1)
    object_mask=valid&(ids==scene['object_id']);table_mask=valid&(ids==scene['table_id'])
    if min(object_mask.sum(),table_mask.sum())<100:raise ValueError('Empty object/table RGB-D')
    quant_error=float(abs(depth[valid]/cam['depth_factor']-metric[valid]).max());table_error=float(abs(world[table_mask,2]-scene['table_top_z_world_m']).max())
    assert quant_error<=.5/cam['depth_factor']+1e-12 and table_error<.00051
    recovered=world[valid]@T_CW[:3,:3].T+T_CW[:3,3];pixel=np.column_stack([recovered[:,0]/recovered[:,2]*K[0,0]+K[0,2],recovered[:,1]/recovered[:,2]*K[1,1]+K[1,2]])
    pixel_error=float(abs(pixel-np.column_stack([u[valid],v[valid]])).max());assert pixel_error<1e-9
    # Compare projected actual triangle vertices and actual object pixels. This
    # checks units/frame/visibility; it does not assert convex contact surfaces.
    import trimesh
    mesh=trimesh.load(scene['spec']['object']['prepared']['visual_mesh'],force='mesh',process=False)
    pos,orn=p.getBasePositionAndOrientation(scene['object_id'],physicsClientId=client_id);R=Rotation.from_quat(orn).as_matrix();vertices=np.asarray(mesh.vertices)@R.T+pos
    actual_visual_aabb=np.array([vertices.min(0),vertices.max(0)]);collision_aabb=np.asarray(before['object']['aabb_world_m'])
    # Bullet's compound broadphase rotates the aggregate local AABB; it is not
    # the tight world AABB of transformed convex vertices. Principal-frame
    # normalization can therefore enlarge that bound without changing geometry.
    collision_vertices=[]
    for part_path in scene['spec']['object']['prepared']['convex_parts']:
        part=trimesh.load(part_path,force='mesh',process=False)
        collision_vertices.append(np.asarray(part.vertices)@R.T+pos)
    collision_vertices=np.concatenate(collision_vertices)
    collision_vertex_aabb=np.array([collision_vertices.min(0),collision_vertices.max(0)])
    aabb_error=float(abs(actual_visual_aabb-collision_vertex_aabb).max());tolerance=scene['spec']['object']['prepared']['AABB_tolerance_m']
    if aabb_error>tolerance:raise ValueError('Actual visual/convex vertex AABB frame/scale mismatch')
    native_contains_convex=bool(np.all(collision_aabb[0]<=collision_vertex_aabb[0]+1e-9) and np.all(collision_aabb[1]>=collision_vertex_aabb[1]-1e-9))
    if not native_contains_convex:raise ValueError('Native broadphase AABB does not contain actual convex geometry')
    after=_state(p,client_id,scene);assert before==after,'Rendering changed physical state'
    Image.fromarray(rgb).save(output_dir/'color.png');Image.fromarray(depth).save(output_dir/'depth.png');scipy.io.savemat(output_dir/'meta.mat',dict(intrinsic_matrix=K,factor_depth=np.array([[cam['depth_factor']]])))
    np.save(output_dir/'depth_m.npy',metric.astype(np.float32));np.save(output_dir/'segmentation_for_validation_only.npy',segmentation);np.save(output_dir/'camera_points_m.npy',points[valid].astype(np.float32))
    validation=dict(actual_renderer='PyBullet DIRECT ER_TINY_RENDERER',integrated_physics_steps=0,render_state_unchanged=True,visible_object_pixels=int(object_mask.sum()),visible_table_pixels=int(table_mask.sum()),
        depth_quantization_max_error_m=quant_error,table_surface_max_error_m=table_error,depth_to_XYZ_to_pixel_max_error_pixels=pixel_error,
        actual_visual_AABB_world_m=actual_visual_aabb.tolist(),actual_collision_AABB_world_m=collision_aabb.tolist(),actual_collision_vertex_AABB_world_m=collision_vertex_aabb.tolist(),visual_collision_AABB_max_disagreement_m=aabb_error,
        geometry_alignment_tolerance_m=tolerance,geometry_alignment_check='True transformed visual and separate convex vertices; native broadphase AABB is only a conservative bound',native_broadphase_contains_actual_convex_geometry=native_contains_convex,native_broadphase_vs_actual_convex_AABB_max_excess_m=float(abs(collision_aabb-collision_vertex_aabb).max()),
        source_convex_part_count=scene['spec']['object']['prepared']['convex_part_count'],actual_collision_shape_count=len(scene['native_object_collision_shape_data']),
        complete_geometry_fed_to_network=False,segmentation_fed_to_original_network=False,meta_contains_no_object_pose_or_class_labels=True,
        geometry_boundary='Real triangle visual with separate source convex decomposition; collision approximation and default convex margins retained; no whole-object single hull or analytic surface/ray admission gate')
    names=['color.png','depth.png','meta.mat','depth_m.npy','segmentation_for_validation_only.npy','camera_points_m.npy'];files=[dict(path=str((output_dir/name).resolve()),bytes=(output_dir/name).stat().st_size,sha256=sha(output_dir/name)) for name in names]
    manifest=dict(scene_module='mesh_scene',scene_signature_sha256=scene['scene_signature_sha256'],spec=scene['spec'],source_assets=scene['source_assets'],actual_state=after,physics_engine_parameters=p.getPhysicsEngineParameters(physicsClientId=client_id),native_object_collision_shape_data=scene['native_object_collision_shape_data'],
        initial_placement_resets_before_physics=1,initial_native_support_distance_m=scene['initial_native_support_distance_m'],base_link_frame_equals_COM_principal_frame=True,body_ids=dict(object=scene['object_id'],table=scene['table_id']),camera=dict(K=K.tolist(),T_WC=T_WC.tolist(),T_CW=T_CW.tolist(),view_matrix_column_major=view,projection_matrix_column_major=projection),validation=validation,files=files,boundary='Zero-step real mesh observation and compound collision port; not original author physical benchmark')
    (output_dir/'scene_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n');(output_dir/'RESULTS.json').write_text(json.dumps(dict(status='real_mesh_render_validation_passed',physical_steps=0,validation=validation,scene_manifest_sha256=sha(output_dir/'scene_manifest.json')),indent=2)+'\n')
    assert_scene_matches_manifest(p,client_id,scene,manifest);return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True);args=ap.parse_args()
    raw=json.loads(args.assets.read_text());assets=raw['assets'] if isinstance(raw,dict) else raw;prepared=[]
    args.output_root.mkdir(parents=True,exist_ok=True)
    for asset in assets:prepared.append(prepare_asset(asset,args.output_root/'prepared'/asset['id']))
    specs=frozen_specs(prepared);(args.output_root/'SCENES.json').write_text(json.dumps(dict(specs=specs,physical_steps=0),indent=2)+'\n')
    import pybullet as p
    original=p.stepSimulation
    def forbidden(*a,**kw):raise RuntimeError('Mesh scene preparation/rendering forbids physics integration')
    p.stepSimulation=forbidden
    try:
        for name,spec in specs.items():
            client=p.connect(p.DIRECT)
            try:
                scene=create_scene(p,client,spec);manifest=render_scene(p,client,scene,args.output_root/'scenes'/name)
                print(json.dumps(dict(scene=name,status='ready',signature=manifest['scene_signature_sha256'],physical_steps=0)),flush=True)
            finally:p.disconnect(client)
    finally:p.stepSimulation=original


if __name__=='__main__':main()
