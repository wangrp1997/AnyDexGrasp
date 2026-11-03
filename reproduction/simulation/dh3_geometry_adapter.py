"""Static DH3 URDF geometry adapter, exact or bounded helperless mode.

Original STLs/inertials remain unchanged. The author swaps tip/outlet position
components while keeping orientations. This adapter represents that operation
with joint coordinates. Exact mode uses one massless counterflex frame; the
explicit helperless mode bounds its omitted source Z delta. Neither moves a mesh
per frame. Its coordinate map is geometric; dynamic actuation is a separate port.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'reproduction/runtime'))
from dh3_author_kinematics import joint_angles,closure_path,source_receipt,SOURCE_SHA256,REV

SOURCE_URDF=ROOT/'generate_mesh_and_pointcloud/dh3_urdf/dh3_urdf/urdf/dh3_urdf.urdf'
MESH_DIR=SOURCE_URDF.parent.parent/'meshes'
GENERATOR=ROOT/'generate_mesh_and_pointcloud/recover_dh3_hand_to_stl.py'
TABLE=ROOT/'generate_mesh_and_pointcloud/dh3_urdf/width_12D_angle_2D_angle.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _vector(text):return np.array([float(value) for value in text.split()])


def adapter_targets(author12,manifest):
    """Return every movable adapter joint target, by name, from original12D.

    These targets may be used as joint coordinates by an independently specified
    controller. This function does not choose forces, timings or gains.
    """
    q=np.asarray(author12,dtype=np.float64)
    if q.shape!=(12,) or not np.isfinite(q).all():raise ValueError('Expected finite original12-vector')
    return {name:float(np.dot(row['weights_author12'],q)+row['offset_rad'])
            for name,row in manifest['movable_joint_map'].items()}


def build_adapter(output_dir,*,helperless=False):
    """Write a static derived URDF and its explicit original12 -> joint map."""
    source=ET.parse(SOURCE_URDF).getroot();derived=copy.deepcopy(source)
    original_joints={j.attrib['name']:j for j in source.findall('joint')}
    joints={j.attrib['name']:j for j in derived.findall('joint')}
    links={link.attrib['name']:link for link in derived.findall('link')}
    mapping={};edits=[]
    def coordinate(name,indices):
        weights=np.zeros(12)
        for i in indices:weights[i]+=1
        mapping[name]={'weights_author12':weights.tolist(),'offset_rad':0.}
    for finger in range(1,4):
        offset=4*(finger-1);prefix='finger'+str(finger)
        base=joints[prefix+'_base_joint'];flex=joints[prefix+'_joint']
        tip=joints[prefix+'_tip_joint'];outlet=joints[prefix+'_outlet_joint']
        if base.attrib['type']!='fixed':coordinate(base.attrib['name'],[offset])
        coordinate(flex.attrib['name'],[offset+1])
        # Outlet mesh is at its original outlet position, with original TIP orientation.
        outlet.find('origin').attrib['rpy']=flex.find('origin').attrib['rpy']
        outlet.find('axis').attrib['xyz']='0 -1 0'
        coordinate(outlet.attrib['name'],[offset+1,offset+2])
        a=_vector(flex.find('origin').attrib['rpy']);b=_vector(original_joints[prefix+'_outlet_joint'].find('origin').attrib['rpy'])
        if np.array_equal(a,b) or helperless:
            # Parent flex rotates-Y*f; +Y*(f+o) cancels f and leaves +Y*o.
            tip.find('axis').attrib['xyz']='0 1 0'
            coordinate(tip.attrib['name'],[offset+1,offset+3])
            if not np.array_equal(a,b):
                edits.append({'approximation':'helperless third-tip orientation uses main-flex fixed Z rotation',
                    'omitted_Z_delta_rad':float(b[2]-a[2]),'finger':finger,
                    'preserved':'Original source unchanged; no mesh scaling, no original entity inertia/mass change'})
        else:
            assert np.array_equal(a[:2],[0,0]) and np.array_equal(b[:2],[0,0])
            helper_name=prefix+'_author_counterflex_frame'
            helper=ET.Element('link',{'name':helper_name})
            inertial=ET.SubElement(helper,'inertial');ET.SubElement(inertial,'origin',{'xyz':'0 0 0','rpy':'0 0 0'})
            ET.SubElement(inertial,'mass',{'value':'0'})
            ET.SubElement(inertial,'inertia',{name:'0' for name in ['ixx','ixy','ixz','iyy','iyz','izz']})
            derived.append(helper)
            counter_name=prefix+'_author_counterflex_joint'
            original_tip_name=tip.attrib['name'];tip.attrib['name']=counter_name
            tip.find('child').attrib['link']=helper_name;tip.find('axis').attrib['xyz']='0 1 0'
            coordinate(counter_name,[offset+1])
            final=ET.Element('joint',{'name':original_tip_name,'type':'revolute'})
            ET.SubElement(final,'origin',{'xyz':'0 0 0','rpy':'0 0 '+repr(float(b[2]-a[2]))})
            ET.SubElement(final,'parent',{'link':helper_name});ET.SubElement(final,'child',{'link':prefix+'_tip_Link'})
            ET.SubElement(final,'axis',{'xyz':'0 1 0'})
            final.append(copy.deepcopy(original_joints[original_tip_name].find('limit')))
            derived.append(final);coordinate(original_tip_name,[offset+3])
            edits.append({'helper':helper_name,'mass_kg':0.,'Z_origin_delta_rad':float(b[2]-a[2]),
                'reason':'Original main/outlet rpy differ. Counterflex before this fixed delta preserves both exact source rotations.'})
    # Wrong original mimic relations cannot remain as an implicit second map.
    for joint in derived.findall('joint'):
        for item in list(joint.findall('mimic')):joint.remove(item)
    # Geometry-domain envelopes, not calibrated actuator force/speed assumptions.
    corners=[joint_angles(p,r) for p in [0,95] for r in [0,100]]
    for joint in derived.findall('joint'):
        if joint.attrib['type']=='fixed':continue
        name=joint.attrib['name'];values=[np.dot(mapping[name]['weights_author12'],q) for q in corners]
        limit=joint.find('limit');limit.attrib['lower']=repr(float(min(values)));limit.attrib['upper']=repr(float(max(values)))
    for link in derived.findall('link'):
        for mesh in link.findall('./visual/geometry/mesh')+link.findall('./collision/geometry/mesh'):
            path=MESH_DIR/Path(mesh.attrib['filename']).name
            if not path.is_file():raise FileNotFoundError(path)
            mesh.attrib['filename']=str(path)
    output_dir.mkdir(parents=True,exist_ok=True);urdf=output_dir/'dh3_author_geometry.urdf'
    ET.indent(derived,space='  ');ET.ElementTree(derived).write(urdf,encoding='utf-8',xml_declaration=True)
    def inertia_signature(link):
        item=link.find('inertial')
        return {node.tag:dict(node.attrib) for node in item}
    original_links={link.attrib['name']:link for link in source.findall('link')}
    for name,link in original_links.items():assert inertia_signature(link)==inertia_signature(links[name])
    manifest={'scope':'Static original-geometry URDF adapter; dynamic controller independently specified',
        'source_revision':REV,'source_URDF':str(SOURCE_URDF),'source_URDF_sha256':sha(SOURCE_URDF),
        'source_generator':str(GENERATOR),'source_generator_sha256':sha(GENERATOR),
        'kinematic_source':source_receipt(),'output_URDF':str(urdf.resolve()),'output_URDF_sha256':sha(urdf),
        'movable_joint_map':mapping,'fixed_original_joint':{'name':'finger2_base_joint','author_index':4,'formula_value':1.,'kept_fixed':True},
        'helper_frames':[x for x in edits if 'helper' in x],'approximations':[x for x in edits if 'approximation' in x],
        'helperless':helperless,'original_entity_links':len(original_links),'total_links':len(derived.findall('link')),
        'original_13_entity_inertials_preserved':True,'original_meshes_unchanged':True,
        'mesh_files':[{'path':str(path),'sha256':sha(path)} for path in sorted(MESH_DIR.glob('*.STL'))],
        'limit_policy':'Only geometry envelopes recomputed from original affine position[0,95]/rotation[0,100] domain. Original existing effort/velocity retained, including zeros; these are not validated motor limits.',
        'mimic_policy':'Original incompatible mimic tags removed. This explicit linear joint map defines geometry; use a separately validated dynamic coupling/controller.',
        'no_per_frame_mesh_pose_override':True,'physics_steps':0,'dynamic_validity_claim':False}
    (output_dir/'ADAPTER.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return urdf,manifest


def rpy_matrix(values):
    roll,pitch,yaw=values;cr,sr=np.cos(roll),np.sin(roll);cp,sp=np.cos(pitch),np.sin(pitch);cy,sy=np.cos(yaw),np.sin(yaw)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def fk(document,targets):
    matrices={'base_Link':np.eye(4)};remaining=list(document.findall('joint'))
    while remaining:
        progressed=False
        for joint in remaining[:]:
            parent=joint.find('parent').attrib['link']
            if parent not in matrices:continue
            origin=joint.find('origin');T=np.eye(4);T[:3,:3]=rpy_matrix(_vector(origin.attrib.get('rpy','0 0 0')));T[:3,3]=_vector(origin.attrib.get('xyz','0 0 0'))
            if joint.attrib['type']!='fixed':
                axis=_vector(joint.find('axis').attrib['xyz']);angle=targets[joint.attrib['name']]
                K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
                rotation=np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)
                T[:3,:3]=T[:3,:3]@rotation
            matrices[joint.find('child').attrib['link']]=matrices[parent]@T
            remaining.remove(joint);progressed=True
        if not progressed:raise ValueError('URDF tree cannot be resolved')
    return matrices


def original_mesh_frames(source_frames):
    """Exact original source component swap, with mesh map assigned by part name."""
    desired={name:matrix.copy() for name,matrix in source_frames.items()}
    for finger in range(1,4):
        tip='finger%d_tip_Link'%finger;outlet='finger%d_outlet_Link'%finger
        # Author mesh-map index2 is OUTLET with R_tip,p_outlet; index3 TIP with R_outlet,p_tip.
        desired[outlet][:3,:3]=source_frames[tip][:3,:3]
        desired[tip][:3,:3]=source_frames[outlet][:3,:3]
    return desired


def stl_vertices(path):
    raw=path.read_bytes();count=int.from_bytes(raw[80:84],'little')
    if len(raw)!=84+count*50:raise ValueError('Expected original binary STL: '+str(path))
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
    return np.frombuffer(raw,offset=84,count=count,dtype=dtype)['vertices'].reshape(-1,3).astype(np.float64)


def validate(urdf,manifest,overlay,output_dir):
    sys.path.insert(0,str(overlay));import pybullet as p
    from transforms3d.quaternions import quat2mat
    scope={'np':np,'quat2mat':quat2mat}
    node=next(n for n in ast.parse(GENERATOR.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='four_meta_to_matrix')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(GENERATOR),'exec'),scope)
    author_matrix=scope['four_meta_to_matrix']
    source_doc=ET.parse(SOURCE_URDF).getroot();adapter_doc=ET.parse(urdf).getroot()
    original_joint_names=[j.attrib['name'] for j in source_doc.findall('joint')]
    mesh_by_link={link.attrib['name']:MESH_DIR/Path(link.find('visual/geometry/mesh').attrib['filename']).name for link in source_doc.findall('link')}
    vertices={name:stl_vertices(path) for name,path in mesh_by_link.items()}
    table=json.loads(TABLE.read_text());cases=[]
    for pose,entries in table.items():
        for width,entry in entries.items():cases.append((pose+'/'+width,np.array(entry['12d'])))
    path=closure_path(94,67,96)
    cases.extend(('close_path/%03d'%i,q) for i,q in enumerate(path['joint_geometry_rad']))
    # Guard all physics integration; resetJointState is only the FK probe input.
    def no_step(*a,**kw):raise RuntimeError('Geometry validation cannot integrate physics')
    p.stepSimulation=no_step;client=p.connect(p.DIRECT);assert client==0
    before=time.monotonic();per_link={name:0. for name in vertices};analytic_error=0.;raw_baseline_error=0.;frame_error=0.
    case_rows=[]
    try:
        source_body=p.loadURDF(str(SOURCE_URDF),useFixedBase=True,flags=p.URDF_USE_INERTIA_FROM_FILE)
        adapter_body=p.loadURDF(str(urdf),useFixedBase=True,flags=p.URDF_USE_INERTIA_FROM_FILE)
        source_ids={p.getJointInfo(source_body,i)[1].decode():i for i in range(p.getNumJoints(source_body))}
        adapter_ids={p.getJointInfo(adapter_body,i)[1].decode():i for i in range(p.getNumJoints(adapter_body))}
        def bullet_frames(body):
            result={'base_Link':np.eye(4)}
            for i in range(p.getNumJoints(body)):
                state=p.getLinkState(body,i,computeForwardKinematics=True);pose=[*state[4],*state[5]]
                name=p.getJointInfo(body,i)[12].decode()
                result[name]=author_matrix(pose).astype(np.float64)
            return result
        for label,q in cases:
            targets=adapter_targets(q,manifest)
            source_targets={name:float(value) for name,value in zip(original_joint_names,q)}
            analytic_source=fk(source_doc,source_targets);analytic_adapter=fk(adapter_doc,targets)
            expected_analytic=original_mesh_frames(analytic_source)
            for name in vertices:analytic_error=max(analytic_error,float(np.max(np.abs(expected_analytic[name]-analytic_adapter[name]))))
            for name,value in source_targets.items():p.resetJointState(source_body,source_ids[name],value)
            for name,value in targets.items():p.resetJointState(adapter_body,adapter_ids[name],value)
            observed_source=bullet_frames(source_body);observed_adapter=bullet_frames(adapter_body)
            expected=original_mesh_frames(observed_source);maximum=0.
            for name,V in vertices.items():
                # Include every source triangle vertex, no surface sampling/subset.
                delta=observed_adapter[name]-expected[name]
                errors=V@delta[:3,:3].T+delta[:3,3]
                err=float(np.sqrt(np.sum(errors*errors,axis=1)).max())
                per_link[name]=max(per_link[name],err);maximum=max(maximum,err)
                frame_error=max(frame_error,float(np.max(np.abs(delta))))
                raw_delta=observed_source[name]-expected[name]
                raw_difference=V@raw_delta[:3,:3].T+raw_delta[:3,3]
                raw_baseline_error=max(raw_baseline_error,float(np.sqrt(np.sum(raw_difference*raw_difference,axis=1)).max()))
            case_rows.append({'case':label,'max_original_mesh_vertex_error_m':maximum})
    finally:p.disconnect(client)
    # Exact analytic algebra; actual Bullet frame values are quantized to float32 by source code.
    helperless=manifest.get('helperless',False)
    if not helperless:assert analytic_error<2e-14,analytic_error
    tolerance=1e-6 if helperless else 2e-7
    assert max(per_link.values())<tolerance,per_link
    assert raw_baseline_error>1e-4,raw_baseline_error
    report={'status':'bounded_helperless_geometry_approximation_validated' if helperless else 'original_geometry_alignment_validated','source_revision':REV,
        'author_source_sha256':SOURCE_SHA256,'adapter_URDF_sha256':sha(urdf),'adapter_manifest_sha256':sha(output_dir/'ADAPTER.json'),
        'preshape_cases':355,'geometric_closed_path_cases':96,'total_cases':len(cases),'entity_links_compared':len(vertices),
        'all_source_triangle_vertices_compared':True,'vertices_per_link':{k:len(v) for k,v in vertices.items()},
        'analytic_transform_max_absolute_error':analytic_error,'PyBullet_author_float32_frame_max_absolute_error':frame_error,
        'all_vertices_max_error_m':max(per_link.values()),'per_link_max_vertex_error_m':per_link,
        'declared_Bullet_vertex_tolerance_m':tolerance,'helperless':helperless,
        'geometric_approximation':manifest.get('approximations',[]),'unadapted_URDF_geometry_negative_control_max_error_m':raw_baseline_error,
        'source_four_meta_to_matrix_executed_unchanged':True,'no_new_surface_samples':True,
        'static_URDF_tree_geometry':True,'per_frame_mesh_reset_used':False,
        'zero_step_joint_reset_used_only_for_FK_validation':True,'physics_steps':0,'dynamic_validity_claim':False,
        'original_13_entity_inertials_preserved':True,'massless_helper_count':len(manifest['helper_frames']),
        'validation_seconds':time.monotonic()-before,
        'boundary':('Bounded helperless geometry approximation only; original third-tip constant Z delta omitted as explicitly recorded. Dynamic coupling/control still independently validated; no physics steps, contacts or lift here.' if helperless else 'Geometry equivalence only. One zero-mass coordinate helper requires separately validated engine dynamics/coupling. Limits are geometry-domain envelopes, not original force limits. No control policy, contacts, lift, networks or asset sampling executed.')}
    (output_dir/'VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
    (output_dir/'case_vertex_errors.json').write_text(json.dumps(case_rows,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/dh3_geometry_adapter_20261103_v1')
    parser.add_argument('--overlay',type=Path,default=ROOT/'outputs/native_deps/python_overlay')
    parser.add_argument('--generate-only',action='store_true')
    parser.add_argument('--helperless',action='store_true',help='Single bounded third-tip Z-delta approximation, no zero-mass moving frame')
    args=parser.parse_args();urdf,manifest=build_adapter(args.output_dir,helperless=args.helperless)
    if args.generate_only:print(json.dumps({'URDF':str(urdf),'validated':False}));return
    report=validate(urdf,manifest,args.overlay,args.output_dir)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
