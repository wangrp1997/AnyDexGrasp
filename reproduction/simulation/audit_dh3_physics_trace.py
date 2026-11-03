"""Read-only, simulator-free audit of saved DH3 physics-port traces.

Recomputes mechanism residuals/finite bounds and observed lifting separately.
No pybullet import, inference, collision query, render or physical step occurs.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text())


def absolute(path):
    path=Path(path)
    return path if path.is_absolute() else ROOT/path


def match(result,key,value,atol=1e-12):
    assert key in result,key
    observed=result[key]
    if isinstance(value,(bool,str)):
        assert observed==value,(key,observed,value)
    else:assert np.allclose(observed,value,rtol=0,atol=atol),(key,observed,value)


def source_audit(source,mode):
    tree=ast.parse(source.read_text())
    calls=[]
    class Calls(ast.NodeVisitor):
        def __init__(self):self.scope=[]
        def visit_ClassDef(self,node):
            self.scope.append(node.name);self.generic_visit(node);self.scope.pop()
        def visit_FunctionDef(self,node):
            self.scope.append(node.name);self.generic_visit(node);self.scope.pop()
        def visit_Call(self,node):
            if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='p':
                calls.append((node.func.attr,tuple(self.scope),node.lineno))
            self.generic_visit(node)
    Calls().visit(tree)
    creators=[c for c in calls if c[0]=='createConstraint']
    resets=[c for c in calls if c[0] in ('resetJointState','resetBasePositionAndOrientation')]
    assert len(creators)==2 and all(c[1]==('CoupledDH3','__init__') for c in creators)
    assert len(resets)==1 and resets[0][1]==('CoupledDH3','__init__')
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_'+mode)
    steps=[c for c in calls if c[0]=='stepSimulation' and c[1]==('run_'+mode,)]
    assert len(steps)==1
    text=ast.unparse(function)
    assert 'p.resetJointState = p.resetBasePositionAndOrientation = forbid_reset' in text or 'p.resetJointState = p.resetBasePositionAndOrientation = forbid_reset' in source.read_text()
    guard=next(n for n in ast.walk(function) if isinstance(n,ast.FunctionDef) and n.name=='forbid_reset')
    assert any(isinstance(n,ast.Raise) for n in guard.body)
    return dict(constraint_creation_calls_only_hand_initialization=[c[2] for c in creators],
        joint_reset_calls_only_hand_initialization=[c[2] for c in resets],
        runtime_reset_guard_raises=True,mode_step_call_sites=1,
        media_step_call_sites=0 if mode=='cube' else None,
        boundary='Frozen source control-flow plus recorded initial constraint inventory; no per-step constraint inventory was recorded')


def contact_summary(contacts):
    return dict(rows=int(len(contacts)),positive_normal_rows=int((contacts[:,6]>0).sum()) if len(contacts) else 0,
        peak_normal_force_N=float(contacts[:,6].max()) if len(contacts) else None,
        min_signed_distance_m=float(contacts[:,5].min()) if len(contacts) else None)


def rigid_model_audit(controller, trace, result):
    """Recompute the frozen ideal coordinate model and geometry-only radius."""
    frozen=controller['frozen_rigid_transmission_model'];path=absolute(frozen['path']);model=load(path)
    assert sha(path)==frozen['sha256'] and model==frozen['model']
    assert model['physics_steps']==0 and model['frozen_before_new_physics']
    assert sha(path.parent/'controller_source_frozen.py')==model['controller_source_sha256']==controller['controller_script_sha256']
    assert model['adapter_manifest_sha256']==controller['adapter_manifest_sha256']
    assert model['URDF_sha256']==controller['derived_urdf_sha256']
    declared=model['parameters'];actual=controller['assumptions_SI_not_hardware_identification']
    different=[key for key in declared if declared[key]!=actual[key]]
    if different:
        assert different==['dt'],different
        plans=[]
        def find_plans(value):
            if isinstance(value,dict):
                for key,item in value.items():
                    if key=='planned_runs':plans.extend(item)
                    else:find_plans(item)
            elif isinstance(value,list):
                for item in value:find_plans(item)
        find_plans(model)
        planned=[row for row in plans if Path(row['output_dir']).resolve()==Path(controller['frozen_controller_source']).parent.resolve()]
        assert len(planned)==1 and planned[0]['time_step_s']==actual['dt']
        assert planned[0]['expected_steps']==len(trace['step'])
    for key in ('prior_rigid_model','prior_time_resolution_plan','prior_time_resolution_model'):
        if key in model:assert sha(model[key]['path'])==model[key]['sha256']
    assert sha(model['primary_engine_source']['path'])==model['primary_engine_source']['sha256']
    for item in model['source_meshes']:assert sha(item['path'])==item['sha256']
    names=controller['joint_order'];masters=model['masters'];rows=controller['two_register_affine_map']
    assert names==model['joint_order'] and masters==controller['active_motors']
    A=np.zeros((len(names),2));b=np.zeros(len(names));C=[];offsets=[]
    for i,name in enumerate(names):
        row=rows[name];col=0 if abs(row['position_slope'])>1e-13 else 1
        key='position_slope' if col==0 else 'rotation_slope';master=masters[col]
        factor=row[key]/rows[master][key];offset=row['offset']-factor*rows[master]['offset']
        A[i,col]=factor;b[i]=offset
        if name not in masters:
            cr=np.zeros(len(names));cr[i]=1;cr[names.index(master)]=-factor;C.append(cr);offsets.append(offset)
    C=np.array(C);offsets=np.array(offsets)
    for actual,declared in ((A,model['A']),(b,model['b']),(C,model['C']),(offsets,model['constraint_offsets'])):
        assert np.array_equal(actual,declared)
    rank=int(np.linalg.matrix_rank(C));assert rank==9==model['constraint_matrix_rank']
    assert np.linalg.matrix_rank(A)==2 and np.max(abs(C@A))==model['CA_max_abs']==0
    assert np.max(abs(C@b-offsets))==model['Cb_minus_offset_max_abs']==0
    assert np.allclose(np.linalg.svd(C,compute_uv=False),model['constraint_singular_values'],rtol=0,atol=1e-14)
    target_residual=trace['joint_target']@C.T-offsets
    assert abs(target_residual).max()<1e-14
    observed_residual=trace['q']@C.T-offsets
    assert np.allclose(observed_residual,trace['gear_residual'],rtol=0,atol=1e-14)
    u=trace['q'][:,[names.index(name) for name in masters]]
    physical_coordinate_deviation=float(abs(trace['q']-(u@A.T+b)).max())
    # Parse source mesh bytes independently, without loading a simulator or the
    # model-builder's own STL/kinematics helper.
    def vertices(path):
        raw=Path(path).read_bytes()
        count=int.from_bytes(raw[80:84],'little') if len(raw)>=84 else -1
        if len(raw)==84+50*count:
            dtype=np.dtype([('normal','<f4',(3,)),('vertex','<f4',(3,3)),('attribute','<u2')])
            return np.frombuffer(raw,offset=84,dtype=dtype,count=count)['vertex'].reshape(-1,3).astype(np.float64)
        lines=[line.split()[1:] for line in raw.decode('ascii').splitlines() if line.strip().startswith('vertex ')]
        return np.asarray(lines,dtype=np.float64)
    doc=ET.parse(absolute(controller['derived_urdf'])).getroot()
    edges={j.find('child').attrib['link']:(j.find('parent').attrib['link'],float(np.linalg.norm(np.fromstring(j.find('origin').attrib.get('xyz','0 0 0'),sep=' ')))) for j in doc.findall('joint')}
    assert all(j.attrib['type'] in ('fixed','revolute','continuous') for j in doc.findall('joint'))
    mesh_hash={item['path']:item['sha256'] for item in model['source_meshes']}
    radius_rows={};mass=0;identity_visuals=0;identity_collisions=0
    for link in doc.findall('link'):
        name=link.attrib['name'];node=name;length=0.;visited=set()
        while node in edges:
            assert node not in visited;visited.add(node);node,extra=edges[node];length+=extra
        visual=link.find('visual');collision=link.find('collision')
        for element in (visual,collision):
            origin=element.find('origin');mesh=element.find('geometry/mesh')
            xyz=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            rpy=np.fromstring(origin.attrib.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            scale=np.fromstring(mesh.attrib.get('scale','1 1 1'),sep=' ')
            assert np.array_equal(xyz,np.zeros(3)) and np.array_equal(rpy,np.zeros(3)) and np.array_equal(scale,np.ones(3))
            assert sha(mesh.attrib['filename'])==mesh_hash[mesh.attrib['filename']]
        assert visual.find('geometry/mesh').attrib['filename']==collision.find('geometry/mesh').attrib['filename']
        identity_visuals+=1;identity_collisions+=1
        norms=np.linalg.norm(vertices(visual.find('geometry/mesh').attrib['filename']),axis=1)
        mesh_norm=float(norms.max());inertial=link.find('inertial')
        com_norm=float(np.linalg.norm(np.fromstring(inertial.find('origin').attrib.get('xyz','0 0 0'),sep=' ')))
        mass+=float(inertial.find('mass').attrib['value'])
        row=dict(tree_origin_path_length_m=length,max_mesh_vertex_local_norm_m=mesh_norm,
            local_COM_norm_m=com_norm,all_relevant_points_radius_bound_m=length+max(mesh_norm,com_norm))
        for key,value in row.items():match(model['per_link_radius_bounds'][name],key,value)
        radius_rows[name]=row
    radius=max(row['all_relevant_points_radius_bound_m'] for row in radius_rows.values());diameter=2*radius
    scale=model['load_scale_terms'];a=controller['assumptions_SI_not_hardware_identification']
    force_norm=np.sqrt(3)*a['wrist_max_force_per_row_N_or_Nm'];gravity=float(np.linalg.norm(a['gravity_m_s2']))
    engineering=force_norm+diameter*(force_norm+(mass+.1)*gravity)+a['position_motor_max_torque_Nm']+a['rotation_motor_max_torque_Nm']
    for key,value in dict(hand_mass_kg=mass,geometry_radius_bound_about_baseLink_m=radius,
        geometry_diameter_bound_m=diameter,wrist_force_norm_scale_N=force_norm,wrist_torque_norm_scale_Nm=force_norm,
        gravity_norm_m_s2=gravity,engineering_moment_scale_Nm=engineering).items():match(scale,key,value)
    cap=a['gear_max_torque_Nm'];metrics=result['rigid_transmission_metrics']
    delta=np.diff(np.vstack([np.zeros((1,observed_residual.shape[1])),trace['gear_residual']]),axis=0)
    products=trace['gear_reaction']*delta;peak=float(abs(trace['gear_reaction']).max())
    for key,value in dict(finite_row_cap_Nm=cap,peak_row_reaction_Nm=peak,minimum_margin_to_row_cap_Nm=cap-peak,
        maximum_cap_utilization=peak/cap,sampled_absolute_constraint_correction_work_proxy_J=float(abs(products).sum()),
        sampled_signed_constraint_correction_work_proxy_J=float(products.sum()),
        sampled_peak_abs_constraint_power_proxy_W=float(abs(products.sum(1)).max()/a['dt'])).items():match(metrics,key,value)
    return dict(model_path=str(path),model_sha256=sha(path),frozen_model_copy_and_controller_SHA=True,
        ideal_map_A_b_C_exact=True,rank_C=rank,nullity_C=len(names)-rank,rank_A=2,CA_max_abs=float(abs(C@A).max()),
        Cb_minus_offset_max_abs=float(abs(C@b-offsets).max()),command_target_Cq_minus_offset_max_abs=float(abs(target_residual).max()),
        actual_q_minus_A_measured_u_plus_b_max_abs_rad=physical_coordinate_deviation,
        actual_Cq_offset_agrees_logged_residual=True,
        geometry_radius_bound_about_baseLink_m=radius,geometry_diameter_bound_m=diameter,
        geometry_boundary='Triangle-inequality bound on source mesh/COM points for revolute/fixed tree; engine collision margins are not included',
        identity_origin_unit_scale_visual_count=identity_visuals,identity_origin_unit_scale_collision_count=identity_collisions,
        hand_mass_kg=mass,engineering_moment_scale_Nm=engineering,finite_structural_row_cap_Nm=cap,
        observed_peak_row_reaction_Nm=peak,cap_saturated=bool(peak>=cap-1e-7),maximum_cap_utilization=peak/cap,
        sampled_constraint_work_metrics=metrics,
        boundary='The geometry radius is a source-geometry bound. Wrench-times-diameter is only a load scale, not a constraint-multiplier theorem or validated hardware rating. Actual q violates the ideal relation by the separately reported residual.')


def audit(run_dir):
    r=load(run_dir/'RESULTS.json');c=load(run_dir/'CONTROLLER.json')
    with np.load(run_dir/'trace.npz',allow_pickle=False) as stored:t={key:stored[key].copy() for key in stored.files}
    mode='cube' if 'object_pose' in t else 'empty'
    a=c['assumptions_SI_not_hardware_identification'];dt=a['dt'];n=len(t['step'])
    assert np.array_equal(t['step'],np.arange(1,n+1))
    assert np.array_equal(t['time'],t['step']*dt)
    assert all(len(value)==n for value in t.values())
    finite=bool(all(np.isfinite(value).all() for value in t.values() if value.dtype.kind in 'fiu'))
    assert sha(run_dir/'trace.npz')==r['trace_sha256']
    assert sha(run_dir/'CONTROLLER.json')==r.get('controller_sha256',r.get('controller_receipt_sha256'))
    snapshot=load(run_dir/'SOURCE_SNAPSHOT.json');frozen=absolute(snapshot['path'])
    assert sha(frozen)==snapshot['sha256']==c['controller_script_sha256']
    source_checks=source_audit(frozen,mode)
    for key,hashkey in (('adapter_manifest','adapter_manifest_sha256'),('derived_urdf','derived_urdf_sha256')):
        assert sha(absolute(c[key]))==c[hashkey]
    assert sha(c['author_kinematics']['source'])==c['author_kinematics']['source_sha256']
    names=c['joint_order'];ix={name:i for i,name in enumerate(names)}
    residual=np.column_stack([t['q'][:,ix[g['dependent']]]-g['factor']*t['q'][:,ix[g['master']]]-g['offset'] for g in c['gears']])
    assert residual.tobytes()==t['gear_residual'].tobytes()
    limits=np.zeros(len(names))
    limits[ix['finger1_joint']]=a['position_motor_max_torque_Nm']
    limits[ix['finger1_base_joint']]=a['rotation_motor_max_torque_Nm']
    motor_bound=bool((abs(t['motor_torque'])<=limits+1e-8).all())
    gear_bound=bool((abs(t['gear_reaction'])<=a['gear_max_torque_Nm']+1e-7).all())
    wrist_force_bound=bool((abs(t['wrist_reaction'])<=a['wrist_max_force_per_row_N_or_Nm']+1e-7).all())
    wrist_bound=bool((t['wrist_position_error']<.002).all())
    gearmax=float(abs(residual).max());speed=float(abs(t['v']).max())
    work_proxy=float(np.sum(abs(t['gear_reaction'][1:]*np.diff(residual,axis=0))))
    stable=bool(finite and gearmax<.01 and speed<1.5 and motor_bound and gear_bound and wrist_bound)
    for key,value in dict(physics_steps=n,physical_seconds=n*dt,finite=finite,max_gear_residual_rad=gearmax,
        peak_joint_velocity_rad_s=speed,motor_torque_bounds_pass=motor_bound,gear_reaction_bounds_pass=gear_bound,
        max_wrist_COM_error_m=float(t['wrist_position_error'].max()),peak_motor_torque_Nm=abs(t['motor_torque']).max(0),
        peak_gear_reaction_Nm=abs(t['gear_reaction']).max(0)).items():match(r,key,value)
    for g,maximum in zip(c['gears'],abs(residual).max(0)):match(r['per_gear_max_residual_rad'],g['dependent'],maximum)
    # Independent initial actual-constraint readback, not the summary zero counter.
    actual=c['actual_constraint_info'];hand=int(actual['wrist'][0]);all_constraints=[actual['wrist']]+actual['gears']
    assert actual['wrist'][2]==-1 and actual['wrist'][4]==4
    assert len(actual['gears'])==len(c['gears'])==9
    assert all(row[0]==hand and row[2]==hand and row[4]==6 for row in actual['gears'])
    assert actual['wrist'][10]==a['wrist_max_force_per_row_N_or_Nm']
    xml=ET.parse(absolute(c['derived_urdf'])).getroot()
    joint_to_link={node.attrib['name']:node.find('child').attrib['link'] for node in xml.findall('joint')}
    link_names={-1:'base_Link'}
    for gear,info in zip(c['gears'],actual['gears']):
        assert np.allclose([info[10],info[11],info[13],info[14]],
            [a['gear_max_torque_Nm'],-gear['factor'],gear['offset'],a['gear_erp']],rtol=0,atol=1e-14)
        link_names[int(info[1])]=joint_to_link[gear['dependent']]
        link_names[int(info[3])]=joint_to_link[gear['master']]
    peak_step,peak_gear=np.unravel_index(abs(residual).argmax(),residual.shape)
    report=dict(run_dir=str(run_dir.resolve()),mode=mode,status='saved_trace_independently_verified',
        source_snapshot_sha256=sha(frozen),source_snapshot_acquisition=snapshot['acquisition'],
        source_control_flow=source_checks,physics_steps=n,physical_seconds=n*dt,finite=finite,
        mechanism=dict(residuals_recomputed_from_all_q_bit_exact=True,max_residual_rad=gearmax,
            per_gear_max_residual_rad={g['dependent']:float(x) for g,x in zip(c['gears'],abs(residual).max(0))},
            peak_joint_velocity_rad_s=speed,motor_torque_bounds_pass=motor_bound,gear_reaction_bounds_pass=gear_bound,
            wrist_reaction_per_row_bound_pass=wrist_force_bound,wrist_position_tracking_pass=wrist_bound,
            peak_wrist_reaction_per_row=abs(t['wrist_reaction']).max(0).tolist(),
            sampled_constraint_correction_work_abs_proxy_J_excluding_initial=work_proxy,
            peak_residual=dict(step=int(t['step'][peak_step]),time_s=float(t['time'][peak_step]),
                phase=str(t['phase'][peak_step]),dependent=c['gears'][peak_gear]['dependent'],
                signed_residual_rad=float(residual[peak_step,peak_gear]),gear_reaction_Nm=float(t['gear_reaction'][peak_step,peak_gear]))),
        constraints=dict(recorded_count=len(all_constraints),hand_body_id=hand,only_hand_to_world_wrist_and_internal_gears=True),
        declared_limits=a,checks=dict(trace_SHA=True,controller_SHA=True,frozen_source_SHA=True,adapter_and_URDF_SHA=True),
        artifact_SHA={name:sha(run_dir/name) for name in ('RESULTS.json','CONTROLLER.json','SOURCE_SNAPSHOT.json','trace.npz')},
        phases=[])
    if 'frozen_rigid_transmission_model' in c:
        report['rigid_transmission_model_audit']=rigid_model_audit(c,t,r)
    report['time_and_control_audit']=time_and_control_audit(c,t,r,mode)
    contacts=None
    if mode=='cube':
        scene_path=absolute(c['scene_manifest']);scene=load(scene_path)
        selection_path=absolute(c['selection_results']);selection=load(selection_path)
        assert sha(scene_path)==c['scene_manifest_sha256']
        assert sha(selection_path)==c['selection_sha256']
        assert sha(absolute(c['candidate_npz']))==c['candidate_sha256']==selection['output']['sha256']
        assert scene['scene_signature_sha256']==c['scene_signature_sha256']
        assert np.array_equal(scene['camera']['T_WC'],c['T_WC'])
        for file in scene['files']:assert sha(file['path'])==file['sha256']
        ids=c['body_ids'];assert ids['hand']==hand
        assert all(ids['object'] not in (row[0],row[2]) for row in all_constraints)
        contacts=np.load(run_dir/'contacts.npy',allow_pickle=False)
        assert contacts.ndim==2 and contacts.shape[1]==9 and np.isfinite(contacts).all()
        assert np.equal(contacts[:,:5],np.rint(contacts[:,:5])).all()
        assert ((contacts[:,0]>=1)&(contacts[:,0]<=n)).all()
        assert ((contacts[:,1]==hand)|(contacts[:,2]==hand)).all()
        def pair(body):return ((contacts[:,1]==hand)&(contacts[:,2]==body))|((contacts[:,2]==hand)&(contacts[:,1]==body))
        object_contacts=contacts[pair(ids['object'])];table_contacts=contacts[pair(ids['table'])]
        final_steps=np.arange(n-round(.5/dt)+1,n+1)
        last=object_contacts[np.isin(object_contacts[:,0],final_steps)&(object_contacts[:,6]>0)]
        counts=np.array([(last[:,0]==step).sum() for step in final_steps])
        force_totals=np.array([last[last[:,0]==step,6].sum() for step in final_steps])
        initial_z=scene['actual_state']['object']['position_world_m'][2]
        lift=float(t['object_pose'][-1,2]-initial_z);lowest=float(t['object_aabb'][-1,0,2])
        maximum=float(t['object_pose'][:,2].max()-initial_z)
        half=np.array(scene['spec']['object']['collision_box_size_m'])/2
        extent=np.abs(Rotation.from_quat(t['object_pose'][:,3:7]).as_matrix())@half
        expected_aabb=np.stack([t['object_pose'][:,:3]-extent,t['object_pose'][:,:3]+extent],axis=1)
        aabb_error=float(abs(expected_aabb-t['object_aabb']).max())
        assert aabb_error<1e-12
        held=bool(lift>.05 and lowest>.04 and len(last)>0)
        match(r,'lift_observed',held);match(r,'port_stability_pass',stable)
        match(r,'status','object_lifted' if held and stable else 'not_lifted_or_port_invalid')
        for key,value in dict(object_final_lift_m=lift,object_final_lowest_world_z_m=lowest,object_max_lift_m=maximum,
            hand_object_contact_rows=len(object_contacts),hand_table_contact_rows=len(table_contacts),
            final_half_second_hand_object_contact_rows=len(last)).items():match(r,key,value)
        match(r,'state_resets_after_first_step',0);match(r,'object_attachment_constraints',0)
        media=load(run_dir/'frames.json');frame_records=media['frames']
        assert media['fps']==20 and np.isclose(1/(media['every_physics_steps']*dt),20,rtol=0,atol=1e-12)
        assert media['actual_simultaneous_render'] and media['zero_additional_physics_steps_for_media']
        expected_steps=np.arange(1,n+1,media['every_physics_steps'])
        assert np.array_equal(expected_steps,[row['physics_step'] for row in frame_records])
        assert len(frame_records)==r['actual_rendered_frames']==len(expected_steps)
        assert len(list((run_dir/'frames').glob('*.png')))==len(frame_records)
        frame_hash=hashlib.sha256()
        for row in frame_records:
            path=run_dir/row['path'];assert abs(row['physical_time_s']-row['physics_step']*dt)<1e-15
            with Image.open(path) as image:
                assert image.size==(640,480) and image.mode=='RGB';image.verify()
            frame_hash.update((row['path']+' '+sha(path)+'\n').encode())
        peak_contacts=contacts[contacts[:,0]==t['step'][peak_step]]
        peak_pairs=[]
        for row in peak_contacts:
            if row[6]<=0:continue
            other=int(row[2] if row[1]==hand else row[1])
            link=int(row[3] if row[1]==hand else row[4])
            peak_pairs.append(dict(other_body='object' if other==ids['object'] else 'table' if other==ids['table'] else 'hand',
                hand_link=link_names.get(link,'index_'+str(link)),normal_force_N=float(row[6]),signed_distance_m=float(row[5])))
        report['mechanism']['peak_residual']['simultaneous_positive_normal_contacts']=peak_pairs
        report.update(lift_observed=held,port_stability_pass=stable,validated_physical_grasp_success=bool(held and stable),
            final_object_lift_m=lift,final_object_lowest_world_z_m=lowest,max_object_lift_m=maximum,
            independent_oriented_cube_AABB_max_error_m=aabb_error,
            hand_object=contact_summary(object_contacts),hand_table=contact_summary(table_contacts),
            final_half_second=dict(steps=len(final_steps),positive_normal_contact_covered_steps=int((counts>0).sum()),
                every_step_positive_normal_contact=bool((counts>0).all()),min_contact_rows_per_step=int(counts.min()),
                max_contact_rows_per_step=int(counts.max()),min_total_normal_force_per_step_N=float(force_totals.min()),
                contact_rows=len(last)),
            perception_binding=dict(scene_manifest_sha256=sha(scene_path),selection_sha256=sha(selection_path),
                candidate_sha256=sha(absolute(c['candidate_npz'])),scene_signature_sha256=c['scene_signature_sha256']),
            media=dict(actual_saved_frames=len(frame_records),every_physics_steps=media['every_physics_steps'],fps=media['fps'],
                implied_video_seconds=len(frame_records)/media['fps'],first_last_physics_steps=[int(expected_steps[0]),int(expected_steps[-1])],
                frames_SHA_sequence=frame_hash.hexdigest(),all_frames_decode=True,zero_additional_physics_steps_source_and_receipt=True,
                encoded_MP4_independently_probed_by_this_audit=False),
            boundary='Actual cube lifting/contact observed but port stability is separate. Recorded finite ideal gears, scripted wrist and convex contacts are simulator assumptions; no hardware or paper success-rate claim.')
        report['artifact_SHA'].update({name:sha(run_dir/name) for name in ('contacts.npy','frames.json')})
    else:
        contacts=np.load(run_dir/'self_contacts.npy',allow_pickle=False)
        assert contacts.ndim==2 and contacts.shape[1]==5 and np.isfinite(contacts).all()
        close_final=int(np.flatnonzero(t['phase']=='hold_closed')[-1])
        error=abs(t['q'][close_final]-t['joint_target'][close_final])
        recent=contacts[(contacts[:,0]>=close_final-round(.1/dt))&(contacts[:,0]<=close_final+1)&(contacts[:,4]>.01)]
        stopped=bool(error.max()>=.04 and len(recent)>0 and abs(t['v'][close_final,ix['finger1_joint']])<.05)
        tracking=bool(error.max()<.04 or stopped)
        empty_pass=bool(stable and tracking)
        match(r,'closed_final_max_target_error_rad',float(error.max()))
        match(r,'closed_final_contact_stopped',stopped);match(r,'closed_final_tracking_or_contact_stop',tracking)
        match(r,'legacy_absolute_correction_work_proxy_J' if 'legacy_absolute_correction_work_proxy_J' in r else 'absolute_constraint_correction_work_upper_proxy_J',work_proxy)
        match(r,'status','passed' if empty_pass else 'failed')
        report.update(empty_mechanism_pass=empty_pass,port_stability_pass=stable,grasp_result=False,
            closed_final_contact_stopped=stopped,closed_final_tracking_or_contact_stop=tracking,
            closed_final_max_target_error_rad=float(error.max()),
            boundary='Empty mechanism check only. A contact stop is not exact endpoint tracking and establishes no loaded-grasp validity.')
        report['artifact_SHA']['self_contacts.npy']=sha(run_dir/'self_contacts.npy')
    # Stage summaries retain transients rather than hiding them in a final hold.
    for phase in dict.fromkeys(t['phase']):
        mask=t['phase']==phase;steps=t['step'][mask]
        item=dict(phase=str(phase),steps=len(steps),first_last_steps=[int(steps[0]),int(steps[-1])],
            max_gear_residual_rad=float(abs(residual[mask]).max()),peak_joint_velocity_rad_s=float(abs(t['v'][mask]).max()),
            max_wrist_COM_error_m=float(t['wrist_position_error'][mask].max()),
            peak_gear_reaction_Nm=float(abs(t['gear_reaction'][mask]).max()))
        if mode=='cube':
            stage=contacts[np.isin(contacts[:,0],steps)]
            item['hand_object']=contact_summary(stage[((stage[:,1]==hand)&(stage[:,2]==ids['object']))|((stage[:,2]==hand)&(stage[:,1]==ids['object']))])
            item['hand_table']=contact_summary(stage[((stage[:,1]==hand)&(stage[:,2]==ids['table']))|((stage[:,2]==hand)&(stage[:,1]==ids['table']))])
            item['object_lift_first_last_m']=[float(t['object_pose'][mask][i,2]-initial_z) for i in (0,-1)]
        report['phases'].append(item)
    return report


def time_and_control_audit(c,t,result,mode):
    """Tie actual dt readback and continuous physical-time command to the log."""
    dt=c['assumptions_SI_not_hardware_identification']['dt'];n=len(t['step'])
    actual=c.get('actual_physics_engine_parameters')
    engine_verified=False;override_verified=False
    if actual:
        assert actual['fixedTimeStep']==dt and actual['numSubSteps']==1
        assert actual['numSolverIterations']==c['assumptions_SI_not_hardware_identification']['solver_iterations']
        assert actual['numNonContactInnerIterations']==c['assumptions_SI_not_hardware_identification']['non_contact_inner_iterations']
        assert [actual['gravityAcceleration'+axis] for axis in 'XYZ']==c['assumptions_SI_not_hardware_identification']['gravity_m_s2']
        engine_verified=True
    if 'actual_time_step_s' in result:match(result,'actual_time_step_s',dt)
    if 'actual_time_step_s' in result.get('rigid_transmission_metrics',{}):match(result['rigid_transmission_metrics'],'actual_time_step_s',dt)
    start=np.arange(n)*dt
    smooth=lambda x: np.clip(x,0,1)**2*(3-2*np.clip(x,0,1))
    target=np.array(t['register_target'],copy=True)
    if mode=='cube':
        scene=load(absolute(c['scene_manifest']))
        initial=np.load(absolute(c['candidate_npz']),allow_pickle=False)['top1_DH3_grasp_array'][15:17]
        position=np.full(n,initial[0]);closing=(start>=3.5)&(start<5.5)
        position[closing]=initial[0]*(1-smooth((start[closing]-3.5)/2));position[start>=5.5]=0
        target=np.column_stack([position,np.full(n,initial[1])])
        pre=np.asarray(c['world_pre_baseLink_m']);goal=np.asarray(c['world_target_baseLink_m'])
        base=np.broadcast_to(pre,(n,3)).copy();approach=(start>=.5)&(start<3.5)
        base[approach]=pre+(goal-pre)*smooth((start[approach]-.5)/3)[:,None];base[start>=3.5]=goal
        base[start>=6.25]=goal+np.array([0,0,.1])*smooth((start[start>=6.25]-6.25)/3)[:,None]
        offset=np.asarray(c['base_link_to_COM'][0]);rotation=Rotation.from_quat(c['baseLink_orientation_xyzw']).as_matrix()
        com_target=base+rotation@offset
        recovered_error=np.linalg.norm(t['com'][:,:3]-com_target,axis=1)
        error_discrepancy=float(abs(recovered_error-t['wrist_position_error']).max())
        assert error_discrepancy<2e-12
        if 'integration_parameter_override' in c:
            over=c['integration_parameter_override']
            assert over['observation_engine_time_step_s']==scene['spec']['physics']['time_step_s']
            assert over['actual_integration_time_step_s']==dt and over['num_sub_steps']==1
            assert over['scene_signature_unchanged'] and over['override_before_first_integration']
            override_verified=True
        else:assert dt==scene['spec']['physics']['time_step_s']
    else:
        initial=result['initial_registers'];position=np.full(n,initial[0]);rotation=np.full(n,initial[1]);other=initial[1]+23 if initial[1]<=77 else initial[1]-23
        out=(start>=.5)&(start<1.5);back=(start>=1.5)&(start<2.5);close=(start>=2.5)&(start<4.5)
        rotation[out]=initial[1]+(other-initial[1])*smooth(start[out]-.5)
        rotation[back]=other+(initial[1]-other)*smooth(start[back]-1.5)
        position[close]=initial[0]*(1-smooth((start[close]-2.5)/2));position[start>=4.5]=0
        target=np.column_stack([position,rotation]);error_discrepancy=None
    register_error=float(abs(target-t['register_target']).max());assert register_error<1e-12
    rows=c['two_register_affine_map'];joint=np.column_stack([rows[name]['offset']+target[:,0]*rows[name]['position_slope']+target[:,1]*rows[name]['rotation_slope'] for name in c['joint_order']])
    joint_error=float(abs(joint-t['joint_target']).max());assert joint_error<1e-12
    return dict(actual_dt_s=dt,recorded_every_external_step=True,num_sub_steps_per_recorded_step=1,
        actual_engine_readback_verified=engine_verified,scene_observation_dt_override_before_steps_verified=override_verified,
        command_time_is_step_start=True,all_step_continuous_register_function_max_error=register_error,
        all_step_affine_joint_target_max_error_rad=joint_error,reconstructed_wrist_COM_target_error_discrepancy_m=error_discrepancy,
        boundary='Command evaluated at (step-1)*dt; recorded state/time after one dt. Engine readback unavailable for historical 240Hz receipts is explicitly marked.')


def time_resolution_series(reports):
    cubes=sorted((r for r in reports if r['mode']=='cube'),key=lambda r:r['declared_limits']['dt'],reverse=True)
    assert len(cubes)>=2
    base=cubes[0];bc=load(Path(base['run_dir'])/'CONTROLLER.json');bt=np.load(Path(base['run_dir'])/'trace.npz',allow_pickle=False)
    dt0=bc['assumptions_SI_not_hardware_identification']['dt'];base_other=dict(bc['assumptions_SI_not_hardware_identification']);base_other.pop('dt')
    comparisons=[]
    for item in cubes[1:]:
        c=load(Path(item['run_dir'])/'CONTROLLER.json');t=np.load(Path(item['run_dir'])/'trace.npz',allow_pickle=False)
        dt=c['assumptions_SI_not_hardware_identification']['dt'];other=dict(c['assumptions_SI_not_hardware_identification']);other.pop('dt')
        assert other==base_other
        for key in ('candidate_sha256','selection_sha256','scene_manifest_sha256','world_pre_baseLink_m','world_target_baseLink_m','world_approach_axis','wrist_script'):
            assert c[key]==bc[key]
        ratio=round(dt0/dt);assert ratio>=1 and ratio*dt==dt0
        idx=np.arange(len(bt['step']))*ratio
        assert (idx*dt==np.arange(len(bt['step']))*dt0).all()
        bits={key:bt[key].tobytes()==t[key][idx].tobytes() for key in ('phase','register_target','joint_target')}
        assert all(bits.values()),bits
        comparisons.append(dict(reference_Hz=round(1/dt0),fine_Hz=round(1/dt),common_command_start_samples=len(idx),
            physical_start_times_bit_exact=True,common_start_phase_register_joint_targets_bit_exact=bits,
            only_declared_runtime_assumption_changed='dt'))
    hz=[round(1/r['declared_limits']['dt']) for r in cubes]
    errors=[r['mechanism']['max_residual_rad'] for r in cubes]
    orders=[float(np.log(errors[i]/errors[i+1])/np.log(hz[i+1]/hz[i])) for i in range(len(hz)-1)]
    return dict(reference_Hz=hz[0],Hz=hz,gear_residual_rad=errors,
        peak_joint_velocity_rad_s=[r['mechanism']['peak_joint_velocity_rad_s'] for r in cubes],
        observed_adjacent_residual_orders=orders,common_start_command_comparisons=comparisons,
        all_lift_observed=all(r['lift_observed'] for r in cubes),all_port_stability_pass=all(r['port_stability_pass'] for r in cubes),
        boundary='Same continuous command sampled at finer dt. Different contact switching and trajectories are retained; observed error trends do not prove asymptotic convergence or hardware accuracy.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--time-resolution-series',action='store_true')
    args=parser.parse_args()
    if args.output.exists():parser.error('Retain evidence; choose a new output')
    try:
        reports=[audit(absolute(path)) for path in args.run_dir]
        cubes=[r for r in reports if r['mode']=='cube']
        comparison=None
        if len(cubes)==2 and not args.time_resolution_series:
            dirs=[Path(r['run_dir']) for r in cubes];controllers=[load(d/'CONTROLLER.json') for d in dirs]
            aa=[dict(c['assumptions_SI_not_hardware_identification']) for c in controllers]
            changed=[key for key in aa[0] if aa[0][key]!=aa[1][key]]
            assert len(changed)==1,changed
            changed_key=changed[0];values=[a.pop(changed_key) for a in aa]
            same_input=all(controllers[0][key]==controllers[1][key] for key in
                ('candidate_sha256','selection_sha256','scene_manifest_sha256','world_approach_axis','world_target_baseLink_m','wrist_script'))
            assert aa[0]==aa[1] and same_input
            traces=[np.load(d/'trace.npz',allow_pickle=False) for d in dirs]
            commands={key:traces[0][key].tobytes()==traces[1][key].tobytes() for key in ('step','time','phase','register_target','joint_target')}
            assert all(commands.values())
            comparison=dict(only_declared_assumption_changed=changed_key,values=values,
                frozen_candidate_scene_and_script_identical=True,command_arrays_bit_identical=commands,
                both_lift_observed=all(r['lift_observed'] for r in cubes),both_port_stability_failed=all(not r['port_stability_pass'] for r in cubes),
                interpretation='One declared assumption changed with identical source candidate/scene and command arrays. Lifting and port stability remain separate; the comparison does not isolate a general physical cause.')
        empties=[r for r in reports if r['mode']=='empty']
        empty_comparison=None
        if len(empties)==2:
            dirs=[Path(r['run_dir']) for r in empties]
            empty_comparison=dict(trace_file_byte_identical=sha(dirs[0]/'trace.npz')==sha(dirs[1]/'trace.npz'),
                self_contact_file_byte_identical=sha(dirs[0]/'self_contacts.npy')==sha(dirs[1]/'self_contacts.npy'))
            assert all(empty_comparison.values())
        result=dict(status='all_saved_physics_traces_independently_verified',runs=reports,paired_comparison=comparison,
            empty_regression_comparison=empty_comparison,
            time_resolution_series=time_resolution_series(reports) if args.time_resolution_series else None,
            audit_source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
            audit_new_physics_steps=0,audit_new_network_forwards=0,audit_new_renders=0)
    except Exception as exc:
        result=dict(status='physics_trace_audit_failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');raise
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],runs=[dict(run_dir=r['run_dir'],mode=r['mode'],max_gear_residual_rad=r['mechanism']['max_residual_rad'],
        lift_observed=r.get('lift_observed'),port_stability_pass=r['port_stability_pass']) for r in reports],output=str(args.output)),indent=2))


if __name__=='__main__':main()
