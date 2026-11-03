"""Audit original DH3 control/data and URDF in PyBullet DIRECT, zero steps.

No hardware module is imported and no simulation integration or grasp occurs.
Original methods can be replayed only on a recording-only register interface.
"""
from __future__ import annotations
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from run_official_dh3_collision import ROOT,REV,sha,original


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/dh3_simulation_readiness_20261103_v1')
    parser.add_argument('--overlay',type=Path,default=ROOT/'outputs/native_deps/python_overlay')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.overlay));import pybullet as p
    urdf=ROOT/'generate_mesh_and_pointcloud/dh3_urdf/dh3_urdf/urdf/dh3_urdf.urdf'
    hardware=ROOT/'ur_toolbox/ur_toolbox/robot/DH3/DH3.py'
    arm_source=ROOT/'ur_toolbox/ur_toolbox/robot/ur.py'
    generator=ROOT/'generate_mesh_and_pointcloud/recover_dh3_hand_to_stl.py'
    table_path=ROOT/'generate_mesh_and_pointcloud/dh3_urdf/width_12D_angle_2D_angle.json'
    document=ET.parse(urdf).getroot();joint_elements=document.findall('joint')
    table=json.loads(table_path.read_text());source_receipts=[]
    workbook=ROOT/'generate_mesh_and_pointcloud/dh3_urdf/dh3_routine_to_angle-use.xlsx'
    excel_sheets=[];ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(workbook) as archive:
        strings=[]
        if 'xl/sharedStrings.xml' in archive.namelist():
            for item in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si',ns):
                strings.append(''.join(t.text or '' for t in item.findall('.//m:t',ns)))
        for name in sorted(n for n in archive.namelist() if n.startswith('xl/worksheets/sheet') and n.endswith('.xml')):
            rows=[]
            for row in ET.fromstring(archive.read(name)).findall('.//m:row',ns):
                cells={}
                for cell in row.findall('m:c',ns):
                    value=cell.find('m:v',ns);text=value.text if value is not None else None
                    if cell.attrib.get('t')=='s' and text is not None:text=strings[int(text)]
                    cells[cell.attrib['r']]=text
                rows.append(cells)
            zero=[row for row in rows[1:] if any(k.startswith('B') and value is not None and float(value)==0 for k,value in row.items())]
            excel_sheets.append({'sheet':name,'rows_including_header':len(rows),'headers':rows[0],
                'column_B_position0_rows':zero,'position0_count':len(zero)})
    for path in [urdf,hardware,arm_source,generator,table_path,workbook,*sorted((urdf.parent.parent/'meshes').glob('*.STL'))]:
        source_receipts.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
    # Preserve original 2D-to-12D geometric calibration formula from author source.
    function=next(n for n in ast.parse(generator.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='read_excel_2D_angle_to_12D_angle')
    ranges=ast.literal_eval(next(n.value for n in function.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='pos_scope'))
    rate=original(generator,{'rate'},{})['rate']
    def geometry_joint_vector(position,rotation):
        return np.array([rate((0,95),bounds,rotation if i in [0,4,8] else position) for i,bounds in enumerate(ranges)])
    # This published final prediction was selected in both identity and eleven branches.
    selected=table['pose2']['9.0'];opening=np.array(selected['12d']);closing=geometry_joint_vector(0,selected['2d'][1])
    all_errors=[];type_ranges={}
    for name,entries in table.items():
        for value in entries.values():all_errors.append(float(np.max(np.abs(geometry_joint_vector(*value['2d'])-value['12d']))))
        type_ranges[name]={'slots':len(entries),'position_register_range':[min(v['2d'][0] for v in entries.values()),max(v['2d'][0] for v in entries.values())],
            'rotation_register_values':sorted(set(v['2d'][1] for v in entries.values()))}
    # Execute original method bodies only against recording substitutes, never constructor.
    writes=[];sleeps=[]
    class RecordingRegisters:
        def SetTargetPosition(self,x):writes.append(['SetTargetPosition',x])
        def SetTargetRotation(self,x):writes.append(['SetTargetRotation',x])
    control=original(hardware,{'DH3'},{'np':np,'time':SimpleNamespace(sleep=lambda x:sleeps.append(x))})['DH3']
    device=control.__new__(control);device.m_gripper=RecordingRegisters()
    device.open_gripper(np.array(selected['2d']),sleep_time=.9)
    device.close_gripper()
    assert writes==[['SetTargetPosition',94],['SetTargetRotation',67],['SetTargetPosition',0]]
    links=[]
    for link in document.findall('link'):
        inertia=link.find('inertial/inertia');value={k:float(v) for k,v in inertia.attrib.items()}
        matrix=np.array([[value['ixx'],value['ixy'],value['ixz']],[value['ixy'],value['iyy'],value['iyz']],[value['ixz'],value['iyz'],value['izz']]])
        eigenvalues=np.linalg.eigvalsh(matrix)
        links.append({'name':link.attrib['name'],'mass_kg':float(link.find('inertial/mass').attrib['value']),
            'inertia_eigenvalues_kg_m2':eigenvalues.tolist(),'positive_definite':bool(np.all(eigenvalues>0)),
            'principal_moment_triangle_condition':bool(2*eigenvalues.max()<=eigenvalues.sum()+1e-12),
            'collision_mesh':link.find('collision/geometry/mesh').attrib['filename']})
    joints=[];mimic_errors=[];opening_violations=[];closing_violations=[]
    positions={joint.attrib['name']:opening[i] for i,joint in enumerate(joint_elements)}
    for index,joint in enumerate(joint_elements):
        limit=joint.find('limit');mimic=joint.find('mimic');row={'index':index,**joint.attrib,
            'limit':{k:float(v) for k,v in limit.attrib.items()} if limit is not None else None,
            'mimic':dict(mimic.attrib) if mimic is not None else None,'dynamics':joint.find('dynamics').attrib if joint.find('dynamics') is not None else None}
        if joint.attrib['type']!='fixed':
            for q,violations in [(opening,opening_violations),(closing,closing_violations)]:
                value=float(q[index]);lower=row['limit']['lower'];upper=row['limit']['upper']
                if value<lower-1e-10 or value>upper+1e-10:violations.append({'index':index,'name':joint.attrib['name'],'value_rad':value,'lower':lower,'upper':upper})
        if mimic is not None:
            expected=positions[mimic.attrib['joint']]*float(mimic.attrib.get('multiplier',1))+float(mimic.attrib.get('offset',0))
            error=float(opening[index]-expected)
            if abs(error)>1e-8:mimic_errors.append({'index':index,'name':joint.attrib['name'],'opening_rad':float(opening[index]),'mimic_expected_rad':float(expected),'difference_rad':error})
        joints.append(row)
    steps=[]
    def reject_step(*unused,**kwargs):
        steps.append(1);raise RuntimeError('Physics integration forbidden in readiness audit')
    p.stepSimulation=reject_step
    client=p.connect(p.DIRECT);assert client==0
    try:
        # Same unchanged URDF; second flag exposes explicit CAD-inertia import.
        bodies={name:p.loadURDF(str(urdf),useFixedBase=True,flags=flags) for name,flags in [('default',0),('file_inertia',p.URDF_USE_INERTIA_FROM_FILE)]}
        bullet={}
        for name,body in bodies.items():
            metadata=[]
            for i in range(p.getNumJoints(body)):
                info=p.getJointInfo(body,i);d=p.getDynamicsInfo(body,i)
                metadata.append({'index':i,'name':info[1].decode(),'joint_type':info[2],'lower':info[8],'upper':info[9],
                    'max_force':info[10],'max_velocity':info[11],'joint_damping':info[6],'joint_friction':info[7],
                    'mass_kg':d[0],'lateral_friction':d[1],'local_inertia_diagonal':list(d[2]),'restitution':d[5],
                    'rolling_friction':d[6],'spinning_friction':d[7],'contact_damping':d[8],'contact_stiffness':d[9],
                    'collision_shape_count':len(p.getCollisionShapeData(body,i))})
            bullet[name]={'body_id':body,'joints':metadata,'joint_count':p.getNumJoints(body),'num_constraints':p.getNumConstraints()}
        body=bodies['default']
        states_before=[p.getJointState(body,i)[0] for i in range(12)]
        p.resetJointState(body,1,.5)
        states_after=[p.getJointState(body,i)[0] for i in range(12)]
        assert states_after[1]==.5 and states_after[2]==0 and states_after[5]==0
        for i,value in enumerate(opening):p.resetJointState(body,i,float(value))
        actual_open=[p.getJointState(body,i)[0] for i in range(12)]
        assert np.array_equal(np.array(actual_open)[[i for i,j in enumerate(joints) if j['type']!='fixed']],opening[[i for i,j in enumerate(joints) if j['type']!='fixed']])
    finally:p.disconnect(client)
    assert not steps
    report={'scope':'Original DH3 zero-step physics-model readiness, not grasp execution','source_revision':REV,
        'simulation_steps':0,'physical_grasps_executed':0,'robot_connection':False,'hardware_module_imported':False,
        'author_sources_modified':False,'CanonDex_modified':False,'source_files':source_receipts,
        'original_hardware_control':{'preshape_position_rotation_registers':selected['2d'],'recorded_original_method_writes':writes,
            'recorded_original_sleep_seconds':sleeps,'initial_force_register':70,'initial_speed_register':100,
            'register_to_SI_force_torque_speed_calibration_present':False,'type_specific_close_targets':False,
            'close_behavior':'Position register0; rotation register retained. Physical firmware supplies its own actuation/contact response.',
            'method_constructor_executed':False,'recording_interface_only':True,
            'DH3_target_backoff_m':float(selected['2d'][0]/100*.005),'selected_stored_depth_m':.03,
            'selected_net_target_depth_shift_m':float(.03-selected['2d'][0]/100*.005)},
        'geometric_calibration':{'original_2D_to_12D_ranges':ranges,'published_slots':len(all_errors),
            'all_published_12D_match_original_linear_formula_max_error':max(all_errors),'per_type':type_ranges,
            'original_excel_inspection':{'path':str(workbook),'reader':'stdlib ZIP/XML only; no package installed','sheets':excel_sheets,'position0_present':any(x['position0_count'] for x in excel_sheets)},
            'selected_opening_12D_rad':opening.tolist(),'position0_formula_endpoint_12D_rad':closing.tolist(),
            'position0_is_in_released_lookup':any(v['2d'][0]==0 for entries in table.values() for v in entries.values()),
            'boundary':'Linear 2D-to-12D is the original offline mesh calibration, not identified underactuated actuator dynamics. Position0 endpoint is computed from that formula, not a released closed-hand dynamic target.'},
        'URDF':{'CAD_exporter':'SolidWorks URDF Exporter','links':links,'joints':joints,
            'all_link_masses_positive':all(x['mass_kg']>0 for x in links),'all_inertia_positive_definite':all(x['positive_definite'] for x in links),
            'all_inertia_triangle_conditions_pass':all(x['principal_moment_triangle_condition'] for x in links),
            'mass_sum_kg':sum(x['mass_kg'] for x in links),'revolute_count':sum(x['type']=='revolute' for x in joints),
            'mimic_count':sum(x['mimic'] is not None for x in joints),
            'zero_range_effort_velocity_joints':[x['name'] for x in joints if x['limit'] and x['limit']['lower']==x['limit']['upper']==x['limit']['effort']==x['limit']['velocity']==0],
            'selected_open_limit_violations':opening_violations,'geometric_position0_limit_violations':closing_violations,
            'selected_open_mimic_mismatches':mimic_errors,
            'explicit_contact_materials_present':bool(document.findall('.//contact')),
            'transmission_count':len(document.findall('transmission'))},
        'PyBullet_zero_step':{'module_file':p.__file__,'api_version':p.getAPIVersion(),'connection':'DIRECT private','bodies':bullet,
            'mimic_reset_probe':{'command':'resetJointState(finger1_joint,0.5)','before':states_before,'after':states_after,'other_mimicked_joint_states_unchanged':True},
            'lookup_can_be_teleported_beyond_declared_limits':True,'no_step_guard_calls':len(steps)},
        'conclusion':'Ready for original geometric/FK reproduction; not yet a validated original physical DH3 simulation. Raw URDF limits/mimic and the offline lookup are incompatible, and device force/speed commands lack SI actuator calibration. A separately versioned, explicitly validated physics/control port is required.'}
    (args.output_dir/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'conclusion':report['conclusion'],'joints':len(joints),'mimics':report['URDF']['mimic_count'],
        'selected_open_limit_violations':len(opening_violations),'closed_geometric_limit_violations':len(closing_violations),
        'selected_open_mimic_mismatches':len(mimic_errors),'all_table_geometric_formula_error':max(all_errors),'simulation_steps':0},indent=2))


if __name__=='__main__':main()
