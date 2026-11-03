"""Pure original DH3 2-register -> 12-angle geometry interface.

Extracts the author's unchanged rate() function and pos_scope assignment.
No hardware, physics, URDF modification or actuator timing is involved.
The position0 endpoint is a formula value, not a measured closed-hand pose.
"""
from __future__ import annotations
import argparse
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'generate_mesh_and_pointcloud/recover_dh3_hand_to_stl.py'
SOURCE_SHA256='3a123241195638c6ca7a7b1cd438fe9c4504e4fd567ceea5acf0d8dc1ea62a81'
TABLE=ROOT/'generate_mesh_and_pointcloud/dh3_urdf/width_12D_angle_2D_angle.json'
TABLE_SHA256='097929420ba7bfcbf3614ab193a7d85dec4bd95c76aca6a96c9dca1ec7ed2a5a'
REV='c9c4a43df33e40860417c7e2dd02f5122d3b2da2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _author_formula():
    if sha(SOURCE)!=SOURCE_SHA256:
        raise ValueError('Original DH3 geometry source differs from pinned author SHA')
    tree=ast.parse(SOURCE.read_text())
    rate=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='rate')
    reader=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='read_excel_2D_angle_to_12D_angle')
    ranges=next(node for node in reader.body if isinstance(node,ast.Assign)
                and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='pos_scope')
    scope={}
    exec(compile(ast.Module(body=[rate,ranges],type_ignores=[]),str(SOURCE),'exec'),scope)
    if len(scope['pos_scope'])!=12:raise ValueError('Unexpected original joint range count')
    return scope['rate'],tuple(scope['pos_scope']),rate.lineno,ranges.lineno


def source_receipt():
    """Pinned source and explicit geometry-only interpretation."""
    _,ranges,rate_line,ranges_line=_author_formula()
    return {'source_revision':REV,'source':str(SOURCE),'source_sha256':SOURCE_SHA256,
        'original_rate_line':rate_line,'original_pos_scope_line':ranges_line,
        'rate_and_pos_scope_AST_unchanged':True,'pos_scope':ranges,
        'position_input_range':[0,95],'rotation_input_range':[0,100],
        'rotation_boundary':'The original denominator is0..95; released rotations also include100. This interface preserves the original affine evaluation for100.',
        'fixed_joint_slot':{'zero_based_index':4,'name':'finger2_base_joint','formula_value_rad':1.0,
            'URDF_type':'fixed','physically_actuated':False,
            'boundary':'The original twelve-value array retains1 here. The fixed joint ignores this value for URDF FK; it is not made movable.'},
        'physics_steps':0,'real_hardware_connected':False,
        'closed0_boundary':'Position0 is the original hardware close command and is evaluable by the author formula. No measured position0 pose exists in released lookup/XLSX; this is a geometric formula endpoint, not validated actuation/contact dynamics.',
        'mesh_frame_boundary':'Original get_meshes swaps position xyz only for links2/3,6/7,10/11 while preserving each rotation. Raw URDF FK is not automatically equivalent to the original assembled collision mesh; that geometry consistency must be checked before a physics port.'}


def joint_angles(position: float, rotation: float) -> np.ndarray:
    """Return the original twelve geometry values in radians, preserving slot4.

    Position must be finite in[0,95], rotation finite in[0,100]. These device
    register values are not SI joint positions, velocities, forces or torques.
    """
    position=float(position);rotation=float(rotation)
    if not np.isfinite(position) or not 0<=position<=95:
        raise ValueError('DH3 position must be finite and in[0,95]')
    if not np.isfinite(rotation) or not 0<=rotation<=100:
        raise ValueError('DH3 rotation must be finite and in[0,100]')
    rate,ranges,_,_=_author_formula()
    return np.array([rate((0,95),bounds,rotation if index in [0,4,8] else position)
                     for index,bounds in enumerate(ranges)],dtype=np.float64)


def closure_path(start_position: float, rotation: float, samples: int=96):
    """Geometric interpolation to original close register0, with no timestamps.

    The normalized path coordinate merely samples the formula; it does not
    specify real speed, duration, torque, contact compliance or a controller.
    """
    if isinstance(samples,bool) or int(samples)!=samples or samples<2:
        raise ValueError('At least two integer samples are required')
    joint_angles(start_position,rotation)
    progress=np.linspace(0,1,int(samples))
    positions=np.linspace(float(start_position),0.,int(samples))
    rotations=np.full(int(samples),float(rotation))
    angles=np.stack([joint_angles(p,r) for p,r in zip(positions,rotations)])
    return {'normalized_path_progress':progress,'position_register':positions,
            'rotation_register':rotations,'joint_geometry_rad':angles}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-dir',type=Path,default=ROOT/'outputs/official_dh3_eleven_branch_20261103_v1/selection')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/dh3_author_kinematics_20261103_v1')
    parser.add_argument('--samples',type=int,default=96)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    if sha(TABLE)!=TABLE_SHA256:raise ValueError('Original published lookup SHA changed')
    table=json.loads(TABLE.read_text());errors=[];per_type={}
    for pose,entries in table.items():
        for entry in entries.values():
            actual=joint_angles(*entry['2d']);expected=np.array(entry['12d'],dtype=np.float64)
            if not np.array_equal(actual,expected):raise AssertionError('Original lookup mapping is not exactly equal: '+pose)
            errors.append(float(np.max(np.abs(actual-expected))))
        per_type[pose]=len(entries)
    selection_path=args.selection_dir/'RESULTS.json';selection=json.loads(selection_path.read_text())
    assert selection['selected'] and selection['physical_grasps_executed']==0
    position,rotation=selection['selected_hand']['actuator2']
    path=closure_path(position,rotation,args.samples)
    assert path['position_register'][0]==position and path['position_register'][-1]==0
    assert np.array_equal(path['joint_geometry_rad'][0],joint_angles(position,rotation))
    assert np.array_equal(path['joint_geometry_rad'][-1],joint_angles(0,rotation))
    assert np.all(path['joint_geometry_rad'][:,4]==1)
    target=args.output_dir/'selected_top1_to_position0_geometric_path.npz'
    np.savez_compressed(target,**path)
    report={'status':'author_kinematics_validated','scope':'Pure original geometry mapping and a parameterized geometric closure path only',
        'source':source_receipt(),'published_lookup':{'path':str(TABLE),'sha256':TABLE_SHA256,
            'slots_compared':len(errors),'per_type_slots':per_type,'all_twelve_values_exactly_equal':True,'maximum_absolute_error':max(errors)},
        'selected_top1':{'selection_RESULTS':str(selection_path),'sha256':sha(selection_path),'position':position,'rotation':rotation,
            'hand_type':selection['selected_hand']['type'],'width_m':selection['selected_hand']['width_m']},
        'path':{'path':str(target.resolve()),'sha256':sha(target),'bytes':target.stat().st_size,'samples':args.samples,
            'start_registers':[position,rotation],'end_registers':[0,rotation],
            'first_geometry_rad':path['joint_geometry_rad'][0].tolist(),'last_geometry_rad':path['joint_geometry_rad'][-1].tolist(),
            'time_seconds_specified':False,'forces_or_torques_specified':False,'real_closed_pose_measurement':False,
            'URDF_limits_or_mimic_modified':False},
        'physics_steps':0,'physical_grasps_executed':0,'hardware_module_imported':False,'CanonDex_modified':False}
    (args.output_dir/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
