"""Run the original DH3 scene-point/hand-voxel collision gate, without hardware.

Uses real locally generated original hand PCLs. The only instrumentation wraps
Open3D's actual inclusion query to retain its unchanged Boolean output/counts.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
REV='c9c4a43df33e40860417c7e2dd02f5122d3b2da2'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def original(path,names,scope):
    items=[]
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names:items.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets):items.append(node)
    exec(compile(ast.Module(body=items,type_ignores=[]),str(path),'exec'),scope)
    return scope


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--hand-run-dir',type=Path,required=True)
    parser.add_argument('--asset-root',type=Path,required=True)
    parser.add_argument('--scene-points',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--grasp-api-source',type=Path,default=ROOT/'outputs/native_deps/graspnetAPI_source/graspnetAPI/grasp.py')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic()
    report={'scope':'original DH3 offline single-frame hand-voxel collision admission',
            'source_revision':REV,'collision_check_executed':False,'physical_grasps_executed':0,
            'simulation_run':False,'robot_connection':False,'grasp_success_claim':False,
            'empty_pointcloud_substituted':False,'collision_gates_relaxed':False,'status':'starting'}
    try:
        import open3d as o3d
        source=ROOT/'utils/collision_detector.py'
        hand_source=ROOT/'ur_toolbox/ur_toolbox/robot/DH3/DH3_grasp.py'
        api=original(args.grasp_api_source,{'GRASP_ARRAY_LEN','RECT_GRASP_ARRAY_LEN','EPS','Grasp','GraspGroup'},{'np':np,'copy':copy})
        hand_scope=original(hand_source,{'grasp_types','MIN_GRASP_WIDTH','MAX_GRASP_WIDTH','DH3_DEFAULT_DEPTH','DH3_ARRAY_LEN','DH3Grasp','DH3GraspGroup'},
                            {'np':np,'copy':copy,'os':os,'json':json,'math':math})
        prediction=args.hand_run_dir/'predicted_DH3_hand_poses_pre_collision.npz'
        with np.load(prediction,allow_pickle=False) as saved:
            hand=hand_scope['DH3GraspGroup'](saved['DH3_grasp_group_array'].copy())
            two=api['GraspGroup'](saved['two_finger_grasp_group_array'].copy())
            features=saved['grasp_features'].copy()
        if args.scene_points:
            scene_path=args.scene_points
            scene=np.load(scene_path,allow_pickle=False)
            source_key=None
        else:
            hand_receipt=json.loads((args.hand_run_dir/'RESULTS.json').read_text())
            frame=Path(hand_receipt['source_frame'])
            scene_path=frame/'representation_outputs.npz'
            source_key='sparse_input_XYZ_features'
            with np.load(scene_path,allow_pickle=False) as saved:scene=saved[source_key].copy()
        assert scene.ndim==2 and scene.shape[1]==3 and len(scene)>0 and np.isfinite(scene).all()
        report['input']={'hand_prediction':str(prediction.resolve()),'hand_prediction_sha256':sha(prediction),
            'scene_points':str(scene_path.resolve()),'scene_file_sha256':sha(scene_path),'npz_key':source_key,
            'scene_count':len(scene),'candidates':len(hand),'scene_selection':'identity quantized original points_down, no merged augmentation scene'}
        # Determine actual original width/remapping slots on clones, never double-expand input.
        planned_hand=copy.deepcopy(hand);planned_two=copy.deepcopy(two)
        planned_two.widths=planned_two.widths*1.7
        planned_min=planned_two.widths>.045
        planned_hand=planned_hand[planned_min];planned_two=planned_two[planned_min]
        planned_hand.graspgroupTR_2_TR(planned_two,str(args.asset_root))
        keys=sorted({row.get_grasp_type_with_finger_name()+'_'+str(round(row.width*100,1)) for row in planned_hand})
        pcls={};receipts=[]
        for key in keys:
            pose,width=key.split('_')
            path=args.asset_root/'meshes/source_pointclouds/voxel_size_3'/pose/(width+'.ply')
            slot=args.asset_root/'slot_receipts'/(key+'.json')
            row=json.loads(slot.read_text())
            expected=next(item for item in row['pointclouds'] if item['voxel_mm']==3)
            assert sha(path)==expected['sha256']
            cloud=o3d.io.read_point_cloud(str(path));points=np.asarray(cloud.points)
            assert len(points)==expected['point_count'] and len(points)>0 and np.isfinite(points).all()
            assert row['physics_steps']==0 and row['original_surface_samples']==300000 and row['original_infill_layers']==15
            pcls[key]=cloud
            receipts.append({'key':key,'path':str(path.resolve()),'sha256':expected['sha256'],
                             'point_count':len(points),'slot_receipt_sha256':sha(slot),'sampling_seed':row['sampling_seed']})
        report['assets']={'root':str(args.asset_root.resolve()),'required_slots':len(keys),'slots':receipts,
            'table_sha256':sha(args.asset_root/'width_12D_angle_2D_angle.json'),
            'source':'pinned original 13 STLs/URDF and original FK/surface/infill/voxel functions; published lookup joints; zero-step PyBullet DIRECT'}
        hits=[];query_masks=[]
        class QueryRecording:
            def __init__(self,grid):self.grid=grid
            def check_if_included(self,points):
                output=self.grid.check_if_included(points)
                value=np.asarray(output,dtype=bool)
                hits.append(int(value.sum()));query_masks.append(value.copy())
                return output
        class VoxelRecording:
            @staticmethod
            def create_from_point_cloud(*values,**kwargs):
                return QueryRecording(o3d.geometry.VoxelGrid.create_from_point_cloud(*values,**kwargs))
        observed_o3d=SimpleNamespace(geometry=SimpleNamespace(PointCloud=o3d.geometry.PointCloud,VoxelGrid=VoxelRecording),utility=o3d.utility)
        detector_scope=original(source,{'ModelFreeCollisionDetectorMultifinger'},
            {'np':np,'copy':copy,'math':math,'time':time,'o3d':observed_o3d,'GraspGroup':api['GraspGroup']})
        detector=detector_scope['ModelFreeCollisionDetectorMultifinger'](scene,voxel_size=.001)
        start=time.monotonic()
        mapped_hand,mapped_two,mask,min_width=detector.detect(hand,two,str(args.asset_root),pcls,
            min_grasp_width=.045,VoxelGrid=.003,DEBUG=False,approach_dist=.05,collision_thresh=0,adjust_gripper_centers=False)
        report['original_detect_seconds']=time.monotonic()-start
        mask=np.asarray(mask,dtype=bool);min_width=np.asarray(min_width,dtype=bool)
        assert len(hits)==len(mask)==len(mapped_hand)
        assert np.array_equal(min_width,planned_min)
        assert np.array_equal(mapped_hand.grasp_group_array,planned_hand.grasp_group_array)
        assert np.array_equal(mapped_two.grasp_group_array,planned_two.grasp_group_array)
        assert np.array_equal(mask,np.array(hits)==0)
        source_indices=np.flatnonzero(min_width)
        all_records=[]
        for i in range(len(min_width)):
            item={'pre_collision_index':i,'min_width_pass':bool(min_width[i])}
            if min_width[i]:
                j=int(np.searchsorted(source_indices,i));item.update(voxel_overlap_points=hits[j],admitted=bool(mask[j]),
                    reason='no_observed_scene_point_in_original_hand_approach_voxels' if mask[j] else 'observed_scene_point_in_original_hand_approach_voxels',
                    hand_type=int(mapped_hand.grasp_types[j]),score=float(mapped_hand.scores[j]))
            else:item.update(admitted=False,reason='original_expanded_width_not_above_45mm')
            all_records.append(item)
        with (args.output_dir/'candidate_collision_records.jsonl').open('w') as stream:
            for item in all_records:stream.write(json.dumps(item)+'\n')
        np.savez_compressed(args.output_dir/'collision_outputs.npz',DH3_grasp_group_array=mapped_hand.grasp_group_array,
            two_finger_grasp_group_array=mapped_two.grasp_group_array,admission_mask=mask,min_width_index=min_width,
            source_indices=source_indices,overlap_point_counts=np.array(hits),scene_inclusion_masks_packed=np.packbits(np.array(query_masks),axis=1),
            admitted_grasp_features=features[min_width][mask],admitted_DH3_grasp_group_array=mapped_hand.grasp_group_array[mask],
            admitted_two_finger_grasp_group_array=mapped_two.grasp_group_array[mask])
        report.update(status='original_collision_admission_complete',collision_check_executed=True,
            stages={'pre_collision_candidates':len(min_width),'after_original_width_gate':len(mask),
                'collision_rejected':int((~mask).sum()),'collision_admitted':int(mask.sum())},
            original_parameters={'scene_voxel_argument_m':.001,'hand_and_voxel_grid_m':.003,'approach_distance_m':.05,
                'min_grasp_width_m':.045,'collision_threshold_points':0,'adjust_gripper_centers':False,'DEBUG':False,
                'width_multiplier':1.7,'original_accumulated_approach_offsets_m':[0,.02,.05,.07]},
            instrumentation={'original_source':str(source),'source_sha256':sha(source),'detect_and_load_function_bodies_unchanged':True,
                'time_import_provided':'Original collision module omits time import; actual time module provided in AST scope',
                'Open3D_query':'passthrough to real VoxelGrid/create_from_point_cloud/check_if_included; original output returned unchanged',
                'Open3D_version':o3d.__version__},
            validation={'original_width_prelude_and_hand_mapping_exact':True,'all_masks_equal_actual_zero_overlap_test':True,
                'all_required_original_pointclouds_nonempty_finite_SHA_verified':True},
            boundary='Visibility-only original hand/scene pointcloud gate. No arm, path, IK, physical lifting, independent-scene evaluation or closed-loop control was executed.')
        report['total_seconds']=time.monotonic()-begin
        report['output_files']=[{'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)} for path in [args.output_dir/'candidate_collision_records.jsonl',args.output_dir/'collision_outputs.npz']]
        (args.output_dir/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({key:report[key] for key in ['status','collision_check_executed','stages','original_detect_seconds','total_seconds']},indent=2),flush=True)
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),total_seconds=time.monotonic()-begin)
        (args.output_dir/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
        print(report['traceback'],flush=True)
        raise


if __name__=='__main__':main()
