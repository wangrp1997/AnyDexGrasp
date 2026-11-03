"""Continue a genuine single-frame CGR run through original DH3 hand scoring.

Original camera/width gates, >.85 learned-score gate, feature canonicalization
and calibrated hand mapping are retained. Hardware and unprovided collision
assets are never replaced by empty checks. One identity branch, not 11 TTA.
"""
from __future__ import annotations
import argparse
import ast
from collections import OrderedDict
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from validate_official_decision import ROOT,sha,load_original_classes,safe_state_dict


def ast_items(path,names,scope):
    tree=ast.parse(path.read_text())
    selected=[]
    for node in tree.body:
        if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in names for target in node.targets):
            selected.append(node)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),scope)
    return scope


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--frame-dir',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=Path('/mnt/public/datasets/AnyDexGrasp/metadata/model_obj140_manifest_20261103.json'))
    p.add_argument('--table-dir',type=Path,default=Path('/mnt/public/datasets/AnyDexGrasp/source_tables/generate_mesh_and_pointcloud/dh3_urdf'))
    p.add_argument('--grasp-api-source',type=Path,default=ROOT/'outputs/native_deps/graspnetAPI_source/graspnetAPI/grasp.py')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/official_dh3_hand_frame')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();torch.set_num_threads(4)
    report={'scope':'original DH3 single-frame learned hand pose prediction only','source_revision':'c9c4a43df33e40860417c7e2dd02f5122d3b2da2',
        'physical_grasps_executed':0,'simulation_run':False,'robot_connection':False,
        'new_training_steps':0,'test_time_augmentation_branches':1,'not_author_eleven_branch_TTA':True,
        'grasp_success_claim':False,'collision_check_executed':False,'status':'starting',
        'RANDOM':False,'source_frame':str(args.frame_dir.resolve()),'torch':torch.__version__,
        'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'stages':{},'timing_seconds':{}}
    try:
        frame=json.loads((args.frame_dir/'RESULTS.json').read_text())
        assert frame['full_representation_forward'] and frame['args']['parser_hand']=='dh3'
        assert frame['calibration']['depth_bounds_m']==[.15,.72]
        report['source_frame_RESULTS_sha256']=sha(args.frame_dir/'RESULTS.json')
        report['source_frame_checkpoint']=frame['checkpoint']
        report['source_frame_calibration']=frame['calibration']
        report['source_frame_timing_seconds']=frame['timing_seconds']
        source=ROOT/'robot_dh3.py'
        scope={'np':np,'torch':torch,'copy':copy,'OrderedDict':OrderedDict,
            'RANDOM':False,'NUM_OF_DH3_DEPTH':4,'NUM_OF_DH3_TYPE':4,'MAX_GRASP_WIDTH':.099,'MIN_GRASP_WIDTH':.045}
        ast_items(source,{'flip_ggarray','get_graspgroup_features','get_DH3_depth_type'},scope)
        report['original_function_overlay']={'source':str(source),'sha256':sha(source),
            'functions':['flip_ggarray','get_graspgroup_features','get_DH3_depth_type'],
            'function_bodies_unchanged':True,'hardware_module_imported':False}
        with np.load(args.frame_dir/'raw_camera_candidates.npz',allow_pickle=False) as saved:
            preds=torch.from_numpy(saved['candidates'].copy())
            features=saved['candidate_features'].copy()
        report['stages']['raw_parser_candidates']=len(preds)
        function=next(node for node in ast.parse(source.read_text()).body if isinstance(node,ast.FunctionDef) and node.name=='get_grasp')
        gates=[]
        for node in function.body:
            if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):
                continue
            name=node.targets[0].id
            if name in ['mask','workspace_mask'] and 'preds[' in ast.unparse(node.value):
                gates.append(node)
        assert len(gates)==2
        gate_scope=dict(scope,preds=preds)
        exec(compile(ast.Module(body=gates,type_ignores=[]),str(source),'exec'),gate_scope)
        geom_mask=gate_scope['mask'];workspace=gate_scope['workspace_mask'];keep=geom_mask&workspace
        report['camera_gates']={'original_AST_assignments':[ast.unparse(node) for node in gates],
            'orientation_only_pass':int((preds[:,9]>.9).sum()),
            'width_only_pass':int(((preds[:,1]<.099)&(preds[:,1]>.045)).sum()),
            'orientation_and_width_pass':int(geom_mask.sum()),'workspace_only_pass':int(workspace.sum())}
        preds=preds[keep];features=features[keep.numpy()]
        report['stages']['after_original_camera_and_width_gates']=len(preds)
        if len(preds)==0:
            report.update(status='no_candidates_after_original_camera_gates',stop_reason='Original DH3 camera orientation/width/workspace conjunction rejected this identity-frame branch')
        else:
            heights=.03*torch.ones((len(preds),1));object_ids=-torch.ones((len(preds),1))
            ggarray=torch.cat([preds[:,0:2],heights,preds[:,2:15],preds[:,15:16],object_ids],dim=-1).numpy()
            ggarray,flips=scope['flip_ggarray'](ggarray)
            features=np.c_[features,flips]
            index=ggarray[:,0].argsort()[::-1][:1000]
            ggarray=ggarray[index];features=features[index]
            report['stages']['after_original_score_top1000']=len(ggarray)
            feature_dict=scope['get_graspgroup_features'](features,None)
            np.savez_compressed(args.output_dir/'predecision_inputs.npz',ggarray=ggarray,
                grasp_features=features,canonical_CGR=feature_dict['grasp_preds_features'])
            train_cls,inference_cls,_=load_original_classes()
            model_items=[item for item in json.loads(args.manifest.read_text())['artifacts'] if item['hand']=='dh3_model']
            assert len(model_items)==4 and sorted(item['grasp_type'] for item in model_items)==[1,2,3,4]
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            models={'480':{}};receipts=[]
            for item in sorted(model_items,key=lambda item:item['grasp_type']):
                path=Path(item['destination']);state,metadata=safe_state_dict(path,train_cls,inference_cls)
                model=inference_cls(input_num=480);model.load_state_dict(state,strict=True);model.to(device);model.eval()
                models['480'][str(item['grasp_type'])]=model
                receipts.append({'grasp_type':item['grasp_type'],'path':str(path),'sha256':sha(path),
                    'strict_state_dict_load':True,'output_channels':int(state['conv7.weight'].shape[0]),'weights_only':metadata['weights_only']})
            report['decision_checkpoints']=receipts
            report['model_loader_boundary']='All four published single-file DH3 heads loaded deterministically. Released get_DH3_model resets its dict inside the class loop; that I/O loader is not invoked. Original scoring function and all heads are preserved.'
            if device.type=='cuda':torch.cuda.synchronize(device)
            begin=time.monotonic()
            depth,types,scores,selected_gg,selected_features=scope['get_DH3_depth_type'](models,feature_dict,ggarray,features)
            if device.type=='cuda':torch.cuda.synchronize(device)
            report['timing_seconds']['original_DH3_learned_scoring']=time.monotonic()-begin
            report['decision_device']=str(device)
            report['stages']['all_original_type_depth_combinations']=int(len(ggarray)*16)
            report['stages']['original_top3500_returned']=len(scores)
            report['learned_scores']={'minimum':float(scores.min()),'maximum':float(scores.max()),
                'threshold':.85,'comparison':'>','sigmoid_probability':False}
            # Original robot_grasp uses this exact threshold before hand mapping.
            mask=scores>.85
            report['stages']['after_original_learned_score_gate']=int(mask.sum())
            np.savez_compressed(args.output_dir/'learned_ranked_proposals.npz',depths=depth,types=types,scores=scores,
                ggarray=selected_gg,grasp_features=selected_features,score_gate_mask=mask)
            if not mask.any():
                report.update(status='no_candidates_after_original_learned_score_gate',stop_reason='No original learned DH3 score exceeds .85; no gate relaxed and no hand pose falsely selected')
            else:
                depth=depth[mask];types=types[mask];scores=scores[mask]
                selected_gg=selected_gg[mask];selected_features=selected_features[mask]
                api_scope={'np':np,'copy':copy}
                ast_items(args.grasp_api_source,{'GRASP_ARRAY_LEN','RECT_GRASP_ARRAY_LEN','EPS','Grasp','GraspGroup'},api_scope)
                hand_source=ROOT/'ur_toolbox/ur_toolbox/robot/DH3/DH3_grasp.py'
                hand_scope={'np':np,'copy':copy,'os':os,'json':json,'math':math}
                ast_items(hand_source,{'grasp_types','MIN_GRASP_WIDTH','MAX_GRASP_WIDTH','DH3_DEFAULT_DEPTH','DH3_ARRAY_LEN','DH3Grasp','DH3GraspGroup'},hand_scope)
                group=api_scope['GraspGroup'](selected_gg)
                hand=hand_scope['DH3GraspGroup']()
                before_widths=group.widths.copy()
                hand.from_graspgroup(group,types,str(args.table_dir))
                hand.object_ids=group.object_ids;hand.scores=scores
                hand.depths=hand.depths+depth+0.0
                assert np.isfinite(hand.grasp_group_array).all()
                top=np.argsort(hand.scores)[::-1][:400]
                hand=hand[top];group=group[top];selected_features=selected_features[top]
                report['stages']['mapped_original_hand_poses_before_top400']=len(scores)
                report['stages']['original_top400_before_collision']=len(hand)
                report['hand_mapping']={'source':str(hand_source),'source_sha256':sha(hand_source),
                    'original_GraspGroup_source':str(args.grasp_api_source),'original_GraspGroup_source_sha256':sha(args.grasp_api_source),
                    'table':str(args.table_dir/'width_12D_angle_2D_angle.json'),
                    'table_sha256':sha(args.table_dir/'width_12D_angle_2D_angle.json'),
                    'API_functions_unchanged':True,'type_counts':{str(key):int((hand.grasp_types==key).sum()) for key in [1,2,3,4]},
                    'width_clamp_count_before_top400':int(np.count_nonzero(before_widths!=selected_gg[:,1])),
                    'pose_frame':'input camera optical; original hand datum with depth separately stored',
                    'array_columns':'score, depth, type, rotation9, translation3, actuator2, object_id, width'}
                np.savez_compressed(args.output_dir/'predicted_DH3_hand_poses_pre_collision.npz',
                    DH3_grasp_group_array=hand.grasp_group_array,two_finger_grasp_group_array=group.grasp_group_array,
                    grasp_features=selected_features)
                pcl=args.table_dir/'meshes/source_pointclouds/voxel_size_3'
                repo_pcl=ROOT/'generate_mesh_and_pointcloud/dh3_urdf/meshes/source_pointclouds/voxel_size_3'
                report['collision_assets']={'required_original_pcl_roots':[str(pcl),str(repo_pcl)],
                    'available_roots':[str(path) for path in [pcl,repo_pcl] if path.is_dir()],
                    'original_collision_voxel_size_m':.001,'original_hand_pointcloud_voxel_m':.003,
                    'original_approach_distance_m':.05,'original_collision_threshold':0,
                    'empty_cloud_or_all_true_mask_substituted':False}
                if not report['collision_assets']['available_roots']:
                    report.update(status='hand_pose_prediction_complete_collision_assets_missing',
                        stop_reason='Original precomputed per-type/per-width DH3 collision pointclouds are absent; stopped before original collision admission, no empty check substituted')
                else:
                    report.update(status='hand_pose_prediction_complete_collision_stage_not_executed',
                        stop_reason='Collision assets root found; full original collision stage has not been validated by this runner')
        report['timing_seconds']['total_continuation']=time.monotonic()-start
        (args.output_dir/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        report['timing_seconds']['total_continuation']=time.monotonic()-start
        (args.output_dir/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        raise


if __name__=='__main__':
    main()
