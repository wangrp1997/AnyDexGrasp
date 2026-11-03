"""Run the author's identity + ten-augmentation DH3 frame pipeline offline.

Original get_grasp recovery/filters and get_ggarray_features aggregation execute
directly from the pinned source. Only five camera constants are replaced with
the input calibration. No hardware module is imported or physical grasp executed.
"""
from __future__ import annotations

import argparse
import ast
import collections.abc
import copy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
from PIL import Image
import scipy.io
import torch

from run_official_rgbd_frame import ROOT, sha, source_overlay


ORIGINAL_FUNCTIONS = {
    'augment_data', 'get_depth', 'get_grasp', 'get_ggarray_features', 'parse_preds'
}


def original_overlay(calibration, scope):
    source = ROOT / 'robot_dh3.py'
    tree = ast.parse(source.read_text())
    nodes = [copy.deepcopy(node) for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in ORIGINAL_FUNCTIONS]
    assert {node.name for node in nodes} == ORIGINAL_FUNCTIONS
    replacement_names = {'fx', 'fy', 'cx', 'cy', 's'}
    replacements = []
    for node in nodes:
        if node.name != 'get_grasp':
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            targets = statement.targets
            if len(targets) != 1:
                continue
            target = targets[0]
            names = ([x.id for x in target.elts] if isinstance(target, ast.Tuple)
                     else [target.id] if isinstance(target, ast.Name) else [])
            if names and set(names) <= replacement_names:
                old = ast.unparse(statement)
                value = ([ast.Constant(float(calibration[name])) for name in names])
                statement.value = ast.Tuple(elts=value, ctx=ast.Load()) if len(names)>1 else value[0]
                replacements.append(dict(names=names, original=old,
                                         calibrated=ast.unparse(statement)))
    assert {name for item in replacements for name in item['names']} == replacement_names
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source), 'exec', dont_inherit=True), scope)
    return dict(source=str(source), sha256=sha(source),
                functions=sorted(ORIGINAL_FUNCTIONS), camera_constant_changes=replacements,
                original_recovery_flip_filters_and_aggregation_unchanged=True,
                hardware_module_imported=False)


def call_cli(module, argv):
    previous = sys.argv
    try:
        sys.argv = [str(Path(module.__file__).resolve()), *map(str, argv)]
        module.main()
    finally:
        sys.argv = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('/mnt/public/datasets/AnyDexGrasp/logs/model/checkpoint.tar.18'))
    parser.add_argument('--manifest', type=Path,
                        default=Path('/mnt/public/datasets/AnyDexGrasp/metadata/model_obj140_manifest_20261103.json'))
    parser.add_argument('--table-dir', type=Path,
                        default=Path('/mnt/public/datasets/AnyDexGrasp/source_tables/generate_mesh_and_pointcloud/dh3_urdf'))
    parser.add_argument('--native-deps', type=Path, default=ROOT/'outputs/native_deps')
    parser.add_argument('--grasp-api-source', type=Path,
                        default=ROOT/'outputs/native_deps/graspnetAPI_source/graspnetAPI/grasp.py')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--scene-manifest', type=Path,
                        help='Bind RGB-D file hashes to an explicitly declared rendered scene; not a network input')
    parser.add_argument('--asset-root', type=Path,
                        help='After hand scoring, incrementally prepare original assets and run original collision')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new output directory; retain prior receipts')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report = dict(scope='Original DH3 eleven-branch offline calibrated frame inference',
                  source_revision='c9c4a43df33e40860417c7e2dd02f5122d3b2da2',
                  args={key:str(value) if isinstance(value, Path) else value
                        for key,value in vars(args).items()},
                  physical_grasps_executed=0, robot_connection=False, simulation_run=False,
                  grasp_success_claim=False, new_training_steps=0,
                  independent_test_generalization_claim=False,
                  test_time_augmentation_branches=11, original_random_augmentations=10,
                  seed=args.seed, collision_check_executed=False,
                  CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  timing_seconds={}, status='starting')
    try:
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if not hasattr(collections, 'Sequence'):
            collections.Sequence = collections.abc.Sequence
        depth_path, color_path, meta_path = [args.image_dir/name
                                            for name in ['depth.png','color.png','meta.mat']]
        depth = np.asarray(Image.open(depth_path))
        color = np.asarray(Image.open(color_path).convert('RGB'))
        meta = scipy.io.loadmat(meta_path)
        intrinsic = np.asarray(meta['intrinsic_matrix'], dtype=np.float64)
        factor = float(np.asarray(meta['factor_depth']).reshape(-1)[0])
        if depth.shape != (720,1280) or color.shape != (720,1280,3):
            raise ValueError('Original shared-frame read has fixed 720x1280 shape')
        if intrinsic.shape != (3,3) or factor<=0 or depth.dtype != np.uint16:
            raise ValueError('Invalid input calibration/depth format')
        calibration = dict(fx=intrinsic[0,0],fy=intrinsic[1,1],
                           cx=intrinsic[0,2],cy=intrinsic[1,2],s=factor)
        report['input_files'] = {str(path):dict(bytes=path.stat().st_size,sha256=sha(path))
                                 for path in [depth_path,color_path,meta_path]}
        if args.scene_manifest:
            scene = json.loads(args.scene_manifest.read_text())
            declared = {Path(item['path']).name:item for item in scene['files']}
            for path in [depth_path,color_path,meta_path]:
                if declared[path.name]['sha256'] != sha(path):
                    raise ValueError('Rendered scene binding differs for '+path.name)
            report['scene_binding'] = dict(manifest=str(args.scene_manifest.resolve()),
                                           manifest_sha256=sha(args.scene_manifest),
                                           scene_signature_sha256=scene['scene_signature_sha256'],
                                           file_hashes_verified=True,T_WC=scene['camera']['T_WC'],
                                           manifest_or_object_pose_fed_to_network=False,
                                           boundary='Scene provenance and later control binding only; original network consumes rendered RGB-D/calibration')
        report['calibration'] = dict(intrinsic_matrix=intrinsic.tolist(),factor_depth=factor,
                                    depth_bounds_m=[.15,.72],voxel_size_m=.005,
                                    workspace_image_mask_used=False,
                                    boundary='Only rig-specific intrinsics/depth factor replaced by meta.mat')
        scope = dict(np=np,torch=torch,copy=copy,time=time,
                     BATCH_SIZE=1,MAX_GRASP_WIDTH=.099,MIN_GRASP_WIDTH=.045,
                     DEBUG=True,POINTCLOUD_AUGMENT_NUM=10,
                     cfgs=SimpleNamespace(use_graspnet_v2=True,half_views=False))
        report['original_source_overlay'] = original_overlay(calibration,scope)
        if args.prepare_only:
            matrices = [np.eye(4)]
            matrices.extend(scope['augment_data'](flip=bool(index%2)) for index in range(10))
            report['augmentation_recipe'] = [dict(branch=index,flip=bool(index and (index-1)%2),
                                                   input_augmentation_matrix=matrix.tolist())
                                             for index,matrix in enumerate(matrices)]
            report.update(status='eleven_branch_recipe_prepared_only',full_representation_forward=False)
        else:
            # Original source explicitly allocates cuda:0. Select the physical GPU
            # with CUDA_VISIBLE_DEVICES before process launch, without rewriting it.
            if not torch.cuda.is_available():
                raise RuntimeError('Original sparse/FPS/cylinder forward requires CUDA')
            torch.cuda.set_device(0)
            device = torch.device('cuda:0')
            paths = [ROOT/'models',ROOT/'utils',ROOT/'pointnet2',
                     args.native_deps/'build_src/MinkowskiEngine',args.native_deps/'pointnet2/lib']
            paths += list((args.native_deps/'pointnet2').glob('lib.*'))
            for path in reversed(paths):
                if path.exists():sys.path.insert(0,str(path))
            import MinkowskiEngine as ME
            import pointnet2_utils
            import open3d as o3d
            from np_utils import transform_point_cloud
            from pt_utils import batch_viewpoint_params_to_matrix
            extension = importlib.import_module('pointnet2._ext')
            if not any(str(extension.__file__).endswith(x)
                       for x in importlib.machinery.EXTENSION_SUFFIXES):
                raise RuntimeError('Original native PointNet2 extension required')
            counters = {name:0 for name in ['furthest_point_sample','cylinder_query','grouping_operation']}
            for name in counters:
                original = getattr(pointnet2_utils,name)
                def counted(*values,_original=original,_name=name,**kwargs):
                    counters[_name] += 1
                    return _original(*values,**kwargs)
                setattr(pointnet2_utils,name,counted)
            begin = time.monotonic()
            cls, overlay = source_overlay()
            unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(args.checkpoint)
            if unsafe:raise RuntimeError('Representation checkpoint requires unexpected pickle globals')
            checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
            state = checkpoint['model_state_dict']
            num_depth = state['grasp_generator.conv3.weight'].shape[0]//(48*2)
            if num_depth != 5:raise ValueError('Original declared DH3 v2 recipe requires five depths')
            model = cls(in_channels=3,num_seed=2048,num_view=300,num_angle=48,
                        num_depth=5,is_training=False,half_views=False)
            model.load_state_dict(state,strict=True);model.to(device);model.eval()
            report['representation_overlay'] = overlay
            report['native_dependencies'] = dict(
                MinkowskiEngine_file=ME.__file__,
                MinkowskiEngine_version=getattr(ME,'__version__',None),
                pointnet2_extension=str(extension.__file__),
                pointnet2_extension_sha256=sha(Path(extension.__file__)),
                cuda_device_name=torch.cuda.get_device_name(device),
                cuda_compute_capability=list(torch.cuda.get_device_capability(device)))
            report['checkpoint'] = dict(path=str(args.checkpoint),sha256=sha(args.checkpoint),
                                        weights_only=True,strict_state_dict_load=True,
                                        num_depth=5,num_seed=2048,model_load_count=1)
            report['timing_seconds']['representation_model_load'] = time.monotonic()-begin
            scope.update(ME=ME,o3d=o3d,transform_point_cloud=transform_point_cloud,
                         batch_viewpoint_params_to_matrix=batch_viewpoint_params_to_matrix)
            original_parse = scope['parse_preds']
            original_grasp = scope['get_grasp']
            branches = []
            frame_dir = args.output_dir/'frame';frame_dir.mkdir()
            branch_dir = frame_dir/'branches';branch_dir.mkdir()

            def observed_parse(*values,**kwargs):
                predictions,features = original_parse(*values,**kwargs)
                branches[-1]['original_parser_candidates_before_recovery_and_gates'] = (
                    len(predictions[0]) if predictions else 0)
                return predictions,features

            def observed_grasp(*values,**kwargs):
                index = len(branches)
                before = np.array(kwargs.get('augment_mat',np.eye(4)),copy=True)
                item = dict(branch=index,flip=bool(kwargs.get('flip',False)),
                            input_augmentation_matrix=before.tolist())
                branches.append(item)
                t = time.monotonic()
                result = original_grasp(*values,**kwargs)
                torch.cuda.synchronize(device)
                gg,_,points,features,sparse = result
                item.update(wall_seconds=time.monotonic()-t,
                            candidates_after_original_recovery_and_gates=0 if gg is None else len(gg),
                            unique_voxels=len(points),
                            post_original_flip_mutation_matrix=np.asarray(kwargs.get('augment_mat',np.eye(4))).tolist())
                arrays = dict(input_augmentation_matrix=before)
                if gg is not None:
                    arrays.update(ggarray=gg.detach().cpu().numpy(),
                                  candidate_features=features.detach().cpu().numpy())
                np.savez_compressed(branch_dir/('branch_%02d.npz'%index),**arrays)
                return result

            scope.update(parse_preds=observed_parse,get_grasp=observed_grasp)
            depth_buffer = SimpleNamespace(buf=np.ascontiguousarray(depth).tobytes())
            color_buffer = SimpleNamespace(buf=np.ascontiguousarray(color.astype(np.float32)/255).tobytes())
            # The author generates all ten transforms before the first forward.
            # Seed here makes that exact source RNG recipe reproducible.
            np.random.seed(args.seed)
            begin = time.monotonic()
            merged,_,identity_points,features,sparse = scope['get_ggarray_features'](
                depth_buffer,color_buffer,model)
            torch.cuda.synchronize(device)
            report['timing_seconds']['eleven_original_get_grasp_branches'] = time.monotonic()-begin
            assert len(branches)==11 and [x['flip'] for x in branches]==[False]+[bool(i%2) for i in range(10)]
            report['branches'] = branches
            report['native_operator_call_counts'] = counters
            report['full_representation_forward'] = True
            report['identity_scene_points'] = 'frame/scene_points_down.npy; original identity points_down only'
            np.save(frame_dir/'scene_points_down.npy',identity_points.detach().cpu().numpy())
            report['merged_candidates_after_original_get_grasp_gates'] = 0 if merged is None else len(merged)
            frame_report = dict(full_representation_forward=True,args=dict(parser_hand='dh3'),
                                calibration=report['calibration'],checkpoint=report['checkpoint'],
                                timing_seconds=report['timing_seconds'],
                                test_time_augmentation_branches=11,branches=branches,
                                raw_candidate_boundary='Original per-branch recovery/gates already applied; then original merge')
            if 'scene_binding' in report:
                frame_report['scene_binding'] = report['scene_binding']
            if merged is None:
                report['status'] = 'no_candidates_after_eleven_original_branches'
            else:
                combined = merged.detach().cpu().numpy()
                features = features.detach().cpu().numpy()
                if not np.isfinite(combined).all() or not np.isfinite(features).all():
                    raise ValueError('Non-finite merged original candidates')
                parsed_columns = np.concatenate([combined[:,:2],combined[:,3:16]],axis=1)
                assert parsed_columns.shape[1]==15
                np.savez_compressed(frame_dir/'raw_camera_candidates.npz',
                                    candidates=parsed_columns,candidate_features=features)
                np.savez_compressed(frame_dir/'original_merged_get_grasp_outputs.npz',
                                    ggarray=combined,candidate_features=features)
            (frame_dir/'RESULTS.json').write_text(json.dumps(frame_report,indent=2)+'\n')
            report['frame_stage'] = dict(directory=str(frame_dir),RESULTS_sha256=sha(frame_dir/'RESULTS.json'))
            if merged is not None:
                import run_official_dh3_frame as hand_module
                begin = time.monotonic()
                hand_dir = args.output_dir/'hand'
                call_cli(hand_module,['--frame-dir',frame_dir,'--manifest',args.manifest,
                                      '--table-dir',args.table_dir,'--grasp-api-source',args.grasp_api_source,
                                      '--output-dir',hand_dir])
                hand_report_path = hand_dir/'RESULTS.json'
                hand_report = json.loads(hand_report_path.read_text())
                hand_report.update(scope='Original DH3 hand scoring/mapping on merged eleven-branch proposals',
                                   test_time_augmentation_branches=11,not_author_eleven_branch_TTA=False,
                                   repeated_camera_gate_boundary='All candidates already passed original get_grasp gates; repeated unchanged hand-runner gates are asserted idempotent')
                if hand_report['stages']['raw_parser_candidates'] != hand_report['stages']['after_original_camera_and_width_gates']:
                    raise ValueError('Repeated original camera gate changed already-admitted merged candidates')
                hand_report_path.write_text(json.dumps(hand_report,indent=2)+'\n')
                report['hand_stage'] = dict(directory=str(hand_dir),RESULTS_sha256=sha(hand_report_path),
                                            status=hand_report['status'],stages=hand_report['stages'],
                                            original_global_top1000_then_top3500_score_gate_and_top400_preserved=True)
                report['timing_seconds']['original_hand_stage'] = time.monotonic()-begin
                report['status'] = 'eleven_branch_representation_and_hand_stage_complete_before_collision'
            report['collision_handoff'] = dict(
                hand_run_dir=str(args.output_dir/'hand'),
                scene_points=str(frame_dir/'scene_points_down.npy'),
                expected_hand_npz='predicted_DH3_hand_poses_pre_collision.npz',
                original_identity_scene_only=True,collision_not_replaced=True)
            if args.asset_root and merged is not None:
                pose_file = args.output_dir/'hand/predicted_DH3_hand_poses_pre_collision.npz'
                if not pose_file.exists():
                    report['status'] = 'eleven_branch_hand_scoring_complete_no_proposal_for_collision'
                else:
                    for module_name, arguments, key in [
                        ('generate_dh3_collision_assets',
                         ['--hand-run-dir',args.output_dir/'hand','--asset-root',args.asset_root,
                          '--output-dir',args.output_dir/'collision_asset_preparation'],
                         'collision_asset_preparation'),
                        ('run_official_dh3_collision',
                         ['--hand-run-dir',args.output_dir/'hand','--asset-root',args.asset_root,
                          '--scene-points',frame_dir/'scene_points_down.npy',
                          '--output-dir',args.output_dir/'collision'],
                         'original_collision'),
                    ]:
                        path = ROOT/'reproduction/runtime'/(module_name+'.py')
                        if not path.exists():
                            raise RuntimeError('Required original-method stage is not ready: '+str(path))
                        spec = importlib.util.spec_from_file_location(module_name,path)
                        stage = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(stage)
                        begin = time.monotonic()
                        call_cli(stage,arguments)
                        report['timing_seconds'][key] = time.monotonic()-begin
                    collision_result = args.output_dir/'collision/RESULTS.json'
                    if not collision_result.exists():
                        raise RuntimeError('Collision stage did not save its own RESULTS.json')
                    collision_report = json.loads(collision_result.read_text())
                    report['collision_stage'] = dict(directory=str(args.output_dir/'collision'),
                                                      RESULTS_sha256=sha(collision_result),
                                                      result=collision_report)
                    report['collision_check_executed'] = collision_report.get('collision_check_executed') is True
                    if not report['collision_check_executed']:
                        raise RuntimeError('Collision receipt does not confirm actual original collision execution')
                    report['status'] = 'eleven_branch_original_offline_pipeline_complete'
        report['timing_seconds']['total'] = time.monotonic()-start
        (args.output_dir/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),
                      traceback=traceback.format_exc())
        report['timing_seconds']['total'] = time.monotonic()-start
        (args.output_dir/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        raise


if __name__=='__main__':
    main()
