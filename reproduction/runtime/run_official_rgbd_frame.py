"""Original AnyDex representation network on one calibrated RGB-D frame.

No robot imports, no simulated grasp, no learned weights changes. Only unused
training-label imports are isolated; original sparse/FPS/cylinder/group kernels
are required and retained. Camera calibration comes from the input metadata,
not from the author's particular hardware rig.
"""
from __future__ import annotations
import argparse
import ast
import collections.abc
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image
import scipy.io
import torch

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def source_overlay():
    """Execute original module after dropping only its training-only import."""
    path=ROOT/'models/minkowski_graspnet_single_point.py'
    tree=ast.parse(path.read_text())
    removed=[];nodes=[]
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module=='loss':
            assert {alias.name for alias in node.names}=={'process_grasp_labels','match_grasp_view_and_label'}
            removed.append(ast.unparse(node))
        else:
            nodes.append(node)
    assert len(removed)==1
    def unavailable_training_function(*args,**kwargs):
        raise RuntimeError('Training-only label code was called during declared inference')
    scope={'__name__':'official_representation_overlay','__file__':str(path),
           'process_grasp_labels':unavailable_training_function,
           'match_grasp_view_and_label':unavailable_training_function}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope['MinkowskiGraspNet'], {'source':str(path),'sha256':sha(path),
        'only_ast_change':'Removed the two unused training-label imports; any call raises',
        'removed_imports':removed,'network_layers_and_forward_unchanged':True}


def parser_overlay(hand):
    path=ROOT/('robot_'+hand+'.py')
    tree=ast.parse(path.read_text())
    node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='parse_preds')
    from pt_utils import batch_viewpoint_params_to_matrix
    scope={'torch':torch,'np':np,'copy':copy,'BATCH_SIZE':1,'MAX_GRASP_WIDTH':(0.099 if hand=='dh3' else 0.1),
           'batch_viewpoint_params_to_matrix':batch_viewpoint_params_to_matrix}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['parse_preds'],{'source':str(path),'sha256':sha(path),'function':'parse_preds',
        'hardware_module_imported':False,'function_body_unchanged':True}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image-dir',type=Path,required=True)
    p.add_argument('--parser-hand',choices=['allegro','dh3'],default='allegro')
    p.add_argument('--checkpoint',type=Path,default=Path('/mnt/public/datasets/AnyDexGrasp/logs/model/checkpoint.tar.18'))
    p.add_argument('--native-deps',type=Path,default=ROOT/'outputs/native_deps')
    p.add_argument('--workspace-mask',type=Path)
    p.add_argument('--voxel-size',type=float,default=.005)
    p.add_argument('--depth-min',type=float,default=.2)
    p.add_argument('--depth-max',type=float,default=.65)
    p.add_argument('--num-seed',type=int,default=2048)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/official_rgbd_frame')
    args=p.parse_args()
    if not (args.voxel_size>0 and 0<=args.depth_min<args.depth_max and args.num_seed>0):
        p.error('Invalid geometric preprocessing parameters')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report={'scope':'original-method calibrated single RGB-D software inference',
        'source_revision':'c9c4a43df33e40860417c7e2dd02f5122d3b2da2',
        'source_checkout':str(ROOT),'physical_grasps_executed':0,'no_robot_connection':True,
        'simulation_run':False,'grasp_success_claim':False,'new_training_steps':0,
        'author_independent_test_generalization_claim':False,
        'example_boundary':'Official GraspNet-team demo RGB-D input; not AnyDex physical test evidence',
        'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,
        'args':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        'timing_seconds':{},'status':'starting'}
    total=time.monotonic()
    try:
        torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
        begin=time.monotonic()
        depth_path=args.image_dir/'depth.png';color_path=args.image_dir/'color.png';meta_path=args.image_dir/'meta.mat'
        depth=np.asarray(Image.open(depth_path))
        color=np.asarray(Image.open(color_path).convert('RGB'))
        meta=scipy.io.loadmat(meta_path)
        intrinsics=np.asarray(meta['intrinsic_matrix'],dtype=np.float64)
        factor=float(np.asarray(meta['factor_depth']).reshape(-1)[0])
        if depth.ndim!=2 or color.shape[:2]!=depth.shape or intrinsics.shape!=(3,3) or factor<=0:
            raise ValueError('Invalid calibrated RGB-D input')
        fx,fy=intrinsics[0,0],intrinsics[1,1];cx,cy=intrinsics[0,2],intrinsics[1,2]
        xmap,ymap=np.meshgrid(np.arange(depth.shape[1]),np.arange(depth.shape[0]))
        z=depth/factor;x=(xmap-cx)/fx*z;y=(ymap-cy)/fy*z
        mask=(z>args.depth_min)&(z<args.depth_max)
        if args.workspace_mask:
            workspace=np.asarray(Image.open(args.workspace_mask)).astype(bool)
            if workspace.shape!=depth.shape:
                raise ValueError('Workspace mask shape differs from depth')
            mask &= workspace
        points=np.stack([x,y,z],axis=-1)[mask].astype(np.float32)
        if not len(points) or not np.isfinite(points).all():
            raise ValueError('No finite points in declared depth/workspace domain')
        # Same camera-point features and integer truncation as author get_grasp.
        coords=np.ascontiguousarray(points/args.voxel_size,dtype=int)
        np.save(args.output_dir/'input_points_m.npy',points)
        np.save(args.output_dir/'input_colors_rgb.npy',color[mask])
        np.save(args.output_dir/'input_coords_before_quantization.npy',coords)
        files=[depth_path,color_path,meta_path]+([args.workspace_mask] if args.workspace_mask else [])
        report['input_files']={str(path):{'bytes':path.stat().st_size,'sha256':sha(path)} for path in files}
        report['calibration']={'intrinsic_matrix':intrinsics.tolist(),'factor_depth':factor,
            'point_units':'metres','sensor_shape':list(depth.shape),'retained_pixels':len(points),
            'point_min_m':points.min(0).tolist(),'point_max_m':points.max(0).tolist(),
            'voxel_size_m':args.voxel_size,'depth_bounds_m':[args.depth_min,args.depth_max],
            'coordinate_quantization':'np.ascontiguousarray(points / voxel_size, dtype=int), author truncation retained',
            'features':'camera XYZ metres, author 3-channel input',
            'calibration_boundary':'Input meta.mat intrinsics/depth scale replace rig-specific constants only',
            'workspace_image_mask_used':bool(args.workspace_mask),'augmentation':'identity only, one author get_grasp branch'}
        report['timing_seconds']['input_preparation']=time.monotonic()-begin
        # Python3.8 exposed this exact ABC alias; restore it only in this process.
        if not hasattr(collections, 'Sequence'):
            collections.Sequence=collections.abc.Sequence
        report['python_compatibility_overlay']={'collections.Sequence':'collections.abc.Sequence (Python3.8 alias restored process-locally)'}
        if args.prepare_only:
            report['status']='inputs_prepared_only'
            report['full_representation_forward']=False
        else:
            begin=time.monotonic()
            native_paths=[args.native_deps/'build_src/MinkowskiEngine',args.native_deps/'pointnet2/lib']
            native_paths+=list((args.native_deps/'pointnet2').glob('lib.*'))
            for path in reversed([ROOT/'models',ROOT/'utils',ROOT/'pointnet2',*native_paths]):
                if path.exists():
                    sys.path.insert(0,str(path))
            import MinkowskiEngine as ME
            import pointnet2_utils
            extension=importlib.import_module('pointnet2._ext')
            extension_path=Path(extension.__file__)
            if not any(str(extension_path).endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
                raise RuntimeError('Original PointNet2 native extension is required; Python replacements refused')
            device=torch.device(args.device)
            if device.type!='cuda' or not torch.cuda.is_available():
                raise RuntimeError('Original PointNet2 cylinder query requires CUDA')
            torch.cuda.set_device(device)
            report['native_dependencies']={'MinkowskiEngine_file':ME.__file__,
                'MinkowskiEngine_version':getattr(ME,'__version__',None),
                'pointnet2_utils_file':pointnet2_utils.__file__,
                'pointnet2_extension':str(extension_path),'pointnet2_extension_sha256':sha(extension_path),
                'cuda_device':str(device),'cuda_device_name':torch.cuda.get_device_name(device),
                'cuda_compute_capability':list(torch.cuda.get_device_capability(device)),
                'LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH','')}
            counters={name:0 for name in ['furthest_point_sample','cylinder_query','grouping_operation']}
            for name in counters:
                original=getattr(pointnet2_utils,name)
                def instrumented(*values,_original=original,_name=name,**kwargs):
                    counters[_name]+=1
                    return _original(*values,**kwargs)
                setattr(pointnet2_utils,name,instrumented)
            model_cls,overlay=source_overlay();report['representation_overlay']=overlay
            unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(args.checkpoint)
            if unsafe:
                raise RuntimeError('Representation checkpoint unexpectedly requires pickle globals: '+repr(unsafe))
            checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
            state=checkpoint['model_state_dict']
            out_channels=state['grasp_generator.conv3.weight'].shape[0]
            num_depth=out_channels//(48*2)
            if num_depth*48*2!=out_channels:
                raise ValueError('Checkpoint output shape inconsistent with original 48-angle CGR')
            model=model_cls(in_channels=3,num_seed=args.num_seed,num_view=300,num_angle=48,
                            num_depth=num_depth,is_training=False,half_views=False)
            model.load_state_dict(state,strict=True);model.to(device);model.eval()
            report['checkpoint']={'path':str(args.checkpoint),'sha256':sha(args.checkpoint),
                'weights_only':True,'strict_state_dict_load':True,'tensor_count':len(state),
                'epoch':checkpoint.get('epoch'),'num_view':300,'num_angle':48,'num_depth':num_depth,
                'num_seed':args.num_seed,'half_views':False}
            report['timing_seconds']['dependency_import_and_model_load']=time.monotonic()-begin
            begin=time.monotonic()
            tensor=torch.from_numpy(points)
            _,indices=ME.utils.sparse_quantize(coords,return_index=True)
            coords_unique=coords[indices];points_unique=tensor[indices]
            coords_batch,points_batch=ME.utils.sparse_collate([coords_unique],[points_unique])
            sinput=ME.SparseTensor(points_batch,coords_batch,device=device)
            report['sparse_input']={'unique_voxels':len(indices),'coordinate_shape':list(sinput.C.shape),
                                   'feature_shape':list(sinput.F.shape)}
            torch.cuda.synchronize(device)
            report['timing_seconds']['sparse_quantization_and_input']=time.monotonic()-begin
            endpoints={'sinput':sinput,'point_clouds':[sinput.F]}
            torch.cuda.reset_peak_memory_stats(device)
            begin=time.monotonic()
            with torch.no_grad():
                endpoints=model(endpoints)
            torch.cuda.synchronize(device)
            report['timing_seconds']['first_full_representation_forward']=time.monotonic()-begin
            report['native_operator_call_counts']=counters
            report['peak_cuda_tensor_memory_bytes']=torch.cuda.max_memory_allocated(device)
            arrays={}
            keys=['stage1_objectness_pred','stage1_heatmap_pred','stage2_seed_xyz','stage2_seed_inds',
                  'stage2_view_xyz','stage2_view_inds','stage2_view_scores','stage3_grasp_scores',
                  'stage3_normalized_grasp_widths','stage3_grasp_features','before_generator','point_features']
            for key in keys:
                array=endpoints[key].detach().cpu().numpy()
                if not np.isfinite(array).all():
                    raise RuntimeError('Non-finite original model output: '+key)
                arrays[key]=array
            arrays['raw_CGR_480']=arrays['stage3_grasp_features'][...,:480].copy()
            arrays['sparse_input_coords']=sinput.C.detach().cpu().numpy()
            arrays['sparse_input_XYZ_features']=sinput.F.detach().cpu().numpy()
            np.savez_compressed(args.output_dir/'representation_outputs.npz',**arrays)
            parse,parse_receipt=parser_overlay(args.parser_hand);report['candidate_parser_overlay']=parse_receipt
            with torch.no_grad():
                preds,features=parse(endpoints,use_v2=(num_depth==5))
            candidates=preds[0].detach().cpu().numpy() if preds else np.empty((0,15),np.float32)
            candidate_features=features[0].detach().cpu().numpy() if features else np.empty((0,0),np.float32)
            if not np.isfinite(candidates).all() or not np.isfinite(candidate_features).all():
                raise RuntimeError('Non-finite raw original parsed candidates')
            np.savez_compressed(args.output_dir/'raw_camera_candidates.npz',candidates=candidates,
                                candidate_features=candidate_features)
            report['output_shapes']={key:list(array.shape) for key,array in arrays.items()}
            report['raw_candidates']={'count':len(candidates),'shape':list(candidates.shape),
                'feature_shape':list(candidate_features.shape),'frame':'input camera optical frame',
                'status':'original parser output before rig-specific width/approach/workspace and collision filtering',
                'not_feasible_grasp_certificate':True}
            report['full_representation_forward']=True;report['status']='representation_forward_complete'
        report['timing_seconds']['total']=time.monotonic()-total
        (args.output_dir/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    except Exception as exc:
        report.update({'status':'blocked_or_failed','error_type':type(exc).__name__,
            'error':str(exc),'traceback':traceback.format_exc(),'full_representation_forward':False})
        report['timing_seconds']['total']=time.monotonic()-total
        (args.output_dir/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        raise


if __name__=='__main__':
    main()
