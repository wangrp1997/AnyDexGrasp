"""One persistent official CGR model, one identity forward per frozen Isaac frame.

Manifest objects accept {object_id, observation_json, representation_dir}.
Aliases {id, observation} match the pilot scheduler. Relative paths are resolved
against the manifest. Defaults write RUN/representations/OBJECT_ID/. No hand
head, DH3 rig filter, candidate selection, robot import, or physics runs here.
"""
from __future__ import annotations
import argparse,collections.abc,copy,hashlib,importlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import scipy.io
import torch

_PARENTS=Path(__file__).resolve().parents
ROOT=_PARENTS[2] if len(_PARENTS)>2 else Path('/home/wangrenpeng/AnyDexGrasp')
# The temporary draft can be contract-checked before root installs it.
if not (ROOT/'reproduction/runtime/run_official_rgbd_frame.py').is_file():
    ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
sys.path.insert(0,str(ROOT/'reproduction/runtime'))
from run_official_rgbd_frame import sha,source_overlay,parser_overlay
REVISION='c9c4a43df33e40860417c7e2dd02f5122d3b2da2'
EXPECTED_SOURCES={
 'models/minkowski_graspnet_single_point.py':'1463ee5804ec72bafbfc0f469041425092817ff6e6b797923db77120e1167b26',
 'robot_dh3.py':'7d227451761a429fff87f689e5dcb9c9e215da5ec3c5c594be1e16f3333fa797'}
KEYS=['stage1_objectness_pred','stage1_heatmap_pred','stage2_seed_xyz','stage2_seed_inds',
 'stage2_view_xyz','stage2_view_inds','stage2_view_scores','stage3_grasp_scores',
 'stage3_normalized_grasp_widths','stage3_grasp_features','before_generator','point_features']

def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def numpy_rng_sha(state):
    return hashlib.sha256(state[1].tobytes()+json.dumps([state[0],state[2],state[3],state[4]]).encode()).hexdigest()

def resolve(value,base):
    p=Path(value).expanduser()
    return (p if p.is_absolute() else base/p).resolve()

def manifest_rows(args):
    manifest=json.loads(args.manifest.read_text());items=manifest['objects']
    if not isinstance(items,list) or not 1<=len(items)<=8:
        raise ValueError('The pilot requires a fixed list of 1..8 observations')
    rows=[]
    for item in items:
        identity=item.get('object_id',item.get('id'))
        if not isinstance(identity,str) or not identity or Path(identity).name!=identity or identity in ('.','..'):
            raise ValueError('Each object needs a unique path-safe object_id')
        observation=item.get('observation_json',item.get('observation'))
        if observation is None:raise ValueError('Each object needs observation_json')
        path=resolve(observation,args.manifest.parent)
        destination=resolve(item['representation_dir'],args.manifest.parent) if item.get('representation_dir') else args.run_dir/'representations'/identity
        row=dict(object_id=identity,observation_json=str(path),output_dir=str(destination.resolve()))
        if item.get('observation_sha256'):row['expected_observation_sha256']=item['observation_sha256']
        rows.append(row)
    if len({r['object_id'] for r in rows})!=len(rows) or len({r['output_dir'] for r in rows})!=len(rows):
        raise ValueError('Duplicate object or output directory')
    return rows

def read_frame(row,args):
    begin=time.monotonic();obs_path=Path(row['observation_json']);obs_digest=sha(obs_path)
    if row.get('expected_observation_sha256',obs_digest)!=obs_digest:raise ValueError('Frozen observation JSON hash differs')
    obs=json.loads(obs_path.read_text());paths={}
    for name,item in obs['files'].items():
        path=resolve(item['path'],obs_path.parent)
        if sha(path)!=item['sha256']:raise ValueError('Frozen observation file changed: '+str(path))
        paths[name]=path
    for name in ['depth.png','color.png','meta.mat']:
        if name not in paths:raise ValueError('Missing calibrated RGB-D file '+name)
    camera=np.asarray(obs['T_WC'],dtype=float)
    if camera.shape!=(4,4) or not np.isfinite(camera).all() or not np.allclose(camera[3],[0,0,0,1],atol=2e-6,rtol=0) or not np.allclose(camera[:3,:3].T@camera[:3,:3],np.eye(3),atol=2e-6,rtol=0) or abs(np.linalg.det(camera[:3,:3])-1)>2e-6:
        raise ValueError('Observation T_WC must be a proper camera-to-world transform')
    if obs.get('object_pose_as_network_input',False):raise ValueError('Object pose is not an allowed network input')
    depth=np.asarray(Image.open(paths['depth.png']));color=np.asarray(Image.open(paths['color.png']).convert('RGB'));meta=scipy.io.loadmat(paths['meta.mat'])
    intrinsics=np.asarray(meta['intrinsic_matrix'],dtype=np.float64);factor=float(np.asarray(meta['factor_depth']).reshape(-1)[0])
    if depth.ndim!=2 or color.shape[:2]!=depth.shape or intrinsics.shape!=(3,3) or not np.isfinite(intrinsics).all() or not np.isfinite(factor) or factor<=0 or intrinsics[0,0]<=0 or intrinsics[1,1]<=0:raise ValueError('Invalid calibrated RGB-D')
    if 'intrinsic_matrix' in obs and not np.allclose(intrinsics,obs['intrinsic_matrix'],atol=1e-7,rtol=0):raise ValueError('Observation and meta.mat intrinsics differ')
    obs_factor=obs.get('depth_factor',obs.get('factor_depth',factor))
    if not np.isclose(factor,float(obs_factor),atol=1e-9,rtol=0):raise ValueError('Observation and meta.mat depth scale differ')
    fx,fy=intrinsics[0,0],intrinsics[1,1];cx,cy=intrinsics[0,2],intrinsics[1,2]
    xmap,ymap=np.meshgrid(np.arange(depth.shape[1]),np.arange(depth.shape[0]));z=depth/factor;x=(xmap-cx)/fx*z;y=(ymap-cy)/fy*z
    mask=(z>args.depth_min)&(z<args.depth_max);points=np.stack([x,y,z],axis=-1)[mask].astype(np.float32)
    if not len(points) or not np.isfinite(points).all():raise ValueError('No finite points in declared depth domain')
    coords=np.ascontiguousarray(points/args.voxel_size,dtype=int)
    binding=dict(observation_json=str(obs_path),observation_sha256=obs_digest,T_WC=camera.tolist(),file_hashes_verified=True,
        source_simulator=obs.get('simulator'),source_renderer=obs.get('renderer'),
        network_inputs=['calibrated RGB-D camera XYZ'],object_pose_as_network_input=False,
        boundary='T_WC and object provenance bind later geometry/execution only; neither is fed to CGR')
    calibration=dict(intrinsic_matrix=intrinsics.tolist(),factor_depth=factor,point_units='metres',sensor_shape=list(depth.shape),retained_pixels=len(points),
        point_min_m=points.min(0).tolist(),point_max_m=points.max(0).tolist(),voxel_size_m=args.voxel_size,depth_bounds_m=[args.depth_min,args.depth_max],
        coordinate_quantization='np.ascontiguousarray(points / voxel_size, dtype=int), author truncation retained',features='camera XYZ metres, author 3-channel input',
        calibration_boundary='meta.mat intrinsics/depth scale; actual observation T_WC is metadata, not a network feature',workspace_image_mask_used=False,
        augmentation='identity only; one original representation forward per object, shared across the three collection gestures')
    input_files={str(paths[name]):dict(bytes=paths[name].stat().st_size,sha256=sha(paths[name])) for name in ['depth.png','color.png','meta.mat']}
    return points,coords,binding,calibration,input_files,time.monotonic()-begin


class PersistentOfficialCGR:
    def __init__(self,args):
        self.args=args;self.counters={n:0 for n in ['furthest_point_sample','cylinder_query','grouping_operation']};self.random_counts={n:0 for n in ['multinomial','randperm']}
        torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
        if not hasattr(collections,'Sequence'):collections.Sequence=collections.abc.Sequence
        native=[args.native_deps/'build_src/MinkowskiEngine',args.native_deps/'pointnet2/lib']+list((args.native_deps/'pointnet2').glob('lib.*'))
        for p in reversed([ROOT/'models',ROOT/'utils',ROOT/'pointnet2',*native]):
            if p.exists():sys.path.insert(0,str(p))
        self.ME=importlib.import_module('MinkowskiEngine');pointnet=importlib.import_module('pointnet2_utils')
        extension=importlib.import_module('pointnet2._ext');me_extension=importlib.import_module('MinkowskiEngineBackend._C')
        for module in [extension,me_extension]:
            if not any(str(module.__file__).endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES):raise RuntimeError('Original compiled native extensions required')
        self.device=torch.device(args.device)
        if self.device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('Original PointNet2 cylinder query requires CUDA')
        torch.cuda.set_device(self.device)
        self.native=dict(MinkowskiEngine_file=self.ME.__file__,MinkowskiEngine_version=getattr(self.ME,'__version__',None),
            MinkowskiEngine_extension=me_extension.__file__,MinkowskiEngine_extension_sha256=sha(me_extension.__file__),
            pointnet2_utils_file=pointnet.__file__,pointnet2_utils_sha256=sha(pointnet.__file__),pointnet2_extension=extension.__file__,pointnet2_extension_sha256=sha(extension.__file__),
            cuda_device=str(self.device),cuda_device_name=torch.cuda.get_device_name(self.device),cuda_compute_capability=list(torch.cuda.get_device_capability(self.device)),
            LD_LIBRARY_PATH=os.environ.get('LD_LIBRARY_PATH',''),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
        for name in self.counters:
            original=getattr(pointnet,name)
            def counted(*values,_f=original,_n=name,**kwargs):self.counters[_n]+=1;return _f(*values,**kwargs)
            setattr(pointnet,name,counted)
        for name in self.random_counts:
            original=getattr(torch,name)
            def counted_random(*values,_f=original,_n=name,**kwargs):self.random_counts[_n]+=1;return _f(*values,**kwargs)
            setattr(torch,name,counted_random)
        for relative,digest in EXPECTED_SOURCES.items():
            if sha(ROOT/relative)!=digest:raise ValueError('Pinned original source differs: '+relative)
        cls,self.overlay=source_overlay()
        unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(args.checkpoint)
        if unsafe:raise RuntimeError('Unexpected checkpoint pickle globals: '+repr(unsafe))
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True);state=checkpoint['model_state_dict'];channels=state['grasp_generator.conv3.weight'].shape[0];self.depths=channels//(48*2)
        if self.depths*48*2!=channels or self.depths!=5:raise ValueError('Require released original 48-angle/5-depth checkpoint')
        constructor_before=torch.get_rng_state().clone();cuda_before=torch.cuda.get_rng_state(self.device).clone()
        self.model=cls(in_channels=3,num_seed=args.num_seed,num_view=300,num_angle=48,num_depth=self.depths,is_training=False,half_views=False)
        incompat=self.model.load_state_dict(state,strict=True)
        if incompat.missing_keys or incompat.unexpected_keys:raise ValueError('Strict original checkpoint load mismatch')
        self.model.to(self.device);self.model.eval();self.parse,self.parser_receipt=parser_overlay('dh3')
        # Reuse the module, while replaying the post-initialization RNG state a
        # cold first-frame process would have. No constructor draw disappears.
        self.cpu_rng=torch.get_rng_state().clone();self.cuda_rng=torch.cuda.get_rng_state(self.device).clone();self.numpy_rng=copy.deepcopy(np.random.get_state())
        self.rng=dict(seed=args.seed,policy='same post-original-initialization RNG state restored for every frame',
            constructor_CPU_before_sha256=array_sha(constructor_before.numpy()),post_initialization_CPU_sha256=array_sha(self.cpu_rng.numpy()),
            post_initialization_CUDA_sha256=array_sha(self.cuda_rng.cpu().numpy()),post_initialization_NumPy_sha256=numpy_rng_sha(self.numpy_rng),
            constructor_CUDA_rng_unchanged=bool(torch.equal(cuda_before,self.cuda_rng)),constructor_skipped_random_draws_replayed=True)
        self.checkpoint=dict(path=str(args.checkpoint),sha256=sha(args.checkpoint),weights_only=True,strict_state_dict_load=True,tensor_count=len(state),epoch=checkpoint.get('epoch'),num_view=300,num_angle=48,num_depth=self.depths,num_seed=args.num_seed,half_views=False)
        stat=args.checkpoint.stat();self.checkpoint_stat=(stat.st_size,stat.st_mtime_ns)
        self.calls=0;del checkpoint,state

    def forward(self,points,coords,directory):
        stat=self.args.checkpoint.stat()
        if (stat.st_size,stat.st_mtime_ns)!=self.checkpoint_stat:raise ValueError('Checkpoint file changed during batch')
        if self.model.training:raise RuntimeError('Persistent model entered training mode')
        torch.set_rng_state(self.cpu_rng);torch.cuda.set_rng_state(self.cuda_rng,self.device);np.random.set_state(self.numpy_rng)
        counters_before=self.counters.copy();random_before=self.random_counts.copy();begin=time.monotonic();tensor=torch.from_numpy(points)
        _,indices=self.ME.utils.sparse_quantize(coords,return_index=True);coords_unique=coords[indices];points_unique=tensor[indices]
        coords_batch,points_batch=self.ME.utils.sparse_collate([coords_unique],[points_unique]);sinput=self.ME.SparseTensor(points_batch,coords_batch,device=self.device)
        torch.cuda.synchronize(self.device);quantization=time.monotonic()-begin;endpoints={'sinput':sinput,'point_clouds':[sinput.F]}
        torch.cuda.reset_peak_memory_stats(self.device);begin=time.monotonic()
        with torch.no_grad():endpoints=self.model(endpoints)
        self.calls+=1;torch.cuda.synchronize(self.device);forward_seconds=time.monotonic()-begin;arrays={}
        for key in KEYS:
            a=endpoints[key].detach().cpu().numpy()
            if not np.isfinite(a).all():raise RuntimeError('Nonfinite original output '+key)
            arrays[key]=a
        arrays['raw_CGR_480']=arrays['stage3_grasp_features'][...,:480].copy();arrays['sparse_input_coords']=sinput.C.detach().cpu().numpy();arrays['sparse_input_XYZ_features']=sinput.F.detach().cpu().numpy()
        begin=time.monotonic()
        with torch.no_grad():preds,features=self.parse(endpoints,use_v2=True)
        candidates=preds[0].detach().cpu().numpy() if preds else np.empty((0,15),np.float32)
        candidate_features=features[0].detach().cpu().numpy() if features else np.empty((0,2261),np.float32)
        if candidates.ndim!=2 or candidates.shape[1]!=15 or candidate_features.shape!=(len(candidates),2261) or not np.isfinite(candidates).all() or not np.isfinite(candidate_features).all():raise RuntimeError('Nonfinite or unpaired original parsed rows')
        parse_seconds=time.monotonic()-begin;begin=time.monotonic()
        np.savez_compressed(directory/'representation_outputs.npz',**arrays);np.savez_compressed(directory/'raw_camera_candidates.npz',candidates=candidates,candidate_features=candidate_features)
        result=dict(sparse_input=dict(unique_voxels=len(indices),coordinate_shape=list(sinput.C.shape),feature_shape=list(sinput.F.shape)),
            native_operator_call_counts={n:self.counters[n]-counters_before[n] for n in self.counters},
            random_forward_call_counts={n:self.random_counts[n]-random_before[n] for n in self.random_counts},
            rng=dict(self.rng,CPU_after_frame_sha256=array_sha(torch.get_rng_state().numpy()),CUDA_after_frame_sha256=array_sha(torch.cuda.get_rng_state(self.device).cpu().numpy())),
            peak_cuda_tensor_memory_bytes=torch.cuda.max_memory_allocated(self.device),output_shapes={k:list(v.shape) for k,v in arrays.items()},
            raw_candidates=dict(count=len(candidates),shape=list(candidates.shape),feature_shape=list(candidate_features.shape),frame='input camera optical frame',status='original parse_preds output before rig-specific width/approach/workspace and collision filtering',not_feasible_grasp_certificate=True),
            timing_seconds=dict(sparse_quantization_and_input=quantization,full_representation_forward=forward_seconds,original_parse_preds=parse_seconds,output_serialization=time.monotonic()-begin))
        del endpoints,sinput,tensor,preds,features,arrays
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--device',default='cuda:1')
    p.add_argument('--checkpoint',type=Path,default=Path('/mnt/public/datasets/AnyDexGrasp/logs/model/checkpoint.tar.18'));p.add_argument('--native-deps',type=Path,default=ROOT/'outputs/native_deps')
    p.add_argument('--seed',type=int,default=20261105);p.add_argument('--num-seed',type=int,default=2048);p.add_argument('--voxel-size',type=float,default=.005);p.add_argument('--depth-min',type=float,default=.2);p.add_argument('--depth-max',type=float,default=.65)
    p.add_argument('--prepare-only',action='store_true',help='Verify manifest/RGB-D only; zero model loads/forwards')
    a=p.parse_args();a.manifest=a.manifest.resolve();a.run_dir=a.run_dir.resolve();a.checkpoint=a.checkpoint.resolve();a.native_deps=a.native_deps.resolve()
    if not (np.isfinite([a.voxel_size,a.depth_min,a.depth_max]).all() and a.voxel_size>0 and 0<=a.depth_min<a.depth_max and a.num_seed>0 and a.seed>=0):p.error('Invalid explicit preprocessing or seed parameters')
    a.run_dir.mkdir(parents=True,exist_ok=True);batch_path=a.run_dir/'BATCH_REPRESENTATION.json'
    if batch_path.exists():p.error('Batch receipt exists; preserve it and use a new run directory')
    rows=manifest_rows(a)
    for row in rows:
        directory=Path(row['output_dir'])
        if directory.exists() and any(directory.iterdir()):p.error('Representation output exists; preserve it: '+str(directory))
    manifest_digest=sha(a.manifest);start=time.monotonic();batch=dict(schema='anydex_boya_pilot_shared_representation_v1',status='starting',manifest=str(a.manifest),manifest_sha256=manifest_digest,
        argv=list(sys.argv),args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},python=sys.version,torch=torch.__version__,torch_cuda=torch.version.cuda,
        source_revision=REVISION,source_checkout=str(ROOT),source_sha256={str(path):sha(path) for path in [Path(__file__),ROOT/'reproduction/runtime/run_official_rgbd_frame.py',ROOT/'models/minkowski_graspnet_single_point.py',ROOT/'robot_dh3.py',ROOT/'utils/pt_utils.py']},
        model_load_count=0,checkpoint_file_load_count=0,strict_state_dict_assignments=0,network_forward_count=0,hand_head_load_count=0,new_training_steps=0,new_physics_steps=0,
        protocol='One identity official representation forward per object; paired predictions/features shared across all three fixed Boya collection gestures',
        objects=[dict(row,status='not_started') for row in rows],timing_seconds={})
    save(batch_path,batch);engine=None;failed=False
    try:
        if not a.prepare_only:
            begin=time.monotonic();engine=PersistentOfficialCGR(a);batch.update(model_load_count=1,checkpoint_file_load_count=1,strict_state_dict_assignments=1,checkpoint=engine.checkpoint,native_dependencies=engine.native,representation_overlay=engine.overlay,candidate_parser_overlay=engine.parser_receipt,rng_policy=engine.rng)
            batch['timing_seconds']['dependency_import_and_one_model_load']=time.monotonic()-begin;save(batch_path,batch)
        for j,row in enumerate(rows):
            if sha(a.manifest)!=manifest_digest:raise ValueError('Frozen manifest changed during batch')
            directory=Path(row['output_dir']);directory.mkdir(parents=True,exist_ok=True);frame_start=time.monotonic()
            report=dict(scope='Fresh frozen Isaac RGB-D, original CGR identity inference for Boya pilot collection',source_revision=REVISION,source_checkout=str(ROOT),
                object_id=row['object_id'],manifest_sha256=manifest_digest,argv=list(sys.argv),args=batch['args'],source_sha256=batch['source_sha256'],seed=a.seed,
                physical_grasps_executed=0,no_robot_connection=True,simulation_run=False,grasp_success_claim=False,new_training_steps=0,hand_head_loaded=False,
                example_boundary='Fresh Isaac native RTX observation supplied by this pilot; not author hardware/demo data, grasp execution, or a Boya learned score',
                full_representation_forward=False,status='starting',test_time_augmentation_branches=1,timing_seconds={})
            save(directory/'RESULTS.json',report)
            try:
                points,coords,binding,calibration,input_files,prep_seconds=read_frame(row,a)
                report.update(scene_binding=binding,calibration=calibration,input_files=input_files,input_points_sha256=array_sha(points),input_coords_sha256=array_sha(coords))
                report['timing_seconds']['input_preparation']=prep_seconds
                if engine is None:report['status']='inputs_prepared_only'
                else:
                    result=engine.forward(points,coords,directory);report['timing_seconds'].update(result.pop('timing_seconds'));report.update(result)
                    report.update(full_representation_forward=True,status='representation_forward_complete',checkpoint=engine.checkpoint,native_dependencies=engine.native,representation_overlay=engine.overlay,candidate_parser_overlay=engine.parser_receipt,persistent_model_reused=j>0)
                    for name in ['raw_camera_candidates.npz','representation_outputs.npz']:report.setdefault('output_files',{})[name]=dict(path=str(directory/name),sha256=sha(directory/name),bytes=(directory/name).stat().st_size)
                # Keep the exact rendered observation/manifest bytes frozen.
                if sha(row['observation_json'])!=binding['observation_sha256'] or sha(a.manifest)!=manifest_digest:raise ValueError('A frozen input changed during this forward')
                for path,value in input_files.items():
                    if sha(path)!=value['sha256']:raise ValueError('A source RGB-D file changed during this forward')
            except Exception as exc:
                report.update(status='blocked_or_failed',full_representation_forward=False,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());failed=True
            report['timing_seconds']['total_frame_wall']=time.monotonic()-frame_start;save(directory/'RESULTS.json',report)
            batch['objects'][j].update(status=report['status'],full_representation_forward=report['full_representation_forward'],result_json=str(directory/'RESULTS.json'),result_sha256=sha(directory/'RESULTS.json'),timing_seconds=report['timing_seconds'],raw_candidate_count=report.get('raw_candidates',{}).get('count'),error=report.get('error'))
            batch['network_forward_count']=engine.calls if engine else 0;batch['timing_seconds']['elapsed_wall']=time.monotonic()-start;save(batch_path,batch)
            print('BOYA_REPRESENTATION '+json.dumps(batch['objects'][j]),flush=True)
            if failed:break
        batch['status']='failed_or_incomplete' if failed else 'inputs_prepared_only' if a.prepare_only else 'all_representation_forwards_complete'
    except Exception as exc:
        batch.update(status='blocked_or_failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());failed=True
    batch['network_forward_count']=engine.calls if engine else 0;batch['timing_seconds']['total_wall']=time.monotonic()-start
    batch['completed_forward_objects']=sum(r.get('full_representation_forward') is True for r in batch['objects'])
    batch['warm_frame_wall_seconds']=[r['timing_seconds']['total_frame_wall'] for r in batch['objects'][1:] if r.get('full_representation_forward')]
    batch['manifest_unchanged_at_end']=sha(a.manifest)==manifest_digest
    if not batch['manifest_unchanged_at_end']:failed=True;batch['status']='frozen_manifest_changed'
    save(batch_path,batch);print(json.dumps({k:batch[k] for k in ['status','model_load_count','checkpoint_file_load_count','network_forward_count','completed_forward_objects','timing_seconds']},indent=2),flush=True)
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
