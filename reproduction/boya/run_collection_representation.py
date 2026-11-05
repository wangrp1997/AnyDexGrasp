"""One frozen official CGR model for a scalable collection manifest.

Reuse the verified pilot inference class and original native/parser overlays.
Only the outer manifest size changes; no decision head is loaded or trained.
"""
from pathlib import Path
import argparse,json,sys,time,traceback
import torch
import numpy as np
from run_pilot_representation import (ROOT,REVISION,sha,save,resolve,array_sha,read_frame,PersistentOfficialCGR)


def manifest_rows(args):
    manifest=json.loads(args.manifest.read_text());items=manifest['objects']
    if not isinstance(items,list) or not 1<=len(items)<=512:
        raise ValueError('Collection requires 1..512 frozen observations')
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

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--device',default='cuda:0')
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
        source_revision=REVISION,source_checkout=str(ROOT),source_sha256={str(path):sha(path) for path in [Path(__file__),Path(__file__).with_name('run_pilot_representation.py'),ROOT/'reproduction/runtime/run_official_rgbd_frame.py',ROOT/'models/minkowski_graspnet_single_point.py',ROOT/'robot_dh3.py',ROOT/'utils/pt_utils.py']},
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
            report=dict(scope='Fresh frozen Isaac RGB-D, original CGR identity inference for Boya declared collection',source_revision=REVISION,source_checkout=str(ROOT),
                object_id=row['object_id'],manifest_sha256=manifest_digest,argv=list(sys.argv),args=batch['args'],source_sha256=batch['source_sha256'],seed=a.seed,
                physical_grasps_executed=0,no_robot_connection=True,simulation_run=False,grasp_success_claim=False,new_training_steps=0,hand_head_loaded=False,
                example_boundary='Fresh Isaac native RTX observation supplied by this collection; not author hardware/demo data, grasp execution, or a Boya learned score',
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
