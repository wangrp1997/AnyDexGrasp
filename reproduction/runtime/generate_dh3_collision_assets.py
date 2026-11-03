"""Generate only required DH3 collision slots using original author functions.

Four isolated CPU processes maximum, each fresh PyBullet DIRECT client and
zero simulation steps. Published lookup joint configurations replace Excel
reading; original FK/mesh assembly/sampling/infill/voxel save code is retained.
"""
from __future__ import annotations
import argparse
import ast
import concurrent.futures
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
REV='c9c4a43df33e40860417c7e2dd02f5122d3b2da2'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def select(path,names,scope):
    nodes=[]
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets):
            nodes.append(node)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope


def original_inputs(table):
    urdf=ROOT/'generate_mesh_and_pointcloud/dh3_urdf/dh3_urdf/urdf/dh3_urdf.urdf'
    meshes=sorted((urdf.parent.parent/'meshes').glob('*.STL'))
    assert len(meshes)==13
    paths=[urdf,*meshes,ROOT/'generate_mesh_and_pointcloud/recover_dh3_hand_to_stl.py',
           ROOT/'generate_mesh_and_pointcloud/angles2stl.py',ROOT/'ur_toolbox/ur_toolbox/robot/DH3/DH3_grasp.py']
    rows=[]
    for path in paths:
        relative=str(path.relative_to(ROOT))
        result=subprocess.run(['git','show',REV+':'+relative],cwd=ROOT,capture_output=True,check=True)
        expected=hashlib.sha256(result.stdout).hexdigest()
        current=sha(path)
        if expected!=current:
            raise ValueError('Original source/input differs from pinned author commit: '+relative)
        rows.append({'path':str(path),'bytes':path.stat().st_size,'sha256':current,'matches_author_commit':True})
    original_table=ROOT/'generate_mesh_and_pointcloud/dh3_urdf/width_12D_angle_2D_angle.json'
    assert sha(table)==sha(original_table)
    return rows


def make_plan(args):
    from_source=ROOT/'ur_toolbox/ur_toolbox/robot/DH3/DH3_grasp.py'
    api_source=ROOT/'outputs/native_deps/graspnetAPI_source/graspnetAPI/grasp.py'
    table=args.table_dir/'width_12D_angle_2D_angle.json'
    source_receipts=original_inputs(table)
    api=select(api_source,{'GRASP_ARRAY_LEN','RECT_GRASP_ARRAY_LEN','EPS','Grasp','GraspGroup'},{'np':np,'copy':copy})
    hand_scope=select(from_source,{'grasp_types','MIN_GRASP_WIDTH','MAX_GRASP_WIDTH','DH3_DEFAULT_DEPTH','DH3_ARRAY_LEN','DH3Grasp','DH3GraspGroup'},
                      {'np':np,'copy':copy,'os':os,'json':json,'math':math})
    input_path=args.hand_run_dir/'predicted_DH3_hand_poses_pre_collision.npz'
    with np.load(input_path,allow_pickle=False) as saved:
        hand=hand_scope['DH3GraspGroup'](saved['DH3_grasp_group_array'].copy())
        two=api['GraspGroup'](saved['two_finger_grasp_group_array'].copy())
    # Exact original detect() width prelude; clones only, no saved prediction mutation.
    two.widths=two.widths*1.7
    mask=two.widths>.045
    hand=hand[mask];two=two[mask]
    hand.graspgroupTR_2_TR(two,str(args.table_dir))
    slots=sorted({(int(row[2]),str(np.round(row[18]*100,1))) for row in hand.grasp_group_array},key=lambda item:(item[0],float(item[1])))
    lookup=json.loads(table.read_text())
    required=[]
    for hand_type,width in slots:
        pose='pose'+str(hand_type);entry=lookup[pose][width]
        required.append({'type':hand_type,'pose':pose,'width_cm_key':width,'joint12_rad':entry['12d'],'actuator2':entry['2d'],
                         'sampling_seed':20261103+hand_type*1000+int(round(float(width)*10))})
    asset_root=args.asset_root or args.output_dir/'assets'
    asset_root.mkdir(parents=True,exist_ok=True)
    copied_table=asset_root/table.name
    if copied_table.exists():
        assert sha(copied_table)==sha(table)
    else:
        shutil.copyfile(table,copied_table)
    for item in required:
        (asset_root/'meshes/source'/item['pose']).mkdir(parents=True,exist_ok=True)
        for scale in range(1,6):
            (asset_root/f'meshes/source_pointclouds/voxel_size_{scale}'/item['pose']).mkdir(parents=True,exist_ok=True)
    (asset_root/'slot_receipts').mkdir(exist_ok=True)
    plan={'scope':'actual slots after exact original collision width*1.7, min-width gate and recalibration',
        'source_revision':REV,'hand_run_dir':str(args.hand_run_dir.resolve()),'input_prediction_sha256':sha(input_path),
        'asset_root':str(asset_root.resolve()),'table_sha256':sha(table),'original_inputs':source_receipts,
        'prelude_candidates':len(hand),'slot_count':len(required),'required_slots':required,
        'original_width_factor':1.7,'original_min_width_m':.045,
        'generator_preserved':'get_meshes(if_source=False), get_mesh, four_meta_to_matrix, save_stl_and_pointcloud',
        'differences_from_author_entry':'Use published 12d lookup rather than Excel/regenerating lookup; DIRECT rather than SHARED_MEMORY; per-slot seed; selected needed slots; output-only paths',
        'physics_steps':0,'shared_memory_connection':False}
    (args.output_dir/'ASSET_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    np.savez_compressed(args.output_dir/'post_original_collision_width_prelude.npz',
        DH3_grasp_group_array=hand.grasp_group_array,two_finger_grasp_group_array=two.grasp_group_array,min_width_index=mask)
    return plan


def worker(args):
    import open3d as o3
    import pybullet as p
    import pybullet_data
    from transforms3d.quaternions import quat2mat
    plan=json.loads(args.plan.read_text());all_slots=plan['required_slots'];asset_root=Path(plan['asset_root'])
    slots=[slot for index,slot in enumerate(all_slots) if index%args.workers==args.worker_index]
    scope={'np':np,'copy':copy,'json':json,'math':math,'os':os,'o3':o3,'o3d':o3,'quat2mat':quat2mat,
           'tqdm':lambda iterable:iterable,'time':time,'p':p}
    source=ROOT/'generate_mesh_and_pointcloud/recover_dh3_hand_to_stl.py'
    hand_scope=select(ROOT/'ur_toolbox/ur_toolbox/robot/DH3/DH3_grasp.py',{'grasp_types'},{})
    scope['grasp_types']=hand_scope['grasp_types']
    angle_scope=select(ROOT/'generate_mesh_and_pointcloud/angles2stl.py',{'DH3Angles2STLs'},dict(scope))
    scope['DH3Angles2STLs']=angle_scope['DH3Angles2STLs']
    original=select(source,{'joint_mesh_mapping','four_meta_to_matrix','get_mesh','save_stl_and_pointcloud','get_meshes'},scope)
    step_calls=[]
    def no_step(*unused,**kwargs):
        step_calls.append(1)
        raise RuntimeError('Physics integration prohibited during original FK asset generation')
    p.stepSimulation=no_step
    clients=[]
    def direct_open(urdf_path):
        client=p.connect(p.DIRECT);clients.append(client)
        assert client==0,'Each fresh asset worker must own a unique DIRECT client0'
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(solverResidualThreshold=0,maxNumCmdPer1ms=1000)
        p.setTimeStep(1./240)
        base=p.getQuaternionFromEuler([0,0,0])
        body=p.loadURDF(urdf_path,[0,0,0],base,useFixedBase=True)
        assert p.getNumJoints(body)==12
        return p,body
    original['open_pybullet']=direct_open
    save_original=original['save_stl_and_pointcloud']
    slots_by_name={(slot['pose'],slot['width_cm_key']):slot for slot in slots}
    finished=[]
    def seeded_save(name,angle,mesh,output_path):
        key=str(np.round(angle[12],1));slot=slots_by_name[(name,key)]
        o3.utility.random.seed(slot['sampling_seed'])
        before=time.monotonic()
        save_original(name,angle,mesh,output_path)
        files=[]
        for scale in range(1,6):
            path=asset_root/f'meshes/source_pointclouds/voxel_size_{scale}'/name/(key+'.ply')
            pointcloud=o3.io.read_point_cloud(str(path))
            array=np.asarray(pointcloud.points)
            if len(array)==0 or not np.isfinite(array).all():
                raise ValueError('Generated original collision PCL is empty or non-finite')
            files.append({'voxel_mm':scale,'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),
                          'point_count':len(array),'min_m':array.min(0).tolist(),'max_m':array.max(0).tolist()})
        stl=asset_root/'meshes/source'/name/(key+'.STL')
        receipt=dict(slot,worker_index=args.worker_index,mesh_triangles=len(mesh.triangles),mesh_vertices=len(mesh.vertices),
            output_mesh={'path':str(stl),'bytes':stl.stat().st_size,'sha256':sha(stl)},pointclouds=files,
            original_surface_samples=300000,original_infill_layers=15,all_five_original_voxel_scales_retained=True,
            generation_and_validation_seconds=time.monotonic()-before,physics_steps=0)
        (asset_root/'slot_receipts'/(name+'_'+key+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        finished.append(receipt)
        print(json.dumps({'worker':args.worker_index,'slot':name+'/'+key,'seconds':receipt['generation_and_validation_seconds'],'pcl3_points':files[2]['point_count']}),flush=True)
    original['save_stl_and_pointcloud']=seeded_save
    angles=[[] for unused in range(4)]
    for slot in slots:
        existing=asset_root/'slot_receipts'/(slot['pose']+'_'+slot['width_cm_key']+'.json')
        if existing.exists():
            receipt=json.loads(existing.read_text())
            assert all(Path(item['path']).exists() and sha(item['path'])==item['sha256'] for item in receipt['pointclouds'])
            finished.append(receipt);continue
        angles[slot['type']-1].append(slot['joint12_rad']+[float(slot['width_cm_key'])]+slot['actuator2'])
    begin=time.monotonic()
    try:
        if any(angles):
            original['get_meshes'](angles,str(ROOT/'generate_mesh_and_pointcloud/dh3_urdf/dh3_urdf/meshes')+'/',
                str(asset_root/'meshes')+'/',str(asset_root/'NOT_WRITTEN_LOOKUP.json'),
                str(ROOT/'generate_mesh_and_pointcloud/dh3_urdf/dh3_urdf/urdf/dh3_urdf.urdf'),if_source=False,vis=False)
    finally:
        for client in clients:
            if p.isConnected(client):p.disconnect(client)
    assert not step_calls
    report={'worker_index':args.worker_index,'slots':len(slots),'completed_slots':len(finished),'physics_steps':0,
        'pybullet_connection':'DIRECT process-private','open3d_version':o3.__version__,
        'worker_wall_seconds':time.monotonic()-begin,'per_slot_sampling_seed_recorded':True}
    (args.plan.parent/f'worker_{args.worker_index}_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--hand-run-dir',type=Path,default=ROOT/'outputs/official_dh3_hand_frame_20261103_v1')
    parser.add_argument('--table-dir',type=Path,default=Path('/mnt/public/datasets/AnyDexGrasp/source_tables/generate_mesh_and_pointcloud/dh3_urdf'))
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/official_dh3_collision_assets_20261103_v1')
    parser.add_argument('--asset-root',type=Path)
    parser.add_argument('--overlay',type=Path,default=ROOT/'outputs/native_deps/python_overlay')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--worker-index',type=int)
    parser.add_argument('--plan',type=Path)
    args=parser.parse_args()
    if not 1<=args.workers<=4:parser.error('One to four isolated processes only')
    if args.worker_index is not None:
        if args.overlay.exists():sys.path.insert(0,str(args.overlay))
        worker(args);return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    plan=make_plan(args)
    if args.plan_only:
        print(json.dumps({'status':'plan_prepared_only','slot_count':plan['slot_count'],'asset_root':plan['asset_root']},indent=2));return
    job=args.output_dir/'ASSET_PLAN.json'
    env=os.environ.copy();env['PYTHONPATH']=str(args.overlay)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='1'
    env['LD_LIBRARY_PATH']=str(args.overlay.parent/'os_libs')+(os.pathsep+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    def launch(index):
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker-index',str(index),'--workers',str(args.workers),'--plan',str(job),'--overlay',str(args.overlay)]
        with (args.output_dir/f'worker_{index}.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        return {'worker_index':index,'returncode':result.returncode,'command':command}
    begin=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes=list(pool.map(launch,range(args.workers)))
    results={'status':'complete' if all(row['returncode']==0 for row in outcomes) else 'worker_failure',
        'required_slots':plan['slot_count'],'workers':outcomes,'wall_seconds':time.monotonic()-begin,
        'physics_steps':0,'asset_root':plan['asset_root'],'full_original_sampling_retained':True}
    (args.output_dir/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
    if any(row['returncode'] for row in outcomes):raise RuntimeError('Original asset generation worker failed; logs preserved')


if __name__=='__main__':
    main()
