"""Replay two archived tall-object grasps while varying only finger closure."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

import author_collection_batch as B
import author_collection_wave_resources as R
import author_mjx_batch_collection_v9 as F
from author_mjx_closure_pair import POLICY, WRISTS, arrays, audit_pair, motion
from author_mjx_fresh_closure_pair import audit_numerics
from author_mjx_shared_pair import STATE, integration_state
from run_author_mjx_expansion import dispatch


SOURCE = B.ROOT/'outputs/boya_mjx_cgr_to100_20261119_v1/collection'
COVERAGE = B.HERE/'evidence/author_mjx_cross_object_closure_20261119_v1/COVERAGE_REVIEW.json'
TARGETS = ('graspnet_007','graspnet_044')
DOF_WIDTHS = {'free':(7,6), 'ball':(4,3), 'slide':(1,1), 'hinge':(1,1)}


def model_addresses(path):
    """Read declared joint order without compiling or loading the old asset bank."""
    sections = []
    joints, actuator_names = {}, []
    nq = nv = 0
    for event, node in ET.iterparse(path,events=('start','end')):
        if event == 'start':
            sections.append(node.tag)
            if len(sections)>2 and sections[1]=='worldbody' and node.tag in ('joint','freejoint'):
                name = node.attrib['name']
                kind = 'free' if node.tag == 'freejoint' else node.get('type','hinge')
                qwidth,vwidth = DOF_WIDTHS[kind]
                if name in joints:
                    raise ValueError('Duplicate source joint '+name)
                joints[name] = (nq,nv,qwidth,vwidth)
                nq += qwidth
                nv += vwidth
            if len(sections)==3 and sections[1]=='actuator':
                actuator_names.append(node.attrib['name'])
        else:
            sections.pop()
            node.clear()
    return joints,actuator_names,nq,nv


def native_frame(segment, tick, lane, wave_start):
    with Path(segment).open() as stream:
        for row_index,line in enumerate(stream,wave_start+1):
            if row_index != tick:
                continue
            row = json.loads(line)
            if row['tick'] != tick or row['overflow'][lane]:
                raise ValueError('Saved source frame is missing or overflowed')
            return {key:np.asarray(row[key][lane],np.float32) for key in ('qpos','qvel','ctrl','native_time_s')}
    raise ValueError('The archived wave contains no frame at the recorded capture tick')


def extract_state(root, index, source, layout, waves):
    from scipy.spatial.transform import Rotation

    record = B.read(source['record']['path'])
    refs = record['bindings']
    B.verify({key:refs[key] for key in ('observation','execution','commands','scene','candidate','trace')})
    assert record['eligible'] and record['label']==1
    live_path = Path(refs['observation']['path']).parent/'LIVE_STATE.json'
    observation = B.read(refs['observation']['path'])
    live = B.read(live_path)
    tick,lane = observation['world_step'],observation['lane']
    assert live['scene_id']==observation['scene_id'] and live['sim_step']==tick
    wave = next(w for w in waves if w['start_world_tick'] < tick <= w['world_tick'])
    archive = Path(wave['audit_snapshot'])/'READ_ONLY_AUDIT.json'
    original_audit = B.read(archive)
    assert original_audit['status']=='read_only_audit_pass'
    B.verify({'wave':wave['segment']})
    frame = native_frame(wave['segment']['path'],tick,lane,wave['start_world_tick'])
    addresses,actuators,nq,nv = layout
    assert frame['qpos'].shape==(nq,) and frame['qvel'].shape==(nv,) and frame['ctrl'].shape==(len(actuators),)
    names = live['joint_names']
    measured = {key:frame[key][[addresses[name][column] for name in names]]
                for key,column in (('qpos',0),('qvel',1))}
    assert all(addresses[name][2:]==(1,1) for name in names)
    np.testing.assert_array_equal(measured['qpos'],live['joint_pos'])
    np.testing.assert_array_equal(measured['qvel'],live['joint_velocity'])
    object_names = observation['object_ids']
    scene = B.read(refs['scene']['path'])
    assert object_names == [obj['object_id'] for obj in scene['objects']]
    poses,velocities = [],[]
    for name in object_names:
        q,v,n,m = addresses[name+'_free']
        assert (n,m)==(7,6)
        pose = frame['qpos'][q:q+n]
        measured_pose = np.asarray(live['objects'][name]['pose_xyzw'])
        np.testing.assert_allclose(pose[:3],measured_pose[:3],rtol=0,atol=1e-7)
        np.testing.assert_allclose(Rotation.from_quat(pose[3:],scalar_first=True).as_matrix(),
                                  Rotation.from_quat(measured_pose[3:]).as_matrix(),rtol=0,atol=2e-7)
        poses.append(pose)
        velocities.append(frame['qvel'][v:v+m])
    state = root/f'SOURCE_STATE_{index:02d}.npz'
    np.savez_compressed(state,joint_names=names,qpos=measured['qpos'],qvel=measured['qvel'],object_ids=object_names,
                        object_qpos=poses,object_qvel=velocities,actuator_names=actuators,ctrl=frame['ctrl'])
    start = root/f'REPLAY_START_{index:02d}.json'
    B.save(start,dict(world_step=0,native_time_s=0,source_world_step=tick,source_native_time_s=float(frame['native_time_s']),
        source_lane=lane,source_state=B.binding(state),source_observation=refs['observation'],source_live_state=B.binding(live_path),
        source_wave=wave['segment'],source_wave_audit=B.binding(archive),
        source_qpos_qvel_ctrl_restored=True,source_solver_warmstart_available=False,solver_warmstart_initialized_zero=True),exclusive=True)
    return start,state,refs


def freeze(root):
    root.mkdir(parents=True,exist_ok=False)
    evidence = B.read(COVERAGE)
    source = B.read(SOURCE/'PROTOCOL.json')
    original_model = B.read(SOURCE/'MODEL_RECEIPT.json')['model']
    waves_file = SOURCE/'WAVES.json'
    inputs = dict(coverage=B.binding(COVERAGE),original_model=original_model,original_waves=B.binding(waves_file),
        original_protocol=B.binding(SOURCE/'PROTOCOL.json'),template=source['inputs']['template:0'],backend=source['inputs']['backend'])
    B.verify(inputs)
    waves = B.read(waves_file)
    layout = model_addresses(original_model['path'])
    cases = []
    for i,target in enumerate(TARGETS):
        ref = evidence['archived_tall_object_positive_controls'][target]
        B.verify({key:ref[key] for key in ('record','commands','execution','scene')})
        start,state,bindings = extract_state(root,i,ref,layout,waves)
        execution = B.read(bindings['execution']['path'])
        old_lifts = [item for item in execution['evaluation']['object_outcomes'] if item['task_success']]
        assert [item['object_id'] for item in old_lifts]==[target]
        commands = arrays(bindings['commands']['path'])
        close_at = int(np.flatnonzero(commands['phase']=='close')[0])
        assert execution['steps']==len(commands['phase']) and close_at>50
        case = dict(index=i,target=target,original_label=1,historical_rise_m=old_lifts[0]['final_reference_rise_m'],
                    close_at=close_at,motion_controls=len(commands['phase']),state=B.binding(state),observation=B.binding(start),
                    **{key:bindings[key] for key in ('commands','scene','execution','candidate','trace')})
        cases.append(case)
        inputs.update({f'case:{i}:{key}':case[key] for key in ('commands','scene','execution','candidate','trace','state','observation')})
    sources = dict(B.read(B.ROOT/'outputs/boya_mjx_cross_object_closure_20261119_v1/collection/PROTOCOL.json')['sources'])
    sources[Path(__file__).name]=B.binding(__file__)
    B.verify(sources)
    for name,value in sources.items():
        destination=root/'source_archive'/name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(value['path'],destination)
    max_ticks=max(case['motion_controls'] for case in cases)
    p=dict(schema='anydex_mjx_tall_closure_replay_v1',cases=cases,inputs=inputs,sources=sources,policy=POLICY,environments=4,
        max_ticks=max_ticks,max_native_environment_steps=max_ticks*4*8,physical_attempts=4,
        restore_scope='Saved capture qpos/qvel/ctrl in reduced object bank, native clock rebased and solver warmstart initialized zero',
        video_recording=False,training_updates=0,new_CGR_forwards=0,exclude_training=True)
    B.save(root/'PROTOCOL.json',p,exclusive=True)
    B.save(root/'READY.json',dict(status='frozen',protocol=B.binding(root/'PROTOCOL.json'),objects=list(TARGETS),
        max_grasps=4,max_native_environment_steps=p['max_native_environment_steps']),exclusive=True)
    print(B.read(root/'READY.json'),flush=True)


def restore(batch, cases):
    import mujoco

    for case in cases:
        scene=B.read(case['scene']['path'])
        for lane in (2*case['index'],2*case['index']+1):
            batch.exchange(lane,scene)
    before = integration_state(batch)
    assert not np.any(before['time']) and not np.any(before['qacc_warmstart'])
    assert not np.any(before['act']) and not np.any(before['history'])
    expected = {name:value.copy() for name,value in before.items()}
    actuators = [mujoco.mj_id2name(batch.model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(batch.model.nu)]
    for case in cases:
        state = arrays(case['state']['path'])
        assert state['actuator_names'].tolist()==actuators
        assert state['joint_names'].tolist()==list(batch.base.planning.joints)
        for lane in (2*case['index'],2*case['index']+1):
            expected['qpos'][lane,batch.base.qindices]=state['qpos']
            expected['qvel'][lane,batch.base.vindices]=state['qvel']
            expected['ctrl'][lane]=state['ctrl']
            for obj,pose,velocity in zip(state['object_ids'],state['object_qpos'],state['object_qvel'],strict=True):
                a,b=batch.base.addresses[obj]
                expected['qpos'][lane,a:a+7]=pose
                expected['qvel'][lane,b:b+6]=velocity
    with batch.wp.ScopedStream(batch.core.stream):
        for key in ('qpos','qvel','ctrl'):
            value=getattr(batch.gpu,key)
            pointer=value.ptr
            value.assign(np.ascontiguousarray(expected[key]))
            assert pointer==value.ptr
        batch.wp.synchronize_stream(batch.core.stream)
    for key,value in integration_state(batch).items():
        np.testing.assert_array_equal(value,expected[key],err_msg=key)
    for i in range(batch.count):
        batch._sync_lane(i,expected['qpos'][i],expected['qvel'][i],expected['time'][i])
    return expected


def collect(root):
    pfile=root/'PROTOCOL.json'
    p=B.read(pfile)
    marker=B.sha(pfile)
    if os.environ.get('ANYDEX_MJX_TALL_REPLAY_ENV')!=marker:
        env=B._environment(B.read(p['inputs']['original_protocol']['path']))
        env.update(PYTHONPATH=os.pathsep.join((str(F.WARP),str(F.DEPS),env['PYTHONPATH'])),CUDA_VISIBLE_DEVICES=F.GPU,
            JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',WARP_CACHE_PATH=str(root/'warp_cache'),
            ANYDEX_MJX_TALL_REPLAY_ENV=marker)
        os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),'collect',str(root)],env)
    from author_resident_prepare import bind_backend

    B.verify(p['inputs']); B.verify(p['sources'])
    assert p['policy']==POLICY
    bind_backend(dict(backend=p['inputs']['backend']['path']))
    from author_compiled_kinematics import CompiledKinematics
    from author_mjx_native_batch import Batch

    template=B.read(p['inputs']['template']['path'])
    B.verify({key:template[key] for key in ('plan','commands')})
    initial=arrays(template['commands']['path'])
    names,home=initial['joint_names'].tolist(),initial['initial_position']
    plan=B.read(template['plan']['path'])
    cases=p['cases']
    scenes=[B.read(case['scene']['path']) for case in cases]
    lock=B.ROOT/'outputs/author_collection_gpu0.lock'
    began=time.monotonic()
    with R.lease(root/'DRIVER.lock'),R.lease(lock),R.active_run(root,lock.with_suffix('.wave_active.json'),B.binding(pfile)):
        B.save(root/'RUN_STARTED.json',dict(process=R.process_identity(os.getpid())),exclusive=True)
        with CompiledKinematics().installed():
            batch=Batch(plan,names,home,scenes,count=p['environments'],max_ticks=p['max_ticks'])
            try:
                batch.set_cost_journal(root/'NATIVE_CALLS.jsonl')
                for case in cases:
                    assert arrays(case['commands']['path'])['joint_names'].tolist()==names
                start=restore(batch,cases)
                np.savez_compressed(root/'RESTORED_STATE.npz',**start)
                columns=[i for i,name in enumerate(names) if name not in batch.base.arm_names and name not in WRISTS]
                assert len(columns)==16
                motion(batch,root,cases,columns)
                B.save(root/'RUNTIME.json',dict(status='complete',world=batch.receipt(),finger_columns=columns,
                    restored=B.binding(root/'RESTORED_STATE.npz'),elapsed_s=time.monotonic()-began),exclusive=True)
            finally:
                batch.close()


def audit(root):
    from author_resident_prepare import bind_backend

    p=B.read(root/'PROTOCOL.json')
    B.verify(p['inputs']); B.verify(p['sources'])
    B.verify({name:dict(path=str(root/'source_archive'/name),sha256=ref['sha256']) for name,ref in p['sources'].items()})
    bind_backend(dict(backend=p['inputs']['backend']['path']))
    from author_compiled_kinematics import CompiledKinematics
    from author_mjx_live_batch_v4 import clock_table
    from audit_author_mujoco_collection import model_for

    result=B.read(root/'RUNTIME.json')
    assert result['status']=='complete'
    B.verify({'restored':result['restored']})
    restored=arrays(result['restored']['path'])
    assert set(restored)==set(STATE)
    clocks=clock_table(p['max_ticks'])
    template=B.read(p['inputs']['template']['path'])
    plan=B.read(template['plan']['path'])
    planning,reference=model_for(template)
    checked=[]
    with CompiledKinematics().installed():
        for case in p['cases']:
            index=case['index'];target=case['target']
            for name,values in restored.items():
                np.testing.assert_array_equal(values[2*index],values[2*index+1],err_msg=name)
            review=audit_pair(case,root,result['finger_columns'])
            audit_numerics(case,root,plan,planning,reference,clocks)
            for branch in review['branches'].values():
                branch['target_lifted']=branch['label']==1 and any(obj['object']==target for obj in branch['lifts'])
            review.update(target=target,historical_rise_m=case['historical_rise_m'],
                original_target_reproduced=review['branches']['original']['target_lifted'],source_numerics_recomputed=True)
            checked.append(review)
    per_tick=p['environments']*8
    rows=0
    with (root/'NATIVE_CALLS.jsonl').open() as stream:
        for rows,line in enumerate(stream,1):
            assert json.loads(line)==dict(completed_native_environment_steps=((rows-1)//2)*per_tick,
                pending_native_environment_steps_upper=per_tick if rows%2==0 else 0)
    ticks=result['world']['global_control_ticks']
    assert rows==2*ticks+1
    steps=ticks*per_tick
    assert steps==result['world']['native_environment_integrations']<=p['max_native_environment_steps']
    with R.lease(root/'DRIVER.lock'),R.lease(B.ROOT/'outputs/author_collection_gpu0.lock'):
        assert B.read(B.ROOT/'outputs/author_collection_gpu0.wave_active.json')['state']=='released'
    reviewed=dict(status='verified_complete',cases=checked,native_environment_steps=steps,unknown_native_steps=0,
        restore_scope=p['restore_scope'],no_new_planning=True,video_recording=False,training_updates=0,exclude_training=True)
    B.save(root/'PAIR_REVIEW.json',reviewed,exclusive=True)
    print(reviewed,flush=True)


def run(root):
    assert B.read(root/'READY.json')['protocol']==B.binding(root/'PROTOCOL.json')
    if subprocess.check_output(['git','status','--porcelain'],cwd=B.ROOT).strip():
        raise ValueError('The frozen comparison expects a committed clean tree')
    B.save(root/'PIPELINE_STARTED.json',dict(git=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=B.ROOT,text=True).strip()),exclusive=True)
    cache=B.ROOT/'outputs/boya_mjx_cross_object_closure_20261119_v1/collection/warp_cache'
    shutil.copytree(cache,root/'warp_cache')
    entry=[sys.executable,'-B',str(Path(__file__).resolve())]
    live=dispatch(entry+['collect',str(root)],root/'timing',1500)
    checked=dispatch(entry+['audit',str(root)],root/'audit_timing',900)
    B.save(root/'PIPELINE_FINISHED.json',dict(run=live,audit=checked,result=B.binding(root/'PAIR_REVIEW.json')),exclusive=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run','collect','audit'))
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    {'freeze':freeze,'run':run,'collect':collect,'audit':audit}[args.action](args.root.resolve())
