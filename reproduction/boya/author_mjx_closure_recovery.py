"""Test a single bounded resumption of closing when a latched grasp loses load."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np

import author_collection_batch as B
import author_collection_wave_resources as R
import author_mjx_closure_pair as closure
import author_mjx_tall_closure_replay as replay
from run_author_mjx_expansion import dispatch


PREVIOUS = B.ROOT/'outputs/boya_mjx_tall_closure_replay_20261119_v1'
RECOVERY = dict(loss_threshold_n=.5, loss_controls=4, only_during_close=True, max_resumptions=1,
                source_close_rows_per_control=1, keep_last_position_at_hold_and_lift=True)


class RecoveringHold(closure.ClosureHold):
    def __init__(self, columns, script):
        super().__init__(columns)
        self.nominal = script['joint_target'][:,columns]
        self.close_end = int(np.flatnonzero(script['phase']=='close')[-1])+1
        self.step = 0
        self.lost_steps = 0

    def apply(self, target):
        self.step += 1
        if self.trigger and 'resume_after_step' in self.trigger:
            progress = min(self.step,self.close_end)-self.trigger['resume_after_step']
            self.target = self.nominal[self.trigger['step']-1+progress]
        super().apply(target)

    def observe(self, row, applied):
        assert row['step']==self.step
        super().observe(row,applied)
        if (row['phase']!='close' or self.step>=self.close_end or self.trigger is None
                or 'resume_after_step' in self.trigger or self.step<=self.trigger['step']):
            return
        threshold = RECOVERY['loss_threshold_n']
        bearing = any(row['objects'][obj]['finger_loads_n'][0]>=threshold
                      and sum(row['objects'][obj]['finger_loads_n'][1:])>=threshold for obj in self.trigger['objects'])
        self.lost_steps = 0 if bearing else self.lost_steps+1
        if self.lost_steps==RECOVERY['loss_controls']:
            self.trigger['resume_after_step'] = self.step


def freeze(root):
    root.mkdir(parents=True,exist_ok=False)
    previous = B.read(PREVIOUS/'FINAL_REVIEW.json')
    assert previous['status']=='verified_complete'
    B.verify({'comparison':previous['comparison'],'loads':previous['load_review']})
    p = B.read(PREVIOUS/'PROTOCOL.json')
    assert p['environments']==4 and p['physical_attempts']==4 and p['policy']==closure.POLICY
    p.update(schema='anydex_mjx_bounded_closure_recovery_v1',recovery_policy=RECOVERY)
    p['inputs'].update(static_comparison=previous['comparison'],static_loads=previous['load_review'],
                       previous_protocol=B.binding(PREVIOUS/'PROTOCOL.json'))
    p['sources'][Path(__file__).name] = B.binding(__file__)
    B.verify(p['inputs']); B.verify(p['sources'])
    archive = root/'source_archive'
    archive.mkdir()
    for name,ref in p['sources'].items():
        shutil.copyfile(ref['path'],archive/name)
    B.save(root/'PROTOCOL.json',p,exclusive=True)
    ready = dict(status='frozen',protocol=B.binding(root/'PROTOCOL.json'),targets=[case['target'] for case in p['cases']],
                 max_grasps=p['physical_attempts'],max_native_environment_steps=p['max_native_environment_steps'],video_recording=False)
    B.save(root/'READY.json',ready,exclusive=True)
    print(ready,flush=True)


def recovery_motion(batch, root, cases, columns):
    scripts = iter(closure.arrays(case['commands']['path']) for case in cases)

    def controller(columns):
        return RecoveringHold(columns,next(scripts))

    with patch.object(closure,'ClosureHold',controller):
        closure.motion(batch,root,cases,columns)


def collect(root):
    protocol = root/'PROTOCOL.json'
    p = B.read(protocol)
    assert p['recovery_policy']==RECOVERY
    marker = B.sha(protocol)
    if os.environ.get('ANYDEX_MJX_TALL_REPLAY_ENV')!=marker:
        env = B._environment(B.read(p['inputs']['original_protocol']['path']))
        frozen = replay.F
        env.update(PYTHONPATH=os.pathsep.join((str(frozen.WARP),str(frozen.DEPS),env['PYTHONPATH'])),
                   CUDA_VISIBLE_DEVICES=frozen.GPU,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   WARP_CACHE_PATH=str(root/'warp_cache'),ANYDEX_MJX_TALL_REPLAY_ENV=marker)
        os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),'collect',str(root)],env)
    with patch.object(replay,'motion',recovery_motion):
        replay.collect(root)


def validate_branch(case, root, mode, columns, script, speed):
    """Independently reconstruct each motor target and both transition steps."""
    import author_mujoco_control as C
    from author_mujoco_sample import classify

    folder = root/f"case_{case['index']:02d}"/mode
    result = B.read(folder/'RESULTS.json')
    outcome = B.read(folder/'OUTCOME.json')
    assert outcome['result']==B.binding(folder/'RESULTS.json') and outcome['decision']==classify(result)
    B.verify({'trace':result['trace']})
    with Path(result['trace']['path']).open() as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows)==result['steps']
    first = case['close_at']
    end = int(np.flatnonzero(script['phase']=='close')[-1])+1
    trigger = None
    recover = None
    if mode=='contact_hold':
        # Start and loss are found in separate passes, using fixed windows.
        initial_window = closure.POLICY['consecutive_close_controls']
        for i in range(first+initial_window-1,min(end,len(rows))):
            sustained = set.intersection(*(closure.opposing_objects(r) for r in rows[i-initial_window+1:i+1]))
            if sustained:
                trigger = dict(step=i+1,objects=sorted(sustained))
                break
        if trigger:
            failed = []
            for row in rows[trigger['step']:min(end-1,len(rows))]:
                load = [row['objects'][obj]['finger_loads_n'] for obj in trigger['objects']]
                threshold = RECOVERY['loss_threshold_n']
                failed.append(all(v[0]<threshold or sum(v[1:])<threshold for v in load))
                n = RECOVERY['loss_controls']
                if len(failed)>=n and all(failed[-n:]):
                    recover = row['step']
                    trigger['resume_after_step'] = recover
                    break
    assert trigger==outcome['trigger']
    advanced = 0
    for i,row in enumerate(rows):
        step = i+1
        assert row['step']==step and row['phase']==script['phase'][i]
        wanted = script['joint_target'][i].copy()
        if trigger and step>trigger['step']:
            count = max(0,min(step,end)-recover) if recover is not None else 0
            assert count>=advanced and count-advanced<=1
            advanced = count
            wanted[columns] = script['joint_target'][trigger['step']-1+count,columns]
        np.testing.assert_array_equal(row['joint_target'],wanted)
        np.testing.assert_array_equal(row['velocity_target'],speed[i])
        np.testing.assert_array_equal(row['applied_position_target_float32'],wanted.astype(np.float32))
        np.testing.assert_array_equal(row['applied_velocity_target_float32'],speed[i].astype(np.float32))
    if result['status']=='complete':
        assert C.LIFT.evaluate_lift(rows,list(rows[0]['objects']),.004)==result['evaluation']
    target = result['evaluation'].get('objects',{}).get(case['target'],{})
    lifts = [dict(object=obj['object_id'],rise_m=obj['final_reference_rise_m'])
             for obj in result['evaluation'].get('object_outcomes',[]) if obj['task_success']]
    return dict(status=outcome['decision']['status'],label=outcome['decision']['label'],trigger=trigger,
        resumed_source_close_controls=advanced,target_lifted=outcome['decision']['label']==1 and bool(target.get('task_success')),
        target_rise_m=target.get('final_reference_rise_m'),loaded_lift_s=target.get('trailing_observed_pickup_run_s'),lifts=lifts,
        strict_1mm_pass=result['gates']['penetration'],penetration_m=result['maxima']['penetration_m'],
        arm_wrist_velocity_and_time_exact=True,trace=result['trace'])


def audit_pair(case,root,columns):
    import author_mujoco_control as C

    script = closure.arrays(case['commands']['path'])
    joints = script['joint_names'].tolist()
    speed = C.source_velocity_targets(script['joint_target'],joints,joints[:6])
    receipt = root/f"case_{case['index']:02d}"/'PRE_CLOSE.json'
    if receipt.exists():
        state = B.read(receipt)
        B.verify({'state':state['state']})
        assert state['tick']==case['close_at'] and state['physics_steps_during_copy']==0
        for name,value in closure.arrays(state['state']['path']).items():
            np.testing.assert_array_equal(value[0],value[1],err_msg=name)
    branches = {mode:validate_branch(case,root,mode,columns,script,speed) for mode in ('original','contact_hold')}
    return dict(index=case['index'],preclose_state_equality_verified=receipt.exists(),branches=branches)


def audit(root):
    with patch.object(replay,'audit_pair',audit_pair):
        replay.audit(root)


def summarize(root):
    p = B.read(root/'PROTOCOL.json')
    result_ref = B.binding(root/'PAIR_REVIEW.json')
    results = B.read(result_ref['path'])
    assert results['status']=='verified_complete'
    previous = {case['target']:case for case in B.read(p['inputs']['static_comparison']['path'])['cases']}
    final = []
    for case in results['cases']:
        original = case['branches']['original']
        recovered = case['branches']['contact_hold']
        static = previous[case['target']]['branches']['contact_hold']
        final.append(dict(target=case['target'],original=original,earlier_static_hold=static,recovering_hold=recovered,
            preclose_state_equality_verified=case['preclose_state_equality_verified'],
            confirmed_recovery=case['original_target_reproduced'] and case['preclose_state_equality_verified']
                 and not static['target_lifted'] and recovered['target_lifted']))
    stages = B.read(root/'PIPELINE_FINISHED.json')
    assert stages['result']==result_ref
    assert all(stages[key]['process_group_exited'] and not R.group_alive(stages[key]['process']['pgid']) for key in ('run','audit'))
    report = dict(status='verified_complete',paired_results=result_ref,policy=RECOVERY,cases=final,
        original_pose_restore=results['restore_scope'],native_environment_steps=results['native_environment_steps'],
        unknown_native_steps=results['unknown_native_steps'],runtime_wall_s=stages['run']['wall_s'],audit_wall_s=stages['audit']['wall_s'],
        exited_process_groups=2,video_recording=False,training_updates=0,exclude_training=True)
    B.save(root/'RECOVERY_REVIEW.json',report,exclusive=True)
    print(dict(status=report['status'],cases=[(v['target'],v['original']['target_lifted'],v['recovering_hold']['target_lifted'],
          v['recovering_hold']['trigger']) for v in final],native_environment_steps=report['native_environment_steps']),flush=True)


def run(root):
    assert B.read(root/'READY.json')['protocol']==B.binding(root/'PROTOCOL.json')
    if subprocess.check_output(['git','status','--porcelain'],cwd=B.ROOT).strip():
        raise ValueError('The frozen comparison expects a committed clean tree')
    B.save(root/'PIPELINE_STARTED.json',dict(git=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=B.ROOT,text=True).strip()),exclusive=True)
    shutil.copytree(PREVIOUS/'warp_cache',root/'warp_cache')
    entry=[sys.executable,'-B',str(Path(__file__).resolve())]
    live=dispatch(entry+['collect',str(root)],root/'timing',1500)
    checked=dispatch(entry+['audit',str(root)],root/'audit_timing',900)
    B.save(root/'PIPELINE_FINISHED.json',dict(run=live,audit=checked,result=B.binding(root/'PAIR_REVIEW.json')),exclusive=True)
    summarize(root)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run','collect','audit','summarize'))
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    {'freeze':freeze,'run':run,'collect':collect,'audit':audit,'summarize':summarize}[args.action](args.root.resolve())
