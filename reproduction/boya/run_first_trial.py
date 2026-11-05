"""One bound AnyDex -> Boya -> existing Isaac complete-arm collection smoke.

Candidate source is frozen before execution. At most four original arm plans
are considered; only the first admitted plan executes, once. Preparation and
infrastructure failures retain null labels. No successful legacy grasp is used.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from asset_adapter import sha
from trial_record import prepare_binding,write_bound_commands,collect_record

ROOT=Path(__file__).resolve().parents[2]


def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--backend-config',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--candidate-dir',type=Path)
    p.add_argument('--observation',type=Path,help='Frozen shared observation JSON; default is run/observation/observation.json')
    p.add_argument('--source-npz',type=Path,help='Frozen paired raw CGR candidates/features; default is run/representation/raw_camera_candidates.npz')
    p.add_argument('--candidate-indices',type=int,nargs='+',help='Explicit ordered subset of this frozen pool; no pool mutation')
    p.add_argument('--max-arm-plans',type=int,default=4)
    a=p.parse_args();run=a.run_dir.resolve();config=json.loads(a.backend_config.read_text())
    pool=a.candidate_dir.resolve() if a.candidate_dir else run/'candidates';gen=json.loads((pool/'generation.json').read_text())
    observation=a.observation.resolve() if a.observation else run/'observation/observation.json'
    source_npz=a.source_npz.resolve() if a.source_npz else run/'representation/raw_camera_candidates.npz'
    obs=json.loads(observation.read_text())
    indices=gen['selected_indices'] if a.candidate_indices is None else a.candidate_indices
    if a.max_arm_plans < 1 or len(set(indices))!=len(indices) or any(i not in gen['selected_indices'] for i in indices):
        raise ValueError('Positive plan budget and unique frozen pool indices required')
    if (run/'trial_summary.json').exists():raise ValueError('Retain previous attempts; use a new run directory')
    shutil.copyfile(config['arm_collision_refinement'],run/'arm_collision_refinements.json')
    env=os.environ.copy();env.update(CANONDEX_COLLISION_PARTITION_CACHE=str(ROOT/'outputs/boya_collision_partition_cache'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    summary=dict(status='preparing',plans=[],actual_grasp_attempts=0,training_labels=0,
        simulator='Isaac Sim / PhysX',grasp_frontend='AnyDex published CGR + new Boya geometry table',
        backend='Existing complete UR16e+Boya native joint-drive executor',
        legacy_experience_or_grasp_sampler_used=False,learned_Boya_head_loaded=False,
        source_observation_sha256=sha(observation),source_npz_sha256=sha(source_npz),
        declared_candidate_indices=indices[:a.max_arm_plans])
    save(run/'trial_summary.json',summary);start=time.monotonic()
    for index in indices[:a.max_arm_plans]:
        plan_dir=run/f'plan_{index:03d}'
        common=['--robot-urdf',config['robot_urdf'],'--reference-root',config['reference_root'],
            '--candidates',str(pool),'--index',str(index),'--object-root',obs['object_root'],
            '--world-object-pose',str(pool/'world_object_pose.json'),'--pregrasp-distance-m','.1','--device','cpu']
        if config.get('restore_object_initial_state',False):
            common.append('--restore-object-initial-state')
        for key in ['hand_urdf','hand_scene','package','physics_config','diagnostic_config']:
            common.extend(['--'+key.replace('_','-'),config[key]])
        cmd=[sys.executable,'-B',str(Path(__file__).with_name('native_arm.py')),
            '--stage','prepare','--output-dir',str(plan_dir),'--arm-select-min-motion','--vectorized-arm-broadphase',*common]
        with (run/f'prepare_{index:03d}.log').open('w') as log:
            proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=360)
        plan=json.loads((plan_dir/'plan.json').read_text()) if (plan_dir/'plan.json').exists() else {}
        row=dict(index=index,returncode=proc.returncode,plan_dir=str(plan_dir),admitted=plan.get('admitted',False),training_label=None)
        summary['plans'].append(row);save(run/'trial_summary.json',summary)
        print('BOYA_ARM_PLAN '+json.dumps(row),flush=True)
        if proc.returncode or not plan.get('admitted'):
            continue
        meta=gen['records'][index]
        with np.load(pool/'candidates.npz',allow_pickle=False) as z:
            spec=dict(world_T_palm=z['proposal_palm'][index].tolist(),
                open_motors=z['proposal_q'][index].tolist(),closed_motors=z['proposal_q_close'][index].tolist(),
                motor_names=z['independent_motor_names'].tolist(),world_T_object=z['world_T_object'].tolist(),
                backend_config_sha256=sha(a.backend_config),candidate_pool_sha256=sha(pool/'candidates.npz'),
                source_observation_sha256=summary['source_observation_sha256'])
        binding=prepare_binding(source_npz,meta['source_candidate_index'],
            ROOT/'robot_dh3.py',gesture_id=gen['gesture_id'],aperture_id=meta['source_slot'],
            aperture_sha256=sha(a.calibration),calibrated_width_m=meta['actual_reference_width_m'],
            extra_depth_index=meta['additional_depth_index'],execution_spec=spec,if_flip=meta['if_flip'])
        save(run/'trial_binding.json',binding)
        commands=plan_dir/'commands.npz'
        with np.load(commands,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        commands_sha=write_bound_commands(plan_dir/'bound_commands.npz',arrays,binding)
        with np.load(plan_dir/'bound_commands.npz',allow_pickle=False) as z:
            if not all(np.array_equal(arrays[k],z[k]) for k in arrays):raise ValueError('Metadata binding changed motion commands')
        (plan_dir/'bound_commands.npz').replace(commands)
        plan['commands_sha256']=commands_sha;plan['trial_binding_sha256']=binding['trial_binding_sha256']
        plan['added_metadata_only_original_control_arrays_unchanged']=True
        save(plan_dir/'plan.json',plan)
        execution=run/'execution'
        cmd=[sys.executable,'-B',str(Path(__file__).with_name('native_arm.py')),
            '--stage','execute','--output-dir',str(execution),'--plan-dir',str(plan_dir),*common]
        summary.update(status='executing',bound_commands_sha256=commands_sha,selected_candidate=meta)
        save(run/'trial_summary.json',summary)
        with (run/'execution.log').open('w') as log:
            proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        ev_path=execution/'evaluation.json'
        if not ev_path.exists():
            summary.update(status='execution_infrastructure_error',returncode=proc.returncode,training_label=None)
            break
        ev=json.loads(ev_path.read_text())
        if ev.get('commands_sha256',commands_sha)!=commands_sha:raise ValueError('Native execution changed command binding')
        steps=ev.get('total_physics_steps',0)
        phases=[]
        if (execution/'trace.npz').exists():
            with np.load(execution/'trace.npz',allow_pickle=False) as z:
                phases=list(dict.fromkeys(str(x) for x in z['phase']))
        eligible=bool(proc.returncode==0 and ev.get('complete') is True and steps==len(arrays['phase']) and 'lift' in phases)
        bound_eval=dict(ev,trial_binding_sha256=binding['trial_binding_sha256'],commands_sha256=commands_sha,
            completed_phases=phases,label_eligible=eligible,backend='Isaac Sim / PhysX',
            original_native_evaluation_sha256=sha(ev_path),
            success_definition='Existing native tabletop-lift task_success; not author real transport-to-bin success')
        save(run/'bound_evaluation.json',bound_eval)
        record=collect_record(binding,run/'bound_evaluation.json',commands)
        save(run/'training_record.json',record)
        summary.update(status='complete' if eligible else 'execution_not_label_eligible',
            actual_grasp_attempts=int(steps>0),physics_steps=steps,training_labels=int(record['label_eligible']),
            training_label=record['result'],outcome_status=record['outcome_status'],
            actual_task_success=ev.get('task_success'),execution_returncode=proc.returncode,
            original_native_evaluation_sha256=sha(ev_path))
        break
    else:
        summary.update(status='no_full_arm_plan_in_declared_budget',training_label=None)
    summary['elapsed_wall_s']=time.monotonic()-start
    save(run/'trial_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
