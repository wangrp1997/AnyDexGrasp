"""One prelift contact-load ablation; accepted controller and commands unchanged.

The banana is removed before the first step; table, gravity, hand self collisions,
script, torque limits and transmission are retained. This is not a grasp trial.
"""
from pathlib import Path
import argparse,dataclasses,hashlib,json,shutil,sys,time,traceback
import numpy as np
ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
sys.path[:0]=[str(ROOT/'reproduction/simulation'),str(ROOT/'reproduction/runtime'),str(ROOT/'outputs/native_deps/python_overlay')]
import dh3_time_refinement_smoke as accepted
from dh3_time_refinement_smoke import CoupledDH3,PhysicsAssumptions,p,sha,smooth
import mesh_scene


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args();out=args.output_dir.absolute()
 if out.exists():raise FileExistsError('Keep all trials; use another directory')
 out.mkdir(parents=True);source=ROOT/'outputs/dh3_real_mesh_generalization_20261103_v1/run_nonapple_v2/physics/banana_yaw000'
 original=json.loads((source/'CONTROLLER.json').read_text());observation=json.loads(Path(original['scene_manifest']).read_text())
 assert sha(accepted.__file__)==original['controller_script_sha256']
 a=PhysicsAssumptions(dt=1/1920,non_contact_inner_iterations=20)
 assert json.loads(json.dumps(dataclasses.asdict(a)))==original['assumptions_SI_not_hardware_identification']
 plan=dict(scope='Exactly one no-banana prelift load ablation, not a grasp evaluation',source_trial=str(source),source_controller_sha256=sha(source/'CONTROLLER.json'),source_trace_sha256=sha(source/'trace.npz'),source_code_sha256=sha(__file__),accepted_controller_sha256=sha(accepted.__file__),scene_manifest_sha256=sha(original['scene_manifest']),intervention='After zero-step scene identity verification and original-ID hand creation remove only object body before any integration',retained=['table','gravity','self collision','two motors','nine ideal gear constraints','time step','solver settings','original hand pose and complete prelift control script'],duration_s=6.25,expected_actual_steps=12000,object_state_after_first_step_changes_forbidden=True,stopping_rule='Stop at 6.25s before lift; no rerun, retuning or object replacement',measured_comparison='Does shared flexion pass the previously blocked 0.395rad? Compare original stored first12000 control targets exactly.',no_grasp_success_claim=True,hardware_identification_claim=False)
 (out/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n');shutil.copyfile(__file__,out/'ablation_source_frozen.py')
 cid=p.connect(p.DIRECT);begin=time.monotonic();rows=[];contacts=[];guarded={};error=None
 try:
  scene=mesh_scene.create_scene(p,cid,observation);mesh_scene.assert_scene_matches_manifest(p,cid,scene,observation)
  removed=scene['object_id']
  p.setTimeStep(a.dt,physicsClientId=cid);p.setPhysicsEngineParameter(numNonContactInnerIterations=a.non_contact_inner_iterations,physicsClientId=cid)
  pre=np.asarray(original['world_pre_baseLink_m']);target=np.asarray(original['world_target_baseLink_m']);orientation=original['baseLink_orientation_xyzw']
  selected=np.load(original['candidate_npz'],allow_pickle=False);registers=selected['top1_DH3_grasp_array'][15:17]
  hand=CoupledDH3(cid,original['adapter_manifest'],pre.tolist(),orientation,*registers,assumptions=a)
  assert hand.body==original['body_ids']['hand']
  p.removeBody(removed,physicsClientId=cid)
  receipt=hand.receipt();receipt.update(scope=plan['scope'],original_trial=str(source),body_ids={'hand':hand.body,'table':scene['table_id']},removed_object_body_id_before_integration=removed,scene_signature_before_intervention=scene['scene_signature_sha256'],intervened_scene_matches_frozen_scene=False)
  accepted.write_controller_receipt(out,receipt)
  for name in ['resetJointState','resetBasePositionAndOrientation','resetBaseVelocity','resetSimulation','removeBody','createConstraint']:guarded[name]=getattr(p,name)
  def forbid(*args,**kwargs):raise RuntimeError('No reset/remove/added constraint after integration begins')
  for name in guarded:setattr(p,name,forbid)
  try:
   for step in range(12000):
    t=step*a.dt;pos=pre.copy();position=float(registers[0]);phase='hold_open'
    if .5<=t<3.5:pos=pre+(target-pre)*smooth((t-.5)/3);phase='approach'
    elif t>=3.5:pos=target.copy();phase='close'
    if 3.5<=t<5.5:position=float(registers[0])*(1-smooth((t-3.5)/2))
    elif t>=5.5:position=0.;phase='hold_closed'
    targets=hand.command(position,float(registers[1]),pos.tolist(),orientation);p.stepSimulation(physicsClientId=cid)
    row=hand.observe();row.update(step=step+1,time=(step+1)*a.dt,phase=phase,register_target=np.array([position,registers[1]]),joint_target=np.array([targets[n] for n in hand.movable_names]),wrist_baseLink_target=pos.copy());rows.append(row)
    for c in p.getContactPoints(physicsClientId=cid):
     if hand.body in (c[1],c[2]):contacts.append([step+1,c[1],c[2],c[3],c[4],c[8],c[9],c[10],c[12]])
  finally:
   for name,method in guarded.items():setattr(p,name,method)
 except Exception as exc:error={'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()}
 finally:p.disconnect(cid)
 if rows:
  arrays={key:np.array([r[key] for r in rows]) for key in rows[0]};np.savez_compressed(out/'trace.npz',**arrays)
 np.save(out/'contacts.npy',np.asarray(contacts,float).reshape(-1,9))
 result=dict(scope=plan['scope'],status='failed' if error else 'completed',error=error,physics_steps=len(rows),physical_seconds=len(rows)*a.dt,wall_seconds=time.monotonic()-begin,new_network_forwards=0,actual_grasp_attempts=0,post_first_step_resets=0,baseline_score_changed=False,plan_sha256=sha(out/'PLAN.json'))
 if not error:
  prior=np.load(source/'trace.npz',allow_pickle=False);n=len(rows);checks={key:np.array_equal(arrays[key],prior[key][:n]) for key in ['time','phase','register_target','joint_target']}
  assert all(checks.values()),checks
  master=hand.movable_names.index('finger1_joint');q=arrays['q'][:,master];oldq=prior['q'][:n,master]
  result.update(control_comparison_exact_arrays=checks,shared_flexion_final_rad=float(q[-1]),original_shared_flexion_same_time_rad=float(oldq[-1]),shared_flexion_max_rad=float(q.max()),closed_flexion_target_rad=float(arrays['joint_target'][-1,master]),passed_previous_blocked_0_395rad=bool(q.max()>.405),peak_joint_velocity_rad_s=float(abs(arrays['v']).max()),max_gear_residual_rad=float(abs(arrays['gear_residual']).max()),max_wrist_COM_error_m=float(arrays['wrist_position_error'].max()),peak_motor_torque_Nm=float(abs(arrays['motor_torque']).max()),peak_gear_reaction_Nm=float(abs(arrays['gear_reaction']).max()),contact_rows=len(contacts),positive_contact_rows=sum(c[6]>0 for c in contacts),trace_sha256=sha(out/'trace.npz'),contacts_sha256=sha(out/'contacts.npy'),boundary='One controlled removal tests load dependence only in declared rigid model. No grasp/lift, no proof of real DH3 mechanics, no passive-DoF identification.')
 (out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
 if error:raise RuntimeError(error['error'])

if __name__=='__main__':main()
