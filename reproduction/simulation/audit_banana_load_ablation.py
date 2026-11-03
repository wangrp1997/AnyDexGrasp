"""Independently recompute saved load-ablation metrics and draw comparison."""
from pathlib import Path
import argparse,hashlib,json,os
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('run_dir',type=Path);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();root=a.run_dir.absolute();out=a.output_dir.absolute()
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);plan=json.loads((root/'PLAN.json').read_text());result=json.loads((root/'RESULTS.json').read_text());c=json.loads((root/'CONTROLLER.json').read_text())
 prior=Path(plan['source_trial']);pc=json.loads((prior/'CONTROLLER.json').read_text());x=np.load(root/'trace.npz',allow_pickle=False);old=np.load(prior/'trace.npz',allow_pickle=False);contacts=np.load(root/'contacts.npy',allow_pickle=False)
 assert sha(prior/'trace.npz')==plan['source_trace_sha256'];assert sha(prior/'CONTROLLER.json')==plan['source_controller_sha256'];assert sha(root/'ablation_source_frozen.py')==plan['source_code_sha256'];assert sha(root/'trace.npz')==result['trace_sha256'];assert sha(root/'contacts.npy')==result['contacts_sha256']
 assert len(x['time'])==12000==result['physics_steps'];assert np.array_equal(x['step'],np.arange(1,12001));assert np.allclose(x['time'],x['step']/1920,rtol=0,atol=1e-12)
 assert pc['assumptions_SI_not_hardware_identification']==c['assumptions_SI_not_hardware_identification'];assert pc['derived_urdf_sha256']==c['derived_urdf_sha256'];assert pc['two_register_affine_map']==c['two_register_affine_map'];assert pc['joint_order']==c['joint_order']
 assert pc['gears']==c['gears'];assert c['body_ids']['hand']==pc['body_ids']['hand'];assert c['body_ids']['table']==pc['body_ids']['table']
 for key in ['time','phase','register_target','joint_target']:assert np.array_equal(x[key],old[key][:12000]),key
 for k in x.files:
  if x[k].dtype.kind in 'fiu':assert np.isfinite(x[k]).all(),k
 j=c['joint_order'].index('finger1_joint');q=x['q'][:,j];oldq=old['q'][:12000,j]
 assert result['shared_flexion_final_rad']==float(q[-1]);assert result['original_shared_flexion_same_time_rad']==float(oldq[-1]);assert result['peak_joint_velocity_rad_s']==float(abs(x['v']).max());assert result['max_gear_residual_rad']==float(abs(x['gear_residual']).max())
 h=c['body_ids']['hand'];table=c['body_ids']['table'];positive=contacts[contacts[:,6]>0];selfrows=positive[(positive[:,1]==h)&(positive[:,2]==h)];tablerows=positive[((positive[:,1]==h)&(positive[:,2]==table))|((positive[:,2]==h)&(positive[:,1]==table))]
 assert len(positive)==len(selfrows)+len(tablerows)
 os.environ.setdefault('MPLCONFIGDIR','/tmp/anydex_diag_matplotlib');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True);s=slice(None,None,8);t=x['time'][s]
 axes[0].plot(t,oldq[s],label='Original: banana present');axes[0].plot(t,q[s],label='Ablation: banana removed before first step');axes[0].plot(t,x['joint_target'][s,j],'k--',label='Exactly identical command');axes[0].set_ylabel('Shared flexion (rad)')
 axes[1].plot(t,old['motor_torque'][:12000,j][s],label='Original motor torque');axes[1].plot(t,x['motor_torque'][s,j],label='No-banana motor torque');axes[1].set_ylabel('Motor torque (Nm)');axes[1].set_xlabel('Recorded time (s)')
 for ax in axes:ax.axvline(3.58125,color='gray',linestyle=':',label='Original first object contact');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('Same rigid hand, table and prelift commands: load-removal test only');fig.tight_layout();fig.savefig(out/'load_removal_comparison.png',dpi=150);plt.close(fig)
 audit=dict(status='saved_load_ablation_verified',run_RESULTS_sha256=sha(root/'RESULTS.json'),plan_sha256=sha(root/'PLAN.json'),trace_sha256=sha(root/'trace.npz'),script_sha256=sha(__file__),checks=['source and frozen script SHA','12000 contiguous actual step/time rows','all prelift control arrays equal original','same assumptions, hand geometry and affine transmission','same constraints and retained body IDs','finite recorded arrays','reported metrics independently recalculated'],positive_contact_rows={'self':len(selfrows),'table':len(tablerows)},first_self_contact_time_s=None if not len(selfrows) else float(selfrows[0,0]/1920),first_table_contact_time_s=None if not len(tablerows) else float(tablerows[0,0]/1920),new_physics_steps=0,new_network_forwards=0,interpretation='Removal allows shared flexion to pass old blocked state; eventual no-object stop is not evidence of exact target tracking. No real DH3 mechanics identified, no grasp score.')
 (out/'RESULTS.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
