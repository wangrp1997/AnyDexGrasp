"""Publish fixed-fixture response measurements; no hardware calibration claim."""
from pathlib import Path
import argparse,csv,hashlib,html,json,os
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fixture-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args();root=args.fixture_dir.absolute();out=args.output_dir.absolute()
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);r=json.loads((root/'RESULTS.json').read_text());rows=[]
 for c in r['cases']:
  m=c['measurement'];means={str(i):0. for i in [1,2,3]};mins=dict(means);maxs=dict(means)
  for pad,v in m['pad_force_summary'].items():
   i=pad.rsplit('finger',1)[1];f=v['normal_force_N_including_zero_steps'];means[i]=f['mean'];mins[i]=f['min'];maxs[i]=f['max']
  rows.append(dict(case_id=c['case_id'],status=c['status'],physics_steps=m['physics_steps'],final_finger1_rad=m['closing_finger_master_angles_final_rad']['finger1'],final_finger2_rad=m['closing_finger_master_angles_final_rad']['finger2'],final_finger3_rad=m['closing_finger_master_angles_final_rad']['finger3'],force_finger1_mean_N=means['1'],force_finger2_mean_N=means['2'],force_finger3_mean_N=means['3'],max_gear_residual_rad=m['max_gear_residual_rad'],peak_joint_speed_rad_s=m['peak_joint_velocity_rad_s'],passive_motor_torque_abs_max_Nm=m['passive_reported_motor_torque_max_Nm'],force_window='last480 actual steps; 0.25s; zero-contact steps included',force_min_N=mins,force_max_N=maxs,source_result_sha256=sha(root/c['case_id']/'RESULTS.json')))
 with (out/'fixture_summary.csv').open('w') as f:
  fields=[k for k in rows[0] if not isinstance(rows[0][k],dict)];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 os.environ.setdefault('MPLCONFIGDIR','/tmp/anydex_calibration_matplotlib');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(12,4.5));ax=axes[0];x=np.arange(len(rows));width=.24
 for j in range(3):
  y=np.array([v[f'force_finger{j+1}_mean_N'] for v in rows]);low=np.array([v['force_min_N'][str(j+1)] for v in rows]);high=np.array([v['force_max_N'][str(j+1)] for v in rows]);err=np.maximum(0,np.vstack([y-low,high-y]));ax.bar(x+(j-1)*width,y,width,label=f'finger{j+1}',yerr=err,capsize=2)
 ax.set_xticks(x,['F1 stop','F2 stop','F3 stop','3 equal stops','F2 early stop'],rotation=20);ax.set_ylabel('Normal reaction per pad (N)');ax.set_title('0.6 Nm common motor, fixed rigid reference');ax.legend();ax.grid(axis='y',alpha=.2)
 case=root/'asymmetric_finger2_045_other_080';d=np.load(case/'trace.npz',allow_pickle=False);c=json.loads((case/'CONTROLLER.json').read_text());s=slice(None,None,8)
 for i in [1,2,3]:
  j=c['joint_order'].index(f'finger{i}_joint');axes[1].plot(d['time'][s],d['q'][s,j],label=f'finger{i}')
 j=c['joint_order'].index('finger1_joint');axes[1].plot(d['time'][s],d['joint_target'][s,j],'k--',label='common command');axes[1].axhline(.8,color='.5',linestyle=':',label='later stop reference');axes[1].set_xlabel('Actual physical time (s)');axes[1].set_ylabel('Proximal flexion (rad)');axes[1].set_title('Asymmetric obstruction: unchanged rigid model');axes[1].legend();axes[1].grid(alpha=.2)
 fig.suptitle('Simulation-port response characterization; hardware transmission and force command remain unidentified',fontsize=11);fig.tight_layout();fig.savefig(out/'force_fixture_summary.png',dpi=170);fig.savefig(out/'force_fixture_summary.pdf');plt.close(fig)
 summary=dict(status='simulation_response_characterized_hardware_not_identified',fixture_results=str(root/'RESULTS.json'),fixture_results_sha256=sha(root/'RESULTS.json'),source_script_sha256=sha(__file__),rows=rows,completed_physics_steps=r['completed_physics_steps'],physical_seconds=r['completed_physics_steps']/1920,fixture_campaign_wall_seconds=r['wall_seconds'],physical_grasp_attempts=0,new_network_forwards=0,publishing_new_physics_steps=0,statistical_boundary='Range bars show observed minima/maxima in a correlated final0.25s window; no independent-sample confidence interval.',calibration_boundary='Current rigid-model input/output response measured. No physical DH3 spring, drive conversion, contact pad or hardware torque-speed parameter has been identified.')
 (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
 table='<table><tr><th>条件</th><th>F1 (N)</th><th>F2 (N)</th><th>F3 (N)</th></tr>'+''.join('<tr><td>'+html.escape(v['case_id'])+'</td>'+''.join('<td>%.3f</td>'%v[f'force_finger{i}_mean_N'] for i in [1,2,3])+'</tr>' for v in rows)+'</table>'
 doc='<!doctype html><html lang="zh"><meta charset="utf-8"><title>DH3 校准资料与测力响应</title><style>body{font-family:system-ui;max-width:1300px;margin:25px auto;padding:15px}img{width:100%}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #bbb}</style><h1>DH3：标定资料核对与五条件测力</h1><p>固定旧仿真模型、原两电机/九gear/0.6Nm闭合输入。5个工装条件全部完成，31200步、16.25秒物理。不是抓取试验，也没有辨识真实DH3。</p><img src="force_fixture_summary.png">'+table+'<p>表中均为最后0.25秒平均法向反力，包含零接触步；曲线/误差线不是独立样本置信区间。</p><p>厂家force70是百分比命令，没有到关节Nm的曲线。原13-link URDF未表达官方STEP中的指内自适应机构；不能据空载角表补出弹簧、摩擦和丝杆传力参数。</p><p><a href="force_fixture_summary.pdf">PDF图</a> · <a href="fixture_summary.csv">CSV</a> · <a href="SUMMARY.json">来源和完整数值</a></p></html>'
 (out/'index.html').write_text(doc);print(json.dumps({k:summary[k] for k in ['status','completed_physics_steps','physical_seconds','fixture_campaign_wall_seconds']},indent=2))
if __name__=='__main__':main()
