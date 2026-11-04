"""Publish saved original/proxy comparisons; rendering and physics are never rerun."""
from pathlib import Path
import argparse,collections,contextlib,hashlib,html,io,json,shutil,sys,time
ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
sys.path[:0]=[str(ROOT/'outputs/native_deps/python_overlay'),str(ROOT/'reproduction/simulation')]
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from encode_recorded_video import encode

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def publish(output):
 output=Path(output);output.mkdir();(output/'videos').mkdir();started=time.monotonic()
 names=['banana_yaw000','banana_yaw090','cube'];versions=['original','v1','v2','v3'];allrows=[];cards=[]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 sheet=Image.new('RGB',(320*len(versions),280*len(names)),(245,245,245));draw=ImageDraw.Draw(sheet)
 for i,name in enumerate(names):
  cells=[]
  for j,version in enumerate(versions):
   if version=='original':
    path=(ROOT/'outputs/dh3_dt1920_cube_physics_20261103_v1' if name=='cube' else ROOT/'outputs/dh3_real_mesh_generalization_20261103_v1/run_nonapple_v2/physics'/name)
   else:path=ROOT/f'outputs/dh3_compliant_comparison_20261104_{version}/physics'/name
   result=read(path/'RESULTS.json');trace=np.load(path/'trace.npz');controller=read(path/'CONTROLLER.json')
   frames=read(path/'frames.json')['frames']
   if not (path/'VIDEO.json').exists():
    with contextlib.redirect_stdout(io.StringIO()):encode(path)
   video=read(path/'VIDEO.json');source=Path(video['video']);target=output/'videos'/f'{version}_{name}.mp4';shutil.copyfile(source,target)
   assert sha(source)==sha(target)
   row=dict(version=version,scene_id=name,physics_output_dir=str(path),result_sha256=sha(path/'RESULTS.json'),trace_sha256=sha(path/'trace.npz'),video=str(target.relative_to(output)),video_sha256=sha(target),video_duration_s=video['video_duration_s'],physics_steps=result['physics_steps'],physical_seconds=result['physical_seconds'],lift_observed=result['lift_observed'],status=result['status'],port_stability_pass=result['port_stability_pass'],object_final_lift_m=result['object_final_lift_m'],candidate_sha256=controller['candidate_sha256'],scene_manifest_sha256=controller['scene_manifest_sha256'],peak_joint_velocity_rad_s=result['peak_joint_velocity_rad_s'],max_gear_residual_rad=result['max_gear_residual_rad'],max_wrist_COM_error_m=result['max_wrist_COM_error_m'],joint_velocity_gate_pass=bool(abs(trace['v']).max()<1.5),gear_residual_gate_pass=bool(abs(trace['gear_residual']).max()<.01),motor_torque_bounds_pass=result['motor_torque_bounds_pass'],gear_reaction_bounds_pass=result['gear_reaction_bounds_pass'],wrist_tracking_pass=result['wrist_tracking_pass'],additional_physics_steps_for_media=0)
   if version!='original':row['compliance']=controller['compliance']
   allrows.append(row)
   lift=f'{result["object_final_lift_m"]*1000:.2f} mm';state='提起且通过旧稳定门' if result['lift_observed'] and result['port_stability_pass'] else ('实际提起，旧稳定门未过' if result['lift_observed'] else '未保持提起')
   cells.append(f'<td><b>{version}</b><p>{state}<br>抬升 {lift}<br>峰值关节速度 {row["peak_joint_velocity_rad_s"]:.3f} rad/s</p><video controls preload="metadata" src="{row["video"]}"></video></td>')
   x=j*320;y=i*280;draw.text((x+5,y+5),f'{name} | {version}',font=font,fill=(25,35,45));draw.text((x+5,y+26),f'lift {lift}; stable {result["port_stability_pass"]}',font=font,fill=(25,35,45))
   im=Image.open(path/frames[-1]['path']).convert('RGB');im.thumbnail((316,232));sheet.paste(im,(x+2,y+51))
  cards.append('<tr>'+''.join(cells)+'</tr>')
 sheet.save(output/'comparison.jpg',quality=92)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
 for ax,version in zip(axes,['rigid','v2']):
  root=ROOT/('outputs/dh3_force_fixture_20261104_v1' if version=='rigid' else 'outputs/dh3_compliant_fixture_20261104_v2')/'asymmetric_finger2_045_other_080'
  z=np.load(root/'trace.npz');namesq=read(root/'CONTROLLER.json')['joint_order']
  for finger in (1,2,3):ax.plot(z['time'],z['q'][:,namesq.index(f'finger{finger}_joint')],label=f'Finger {finger}')
  ax.set(xlabel='Time (s)',ylabel='Proximal angle (rad)',title=version+' | same fixed obstacle positions',ylim=(-.05,1.4));ax.legend()
 fig.savefig(output/'blocked_finger_comparison.png',dpi=180);fig.savefig(output/'blocked_finger_comparison.pdf');plt.close(fig)
 # Aggregate only new work; original trajectories remain comparison references.
 cost=[]
 for version in ('v1','v2'):
  result=read(ROOT/f'outputs/dh3_compliant_fixture_20261104_{version}/RESULTS.json')
  cost.append(dict(kind='force_fixtures',version=version,physics_steps=result['completed_physics_steps'],physical_seconds=result['completed_physics_steps']/1920,wall_seconds=result['wall_seconds'],grasp_attempts=0))
 for version in ('v1','v2','v3'):
  result=read(ROOT/f'outputs/dh3_compliant_comparison_20261104_{version}/RESULTS.json')
  cost.append(dict(kind='three_saved_grasps',version=version,physics_steps=result['physics_steps'],physical_seconds=result['physical_seconds'],wall_seconds=result['wall_seconds'],grasp_attempts=3))
 summary=dict(scope='Saved original grasp poses, declared common-shaft elastic proxy, two parameter sets and one numerical refinement; not a fresh object success-rate benchmark or hardware dynamics reproduction',original_baseline_success=8,original_baseline_scenes=12,original_baseline_unchanged=True,rows=allrows,cost=cost,new_physics_steps=sum(c['physics_steps'] for c in cost),new_physical_seconds=sum(c['physical_seconds'] for c in cost),new_grasp_attempts=sum(c['grasp_attempts'] for c in cost),new_network_forwards=0,additional_physics_steps_for_media=0,media_wall_seconds=time.monotonic()-started,protocol_note='Original gate outcomes retained. Actual lifting is reported separately from port stability; no failed original result or proxy version is erased.',metadata_erratum='Inherited v1/v2 fixture summaries passive_reported_motor_torque_max_Nm and motor_work_endpoint_sample_proxy_J do not handle the virtual actuator name/mapping correctly. Independent proxy audit recomputes true virtual-shaft input work, passive branch torque and energy; do not use those inherited two fields as passive drive/work evidence. Wrapper prose saying no constraints removed refers to old controller and is superseded by explicit proxy receipt: 2 cross-finger gears replaced, 6 local +1 spread retained.')
 save(output/'SUMMARY.json',summary)
 doc='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DH3 共同驱动弹性近似：固定抓姿对照</title><style>body{font-family:system-ui;margin:20px;color:#182231}table{border-collapse:collapse;width:100%;table-layout:fixed}td{border:1px solid #ccd4df;padding:10px;vertical-align:top}video{width:100%}img{max-width:100%}p{line-height:1.6}.note{background:#fff5d6;padding:12px}</style><h1>DH3 共同驱动弹性近似：固定抓姿对照</h1><p>列顺序：原刚性模型 / 弹性 v1 / 提高阻尼 v2 / 同 v2 参数、3840 Hz 数值细化 v3。每个物体沿用原 top1 抓姿、尺寸、质量、摩擦和腕部脚本，每版本实际执行一次，无网络重跑。</p><p class="note">这是我们新增的动力学近似，不能当成真实 DH3 机构或原论文成功率复现。实际提起与旧稳定门分别显示；原 12 场景 8 成功的统计保持。</p><p><a href="SUMMARY.json">全部结果与成本</a> | <a href="comparison.jpg">末帧对照图</a></p><table>'''+''.join(cards)+'''</table><h2>同一阻挡条件下的三指响应</h2><img src="blocked_finger_comparison.png"><p>右图共同输入经弹性支路传动；一指受阻，其余指可继续到达另外两个固定测力垫。所有支路反作用返回同一有界驱动轴。尚无额外指内远节自由弯曲机构。</p></html>'''
 (output/'index.html').write_text(doc)
 print(json.dumps(dict(index=str(output/'index.html'),summary=str(output/'SUMMARY.json'),new_physics_steps=summary['new_physics_steps'],videos=len(allrows),media_wall_seconds=summary['media_wall_seconds']),indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();publish(a.output)
