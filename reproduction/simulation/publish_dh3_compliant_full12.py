"""Publish existing full12 trajectories; no new simulation or inference."""
from __future__ import annotations
import argparse,collections,contextlib,hashlib,html,io,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'outputs/native_deps/python_overlay'),str(ROOT/'reproduction/simulation')]
from encode_recorded_video import encode
from PIL import Image,ImageDraw,ImageFont

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run_dir',type=Path);a=ap.parse_args();root=a.run_dir.resolve()
 report=read(root/'RESULTS.json');plan=read(root/'PLAN.json');assert report['status']=='completed'
 media=root/'videos';media.mkdir();started=time.monotonic();cards=[];records=[]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
 sheet=Image.new('RGB',(640*3,235*4),(245,245,245));draw=ImageDraw.Draw(sheet)
 for i,row in enumerate(report['scenes']):
  case=next(c for c in plan['cases'] if c['scene_id']==row['scene_id']);name=row['scene_id'];scene=Path(case['scene']['path']).parent
  item={k:row.get(k) for k in ['scene_id','mode','functional_status','lift_observed','port_stability_pass','physics_steps','new_physics_steps','object_final_lift_m','physics_output_dir']}
  title=name+' | '+row['functional_status'];x=i%3*640;y=i//3*235
  draw.text((x+7,y+6),title,font=font,fill=(0,100,35) if row['lift_observed'] else (165,45,25))
  if row['physics_output_dir']:
   path=Path(row['physics_output_dir']);result=read(path/'RESULTS.json');frames=read(path/'frames.json')['frames']
   if not (path/'VIDEO.json').exists():
    with contextlib.redirect_stdout(io.StringIO()):encode(path)
   video=read(path/'VIDEO.json');dst=media/(name+'.mp4');shutil.copyfile(video['video'],dst);assert sha(dst)==video['video_sha256']
   item.update(video=str(dst),video_sha256=sha(dst),video_duration_s=video['video_duration_s'],peak_joint_velocity_rad_s=result['peak_joint_velocity_rad_s'],max_gear_residual_rad=result['max_gear_residual_rad'],max_wrist_COM_error_m=result['max_wrist_COM_error_m'])
   lift='%.2f mm'%(row['object_final_lift_m']*1000)
   draw.text((x+7,y+27),'final COM lift '+lift+' | old numeric gates '+str(row['port_stability_pass']),font=font,fill=(25,35,45))
   images=[path/frames[k]['path'] for k in (0,len(frames)//2,len(frames)-1)]
   state='抓起并保持' if row['lift_observed'] else '未保持提起'
   content=f'<p>{state}，末态抬升 {lift}；'+('复用同参数已完成记录' if row['mode']=='reuse' else '本轮一次实际执行')+f'</p><video controls preload="metadata" src="videos/{name}.mp4"></video><details><summary>数值诊断</summary><p>旧数值检查通过：{row["port_stability_pass"]}；峰速 {item["peak_joint_velocity_rad_s"]:.4f} rad/s；gear残差 {item["max_gear_residual_rad"]:.6f} rad；腕误差 {item["max_wrist_COM_error_m"]*1000:.4f} mm。</p></details>'
  else:
   dst=root/(name+'_input.png');shutil.copyfile(scene/'color.png',dst);images=[dst]*3
   draw.text((x+7,y+27),'original collision rejected all candidates | zero physics',font=font,fill=(25,35,45))
   content=f'<p>官方原碰撞筛选无准入抓姿，零实际抓取，没有动作录像。</p><img src="{dst.name}">'
  for j,p in enumerate(images):
   im=Image.open(p).convert('RGB');im.thumbnail((208,174));sheet.paste(im,(x+j*212+3,y+55))
  cards.append('<article><h2>'+html.escape(name)+'</h2>'+content+'</article>');records.append(item)
 sheet.save(root/'contact_sheet.jpg',quality=92)
 lifted=sum(bool(r['lift_observed']) for r in report['scenes']);selected=sum(c['mode']!='no_original_candidate' for c in plan['cases'])
 summary=dict(scope=report['scope'],source_result_sha256=sha(root/'RESULTS.json'),counts=dict(total_scenes=12,independent_object_meshes=6,original_top1_coverage=selected,lifted_and_held=lifted,no_original_candidate=3,executed_not_held=selected-lifted),functional_scene_rate=lifted/12,conditional_execution_rate=lifted/selected,old_numeric_gate_passes=report['numeric_gate_passes'],new_physics_steps=report['new_physics_steps'],reused_physics_steps=report['reused_physics_steps'],new_physical_seconds=report['new_physical_seconds'],new_grasp_attempts=7,reused_grasp_attempts=2,new_network_forwards=0,new_physics_steps_for_media=0,media_wall_seconds=time.monotonic()-started,records=records)
 save(root/'MEDIA_SUMMARY.json',summary)
 doc=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AnyDexGrasp 固定近似模型：12场景</title><style>body{{font-family:system-ui;margin:22px;background:#f4f6f9;color:#172233}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}}article{{background:white;padding:16px;border-radius:10px}}video,img{{width:100%}}h2{{font-size:18px}}p{{line-height:1.6}}.note{{background:#fff3d3;padding:12px}}</style><h1>AnyDexGrasp：固定近似模型，原12场景</h1><p><b>抓起并保持 {lifted}/12；原抓姿准入 {selected}/12；已有抓姿执行 {lifted}/{selected}。</b> 六个实体网格各两个方向，只有六个独立物体。</p><p>模型/参数/物体/原top1固定；7次新增执行，2次香蕉复用同模型记录，3项原碰撞无候选保留。4路CPU并行，3840 Hz；无网络重跑、重新选姿或物体特设。</p><p class="note">这是原方法在声明仿真近似下的小规模功能复现。成功指抬起并保持，不包含论文的机械臂搬运投箱，不能直接对比论文实机成功率。旧数值门单列，原8/12历史统计保持。</p><p><a href="RESULTS.json">执行结果和成本</a> | <a href="MEDIA_SUMMARY.json">录像目录</a> | <a href="contact_sheet.jpg">全场景过程图</a></p><main>'''+''.join(cards)+'</main></html>'
 (root/'index.html').write_text(doc)
 print(json.dumps(dict(index=str(root/'index.html'),counts=summary['counts'],videos=sum('video' in r for r in records),additional_physics_steps=0,media_wall_seconds=summary['media_wall_seconds']),indent=2))
if __name__=='__main__':main()
