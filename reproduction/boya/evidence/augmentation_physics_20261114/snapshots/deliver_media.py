"""Prepare one-process-per-trace replay with exact historical video reuse."""
from pathlib import Path
import sys,os,html,json
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');sys.path.insert(0,str(ROOT/'reproduction/boya'))
import publish_boya_fixed_pool as p
run=Path(__file__).resolve().parent;media=run/'media_v1';mode=sys.argv[1]
if mode=='prepare':
 rows=p.inputs(run);p.require(not media.exists(),'Fresh media directory required');media.mkdir()
 frozen=p.frozen_sources(run);frozen[str(Path(__file__))]=p.sha(__file__)
 retained=[];pending=[]
 for row in rows:
  cid=row['candidate_id'];job=Path(row['job_path']);original=p.read(job)['object_id']
  if row['kind']=='reused':
   candidates=[job.parent.parent/'videos'/original,job.parent.parent/'delivery_videos'/original,ROOT/'outputs/boya_alignment_media_20261108_v1/videos'/original]
   target=next((d for d in candidates if (d/'render_check.json').exists()),None)
   p.require(target is not None,'Missing bound historical video: '+str(job))
   retained.append(p.verify_media(target,row,original));continue
  view=media/cid;view.mkdir();(view/'jobs').mkdir()
  p.save(view/'protocol.json',dict(media_only=True,cases=[dict(object_id=cid,old_execution=None)],source_job=p.ref(job),new_grasp_executions=0))
  p.save(view/'RESULTS.json',dict(status='complete',protocol_sha256=p.sha(view/'protocol.json'),results=[row],new_physics_steps=0))
  (view/'jobs'/(cid+'.json')).write_bytes(job.read_bytes())
  pending.append(dict(candidate_id=cid,view=str(view),protocol=p.ref(view/'protocol.json'),results=p.ref(view/'RESULTS.json'),copied_job=p.ref(view/'jobs'/(cid+'.json'))))
 p.save(media/'DISPATCH.json',dict(media_only=True,frozen_input_sha256=frozen,retained_cases=retained,pending_cases=pending,workers=2,attempts_per_pending_case=1,new_grasp_executions=0,physics_steps_during_replay=0))
 print('MEDIA_PREPARED',len(retained),'reused',len(pending),'new',flush=True)
elif mode=='render':
 report=p.render(run,'media_v1',600);p.require(report['complete'],'Media failed; physical outcomes retained')
elif mode=='publish':
 report=p.publish(run,'media_v1');p.require(report['complete'],'All videos must verify')
 results=p.read(run/'RESULTS.json');audit=p.read(run/'INDEPENDENT_PHYSICS_AUDIT.json');rows={r['candidate_id']:r for r in results['results']}
 parts=['<!doctype html><meta charset="utf-8"><title>Boya 原增强实抓对照</title><style>body{font:17px system-ui;max-width:1250px;margin:30px auto;padding:0 20px;background:#eef1f5}table{border-collapse:collapse;background:white}th,td{padding:10px;border:1px solid #bbb}section{background:white;padding:18px;margin:20px 0}video{width:100%}</style><h1>Boya：补齐原增强后的受控腕实抓对照</h1><p>A3/A6/E0 为三件已用诊断物体；A0 正对照单列。每条件首个几何准入，物理数组相同则复用历史或共用本轮一次执行。没有臂规划、重抓、重训或调阈值。</p><table><tr><th>输入</th><th>评分</th><th>A3</th><th>A6</th><th>E0</th><th>三件</th><th>A0 单列</th></tr>']
 for m in audit['metrics']:
  cells=[m['input_kind'],m['recipe'],*[('成功' if m['outcomes'][o] else '失败')for o in ('egad_A3','egad_A6','egad_E0')],str(m['success'])+'/3','成功' if m['A0_positive_control'] else '失败']
  parts.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>'for c in cells)+'</tr>')
 parts.append('</table><p>视频 Object rise 字幕为物体参考点位移；成功还需源网格底面离桌和末态双侧承载。本结果不是独立外部泛化或完整机械臂成功率。</p>')
 for oid in ('egad_A3','egad_A6','egad_E0','egad_A0'):
  parts.append('<h2>'+oid+('（正对照单列）'if oid=='egad_A0'else '')+'</h2>')
  for c in [c for c in results['conditions'] if c['source_object_id']==oid]:
   cid=c['candidate_id'];r=rows.get(cid);parts.append('<section><h3>'+html.escape(c['input_kind']+' / '+c['recipe'])+' · '+('成功'if r and r['task_success']else'失败')+'</h3>')
   if r:
    parts.append('<p>评分名次（1基）：'+str(c['rank']+1)+'；'+('复用历史执行'if r['kind']=='reused'else'本轮一次执行')+'；物理姿态编号 '+html.escape(cid)+'</p><video controls preload="none" src="delivery_videos/'+html.escape(cid)+'/hand_only.mp4"></video>')
   else:parts.append('<p>无几何候选，未执行。</p>')
   parts.append('</section>')
 parts.append('<p><a href="INDEPENDENT_PHYSICS_AUDIT.json">独立物理核验</a> · <a href="delivery_videos/RESULTS.json">视频全帧解码和源绑定核验</a></p>')
 (run/'index.html').write_text('\n'.join(parts))
