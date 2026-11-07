"""One RTX session for saved proxy/timing alignment traces, zero new physics."""
from pathlib import Path
import argparse,json,sys,os,html
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');B=ROOT/'reproduction/boya';sys.path.insert(0,str(B))
from run_egad_floating_three import read,save,sha
SOURCES=(('proxy','boya_contact_proxy_alignment_20261108_v2'),('timing','boya_position_timing_alignment_20261108_v1'))
def prepare(out):
 if out.exists():raise ValueError('Fresh media collection directory required')
 out.mkdir();(out/'jobs').mkdir();rows=[];cases=[];inputs={}
 for variant,name in SOURCES:
  source=ROOT/'outputs'/name;result=read(source/'RESULTS.json');assert result['status']=='complete'and not result['errors']
  inputs[str(source/'RESULTS.json')]=sha(source/'RESULTS.json')
  for row in result['results']:
   oid=row['object_id'];key=oid+'_'+variant;job=read(source/'jobs'/(oid+'.json'))
   rows.append(dict(row,object_id=key,source_object_id=oid,source_protocol=str(source/'protocol.json')))
   save(out/'jobs'/(key+'.json'),job);cases.append(dict(object_id=key,old_execution=None))
 save(out/'protocol.json',dict(object_ids=[r['object_id']for r in rows],cases=cases,source_results_sha256=inputs,
  scope='Video collection of9 actual independent proxy/timing diagnostic trials; not one continuous regrasp or overall success rate'))
 save(out/'RESULTS.json',dict(status='complete',protocol_sha256=sha(out/'protocol.json'),results=rows))
 return out

def run(out):
 import render_egad_floating_three as original
 original.run(argparse.Namespace(run=out))
 result=read(out/'RESULTS.json');parts=['<!doctype html><meta charset="utf-8"><title>Boya适配对齐对照</title><style>body{font:18px system-ui;max-width:1280px;margin:30px auto;background:#eef1f5}section{background:white;padding:18px;margin:20px 0}video{width:100%}</style><h1>Boya适配对齐对照录像</h1><p>9次独立真实PhysX轨迹：碰撞代理对照5次，位置闭合时间序列对照4次。原外部结果1/12保持；本页不是连续现场重抓或总体成功率。</p>']
 for row in result['results']:
  key=row['object_id'];label='成功'if row['task_success']else'未抓起'
  parts.append(f'<section><h2>{html.escape(key)} — {label}</h2><video controls preload="metadata" src="{key}/hand_only.mp4"></video></section>')
 (out/'videos/index.html').write_text('\n'.join(parts))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=prepare(a.output.resolve())
 os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
 from isaaclab.app import AppLauncher
 launcher=AppLauncher(headless=True,device='cpu',enable_cameras=True,visualizer=None,kit_args='--/log/file=/dev/null --/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false')
 code=1
 try:run(out);code=0
 finally:launcher.app.close(exit_code=code)
