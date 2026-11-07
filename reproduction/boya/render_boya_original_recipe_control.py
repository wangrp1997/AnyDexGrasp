"""RTX replay of four known-scene original-selected-MSE candidate trials."""
from pathlib import Path
import argparse,sys,os,html
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');sys.path.insert(0,str(ROOT/'reproduction/boya'))
from run_egad_floating_three import read

def run(folder):
 import render_egad_floating_three as original
 original.run(argparse.Namespace(run=folder))
 result=read(folder/'RESULTS.json');parts=['<!doctype html><meta charset="utf-8"><title>原始MSE评分器选姿对照</title><style>body{font:18px system-ui;max-width:1280px;margin:30px auto;background:#eef1f5}section{background:white;padding:18px;margin:20px 0}video{width:100%}</style><h1>原始MSE评分器选姿对照</h1><p>四个已知场景；缓存CGR、Boya标定、粗碰撞代理与名义闭合保持，换回未改损失的code100评分器选姿。每件只执行首次几何准入候选一次。此处为受控腕部诊断，不含机械臂规划，不是新的独立外部成功率。</p>']
 for row in result['results']:
  oid=row['object_id'];e=read(row['evaluation']);label='成功'if row['task_success']else'未抓起'
  if e.get('failure_reason'):label+='，'+e['failure_reason']
  parts.append(f'<section><h2>{html.escape(oid)} — {html.escape(label)}</h2><video controls preload="metadata" src="{oid}/hand_only.mp4"></video></section>')
 (folder/'videos/index.html').write_text('\n'.join(parts))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
 os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
 from isaaclab.app import AppLauncher
 launcher=AppLauncher(headless=True,device='cpu',enable_cameras=True,visualizer=None,kit_args='--/log/file=/dev/null --/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false')
 code=1
 try:run(a.run.resolve());code=0
 finally:launcher.app.close(exit_code=code)
