"""Frozen external12 Boya evaluation driver; no result-guided adaptation."""
from pathlib import Path
import argparse,json,hashlib,os,subprocess,sys,time,traceback
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');B=ROOT/'reproduction/boya'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def run(a):
 root=a.run_dir.resolve();pp=root/'protocol.json';protocol=read(pp);scene=a.scene_manifest.resolve() if a.scene_manifest else root/'scenes/objects.json'
 if read(scene)['schema']!='anydex_boya_external_scene_manifest_v1':raise ValueError('Completed external assets required')
 if (root/'EVALUATION_FREEZE.json').exists():raise ValueError('A frozen external evaluation already exists; do not replay')
 sources={str(p):sha(p)for p in B.glob('*.py')};sources[str(Path(__file__).resolve())]=sha(__file__)
 inputs={str(pp):sha(pp),str(scene):sha(scene),**protocol['input_sha256'],**protocol['source_sha256']}
 def verify():
  for p,h in {**inputs,**sources}.items():
   if sha(p)!=h:raise ValueError('Frozen evaluation source/input changed '+p)
 verify();save(root/'EVALUATION_FREEZE.json',dict(schema='anydex_boya_external_evaluation_freeze_v1',protocol_ref=dict(path=str(pp),sha256=sha(pp)),source_sha256=sources,input_sha256=inputs,ranking_and_outcomes_not_yet_observed=True,model_selection=False,training_writeback=False))
 began=time.monotonic();stages=[];status=dict(status='running',stages=stages,object_count=12,successes=None,physics_steps=0,external_protocol_ref=dict(path=str(pp),sha256=sha(pp)))
 save(root/'EVALUATION_PROGRESS.json',status)
 def call(stage,script,*args):
  verify();p=root/(stage+'.log');status.update(stage=stage,status='running',elapsed_s=time.monotonic()-began);save(root/'EVALUATION_PROGRESS.json',status);print('EXTERNAL_STAGE '+stage,flush=True);t=time.monotonic()
  cmd=[sys.executable,'-B',str(B/script),*map(str,args)]
  with p.open('xb')as stream:proc=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
  rec=dict(stage=stage,argv=cmd,returncode=proc.returncode,wall_s=time.monotonic()-t,log=str(p));stages.append(rec);status['elapsed_s']=time.monotonic()-began;save(root/'EVALUATION_PROGRESS.json',status);verify()
  if proc.returncode:raise RuntimeError('External stage failed: '+stage+'; see '+str(p))
 try:
  call('initialize','prepare_boya_external_runtime.py','--scene-manifest',scene,'--run-dir',root,'--workers','4')
  call('observe','render_boya_external_observations.py','--objects-json',root/'objects.json','--output-dir',root/'observations')
  # A source-bound overview also covers objects that never get a physical plan.
  from PIL import Image,ImageOps,ImageDraw
  canvas=Image.new('RGB',(4*480,3*294),'white');pen=ImageDraw.Draw(canvas);contact_inputs={}
  for j,row in enumerate(read(root/'objects.json')['objects']):
   image=Path(row['observation_json']).parent/'color.png';contact_inputs[str(image)]=sha(image)
   tile=ImageOps.fit(Image.open(image).convert('RGB'),(480,270));x=(j%4)*480;y=(j//4)*294;canvas.paste(tile,(x,y));pen.text((x+10,y+274),row['object_id'],fill='black')
  canvas.save(root/'observation_contact_sheet.jpg',quality=95)
  save(root/'OBSERVATION_CONTACT_SHEET.json',dict(source_images_sha256=contact_inputs,image_sha256=sha(root/'observation_contact_sheet.jpg'),new_physics_steps=0))
  call('represent','run_collection_representation.py','--manifest',root/'objects.json','--run-dir',root/'representation_stage','--device','cuda:0','--checkpoint',protocol['representation']['checkpoint']['path'],'--seed',str(protocol['representation']['seed']))
  call('diagnostic','evaluate_boya_external.py','--objects-manifest',root/'objects.json','--protocol',pp,'--training-results',protocol['training_results']['path'],'--calibration',ROOT/'outputs/boya_adapter_preparation_20261105_v2/calibration.json','--domain-policy',B/'evidence/s40_hand_domain_20261107/POLICY.json','--output-dir',root/'diagnostic','--geometry-workers','4')
  call('plan','prepare_boya_external_top1.py','--protocol',pp,'--diagnostic',root/'diagnostic','--backend-config',ROOT/'outputs/boya_pilot8_20261105_v1/backend_pilot.json','--calibration',ROOT/'outputs/boya_adapter_preparation_20261105_v2/calibration.json','--native-preflight',root/'native_preprocessing/SUMMARY.json','--output-dir',root/'planning','--workers','4')
  call('execute','run_boya_external_trials.py','--queue',root/'planning/queue_manifest.json','--diagnostic',root/'diagnostic','--output',root/'execution','--workers','4','--render')
  result=read(root/'execution/RESULTS.json');status.update(status='complete',stage='complete',successes=result['successes'],physics_steps=result['physical_steps'],elapsed_s=time.monotonic()-began,results_ref=dict(path=str(root/'execution/RESULTS.json'),sha256=sha(root/'execution/RESULTS.json')))
 except BaseException:
  status.update(status='infrastructure_error',error=traceback.format_exc(),elapsed_s=time.monotonic()-began);raise
 finally:save(root/'EVALUATION_PROGRESS.json',status)
 print(json.dumps(status,indent=2),flush=True);return status
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--scene-manifest',type=Path);run(p.parse_args())
