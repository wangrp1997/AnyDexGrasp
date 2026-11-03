"""Optional original-generator dependencies: local wheels only, no shared installs."""
import hashlib,json,pathlib,requests,shutil,stat,subprocess,sys,tarfile,zipfile,os
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
here=pathlib.Path(__file__).resolve().parent;repo=here.parents[1];root=repo/'outputs/native_deps'
meta=json.loads((here/'pybullet_components.json').read_text())
for name in ['archives','sources','build_src','wheels','pybullet_tmp','python_overlay']:(root/name).mkdir(parents=True,exist_ok=True)
def download(entry,name):
 target=root/'archives'/name
 if not target.exists():
  with requests.get(entry['url'],stream=True,timeout=(20,120)) as response:
   response.raise_for_status()
   with target.with_suffix('.partial').open('wb') as f:
    for chunk in response.iter_content(1024*1024):
     if chunk:f.write(chunk)
  target.with_suffix('.partial').replace(target)
 if target.stat().st_size!=entry['bytes'] or hashlib.sha256(target.read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('official package identity mismatch: '+name)
 return target
transforms=download(meta['transforms3d'],meta['transforms3d']['filename'])
archive=download(meta['pybullet'],'pybullet-3.2.7.tar.gz')
with tarfile.open(archive) as tar:tar.extractall(root/'sources',filter='data')
original=root/'sources/pybullet-3.2.7';work=root/'build_src/pybullet-3.2.7'
if not work.exists():shutil.copytree(original,work,symlinks=True)
setup=work/'setup.py';identity=json.loads((here/'evidence/pybullet_build_patch.json').read_text())
current=hashlib.sha256(setup.read_bytes()).hexdigest()
if current!=identity['after_sha256']:
 if current!=identity['before_sha256']:raise RuntimeError('unexpected pybullet setup working copy')
 subprocess.run(['patch','--batch','-p1','-i',str(here/'pybullet_build_workers.patch')],cwd=work,check=True)
existing=list((root/'wheels').glob('pybullet-3.2.7-*.whl'))
compatible=[p for p in existing if parse_wheel_filename(p.name)[3].intersection(sys_tags())]
if not compatible:
 env=os.environ.copy();env.update(MAX_JOBS='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TMPDIR=str(root/'pybullet_tmp'),CC='/usr/bin/gcc',CXX='/usr/bin/g++')
 with (root/'pybullet_build_v1.log').open('w') as log:
  subprocess.run([sys.executable,'-B','setup.py','build','--build-base',str(root/'pybullet_build'),'bdist_wheel','--dist-dir',str(root/'wheels'),'--bdist-dir',str(root/'pybullet_bdist')],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 compatible=[p for p in (root/'wheels').glob('pybullet-3.2.7-*.whl') if parse_wheel_filename(p.name)[3].intersection(sys_tags())]
if len(compatible)!=1:raise RuntimeError('expected one matching local pybullet wheel')
overlay=root/'python_overlay';wheels=[transforms,compatible[0]];receipt=[]
for wheel in wheels:
 count=0
 with zipfile.ZipFile(wheel) as z:
  if z.testzip():raise RuntimeError('wheel CRC failure')
  for info in z.infolist():
   target=overlay/info.filename
   if not target.resolve().is_relative_to(overlay.resolve()) or stat.S_ISLNK(info.external_attr>>16):raise RuntimeError('unsafe wheel path')
   if info.is_dir():target.mkdir(parents=True,exist_ok=True);continue
   data=z.read(info);target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists() and target.read_bytes()!=data:raise RuntimeError('overlay conflict: '+info.filename)
   target.write_bytes(data);count+=1
 receipt.append({'filename':wheel.name,'bytes':wheel.stat().st_size,'sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'unpacked_files':count})
sys.path.insert(0,str(overlay))
import pybullet as p,pybullet_data,transforms3d
client=p.connect(p.DIRECT)
if client<0:raise RuntimeError('DIRECT connect failed')
count=p.getNumBodies(physicsClientId=client);p.disconnect(physicsClientId=client)
result={'source_components':meta,'local_wheels':receipt,'overlay':str(overlay),'pybullet_file':p.__file__,'pybullet_data_path':pybullet_data.getDataPath(),'transforms3d_file':transforms3d.__file__,'DIRECT_client':client,'empty_world_bodies':count,'disconnected':True,'physics_steps':0,'all_passed':True,'env_sh_modified':False,'shared_install':False,'new_environment':False}
(root/'python_overlay_rebuild_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
