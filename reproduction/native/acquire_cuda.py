import concurrent.futures, hashlib, json, pathlib, requests, tarfile, os, time, shutil, torch
root=pathlib.Path(__file__).resolve().parents[2]/'outputs/native_deps'
root.mkdir(parents=True,exist_ok=True)
meta=json.loads((pathlib.Path(__file__).parent/'nvidia_components_13.0.0.json').read_text())
archive_root=root/'archives';archive_root.mkdir(exist_ok=True)
extract_root=root/'redist';extract_root.mkdir(exist_ok=True)
toolkit=root/'cuda-13.0';toolkit.mkdir(exist_ok=True)
def acquire(name):
 entry=meta[name]['linux-x86_64'];url='https://developer.download.nvidia.com/compute/cuda/redist/'+entry['relative_path'];dest=archive_root/pathlib.Path(entry['relative_path']).name
 started=time.monotonic()
 if not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest()!=entry['sha256']:
  temporary=dest.with_suffix(dest.suffix+'.partial')
  with requests.get(url,stream=True,timeout=(20,90)) as r:
   r.raise_for_status()
   with temporary.open('wb') as f:
    for chunk in r.iter_content(1024*1024):
     if chunk:f.write(chunk)
  digest=hashlib.sha256(temporary.read_bytes()).hexdigest()
  if digest!=entry['sha256']:raise RuntimeError(name+' sha256 mismatch')
  temporary.replace(dest)
 digest=hashlib.sha256(dest.read_bytes()).hexdigest()
 if dest.stat().st_size!=int(entry['size']):raise RuntimeError(name+' size mismatch')
 with tarfile.open(dest) as tar:
  members=tar.getmembers();top=members[0].name.split('/')[0]
  for member in members:
   if not (extract_root/member.name).resolve().is_relative_to(extract_root.resolve()):raise RuntimeError('unsafe archive path')
  tar.extractall(extract_root,filter='data')
 return {'name':name,'url':url,'archive':str(dest),'extracted':str(extract_root/top),'sha256':digest,'bytes':dest.stat().st_size,'seconds':time.monotonic()-started}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(acquire,['cuda_nvcc','cuda_cudart','cuda_cccl','cuda_crt','libnvvm']))
links=[]
def merge(source,kind):
 for path in source.rglob('*'):
  rel=path.relative_to(source);dst=toolkit/rel
  if path.is_dir():dst.mkdir(parents=True,exist_ok=True);continue
  dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists() or dst.is_symlink():continue
  dst.symlink_to(path);links.append({'destination':str(dst),'source':str(path),'kind':kind})
for row in rows:merge(pathlib.Path(row['extracted']),row['name'])
existing=pathlib.Path(torch.__file__).resolve().parent.parent/'nvidia/cu13'
if not existing.is_dir():raise RuntimeError('This recipe requires an existing CUDA13 PyTorch runtime; do not install into a shared environment automatically')
for name in ['include','lib']:
 if (existing/name).exists():
  for path in (existing/name).rglob('*'):
   rel=path.relative_to(existing/name);dst=toolkit/('lib64' if name=='lib' else 'include')/rel
   if path.is_dir():dst.mkdir(parents=True,exist_ok=True);continue
   dst.parent.mkdir(parents=True,exist_ok=True)
   if not dst.exists() and not dst.is_symlink():dst.symlink_to(path);links.append({'destination':str(dst),'source':str(path),'kind':'existing nvidia cu13 wheel'})
# NVIDIA redistrib uses lib/. Torch extension tooling expects lib64/.
lib=toolkit/'lib';lib64=toolkit/'lib64';lib64.mkdir(exist_ok=True)
if lib.exists():
 for path in lib.iterdir():
  dst=lib64/path.name
  if not dst.exists() and not dst.is_symlink():dst.symlink_to(path)
for name in ['cublas','cublasLt','cusparse','cudart','nvrtc']:
 dst=lib64/('lib'+name+'.so')
 candidates=sorted(lib64.glob('lib'+name+'.so.*'))
 if not dst.exists() and candidates:dst.symlink_to(candidates[0])
receipt={'downloaded':rows,'total_download_bytes':sum(r['bytes'] for r in rows),'local_toolkit':str(toolkit),'links':links,'shared_environment_changed':False,'global_install':False}
(root/'cuda_acquisition_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ['downloaded','total_download_bytes','local_toolkit','shared_environment_changed','global_install']},indent=2))

# Actual follow-up corrections used before the accepted build.
nvcc=toolkit/'bin/nvcc'
if nvcc.is_symlink():
 source=nvcc.resolve();nvcc.unlink();shutil.copy2(source,nvcc)
# Keep the existing cuBLAS declarations and their matching type header together.
p=toolkit/'include/library_types.h';matching=existing/'include/library_types.h'
if not matching.exists():raise RuntimeError('Matching installed CUDA13 library_types.h is required')
if p.is_symlink():p.unlink()
if not p.exists():p.symlink_to(matching)
