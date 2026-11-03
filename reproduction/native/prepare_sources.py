"""Prepare the exact accepted local sources/libraries; no package installation/build."""
import hashlib,json,pathlib,re,requests,shutil,subprocess,tarfile,torch
here=pathlib.Path(__file__).resolve().parent;repo=here.parents[1];root=repo/'outputs/native_deps'
(root/'archives').mkdir(parents=True,exist_ok=True)
components=json.loads((here/'components.json').read_text())
def download(url,name,sha,size):
 path=root/'archives'/name
 if not path.exists():
  with requests.get(url,stream=True,timeout=(20,90)) as response:
   response.raise_for_status()
   with path.with_suffix('.partial').open('wb') as stream:
    for chunk in response.iter_content(1024*1024):
     if chunk:stream.write(chunk)
  path.with_suffix('.partial').replace(path)
 if path.stat().st_size!=size or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
  raise RuntimeError('Source identity mismatch: '+str(path))
 return path
for entry in components['openblas']:
 name=pathlib.Path(entry['archive']).name
 package=download(entry['url'],name,entry['sha256'],entry['bytes'])
 subprocess.run(['dpkg-deb','-x',str(package),str(root/'ubuntu_openblas')],check=True)
local=root/'openblas';(local/'lib').mkdir(parents=True,exist_ok=True);(local/'include').mkdir(exist_ok=True)
for source,target in [(root/'ubuntu_openblas/usr/lib/x86_64-linux-gnu/openblas-pthread',local/'lib'),(root/'ubuntu_openblas/usr/include/x86_64-linux-gnu/openblas-pthread',local/'include')]:
 for path in source.iterdir():
  dest=target/path.name
  if not dest.exists() and not dest.is_symlink():dest.symlink_to(path)
site=pathlib.Path(torch.__file__).resolve().parent.parent
# The accepted machine reused existing wheel runtimes; no shared package is installed.
fortran=sorted((site/'scipy.libs').glob('libgfortran*.so.5*'))+sorted((site/'numpy.libs').glob('libgfortran*.so.5*'))
quad=sorted((site/'scipy.libs').glob('libquadmath*.so.0*'))+sorted((site/'numpy.libs').glob('libquadmath*.so.0*'))
if not fortran or not quad:raise RuntimeError('Existing wheel Fortran/Quadmath runtimes required; provision locally rather than install into a shared environment')
for name,source in [('libgfortran.so.5',fortran[0]),('libquadmath.so.0',quad[0])]:
 dest=local/'lib'/name
 if not dest.exists():dest.symlink_to(source)
needed=subprocess.run(['readelf','-d',str(fortran[0])],check=True,text=True,capture_output=True).stdout
for name in re.findall(r'\[(libquadmath[^\]]+)\]',needed):
 dest=local/'lib'/name
 if not dest.exists():dest.symlink_to(next((path for path in quad if path.name==name),quad[0]))
entry=components['minkowski_source'];name='MinkowskiEngine-'+entry['git_sha']+'.tar.gz'
archive=download(entry['url'],name,entry['archive_sha256'],entry['bytes'])
sources=root/'sources';sources.mkdir(exist_ok=True)
with tarfile.open(archive) as tar:
 top=tar.getmembers()[0].name.split('/')[0];tar.extractall(sources,filter='data')
author=sources/top;work=root/'build_src/MinkowskiEngine';work.parent.mkdir(exist_ok=True)
if not work.exists():shutil.copytree(author,work,symlinks=True)
changes=json.loads((here/'evidence/me_local_patch_receipt.json').read_text())['changes']
if not all(hashlib.sha256((work/c['file']).read_bytes()).hexdigest()==c['after_sha256'] for c in changes):
 if not all(hashlib.sha256((work/c['file']).read_bytes()).hexdigest()==c['before_sha256'] for c in changes):raise RuntimeError('ME working source is neither pristine nor the accepted patched version')
 subprocess.run(['patch','--batch','-p1','-i',str(here/'me_local_compat.patch')],cwd=work,check=True)
(work/'MinkowskiEngineBackend').mkdir(exist_ok=True)
pointnet=repo/'pointnet2';identity=json.loads((here/'evidence/pointnet2_source_receipt.json').read_text())['sha256']
if not all(hashlib.sha256((pointnet/path).read_bytes()).hexdigest()==sha for path,sha in identity.items()):raise RuntimeError('Official pointnet2 source differs from the accepted build identity')
pointnet_work=root/'build_src/pointnet2'
if not pointnet_work.exists():shutil.copytree(pointnet,pointnet_work,symlinks=True)
print('Prepared accepted local sources and OpenBLAS; no install or compilation performed')
