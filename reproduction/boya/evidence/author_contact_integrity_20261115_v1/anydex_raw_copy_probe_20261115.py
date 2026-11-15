import concurrent.futures,hashlib,json,os,time
from pathlib import Path
root=Path('/tmp/anydex_raw_copy_probe_20261115_v1');root.mkdir(exist_ok=False)
source=Path('/tmp/anydex_verified_full_control_20261115_v1/eb25429ed0e703833ab1c9d879a337cf2a0c79554d05463b941045c9e55923ba/CONTACTS.raw.jsonl')
expected='e62a6e4d93d3280669686a40cabcb4a3d46c796e9a3c768109b6a52485965716'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def one(i):
 p=root/f'COPY_{i}.raw.jsonl';h=hashlib.sha256();n=size=0;t=time.monotonic()
 with source.open('rb')as src,p.open('xb')as out:
  for b in src:
   h.update(b);size+=len(b);n+=1
   if out.write(b)!=len(b):raise IOError('short write')
  out.flush();os.fsync(out.fileno())
 st=p.stat();observed=sha(p);f=os.open(p,os.O_RDONLY);holes=[];pos=0
 try:
  while pos<st.st_size:
   hole=os.lseek(f,pos,os.SEEK_HOLE)
   if hole>=st.st_size:break
   data=os.lseek(f,hole,os.SEEK_DATA);holes.append([hole,data]);pos=data
 finally:os.close(f)
 v=dict(worker=i,path=str(p),expected=expected,before_write=h.hexdigest(),after_fsync=observed,bytes=size,records=n,holes=holes,wall_s=time.monotonic()-t,status='verified'if observed==h.hexdigest()==expected else'failed')
 print(json.dumps(v),flush=True);return v
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:rows=list(pool.map(one,range(4)))
value=dict(scope='Four standalone raw copies using original complete JSONL row write boundaries; no JSON reserialization, compression, Kit, physics, or training',source=str(source),rows=rows,source_sha256_after=sha(source),actual_Kit_starts=0,physics_steps=0)
p=Path('/home/wangrenpeng/AnyDexGrasp/outputs/boya_contact_integrity_20261115_v1/RAW_COPY_PROBE.json');p.write_text(json.dumps(value,indent=2)+'\n')
