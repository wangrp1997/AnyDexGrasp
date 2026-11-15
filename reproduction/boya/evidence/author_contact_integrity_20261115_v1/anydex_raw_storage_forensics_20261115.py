import hashlib,json,time
from pathlib import Path
root=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/verified_full_control_20261115_v1')
rows=[]
for i in range(4):
 p=root/f'cpu_4/worker_{i:02d}/result/env_0/execution/MULTI_OBJECT_CONTACTS.jsonl.gz.STATE.json'
 s=json.loads(p.read_text()); rows.append(dict(worker=i,state=str(p),path=s['raw_spool'],expected=s['expected']))
reference=Path(rows[3]['path']); began=time.monotonic()
for row in rows:
 path=Path(row['path']);digest=hashlib.sha256();spans=[];different=zero_different=offset=0
 before=path.stat()
 with path.open('rb') as stream, reference.open('rb') as ref:
  while True:
   a=stream.read(4<<20); b=ref.read(4<<20)
   if not a and not b:break
   digest.update(a)
   if a!=b:
    start=None
    for j,(x,y) in enumerate(zip(a,b)):
     if x!=y:
      different+=1;zero_different+=x==0
      if start is None:start=offset+j
     elif start is not None:
      if len(spans)<64:spans.append([start,offset+j])
      start=None
    if start is not None and len(spans)<64:spans.append([start,offset+len(a)])
   offset+=len(a)
 after=path.stat(); row.update(observed=dict(sha256=digest.hexdigest(),bytes=offset),
  same_stat=(before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),
  differs_from_worker03_bytes=different,zero_bytes_where_reference_nonzero=zero_different,first_difference_spans=spans)
 print(json.dumps({k:v for k,v in row.items()if k not in('path','state')},ensure_ascii=False),flush=True)
value=dict(scope='Read-only comparison; no repair, replay, recompression, physics, or labels',rows=rows,wall_s=time.monotonic()-began)
out=Path('/home/wangrenpeng/AnyDexGrasp/outputs/boya_contact_integrity_20261115_v1/RAW_FORENSICS.json')
out.write_text(json.dumps(value,indent=2)+'\n')
