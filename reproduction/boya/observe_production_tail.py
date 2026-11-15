"""Read-only owned process, raw-tmpfs journal growth and GPU0 observer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import author_production_tail as D

B,R=D.B,D.R
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();p=D.verify(a.run_dir)
first=B.read(B.read(p['window_index']['path'])['windows'][0]['protocol']['path'])
start=time.monotonic();previous={};a.output.parent.mkdir(parents=True,exist_ok=True)
B.save(a.output.with_suffix('.STARTED.json'),dict(process=R.process_identity(os.getpid()),
    protocol=B.binding(a.run_dir/D.PROTOCOL),source=B.binding(__file__),started_unix_s=time.time()),exclusive=True)
with a.output.open('x')as stream:
    while True:
        owned=[]
        for file in a.run_dir.glob('resource_slots/*/*.json'):
            value=B.read(file);identity=value.get('process')
            if value.get('state')!='running'or not identity or R.process_identity(identity['pid'])!=identity:continue
            try:stat=Path('/proc',str(identity['pid']),'stat').read_text().rsplit(')',1)[1].split()
            except(FileNotFoundError,ProcessLookupError):continue
            owned.append(dict(kind=value['resource'],slot=value['slot'],process=identity,stage_dir=value['stage_dir'],
                rss_bytes=int(stat[21])*os.sysconf('SC_PAGE_SIZE'),cpu_ticks=int(stat[11])+int(stat[12])))
        sizes={str(file):file.stat().st_size for file in a.run_dir.glob('windows/*/scenes/*/g*/execution/MULTI_OBJECT_CONTACTS.jsonl.gz')}
        for intent in a.run_dir.glob('windows/*/scenes/*/g*/execution/*.INTENT.json'):
            file=Path(B.read(intent)['spool'])/'CONTACTS.raw.jsonl'
            if file.exists():sizes[str(file)]=file.stat().st_size
        growth={key:size-previous.get(key,size)for key,size in sizes.items()};previous=sizes
        row=dict(unix_s=time.time(),workers=owned,active_kits=sum(w['kind']=='kit'for w in owned),
            active_audits=sum(w['kind']=='audit'for w in owned),journal_bytes=sizes,journal_growth_bytes=growth)
        try:
            q=subprocess.run([first['inputs']['nvidia_smi']['path'],'--id='+D.P.GPU_UUID,
                '--query-gpu=uuid,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],
                check=True,capture_output=True,text=True,timeout=5.)
            row['gpu0_nvml_csv']=q.stdout.strip()
        except Exception as error:row['sampling_error']=repr(error)
        stream.write(json.dumps(row)+'\n');stream.flush()
        if(a.run_dir/'OUTER_WALL.json').exists():break
        if time.monotonic()-start>15000:raise RuntimeError('Observer deadline; no simulator signal sent')
        time.sleep(2.)
B.save(a.output.with_suffix('.FINISHED.json'),dict(finished_unix_s=time.time(),samples=B.binding(a.output),
    new_Kit_launches=0,new_physics_steps=0),exclusive=True)
