"""Read only this pilot's owned resource slots, journal growth and fixed GPU0."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import author_production_collection as D

B,R=D.B,D.R
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();protocol=D.verify(args.run_dir)
first=D.B.read(D.B.read(protocol['window_index']['path'])['windows'][0]['protocol']['path'])
args.output.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic();previous={}
B.save(args.output.with_suffix('.STARTED.json'),dict(process=R.process_identity(os.getpid()),
    protocol=B.binding(args.run_dir/D.PROTOCOL),source=B.binding(__file__),started_unix_s=time.time()),exclusive=True)
with args.output.open('x')as stream:
    while True:
        owned=[]
        for path in args.run_dir.glob('resource_slots/*/*.json'):
            value=B.read(path);identity=value.get('process')
            if value.get('state')!='running'or not identity or R.process_identity(identity['pid'])!=identity:continue
            try:stat=Path('/proc',str(identity['pid']),'stat').read_text().rsplit(')',1)[1].split()
            except(FileNotFoundError,ProcessLookupError):continue
            owned.append(dict(kind=value['resource'],slot=value['slot'],process=identity,stage_dir=value['stage_dir'],
                rss_bytes=int(stat[21])*os.sysconf('SC_PAGE_SIZE'),cpu_ticks=int(stat[11])+int(stat[12])))
        sizes={str(p):p.stat().st_size for p in args.run_dir.glob('windows/*/scenes/*/g*/execution/MULTI_OBJECT_CONTACTS.jsonl.gz')}
        growth={key:size-previous.get(key,size)for key,size in sizes.items()};previous=sizes
        row=dict(unix_s=time.time(),workers=owned,active_kits=sum(w['kind']=='kit'for w in owned),
            active_audits=sum(w['kind']=='audit'for w in owned),journal_bytes=sizes,journal_growth_bytes=growth)
        try:
            smi=subprocess.run([first['inputs']['nvidia_smi']['path'],'--id='+D.P.GPU_UUID,
                '--query-gpu=uuid,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],
                check=True,capture_output=True,text=True,timeout=5.)
            row['gpu0_nvml_csv']=smi.stdout.strip()
        except Exception as error:row['sampling_error']=repr(error)
        stream.write(json.dumps(row)+'\n');stream.flush()
        if(args.run_dir/'OUTER_WALL.json').exists():break
        if time.monotonic()-start>15000:raise RuntimeError('Observer deadline; no simulator signal sent')
        time.sleep(2.)
B.save(args.output.with_suffix('.FINISHED.json'),dict(finished_unix_s=time.time(),samples=B.binding(args.output),
    new_Kit_launches=0,new_physics_steps=0),exclusive=True)
