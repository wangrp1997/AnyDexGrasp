"""Bounded read-only host observations of this run's slots, journals and GPU0."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import author_collection_batch as B
import author_collection_wave_resources as R
import author_pipeline_collection as P

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();protocol=P.verify(args.run_dir)
args.output.parent.mkdir(parents=True,exist_ok=True)
start=time.monotonic();previous={}
B.save(args.output.with_suffix('.STARTED.json'),dict(process=R.process_identity(os.getpid()),
    started_unix_s=time.time(),protocol=B.binding(args.run_dir/P.PROTOCOL),observer=B.binding(__file__),
    new_Kit_launches=0,new_physics_steps=0),exclusive=True)
with args.output.open('x') as stream:
    while True:
        owned=[]
        for path in args.run_dir.glob('resource_slots/*/*.json'):
            value=B.read(path);identity=value.get('process')
            if value.get('state')!='running' or not identity or R.process_identity(identity['pid'])!=identity:continue
            try:stat=Path('/proc',str(identity['pid']),'stat').read_text().rsplit(')',1)[1].split()
            except (FileNotFoundError,ProcessLookupError):continue
            owned.append(dict(kind=value['resource'],slot=value['slot'],process=identity,
                stage_dir=value['stage_dir'],rss_bytes=int(stat[21])*os.sysconf('SC_PAGE_SIZE'),
                cpu_ticks=int(stat[11])+int(stat[12])))
        sizes={str(path):path.stat().st_size for path in args.run_dir.glob('scenes/*/g*/execution/MULTI_OBJECT_CONTACTS.jsonl.gz')}
        growth={key:size-previous.get(key,size) for key,size in sizes.items()};previous=sizes
        row=dict(unix_s=time.time(),monotonic_s=time.monotonic(),workers=owned,
            active_kits=sum(r['kind']=='kit' for r in owned),active_audits=sum(r['kind']=='audit' for r in owned),
            journal_bytes=sizes,journal_growth_bytes=growth,cpu_clock_ticks_per_second=os.sysconf('SC_CLK_TCK'))
        try:
            result=subprocess.run([protocol['inputs']['nvidia_smi']['path'],'--id='+protocol['gpu_uuid'],
                '--query-gpu=uuid,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],
                check=True,capture_output=True,text=True,timeout=5.)
            row['gpu0_nvml_csv']=result.stdout.strip()
        except Exception as error:row['gpu_sampling_error']=repr(error)
        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        if (args.run_dir/'OUTER_WALL.json').exists():break
        if time.monotonic()-start>3600:
            raise RuntimeError('Observer deadline reached; no simulator signal was sent')
        time.sleep(2.)
B.save(args.output.with_suffix('.FINISHED.json'),dict(finished_unix_s=time.time(),
    samples=B.binding(args.output),new_Kit_launches=0,new_physics_steps=0),exclusive=True)
