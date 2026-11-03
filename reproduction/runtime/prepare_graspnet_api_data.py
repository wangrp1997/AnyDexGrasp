"""Fetch only pinned official GraspNet API data-class source; install nothing."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
COMMIT='eb57dd2092d8dbe05312a29c3d0c22f3226efbfc'
FILES={
 'graspnetAPI/grasp.py':'344ad4b8b9216abdfee6ff810ae37fe21890cd185760ea10c0361a7a0a19d4d1',
 'LICENSE':'02971c8d6576ae8417c9701c645b79d50d3fc978daf6c2138fd0e7ddcbf7de5a',
}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/native_deps/graspnetAPI_source')
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args();rows=[]
    for relative,expected in FILES.items():
        path=args.output_dir/relative
        url='https://raw.githubusercontent.com/graspnet/graspnetAPI/'+COMMIT+'/'+relative
        if path.exists():
            content=path.read_bytes()
        elif args.offline:
            raise FileNotFoundError('Pinned source missing in offline mode: '+str(path))
        else:
            with urllib.request.urlopen(url,timeout=45) as response:
                content=response.read()
        digest=hashlib.sha256(content).hexdigest()
        if digest!=expected:
            raise ValueError('Pinned official source SHA mismatch; refusing to publish '+relative)
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            partial=path.with_name(path.name+'.partial');partial.write_bytes(content);partial.replace(path)
        rows.append({'path':str(path.resolve()),'source':url,'bytes':len(content),'sha256':digest})
    receipt={'source_repo':'https://github.com/graspnet/graspnetAPI','commit':COMMIT,
        'package_installed':False,'role':'Original pure-data Grasp/GraspGroup AST source; no hardware/NMS execution',
        'files':rows}
    (args.output_dir/'SOURCE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
