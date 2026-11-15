"""Complete old CPU diagnostic with isolated verified storage; excluded training."""
from pathlib import Path

import author_scaling_full_execution as ORIGINAL
import author_verified_execution as STORE


def run(prepared_dir,output_dir,*,protocol_path):
    vector=ORIGINAL.V;original=vector.source_bindings
    def sources():
        result=dict(original());result.update(STORE.source_bindings())
        result[str(Path(__file__).resolve())]=STORE.W.sha(__file__)
        return result
    vector.source_bindings=sources
    try:
        with STORE.install():return ORIGINAL.run(prepared_dir,output_dir,protocol_path=protocol_path)
    finally:vector.source_bindings=original


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);a=p.parse_args()
    run(a.prepared_dir,a.output_dir,protocol_path=a.protocol)
