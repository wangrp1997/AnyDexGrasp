"""Fixed-world worker with unchanged physics and accelerated geometry checks."""
import argparse
from pathlib import Path
import author_live_collection_service as ORIGINAL
from author_fast_live_world import World

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--lease-fd',type=int,required=True);a=p.parse_args()
    ORIGINAL.World=World;ORIGINAL.serve(a.root,a.protocol,a.lease_fd)
