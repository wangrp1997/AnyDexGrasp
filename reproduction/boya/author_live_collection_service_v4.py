"""Persistent same-world worker using original capture stationarity policy."""
import argparse
from pathlib import Path
import author_live_collection_service as ORIGINAL
from author_live_world_v4 import World


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--lease-fd',type=int,required=True);a=p.parse_args()
    ORIGINAL.World=World;ORIGINAL.serve(a.root,a.protocol,a.lease_fd)
