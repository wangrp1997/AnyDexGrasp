"""Read-only reuse of the existing Isaac complete-arm backend for AnyDex inputs."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[2]
CANON=Path('/home/wangrenpeng/CanonDex')
os.environ.setdefault('CANONDEX_COLLISION_PARTITION_CACHE',str(ROOT/'outputs/boya_collision_partition_cache'))
sys.path[:0]=[str(CANON/'scripts/research'),str(CANON)]
import run_template_arm_positive as backend
if __name__=='__main__':backend.main(sys.argv[1:])
