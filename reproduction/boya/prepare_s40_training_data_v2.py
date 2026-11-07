"""Reuse S40 exporter with the explicit all-arm-segment temporal contract.

The adapter is frozen by the resumed collection protocol. Existing artifact,
label, canonical480 and terminal-prefix checks remain unchanged.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path('/home/wangrenpeng/AnyDexGrasp/reproduction/boya')))
from execute_collection_queue_v3 import install_temporal_contract
install_temporal_contract()
from prepare_s40_training_data import main
if __name__=='__main__':main()
