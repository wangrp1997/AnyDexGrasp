"""Stream only original top-level record IDs; do not load gigabytes of features."""
from __future__ import annotations
import hashlib
import json
import mmap
from pathlib import Path
import re
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/mnt/public/datasets/AnyDexGrasp/logs/data/decision_model')
PAIRS={
 'Allegro':(BASE/'allegro/Allegro_information_single_point_train.json',BASE/'allegro/Allegro_information_single_point_test.json'),
 'DH3':(BASE/'dh3/obj140/train_sr_single_point.json',BASE/'dh3/obj140/test_sr_single_point.json'),
 'Inspire':(BASE/'inspire/obj140/train_sr_new_type10_single_point.json',BASE/'inspire/obj140/test_sr_new_type10_single_point.json'),
}
PATTERN=re.compile(rb'^    "([^"\n]+)": \{',re.MULTILINE)


def inspect(path):
    with path.open('rb') as stream:
        with mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as contents:
            keys=[match.group(1).decode('utf-8') for match in PATTERN.finditer(contents)]
            digest=hashlib.sha256(contents).hexdigest()
    if not keys:
        raise ValueError('Expected author pretty-JSON top-level ID structure was not found')
    return set(keys),{'path':str(path),'bytes':path.stat().st_size,'sha256':digest,
        'record_count':len(keys),'unique_record_ids':len(set(keys)),
        'duplicate_top_level_ids':len(keys)-len(set(keys)),
        'sorted_record_ids_sha256':hashlib.sha256('\n'.join(sorted(set(keys))).encode()).hexdigest()}


def main():
    start=time.monotonic();rows={}
    for hand,(train,test) in PAIRS.items():
        tr,tm=inspect(train);te,em=inspect(test)
        overlap=sorted(tr&te)
        rows[hand]={'train':tm,'test':em,'overlap_count':len(overlap),'overlap_ids':overlap,
                    'train_test_record_id_disjoint':not overlap}
    report={'scope':'official raw train/test record-ID overlap only','read_only_originals':True,
        'method':'mmap regex for exactly four-space top-level JSON object keys; full file SHA256 binding',
        'does_not_establish_object_or_feature_disjointness':True,
        'checkpoint_selection_may_use_released_test':True,
        'independent_generalization_claim':False,
        'hands':rows,'all_raw_record_ids_disjoint':all(row['train_test_record_id_disjoint'] for row in rows.values()),
        'wall_seconds':time.monotonic()-start}
    out=ROOT/'outputs/official_decision_validation_all_obj140/TRAIN_TEST_RECORD_ID_AUDIT.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
