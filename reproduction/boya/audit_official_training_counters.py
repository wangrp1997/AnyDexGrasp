#!/usr/bin/env python3
"""Read released decision-head BN counters without training, execution or pickle fallback."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
sys.path.insert(0,str(ROOT/'reproduction/runtime'))
import validate_official_decision as original

def run(output):
 output=Path(output).resolve()
 if output.exists():raise ValueError('Fresh evidence output required')
 paths=sorted(Path('/mnt/public/datasets/AnyDexGrasp/logs/model').rglob('*.pth'))
 if len(paths)!=31:raise ValueError('Expected exactly31 downloaded decision files')
 train,inference,_=original.load_original_classes();records=[]
 inputs={str(p):original.sha(p)for p in paths}
 for path in paths:
  state,metadata=original.safe_state_dict(path,train,inference)
  counts=[int(state[f'bn{i}.num_batches_tracked'])for i in range(1,8)]
  epoch=int(path.stem.rsplit('_',1)[-1]);parts=path.relative_to('/mnt/public/datasets/AnyDexGrasp/logs/model').parts
  if len(set(counts))!=1:raise ValueError('Unexpected unequal layer counters')
  records.append(dict(path=str(path),sha256=inputs[str(path)],hand=parts[0],BN_train_forward_count_per_layer=counts,
   filename_epoch_zero_based=epoch,BN_count_divided_by_filename_epoch_plus_one=counts[0]/(epoch+1),
   conv7_output_count=int(state['conv7.weight'].shape[0]),metadata=metadata,
   optimizer_state_observed=False,optimizer_steps_inferred=False))
 groups={}
 for hand in sorted({r['hand']for r in records}):
  rows=[r for r in records if r['hand']==hand];counts=[r['BN_train_forward_count_per_layer'][0]for r in rows]
  groups[hand]=dict(files=len(rows),BN_counts=counts,min=min(counts),max=max(counts),greater_than_boya100=sum(x>100 for x in counts))
 sources={str(p):original.sha(p)for p in [Path(__file__).resolve(),ROOT/'reproduction/runtime/validate_official_decision.py',ROOT/'train_multifinger.py',ROOT/'models/minkowski_graspnet.py']}
 report=dict(schema='anydex_official_training_counter_audit_v1',status='complete',file_count=31,by_hand=groups,records=records,
  source_sha256=sources,input_sha256=inputs,
  all_layers_equal=True,all_outputs20=all(r['conv7_output_count']==20 for r in records),
  all_BN_counts_integer_multiple_of_filename_epoch_plus_one=all(r['BN_count_divided_by_filename_epoch_plus_one'].is_integer()for r in records),
  greater_than_boya100_total=sum(r['BN_train_forward_count_per_layer'][0]>100 for r in records),
  train_forward_counter_scope='PyTorch BN increments in training forward, not eval(); author released train loop has one training forward per optimizer batch, but BN resets/extra training forwards/resume history are not known.',
  optimizer_metadata_scope='Author save_model torch.save(multifinger_net) writes the model, not optimizer.state_dict. This weights-only audit sees no optimizer state; BN counts are not asserted as exact optimizer-step counts.',
  released_training_history_boundary='Ratios8/9/10/14 in some DH3/Inspire checkpoints cannot arise from the checked-in first100+batch128 recipe with one batch/epoch and no extra forwards. Published files and current script do not encode one fully recoverable training history.',
  conclusion='Published heads are not uniformly trained for more updates than Boya100. Need not increase epochs or change loss based solely on low Boya success.',
  new_training_steps=0,new_simulation_steps=0,no_unsafe_pickle_fallback=True,
  all_inputs_unchanged=all(original.sha(Path(p))==h for p,h in inputs.items()),
  all_sources_unchanged=all(original.sha(Path(p))==h for p,h in sources.items()))
 if not report['all_inputs_unchanged']or not report['all_sources_unchanged']:raise ValueError('Frozen file changed')
 output.mkdir(parents=True);(output/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items()if k not in ['input_sha256','source_sha256','records']},indent=2));return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);run(p.parse_args().output_dir)
if __name__=='__main__':main()
