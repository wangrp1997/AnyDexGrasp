"""Accept original planner time scaling for all three arm motion segments.

Reuse the locked v2 queue/worker/controller. Only source-bound command admission
and its provenance change; no target/time/phase array is modified. Native task
stop certificates use the same explicit temporal-contract adapter.
"""
from __future__ import annotations
import hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');BOYA=ROOT/'reproduction/boya'
sys.path.insert(0,str(BOYA))
BASE=BOYA/'execute_collection_queue_v2.py';BASE_SHA='e83c87b593e77a0b797dc4e1a8257d018deebfce99aec67d270bfb54f7d6210f'
if hashlib.sha256(BASE.read_bytes()).hexdigest()!=BASE_SHA:raise ValueError('Read-only v2 executor changed')
import execute_collection_queue_v2 as base
_ORIGINAL_PREPARE=getattr(base,'_anydex_v2_original_prepare',base.prepare)
base._anydex_v2_original_prepare=_ORIGINAL_PREPARE

def command_contract(plan,arrays):
 n=plan.get('total_command_steps');dt=plan.get('dt_s')
 if type(n)is not int or n<2100 or dt!=.004:raise ValueError('Full original plan at original dt required')
 if any(arrays[k].dtype.hasobject for k in arrays.files):raise ValueError('Pickled command arrays forbidden')
 names=['settle','home_to_pregrasp','approach','close','hold','lift'];fixed={'settle':50,'close':250,'hold':50};minimum={'home_to_pregrasp':750,'approach':500,'lift':500}
 timing=plan.get('segment_timing',{})
 if set(timing)!=set(minimum):raise ValueError('Exactly three original arm timed segments required')
 counts=dict(fixed)
 for name,floor in minimum.items():
  t=timing[name];count=t.get('steps')
  if type(count)is not int or count<floor:raise ValueError('Original segment minimum violated: '+name)
  fields=['duration_s','velocity_ratio','max_velocity_rad_s','max_acceleration_rad_s2','max_jerk_rad_s3','acceleration_limit_rad_s2','jerk_limit_rad_s3']
  if any(type(t.get(k))not in(int,float)or not np.isfinite(t[k])or t[k]<0 for k in fields):raise ValueError('Finite original timing required: '+name)
  if (not np.isclose(t['duration_s'],count*dt,rtol=0,atol=1e-12)or t['velocity_ratio']>1.+1e-8
      or t['acceleration_limit_rad_s2']!=3. or t['jerk_limit_rad_s3']!=30.
      or t['max_acceleration_rad_s2']>3.+1e-7 or t['max_jerk_rad_s3']>30.+1e-7):raise ValueError('Original timing bounds differ/exceeded: '+name)
  counts[name]=count
 if sum(counts.values())!=n:raise ValueError('Total differs from complete original segment counts')
 phase=arrays['phase'];expected=np.repeat(names,[counts[x]for x in names])
 if phase.shape!=(n,)or not np.array_equal(phase,expected):raise ValueError('Original contiguous phase arrays differ from metadata')
 width=len(arrays['joint_names'])
 if width<=0 or len(set(map(str,arrays['joint_names'])))!=width:raise ValueError('Unique original joint names required')
 for key,shape in {'joint_target':(n,width),'desired_palm':(n,4,4),'time_s':(n,),'initial_position':(width,)}.items():
  if arrays[key].shape!=shape or not np.isfinite(arrays[key]).all():raise ValueError('Original finite shape differs: '+key)
 if not np.array_equal(arrays['time_s'],np.arange(1,n+1)*dt):raise ValueError('Original time grid differs')
 start=51;boundaries=[]
 for name in names[1:]:boundaries.append(dict(name=name,start_step=start,end_step=start+counts[name]-1));start+=counts[name]
 if plan.get('phases')!=boundaries:raise ValueError('Original boundaries differ from complete phase arrays')
 return dict(expected_command_steps=n,expected_phase_counts=counts,expected_phase_boundaries=boundaries,original_dt_s=dt,
  approach_extension_steps=counts['approach']-500,segment_extension_steps={k:counts[k]-minimum[k]for k in minimum},
  command_policy='All source planner home/approach/lift timing retained; fixed settle/close/hold; immutable complete commands, no resampling')

def prepare(args):
 result=_ORIGINAL_PREPARE(args);p=Path(__file__).resolve();result['source_sha256'][str(p)]=base.sha(p)
 result['source_sha256'][str(BASE)]=BASE_SHA
 result['temporal_contract_adapter']=dict(path=str(p),sha256=base.sha(p),all_other_queue_and_worker_functions_reused=True,original_worker_path=str(BASE))
 return result

def install_temporal_contract():
 base.command_contract=command_contract
 # The certificate module captures the imported function at import time. Keep
 # its one admission dependency aligned; all physical prefix checks are reused.
 import terminal_collection_outcome
 terminal_collection_outcome.command_contract=command_contract
 return dict(path=str(Path(__file__).resolve()),sha256=base.sha(__file__),base_sha256=BASE_SHA)

install_temporal_contract();base.prepare=prepare;base.SCHEMA='anydex_boya_collection_queue_v3'
if __name__=='__main__':base.main()
