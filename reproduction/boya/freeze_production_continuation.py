"""Freeze only unstarted original scenes, preserving draws and verified dependencies."""
import argparse
import copy
from pathlib import Path
import subprocess

import author_production_continuation as D

B=D.B
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--original-root',type=Path,required=True)
parser.add_argument('--output-root',type=Path,required=True);args=parser.parse_args()
old=args.original_root.resolve();root=args.output_root.resolve()
original=D.ORIGINAL.verify(old/'run');result=B.read(old/'run/PRODUCTION_RESULTS.json')
assert result['status']=='stopped'and result['known_final_execution_physics_steps']==0
assert not any(old.glob('run/windows/*/scenes/*/g*/RESIDENT_EXECUTION_ENTRY.json'))
first=original['queue']['scenes'][0];scene=old/'run/windows/00/scenes'/f'0000000_{first["scene_id"]}'
error=B.read(scene/'capture/ERROR.json')
assert error['physics_ticks']==error['initialization_object_steps']==0
assert 'ModuleNotFoundError' in error['error']and 'clarabel'in error['error']
assert len(list(old.glob('run/windows/*/scenes/*/stages/capture/STAGE.json')))==1
for row in original['queue']['scenes'][1:]:
    assert not any(old.glob(f'run/windows/*/scenes/*_{row["scene_id"]}/stages/*/STAGE.json'))
registry=B.read(D.P.OLD.REGISTRY);assert registry['state']=='released';D.R.assert_quiescent(old/'run')
root.mkdir(parents=True,exist_ok=False)
continuation=dict(original_protocol=B.binding(old/'run'/D.PROTOCOL),original_result=B.binding(old/'run/PRODUCTION_RESULTS.json'),
    failed_capture_error=B.binding(scene/'capture/ERROR.json'),excluded_failed_scene=first,
    remaining_original_scene_indices=list(range(1,32)),remaining_scenes=31,remaining_slots=124,
    combined_original_scene_budget=32,combined_original_slot_budget=128,failed_scene_retries=0,
    seed=original['queue']['seed'],sampler_redraws=0,source_adapter=D.AST_ADAPTATIONS)
B.save(root/'CONTINUATION_PLAN.json',continuation,exclusive=True)
queue=copy.deepcopy(original['queue']);queue['scenes']=queue['scenes'][1:]
queue.update(scene_counts=dict(fit=15,internal=16),maximum_gesture_scene_attempts=124,
    continuation=B.binding(root/'CONTINUATION_PLAN.json'),original_scene_indices=list(range(1,32)))
queue['inputs']=B.normalized_bindings(queue['inputs'])
queue['inputs'].update(continuation=B.binding(root/'CONTINUATION_PLAN.json'),original_protocol=continuation['original_protocol'],
    original_result=continuation['original_result'],failed_capture_error=continuation['failed_capture_error'])
B.save(root/'QUEUE.json',queue,exclusive=True)
reference=B.read('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/development/batch_v1/BATCH_PROTOCOL.json')
options={k:reference['inputs'][k]['path']for k in('calibration','cloud_manifest','backend','checkpoint','python')}
base=B.freeze(root/'QUEUE.json',root/'source_batch',**options,domain=reference['domain'],
    native_deps=reference['native_deps'],fit_per_gesture=1000,internal_per_gesture=200,stage_timeout_s=1200.)
base['inherited_env']=copy.deepcopy(reference['inherited_env'])
base['sources'].update(original_production_adapter=B.binding(D.ORIGINAL.__file__))
overlay=B.ROOT/'outputs/native_deps/clarabel_0111_overlay/clarabel'
for path in sorted(overlay.iterdir()):
    if path.is_file():base['inputs']['clarabel_dependency:'+path.name]=B.binding(path)
B.save(root/'source_batch/BATCH_PROTOCOL.json',base)
# Exercise exactly the subprocess environment before any Kit/physical dispatch.
probe=subprocess.run([options['python'],'-B','-c',
    'import sys,clarabel,torch,numpy; sys.path.insert(0,'+repr(str(B.ROOT.parent/'CanonDex'))+'); import canondex.grasp.cooking_audit; print(clarabel.__file__); print(torch.__version__)'],
    env=B._environment(base),capture_output=True,text=True,timeout=30.)
B.save(root/'DEPENDENCY_PREFLIGHT.json',dict(returncode=probe.returncode,stdout=probe.stdout,stderr=probe.stderr,
    inherited_env=base['inherited_env'],reference=B.binding(Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/development/batch_v1/BATCH_PROTOCOL.json')),
    Kit_launches=0,physics_steps=0,model_loads=0),exclusive=True)
if probe.returncode!=0:raise RuntimeError('Dependency preflight failed before Kit: '+probe.stderr)
protocol=D.freeze(root/'source_batch',root/'run')
print(dict(status='frozen',remaining_scenes=31,remaining_slots=124,failed_scene_retries=0,
    seed=queue['seed'],run_dir=str(root/'run'),protocol=B.binding(root/'run'/D.PROTOCOL)))
