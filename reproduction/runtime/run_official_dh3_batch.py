"""Reuse official models while executing the existing eleven-branch frame runner.

This wrapper changes model/checkpoint I/O only. The existing calibrated-frame,
original AST algorithms, original gates/collision and final top1 selection execute
unchanged. Each frame starts from the same declared NumPy/Torch seed. Scene
manifests bind rendered RGB-D provenance and are never network inputs.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import types

PROCESS_START = time.monotonic()
PROCESS_START_UNIX = time.time()
ROOT = Path(__file__).resolve().parents[2]


class OfficialModelCache:
    """Process-local checkpoint/model cache, with exact state-object guards.

    The first construction/loading follows the existing runner. Later requests
    for the same constructor signature and checkpoint return the same nn.Module;
    a strict load of the exact cached state object is a verified I/O no-op.
    No model forward, source function, candidate array or physical result is
    cached. The constructor-cache is scoped to these original DH3 checkpoints.
    """

    def __init__(self, runner, checkpoint, manifest):
        import torch
        from run_official_rgbd_frame import sha
        import validate_official_decision as decision
        self.torch = torch
        self.runner = runner
        self.checkpoint = checkpoint.resolve()
        self.manifest = manifest.resolve()
        self.original_torch_load = torch.load
        self.original_random_functions = {name:getattr(torch,name) for name in ('multinomial','randperm')}
        self.constructor_rng = {}
        self.original_source_overlay = runner.source_overlay
        self.original_call_cli = runner.call_cli
        self.original_load_classes = decision.load_original_classes
        self.original_safe_state_dict = decision.safe_state_dict
        artifacts = json.loads(manifest.read_text())['artifacts']
        self.head_items = [x for x in artifacts if x['hand'] == 'dh3_model']
        if len(self.head_items) != 4 or sorted(x['grasp_type'] for x in self.head_items) != [1,2,3,4]:
            raise ValueError('Require exactly four unambiguous released DH3 heads')
        self.allowed_paths = {Path(x['destination']).resolve() for x in self.head_items} | {self.checkpoint}
        self.file_signatures = {str(path):dict(sha256=sha(path),bytes=path.stat().st_size,
                                              mtime_ns=path.stat().st_mtime_ns)
                                for path in self.allowed_paths}
        self.representation_class = None
        self.representation_overlay = None
        self.representation_model = None
        self.representation_signature = None
        self.representation_checkpoint = None
        self.decision_classes = None
        self.decision_states = {}
        self.decision_models = {}
        self.pending_decision_path = None
        self.model_ids = {}
        self.events = []
        self.frame_start_counts = None
        self.counts = dict(representation_checkpoint_file_loads=0,decision_checkpoint_file_loads=0,
                           representation_model_constructions=0,decision_model_constructions=0,
                           representation_constructor_requests=0,decision_constructor_requests=0,
                           strict_state_dict_assignments=0,identical_cached_strict_load_skips=0,
                           representation_checkpoint_cache_hits=0,decision_state_cache_hits=0,
                           constructor_CPU_RNG_replays=0,constructor_CPU_RNG_pre_mismatches=0,
                           torch_multinomial_calls=0,torch_randperm_calls=0)

    def _check_file(self, path):
        path = Path(path).resolve()
        info = path.stat()
        initial = self.file_signatures[str(path)]
        if info.st_size != initial['bytes'] or info.st_mtime_ns != initial['mtime_ns']:
            raise RuntimeError('A pinned checkpoint changed while batch was running: '+str(path))
        return path

    def _guard_model(self, model, label):
        actual_load = model.load_state_dict
        first = {}
        def guarded_load(_model, state, strict=True, assign=False):
            if strict is not True or assign is not False:
                raise ValueError('Batch cache supports only the existing strict, non-assign loader')
            if first:
                if state is not first['state']:
                    raise RuntimeError('Cached model received a different state object: '+label)
                if _model.training:
                    raise RuntimeError('Cached inference model unexpectedly entered training mode')
                self.counts['identical_cached_strict_load_skips'] += 1
                self.events.append(dict(event='identical_cached_strict_load_skip',model=label,
                                        state_object_identity_verified=True))
                return first['result']
            result = actual_load(state, strict=True, assign=False)
            first.update(state=state,result=result)
            self.counts['strict_state_dict_assignments'] += 1
            self.events.append(dict(event='strict_state_dict_assignment',model=label,
                                    missing_keys=list(result.missing_keys),unexpected_keys=list(result.unexpected_keys)))
            return result
        model.load_state_dict = types.MethodType(guarded_load, model)
        self.model_ids[label] = id(model)
        return model

    def torch_load(self, file, *args, **kwargs):
        try:
            path = Path(file).resolve()
        except TypeError:
            return self.original_torch_load(file,*args,**kwargs)
        if path not in self.allowed_paths:
            return self.original_torch_load(file,*args,**kwargs)
        self._check_file(path)
        if kwargs.get('weights_only') is not True or kwargs.get('map_location') != 'cpu' or args:
            raise RuntimeError('Unexpected checkpoint load signature; cache will not weaken safety')
        if path == self.checkpoint:
            if self.representation_checkpoint is None:
                self.representation_checkpoint = self.original_torch_load(file,**kwargs)
                self.counts['representation_checkpoint_file_loads'] += 1
            else:
                self.counts['representation_checkpoint_cache_hits'] += 1
            return self.representation_checkpoint
        self.counts['decision_checkpoint_file_loads'] += 1
        return self.original_torch_load(file,**kwargs)

    def _construct(self, constructor, label, args, kwargs):
        before=self.torch.get_rng_state().clone()
        cuda_before=self.torch.cuda.get_rng_state_all()
        model=self._guard_model(constructor(*args,**kwargs),label)
        after=self.torch.get_rng_state().clone()
        cuda_after=self.torch.cuda.get_rng_state_all()
        cuda_unchanged=all(self.torch.equal(x,y) for x,y in zip(cuda_before,cuda_after))
        if not cuda_unchanged:
            raise RuntimeError('Original constructor unexpectedly consumes CUDA RNG; explicit transition needed')
        self.constructor_rng[label]=(before,after)
        self.events.append(dict(event='constructor_rng_recorded',model=label,
            CPU_before_sha256=hashlib.sha256(before.numpy().tobytes()).hexdigest(),
            CPU_after_sha256=hashlib.sha256(after.numpy().tobytes()).hexdigest(),
            CUDA_constructor_rng_unchanged=True))
        return model

    def _replay_constructor_rng(self, label):
        before,after=self.constructor_rng[label]
        match=self.torch.equal(self.torch.get_rng_state(),before)
        if match:
            self.torch.set_rng_state(after)
            self.counts['constructor_CPU_RNG_replays']+=1
        else:
            # A different forward can consume a different number of CPU draws.
            # Never reset that meaningful state to an unrelated saved value.
            self.counts['constructor_CPU_RNG_pre_mismatches']+=1
        self.events.append(dict(event='cached_constructor_rng_transition',model=label,
            incoming_CPU_state_equals_first_constructor_pre_state=match,
            recorded_CPU_post_state_replayed=match))

    def representation_factory(self, *args, **kwargs):
        signature = (args,tuple(sorted(kwargs.items())))
        self.counts['representation_constructor_requests'] += 1
        if self.representation_model is None:
            self.representation_signature = signature
            self.representation_model = self._construct(self.representation_class,'representation',args,kwargs)
            self.counts['representation_model_constructions'] += 1
        elif signature != self.representation_signature:
            raise RuntimeError('Representation constructor recipe changed between frames')
        else:
            self._replay_constructor_rng('representation')
        return self.representation_model

    def source_overlay(self):
        if self.representation_class is None:
            self.representation_class,self.representation_overlay = self.original_source_overlay()
        return self.representation_factory,self.representation_overlay

    def load_classes(self):
        if self.decision_classes is None:
            self.decision_classes = self.original_load_classes()
        train,inference,dataset = self.decision_classes
        return train,self.decision_factory,dataset

    def safe_state_dict(self, path, _train, _inference):
        path = self._check_file(path)
        if path not in {Path(x['destination']).resolve() for x in self.head_items}:
            raise ValueError('Unexpected decision checkpoint in DH3-only cache')
        if self.decision_classes is None:
            self.decision_classes = self.original_load_classes()
        if path not in self.decision_states:
            train,inference,_ = self.decision_classes
            # Keep the original actual classes for the narrowly allowlisted
            # weights-only unpickler; the factory is never pickle-allowlisted.
            self.decision_states[path] = self.original_safe_state_dict(path,train,inference)
        else:
            self.counts['decision_state_cache_hits'] += 1
        self.pending_decision_path = path
        return self.decision_states[path]

    def decision_factory(self, *args, **kwargs):
        path = self.pending_decision_path
        if path is None or path not in self.decision_states:
            raise RuntimeError('Decision construction must follow its safe checkpoint loading')
        self.counts['decision_constructor_requests'] += 1
        signature = (args,tuple(sorted(kwargs.items())))
        if signature != ((),(('input_num',480),)):
            raise ValueError('Unexpected original DH3 decision recipe')
        if path not in self.decision_models:
            _,inference,_ = self.decision_classes
            label = 'decision:'+str(path)
            self.decision_models[path] = self._construct(inference,label,args,kwargs)
            self.counts['decision_model_constructions'] += 1
        else:
            self._replay_constructor_rng('decision:'+str(path))
        return self.decision_models[path]

    def call_cli(self, module, argv):
        if module.__name__ != 'run_official_dh3_frame':
            return self.original_call_cli(module,argv)
        module.load_original_classes = self.load_classes
        module.safe_state_dict = self.safe_state_dict
        # Correct the standalone runner's historical requested-load counter
        # before the hand stage binds the frame receipt SHA. This touches only
        # this new batch's receipts, never old single-frame outputs.
        values=list(map(str,argv))
        frame=Path(values[values.index('--frame-dir')+1])
        frame_path=frame/'RESULTS.json'
        frame_report=json.loads(frame_path.read_text())
        actual_rep_loads=(self.counts['representation_checkpoint_file_loads']-
                         self.frame_start_counts['representation_checkpoint_file_loads'])
        frame_report['checkpoint'].update(model_load_count=actual_rep_loads,
            shared_model_object_id=self.model_ids['representation'],
            strict_state_dict_load_this_frame=actual_rep_loads == 1,
            strict_model_verified_on_initial_load=True)
        write_json(frame_path,frame_report)
        result=self.original_call_cli(module,argv)
        hand_path=Path(values[values.index('--output-dir')+1])/'RESULTS.json'
        hand_report=json.loads(hand_path.read_text())
        hand_report['model_cache']=dict(shared_models=True,
            decision_checkpoint_file_loads_this_frame=(self.counts['decision_checkpoint_file_loads']-
                  self.frame_start_counts['decision_checkpoint_file_loads']),
            decision_model_objects={str(path):id(model) for path,model in self.decision_models.items()},
            all_shared_models_initially_strict_loaded=True)
        write_json(hand_path,hand_report)
        return result

    @contextlib.contextmanager
    def active(self):
        self.torch.load = self.torch_load
        self.runner.source_overlay = self.source_overlay
        self.runner.call_cli = self.call_cli
        for name,original in self.original_random_functions.items():
            def counted(*args,_name=name,_original=original,**kwargs):
                self.counts['torch_'+_name+'_calls']+=1
                return _original(*args,**kwargs)
            setattr(self.torch,name,counted)
        try:
            yield self
        finally:
            self.torch.load = self.original_torch_load
            self.runner.source_overlay = self.original_source_overlay
            self.runner.call_cli = self.original_call_cli
            for name,original in self.original_random_functions.items():setattr(self.torch,name,original)

    def write_rng(self, path):
        import numpy as np
        arrays={}
        labels=[]
        for index,(label,(before,after)) in enumerate(self.constructor_rng.items()):
            arrays['model_%d_before'%index]=before.numpy()
            arrays['model_%d_after'%index]=after.numpy()
            labels.append(label)
        arrays['model_labels']=np.array(labels)
        np.savez_compressed(path,**arrays)

    def receipt(self):
        return dict(counts=dict(self.counts),model_load_count=(self.counts['representation_checkpoint_file_loads']+
                    self.counts['decision_checkpoint_file_loads']),live_model_object_ids=dict(self.model_ids),
                    checkpoint_signatures=self.file_signatures,events=list(self.events),
                    constructor_CPU_rng_states='CONSTRUCTOR_RNG_STATES.npz',
                    original_forward_random_API_observers=['torch.multinomial','torch.randperm'],
                    boundary='Only checkpoint deserialization, constructor and repeated identical strict-state I/O are cached; all eleven forwards, scoring, collision queries and top1 ranking execute for every frame')


def write_json(path, obj):
    path.write_text(json.dumps(obj,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-manifest',type=Path,action='append',required=True,
                        help='Repeat in the frozen evaluation order; RGB-D lives beside each manifest')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--asset-root',type=Path,default=ROOT/'outputs/official_dh3_collision_assets_20261103_v2/assets')
    parser.add_argument('--checkpoint',type=Path,default=Path('/mnt/public/datasets/AnyDexGrasp/logs/model/checkpoint.tar.18'))
    parser.add_argument('--manifest',type=Path,default=Path('/mnt/public/datasets/AnyDexGrasp/metadata/model_obj140_manifest_20261103.json'))
    parser.add_argument('--table-dir',type=Path,default=Path('/mnt/public/datasets/AnyDexGrasp/source_tables/generate_mesh_and_pointcloud/dh3_urdf'))
    parser.add_argument('--native-deps',type=Path,default=ROOT/'outputs/native_deps')
    parser.add_argument('--grasp-api-source',type=Path,default=ROOT/'outputs/native_deps/graspnetAPI_source/graspnetAPI/grasp.py')
    parser.add_argument('--seed',type=int,default=20261103)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a fresh batch output directory; old results are retained')
    scenes = [path.resolve() for path in args.scene_manifest]
    if len({path.parent.name for path in scenes}) != len(scenes):
        parser.error('Scene directory names must be unique')
    for scene in scenes:
        if not scene.is_file():parser.error('Missing frozen scene manifest: '+str(scene))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report = dict(scope='Shared-model batch of original DH3 eleven-branch RGB-D offline inference',
                  source_revision='c9c4a43df33e40860417c7e2dd02f5122d3b2da2',seed_per_frame=args.seed,
                  process_start_unix=PROCESS_START_UNIX,
                  original_source_modified=False,physical_grasps_executed=0,robot_connection=False,
                  simulation_run=False,training_steps=0,success_rate_claim=False,status='starting',
                  ground_truth_pose_or_object_geometry_fed_to_network=False,frames=[],
                  timing_boundary='Process monotonic wall starts immediately after Python standard-library imports; includes dependency imports, checkpoint loads, new collision assets and every per-frame stage. External interpreter startup is excluded.',
                  timing_seconds={})
    cache = None
    try:
        before_imports = time.monotonic()
        import torch
        import run_official_dh3_augmented_frame as runner
        import select_original_dh3_pose as selection
        from run_official_rgbd_frame import sha
        report['timing_seconds']['runtime_module_imports'] = time.monotonic()-before_imports
        report['source_files'] = {str(path):sha(path) for path in [Path(__file__).resolve(),
            Path(runner.__file__).resolve(),Path(selection.__file__).resolve(),ROOT/'robot_dh3.py',
            ROOT/'reproduction/runtime/run_official_dh3_frame.py']}
        begin = time.monotonic()
        cache = OfficialModelCache(runner,args.checkpoint,args.manifest)
        report['timing_seconds']['cache_preparation_and_checkpoint_hashes'] = time.monotonic()-begin
        with cache.active():
            for index,scene in enumerate(scenes):
                frame_dir = args.output_dir/scene.parent.name
                frame_record = dict(name=scene.parent.name,scene_manifest=str(scene),
                    scene_manifest_sha256=sha(scene),output_dir=str(frame_dir.resolve()),
                    warm_models=(cache.counts['representation_model_constructions']==1 and
                                 cache.counts['decision_model_constructions']==4),
                    seed=args.seed,starting_cache_counts=dict(cache.counts),status='starting')
                report['frames'].append(frame_record)
                cache.frame_start_counts = dict(cache.counts)
                write_json(args.output_dir/'RESULTS.json',report)
                frame_start = time.monotonic()
                # The original runner counts actual native calls by wrapping
                # PointNet2 functions. Restore them after a frame to avoid
                # stacking observers across otherwise unchanged forwards.
                pointnet = sys.modules.get('pointnet2_utils')
                native_names = ['furthest_point_sample','cylinder_query','grouping_operation']
                saved_native = {name:getattr(pointnet,name) for name in native_names} if pointnet else None
                try:
                    argv=['--image-dir',scene.parent,'--scene-manifest',scene,
                          '--checkpoint',args.checkpoint,'--manifest',args.manifest,
                          '--table-dir',args.table_dir,'--native-deps',args.native_deps,
                          '--grasp-api-source',args.grasp_api_source,'--output-dir',frame_dir,
                          '--seed',args.seed,'--asset-root',args.asset_root]
                    runner.call_cli(runner,argv)
                    run_path = frame_dir/'RESULTS.json'
                    run = json.loads(run_path.read_text())
                    if run['status'] == 'failed':
                        raise RuntimeError('Existing single-frame runner reported failure: '+run.get('error',''))
                    frame_record.update(pipeline_status=run['status'],
                        representation_candidates=run.get('merged_candidates_after_original_get_grasp_gates'),
                        native_operator_call_counts=run.get('native_operator_call_counts'),
                        stage_timing_seconds=run.get('timing_seconds'),
                        hand_counts=run.get('hand_stage',{}).get('stages'),
                        collision_counts=run.get('collision_stage',{}).get('result',{}).get('stages'))
                    collision_path = frame_dir/'collision/RESULTS.json'
                    if collision_path.exists() and json.loads(collision_path.read_text()).get('collision_check_executed') is True:
                        selection_start = time.monotonic()
                        runner.call_cli(selection,['--collision-dir',frame_dir/'collision',
                                                   '--output-dir',frame_dir/'selection'])
                        selection_path = frame_dir/'selection/RESULTS.json'
                        chosen = json.loads(selection_path.read_text())
                        run['selection_stage'] = dict(directory=str(frame_dir/'selection'),
                            RESULTS_sha256=sha(selection_path),selected=chosen['selected'])
                        run['timing_seconds']['original_final_selection'] = time.monotonic()-selection_start
                        run['status'] = ('eleven_branch_original_offline_pipeline_and_top1_complete' if chosen['selected']
                                         else 'no_candidate_after_original_collision')
                        frame_record.update(selected=chosen['selected'],selection_dir=str((frame_dir/'selection').resolve()),
                            selected_hand=chosen.get('selected_hand'),status=run['status'])
                    else:
                        frame_record.update(selected=False,status=run['status'],
                                            stop_reason='Original gate yielded no pose for subsequent stages; no gate relaxed')
                    frame_record['wall_seconds'] = time.monotonic()-frame_start
                    run['checkpoint'].update(model_load_count=(
                        cache.counts['representation_checkpoint_file_loads']-
                        frame_record['starting_cache_counts']['representation_checkpoint_file_loads']),
                        shared_model_object_id=cache.model_ids.get('representation'))
                    run['frame_stage']['RESULTS_sha256']=sha(frame_dir/'frame/RESULTS.json')
                    run['batch_cache'] = dict(shared_models=True,cache_receipt='../MODEL_CACHE.json',
                        first_frame=index==0,per_frame_seed=args.seed,
                        model_load_count_this_frame=(cache.counts['representation_checkpoint_file_loads']+
                            cache.counts['decision_checkpoint_file_loads']-
                            frame_record['starting_cache_counts']['representation_checkpoint_file_loads']-
                            frame_record['starting_cache_counts']['decision_checkpoint_file_loads']),
                        global_model_objects=cache.model_ids)
                    run['timing_seconds']['batch_frame_with_selection_wall'] = frame_record['wall_seconds']
                    write_json(run_path,run)
                    frame_record['RESULTS_sha256'] = sha(run_path)
                except Exception as exc:
                    frame_record.update(status='failed',error_type=type(exc).__name__,error=str(exc),
                                        traceback=traceback.format_exc(),wall_seconds=time.monotonic()-frame_start)
                    print(json.dumps(dict(frame=frame_record['name'],status='failed',error=str(exc))),flush=True)
                finally:
                    pointnet = sys.modules.get('pointnet2_utils')
                    if pointnet is not None:
                        if saved_native:
                            for name,value in saved_native.items():setattr(pointnet,name,value)
                        else:
                            # First frame imported PointNet2 itself. Its original
                            # observers bind the actual function as _original.
                            for name in native_names:
                                fn=getattr(pointnet,name)
                                original=getattr(fn,'__kwdefaults__',{}).get('_original')
                                if original is not None:setattr(pointnet,name,original)
                    if torch.cuda.is_available():torch.cuda.synchronize(0)
                    frame_record['ending_cache_counts'] = dict(cache.counts)
                    cache.write_rng(args.output_dir/'CONSTRUCTOR_RNG_STATES.npz')
                    write_json(args.output_dir/'MODEL_CACHE.json',cache.receipt())
                    report['timing_seconds']['process_wall_so_far'] = time.monotonic()-PROCESS_START
                    write_json(args.output_dir/'RESULTS.json',report)
                    print(json.dumps(dict(frame=frame_record['name'],status=frame_record['status'],
                        wall_seconds=frame_record['wall_seconds'],selected=frame_record.get('selected'))),flush=True)
        report.update(status='batch_finished' if not any(x['status']=='failed' for x in report['frames']) else 'batch_finished_with_errors',
            completed_frames=len(report['frames']),selected_frames=sum(bool(x.get('selected')) for x in report['frames']),
            cache=cache.receipt())
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        if cache:report['cache']=cache.receipt()
        raise
    finally:
        report['process_end_unix'] = time.time()
        report['timing_seconds']['total_process_internal_wall'] = time.monotonic()-PROCESS_START
        report['timing_seconds']['cold_first_frame'] = report['frames'][0].get('wall_seconds') if report['frames'] else None
        report['timing_seconds']['warm_subsequent_frames'] = [x.get('wall_seconds') for x in report['frames'] if x['warm_models']]
        write_json(args.output_dir/'RESULTS.json',report)
        print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
