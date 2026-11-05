"""Resume a frozen Boya pilot: shared inputs, at most two independent CPU trials.

Protocol (paths relative to protocol file):
 {"objects":[{"id":"...","observation":".../observation.json",
   "representation_dir":".../representation","pools":{"1":".../pool_g01"}}],
  "gesture_ids":[1,6,8], "limits":{"max_rounds":2,"max_arm_per_trial":4,
   "max_arm_per_object_gesture":8,"max_actual_per_object_gesture":2},
  "source_sha256":{...}}

--stage prepare freezes already prepared observation/representation/pool artifacts;
it never renders, runs a network or builds geometry. Those stages run once per
object outside this CPU dispatcher (geometry preparation must be serialized).
--stage execute uses only those pools and run_first_trial.py. Success does not
stop round two. Explicit infrastructure failures stop new dispatch; live trials
finish safely. Resume acknowledges retained dead rounds without rerunning them. A versioned
continuation may inherit a stopped parent ledger into a separate output run;
parent results stay read-only and historical/local costs are reported separately.
Progress/cost are atomic JSON.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import numpy as np

CAPS = dict(max_rounds=2, max_arm_per_trial=4, max_arm_per_object_gesture=8,
            max_actual_per_object_gesture=2)
TERMINAL_TRIAL = {'complete', 'no_full_arm_plan_in_declared_budget',
                  'execution_infrastructure_error', 'execution_not_label_eligible'}
INFRASTRUCTURE_TRIAL = {'execution_infrastructure_error', 'interrupted_infrastructure_null'}


def infrastructure_failure(result):
    return (result.get('status') in INFRASTRUCTURE_TRIAL or
            bool(result.get('unconfirmed_execution_attempts')) or
            bool(result.get('evaluation_infrastructure_error')))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path, default=None):
    try: return json.loads(Path(path).read_text())
    except FileNotFoundError: return default


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()) + '.' + str(threading.get_ident()))
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def now(): return datetime.now(timezone.utc).isoformat()


def resolve(path, base):
    p = Path(path); return (base / p).resolve() if not p.is_absolute() else p.resolve()


def proc_info(pid):
    try:
        # /proc stat comm can contain spaces/parentheses; fields after final ')'.
        s = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return {'state': s[0], 'pgrp': int(s[2]), 'start_ticks': s[19]}
    except (FileNotFoundError, ProcessLookupError, PermissionError): return None


def group_live(receipt):
    pid = receipt.get('pid')
    if not pid: return False
    own = proc_info(pid)
    if own and own['start_ticks'] != receipt.get('start_ticks'): return False
    if own and own['state'] != 'Z': return True
    # A crashed trial wrapper may leave its native child in the same session.
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            info = proc_info(int(p.name))
            if info and info['state'] != 'Z' and info['pgrp'] == pid: return True
    return False


def find_trial_process(trial_dir, script):
    """Recover the narrow launch->PID-receipt crash gap, without reading env."""
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try: argv = (p / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
        words = [v.decode(errors='replace') for v in argv if v]
        if str(script) in words and '--run-dir' in words:
            i = words.index('--run-dir')
            if i + 1 < len(words) and words[i+1] == str(trial_dir):
                info = proc_info(int(p.name))
                if info and info['state'] != 'Z': return dict(pid=int(p.name), **info)
    return None


def consumed_indices(trial_dir):
    try: summary = read(Path(trial_dir) / 'trial_summary.json', {})
    except json.JSONDecodeError: summary = {}
    used = {int(p['index']) for p in summary.get('plans', [])}
    # run_first_trial opens this log before starting each prepare subprocess.
    for log in Path(trial_dir).glob('prepare_*.log'):
        try: used.add(int(log.stem.split('_')[-1]))
        except ValueError: pass
    return sorted(used)


def pending_indices(selected, consumed, rounds_done, limits):
    if rounds_done >= limits['max_rounds']: return []
    left = limits['max_arm_per_object_gesture'] - len(set(consumed))
    return [int(i) for i in selected if int(i) not in set(consumed)][:max(0, min(left, limits['max_arm_per_trial']))]


def round_cost(rounds):
    """Counts only the supplied records; inherited and local ledgers stay separate."""
    return dict(arm_prepare_calls=sum(len(r.get('consumed_indices',[])) for r in rounds),
        actual_grasp_attempts=sum(r.get('actual_grasp_attempts',0) for r in rounds),
        unconfirmed_execution_attempts=sum(r.get('unconfirmed_execution_attempts',0) for r in rounds),
        physics_steps=sum(r.get('physics_steps',0) for r in rounds),
        training_labels=sum(r.get('training_label') is not None for r in rounds),
        positive_labels=sum(r.get('training_label')==1 for r in rounds),
        negative_labels=sum(r.get('training_label')==0 for r in rounds),
        null_trials=sum(r.get('terminal',False) and r.get('training_label') is None for r in rounds),
        infrastructure_failure_trials=sum(infrastructure_failure(r) for r in rounds),
        zero_step_trials=sum(r.get('terminal',False) and r.get('physics_steps_confirmed') is True and r.get('physics_steps')==0 for r in rounds),
        no_arm_admission_trials=sum(r.get('status')=='no_full_arm_plan_in_declared_budget' for r in rounds),
        reported_trial_wall_s_sum=sum(r.get('reported_trial_wall_s') or 0 for r in rounds),
        wall_sum_is_not_parallel_end_to_end=True)


class Pilot:
    def __init__(self, protocol_path, run_dir, backend_config, calibration, workers=2, root=None):
        self.root = Path(root or Path(__file__).resolve().parent.parent.parent).resolve()
        self.protocol_path = Path(protocol_path).resolve(); self.protocol = read(self.protocol_path)
        self.run = Path(run_dir).resolve(); self.run.relative_to(self.root / 'outputs')
        self.backend = resolve(backend_config or self.protocol['backend_config'], self.protocol_path.parent)
        self.calibration = resolve(calibration or self.protocol['calibration'], self.protocol_path.parent)
        if not 1 <= workers <= 2: raise ValueError('CPU worker concurrency is limited to 1..2')
        self.continuation = self.protocol.get('continuation')
        self.inherited_rounds = set()
        self.workers = workers; self.lock = threading.RLock(); self.stop = threading.Event()
        self.limits = {k: self.protocol.get('limits', {}).get(k, v) for k,v in CAPS.items()}
        for k,v in self.limits.items():
            if type(v) is not int or not 1 <= v <= CAPS[k]: raise ValueError('Invalid authorized budget: ' + k)
        if self.limits['max_rounds'] > self.limits['max_actual_per_object_gesture']:
            raise ValueError('Each round may execute one grasp; round cap exceeds actual cap')
        self.gestures = self.protocol['gesture_ids']; objects = self.protocol['objects']
        if not 1 <= len(objects) <= 8 or not 1 <= len(self.gestures) <= 3:
            raise ValueError('Pilot is bounded by 8 objects and 3 gestures')
        if any(type(g) is not int or g not in (1,6,8) for g in self.gestures):
            raise ValueError('This pilot is frozen to source gesture IDs 1,6,8')
        if len(set(self.gestures)) != len(self.gestures): raise ValueError('Duplicate gestures')
        self.jobs = []
        for obj in objects:
            oid = obj['id']
            if not oid or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in oid):
                raise ValueError('Object ID must be a simple stable directory identifier')
            for gesture in self.gestures:
                entry = obj.get('pools', {}).get(str(gesture), str(self.run/'pools'/oid/f'g{int(gesture)}'))
                directory = entry['directory'] if isinstance(entry, dict) else entry
                self.jobs.append(dict(id=f'{oid}__g{int(gesture):02d}', object_id=oid, gesture_id=int(gesture),
                    observation=str(resolve(obj.get('observation', str(self.run/'observations'/oid/'observation.json')), self.protocol_path.parent)),
                    representation_dir=str(resolve(obj.get('representation_dir',str(self.run/'representations'/oid)), self.protocol_path.parent)),
                    pool=str(resolve(directory, self.protocol_path.parent))))
                repair = obj.get('collision_refinement')
                if repair:
                    value = repair.get('path') if isinstance(repair,dict) else repair
                    self.jobs[-1]['collision_refinement'] = str(resolve(value,self.protocol_path.parent))
                    if isinstance(repair,dict) and repair.get('sha256'):
                        self.jobs[-1]['collision_refinement_sha256'] = repair['sha256']
        if len({j['id'] for j in self.jobs}) != len(self.jobs): raise ValueError('Duplicate object/gesture jobs')
        self.first = self.root / 'reproduction/boya/run_first_trial.py'
        self.progress = {'schema':'anydex_boya_pilot_progress_v1','protocol_sha256':sha(self.protocol_path),
            'created_at':now(),'updated_at':now(),'phase':'not_started','limits':self.limits,
            'max_cpu_workers':workers,'network_forward_dispatches':0,'geometry_dispatches':0,
            'scope':'Frozen-pool Isaac training collection; independent scenes, not same-scene regrasp',
            'jobs':{j['id']:dict(status='pending', rounds=[], **j) for j in self.jobs}}


    def continuation_inputs(self, jobs, bind):
        """Validate a stopped parent, preserving original source via its Git commit.

        Updated source is a new protocol segment. Historical Python inputs may
        be verified against their declared commit; assets/results must retain
        their original bytes. Nothing is written under the parent run.
        """
        spec=self.continuation
        if not spec:return None
        parent=resolve(spec['parent_run'],self.protocol_path.parent)
        if parent==self.run or parent in self.run.parents or self.run in parent.parents:
            raise ValueError('Continuation needs a separate sibling output run')
        parent.relative_to(self.root/'outputs')
        pp=parent/'PREPARED.json';pr=parent/'progress.json'
        bind(pp,spec['prepared_sha256']);bind(pr,spec['progress_sha256'])
        old=read(pp);progress=read(pr)
        if old['protocol_sha256']!=progress['protocol_sha256']:
            raise ValueError('Parent protocol/PREPARED/progress disagree')
        if progress['phase'] not in ('paused_infrastructure','blocked_frozen_input','complete_with_infrastructure_errors','complete'):
            raise ValueError('Parent is not safely stopped')
        revision=spec['source_revision']
        if not isinstance(revision,str) or not 7<=len(revision)<=40 or any(c not in '0123456789abcdefABCDEF' for c in revision):
            raise ValueError('Explicit parent source commit required')
        commit=subprocess.check_output(['git','rev-parse','--verify',revision+'^{commit}'],cwd=self.root,text=True).strip()
        archived={}
        for path,expected in old['frozen_inputs'].items():
            path=Path(path)
            if sha(path)==expected:continue
            if not path.is_relative_to(self.root) or path.suffix!='.py':
                raise ValueError('Parent non-source frozen input changed: '+str(path))
            relative=path.relative_to(self.root).as_posix()
            raw=subprocess.check_output(['git','show',commit+':'+relative],cwd=self.root)
            if hashlib.sha256(raw).hexdigest()!=expected:
                raise ValueError('Parent source differs from declared historical commit: '+relative)
            archived[str(path)]=dict(revision=commit,git_path=relative,sha256=expected,
                                     current_segment_sha256=sha(path))
        oldjobs={j['id']:j for j in old['jobs']}
        if set(oldjobs)!={j['id'] for j in jobs} or self.limits!=old['limits']:
            raise ValueError('Continuation changed fixed jobs or budgets')
        inherited={};completed=[]
        for job in jobs:
            previous=oldjobs[job['id']]
            for key in ['object_id','gesture_id','observation','representation_dir','pool','selected_indices','pool_status']:
                if job[key]!=previous[key]:raise ValueError('Continuation changed source pool/observation: '+key)
            pj=progress['jobs'][job['id']];rows=pj.get('rounds',[])
            if len(rows)>self.limits['max_rounds'] or any(not r.get('terminal') for r in rows):
                raise ValueError('Parent contains nonterminal rounds; reconcile before continuation')
            consumed=[]
            for i,row in enumerate(rows):
                directory=Path(row['directory']).resolve();directory.relative_to(parent/'trials')
                receipt=read(directory/'process.json')
                if (receipt and group_live(receipt)) or (not receipt and find_trial_process(directory,self.first)):
                    raise ValueError('Parent trial process remains alive; no new dispatch')
                used=row.get('consumed_indices',[])
                if len(used)>self.limits['max_arm_per_trial'] or len(set(used))!=len(used) or set(used)&set(consumed) or not set(used)<=set(job['selected_indices']):
                    raise ValueError('Parent consumed indices violate frozen budget')
                consumed+=used
                required=['REQUEST.json','BATCH_RESULT.json','trial_summary.json']
                for name in required:bind(directory/name)
                optional=['process.json','trial_binding.json','training_record.json','bound_evaluation.json',
                    'execution/evaluation.json','execution/trace.npz','execution/native_audit.json',
                    'execution/initial_state_audit.json','execution/initial_object_restore.json']
                for name in optional:
                    if (directory/name).exists():bind(directory/name)
                for pattern in ['plan_*/plan.json','plan_*/commands.npz','plan_*/admission_failure.json','prepare_*.log']:
                    for path in sorted(directory.glob(pattern)):bind(path)
            if len(consumed)>self.limits['max_arm_per_object_gesture'] or sum(r.get('actual_grasp_attempts',0)+r.get('unconfirmed_execution_attempts',0) for r in rows)>self.limits['max_actual_per_object_gesture']:
                raise ValueError('Parent cumulative budget exceeded')
            inherited[job['id']]=copy.deepcopy(rows)
            if pj['status'] in ('complete','no_candidate'):completed.append(job['id'])
        audits=list((parent/'audits').glob('*.json')) if (parent/'audits').exists() else []
        audits += [resolve(p,self.protocol_path.parent) for p in spec.get('audit_paths',[])]
        for path in audits:bind(path)
        return dict(parent_run=str(parent),prepared_sha256=spec['prepared_sha256'],
            progress_sha256=spec['progress_sha256'],parent_protocol_sha256=old['protocol_sha256'],
            source_revision=commit,historical_source_verification=archived,
            inherited_rounds=inherited,inherited_complete_jobs=completed,
            parent_final_cost=copy.deepcopy(progress['cost']),audit_paths=[str(p) for p in audits],
            boundary='Read-only parent results; current source is an explicitly different protocol segment')

    def seed_continuation(self, prepared):
        lineage=prepared.get('continuation')
        if not lineage:return
        self.inherited_rounds={(jid,i) for jid,rows in lineage['inherited_rounds'].items() for i in range(len(rows))}
        self.progress['continuation']=dict(parent_run=lineage['parent_run'],
            prepared_sha256=lineage['prepared_sha256'],progress_sha256=lineage['progress_sha256'],
            source_revision=lineage['source_revision'],parent_final_cost=lineage['parent_final_cost'])
        self.progress['inherited_round_keys']=[[j,i] for j,i in sorted(self.inherited_rounds)]
        for jid,rows in lineage['inherited_rounds'].items():
            self.progress['jobs'][jid]['rounds']=copy.deepcopy(rows)
            if jid in lineage['inherited_complete_jobs']:
                self.progress['jobs'][jid]['status']='complete'
            elif rows:self.progress['jobs'][jid]['status']='continuation_pending'

    def freeze_inputs(self):
        inputs = {}; jobs = []; shared = [self.protocol_path, self.backend, self.calibration, self.root/'robot_dh3.py']
        shared += list((self.root/'reproduction/boya').glob('*.py'))
        config = read(self.backend)
        if config.get('reference_plan'): shared.append(Path(config['reference_plan']))
        shared += [Path(config[k]) for k in ['robot_urdf','hand_urdf','hand_scene','package',
                   'physics_config','diagnostic_config','arm_collision_refinement']]
        def bind(path, expected=None):
            path = Path(path).resolve(); digest = sha(path)
            if expected is not None and expected != digest: raise ValueError('Frozen input mismatch: '+str(path))
            if str(path) in inputs and inputs[str(path)] != digest: raise ValueError('Input changed during freeze')
            inputs[str(path)] = digest; return str(path)
        shared_paths = [bind(p) for p in shared]
        for path, digest in self.protocol.get('source_sha256', {}).items(): bind(resolve(path,self.protocol_path.parent), digest)
        if not all(flag in self.first.read_text() for flag in ['--observation','--source-npz','--candidate-indices']):
            raise ValueError('run_first_trial.py needs the explicit shared-input/index API')
        object_cache = {}
        for job in self.jobs:
            obs_path = Path(job['observation']); frame = Path(job['representation_dir']); pool = Path(job['pool'])
            key = (str(obs_path),str(frame))
            if key not in object_cache:
                obs = read(obs_path); receipt = read(frame/'RESULTS.json')
                if not obs or not receipt or receipt.get('full_representation_forward') is not True:
                    raise ValueError('Actual shared observation and representation required: '+job['object_id'])
                paths = [bind(obs_path), bind(frame/'RESULTS.json'), bind(frame/'raw_camera_candidates.npz')]
                if (frame/'representation_outputs.npz').exists(): paths.append(bind(frame/'representation_outputs.npz'))
                paths += [bind(item['path'],item['sha256']) for item in obs['files'].values()]
                with np.load(frame/'raw_camera_candidates.npz',allow_pickle=False) as z:
                    n = len(z['candidates'])
                    if z['candidates'].shape != (n,15) or z['candidate_features'].shape != (n,2261):
                        raise ValueError('Source candidate/feature contract changed')
                object_cache[key] = (paths,n)
            paths,n = object_cache[key]; job_paths = shared_paths + paths
            gen = read(pool/'generation.json'); empty = read(pool/'no_candidate.json')
            item = dict(job, input_paths=list(job_paths))
            if job.get('collision_refinement'):
                if '--collision-refinement' not in self.first.read_text():raise ValueError('Trial wrapper lacks object refinement API')
                item['input_paths'].append(bind(job['collision_refinement'],job.get('collision_refinement_sha256')))
            if gen is None and empty is not None:
                item.update(pool_status='no_candidate', selected_indices=[])
                item['input_paths'].append(bind(pool/'no_candidate.json'))
            else:
                if not gen or gen.get('complete') is not True or gen.get('gesture_id') != job['gesture_id']:
                    raise ValueError('Complete declared gesture pool required: '+str(pool))
                if gen.get('source_frame') and Path(gen['source_frame']).resolve()!=frame.resolve():
                    raise ValueError('Pool uses a different representation')
                if gen.get('source_observation') and Path(gen['source_observation']).resolve()!=obs_path.resolve():
                    raise ValueError('Pool uses a different observation')
                item['input_paths'] += [bind(pool/'generation.json'),bind(pool/'candidates.npz',gen['candidates_sha256']),bind(pool/'world_object_pose.json')]
                for path,digest in gen['input_sha256'].items(): item['input_paths'].append(bind(path,digest))
                selected = gen['selected_indices']; records = gen['records']
                if len(set(selected))!=len(selected) or any(type(i) is not int or not 0<=i<len(records) for i in selected):
                    raise ValueError('Invalid or duplicate frozen pool indices')
                for i in selected:
                    meta = records[i]
                    if not 0<=meta['source_candidate_index']<n or meta.get('additional_depth_index')!=0:
                        raise ValueError('Pool source index/depth0 collection contract changed')
                with np.load(pool/'candidates.npz',allow_pickle=False) as z:
                    if len(z['proposal_q']) != len(records): raise ValueError('Pool records/commands differ')
                item.update(pool_status='ready', selected_indices=selected)
            jobs.append(item)
        lineage=self.continuation_inputs(jobs,bind)
        result=dict(schema='anydex_boya_pilot_prepared_v1',protocol_sha256=sha(self.protocol_path),
            backend_config=str(self.backend),calibration=str(self.calibration),limits=self.limits,
            frozen_inputs=inputs,jobs=jobs,shared_input_objects=len(object_cache),network_forward_dispatches=0,
            geometry_dispatches=0,prepared_at=now())
        if lineage:result['continuation']=lineage
        return result

    def prepare(self):
        prepared = self.freeze_inputs(); path = self.run/'PREPARED.json'
        existing = read(path)
        if existing:
            for key in ['protocol_sha256','backend_config','calibration','limits','frozen_inputs','jobs','continuation']:
                if existing.get(key) != prepared.get(key): raise ValueError('Prepared batch inputs changed: '+key)
            return existing
        self.run.mkdir(parents=True,exist_ok=True); atomic(path,prepared)
        self.seed_continuation(prepared)
        self.progress['phase']='prepared'; self.publish(); return prepared

    def request_stop(self, kind, reason, job_id=None):
        with self.lock:
            self.stop.set()
            self.progress.setdefault('pause_reasons', []).append(
                dict(kind=kind, reason=reason, job_id=job_id, observed_at=now()))
            self.publish()

    def verify(self, prepared, job=None):
        paths = job['input_paths'] if job else prepared['frozen_inputs']
        for path in paths:
            if sha(path) != prepared['frozen_inputs'][path]:
                self.request_stop('frozen_input', 'Frozen input changed: '+path, job.get('id') if job else None)
                raise ValueError('Frozen input changed; no new dispatch: '+path)

    def publish(self):
        with self.lock:
            rounds=[r for j in self.progress['jobs'].values() for r in j.get('rounds',[])]
            self.progress['updated_at']=now()
            self.progress['cost']=dict(round_cost(rounds),
                no_candidate_jobs=sum(j['status']=='no_candidate' for j in self.progress['jobs'].values()),
                jobs_complete=sum(j['status'] in ['complete','no_candidate'] for j in self.progress['jobs'].values()),
                total_jobs=len(self.jobs),network_forward_dispatches=0,geometry_dispatches=0)
            if self.continuation:
                old=[];local=[]
                for jid,job in self.progress['jobs'].items():
                    for i,row in enumerate(job.get('rounds',[])):
                        (old if (jid,i) in self.inherited_rounds else local).append(row)
                self.progress['inherited_cost']=round_cost(old)
                self.progress['segment_cost']=round_cost(local)
            atomic(self.run/'progress.json',self.progress)

    def trial_snapshot(self, directory, terminal=False):
        directory=Path(directory)
        try: summary=read(directory/'trial_summary.json',{})
        except json.JSONDecodeError: summary={}
        try: evaluation=read(directory/'execution/evaluation.json',{})
        except json.JSONDecodeError:
            if terminal: evaluation={'infrastructure_error':'Malformed native evaluation JSON'}
            else: evaluation={}
        steps=evaluation.get('total_physics_steps',summary.get('physics_steps'))
        if type(steps) is not int or steps<0:
            trace=directory/'execution/trace.npz'; steps=None
            if trace.exists():
                with np.load(trace,allow_pickle=False) as z:
                    if 'phase' in z: steps=len(z['phase'])
        started=(directory/'execution.log').exists()
        if terminal and not started and steps is None: steps=0
        result=dict(directory=str(directory),status=summary.get('status','incomplete'),terminal=terminal,
            consumed_indices=consumed_indices(directory),physics_steps=steps or 0,
            physics_steps_confirmed=steps is not None,actual_grasp_attempts=int(steps is not None and steps>0),
            unconfirmed_execution_attempts=int(started and steps is None),training_label=None,
            failure_reason=summary.get('outcome_status'),plans=summary.get('plans',[]),
            selected_candidate=summary.get('selected_candidate'),
            reported_trial_wall_s=summary.get('elapsed_wall_s'))
        if terminal:
            if evaluation.get('infrastructure_error'):
                result['evaluation_infrastructure_error']=evaluation['infrastructure_error']
            try: record=read(directory/'training_record.json')
            except json.JSONDecodeError: record=None;result['failure_reason']='Malformed training record; no label'
            if record:
                spec=importlib.util.spec_from_file_location('pilot_trial_record',self.root/'reproduction/boya/trial_record.py')
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                original=record['execution_artifacts']
                verified=module.collect_record(record['binding'],original['evaluation_json'],original['commands_npz'])
                for key in ['result','label_eligible','outcome_status']:
                    if verified[key]!=record[key]: raise ValueError('Training record differs from actual artifacts')
                if verified['label_eligible']:result['training_label']=verified['result']
                result['failure_reason']=verified['outcome_status'];result['training_record_sha256']=sha(directory/'training_record.json')
            if not summary or summary.get('status') not in TERMINAL_TRIAL:
                result['status']='interrupted_infrastructure_null';result['failure_reason']='Dead/incomplete execution retained; not rerun or negative-labelled'
            if result['actual_grasp_attempts'] and result['training_label'] is None:
                result['failure_reason']=result['failure_reason'] or 'Executed but not label eligible'
        if (directory/'execution/evaluation.json').exists():result['evaluation_sha256']=sha(directory/'execution/evaluation.json')
        return result

    def update_round(self, job_id, round_index, value):
        with self.lock:
            rows=self.progress['jobs'][job_id]['rounds']
            while len(rows)<=round_index:rows.append({})
            rows[round_index]=value; self.publish()

    def run_round(self, job, round_index, indices, prepared):
        directory=self.run/'trials'/job['id']/f'round_{round_index+1:02d}'
        receipt_path=directory/'process.json'; receipt=read(receipt_path)
        process=None
        if directory.exists():
            request=read(directory/'REQUEST.json')
            if not request:raise ValueError('Existing trial directory has no frozen request; inspect without rerun')
            if request['candidate_indices']!=indices:raise ValueError('Resume candidate selection changed')
            if not receipt:
                found=find_trial_process(directory,self.first)
                if found:
                    receipt=dict(found,status='adopted',adopted_at=now());atomic(receipt_path,receipt)
                elif request.get('launch_reserved'):
                    # No evidence that dynamics ever started: preserve the round; do not launch it again.
                    return self.trial_snapshot(directory,terminal=True)
        else:
            self.verify(prepared,job);directory.mkdir(parents=True)
            request=dict(candidate_indices=indices,created_at=now(),launch_reserved=True,job_id=job['id'],
                protocol_sha256=prepared['protocol_sha256'],pool=job['pool'])
            atomic(directory/'REQUEST.json',request)
            argv=[sys.executable,'-B',str(self.first),'--run-dir',str(directory),
                '--backend-config',str(self.backend),'--calibration',str(self.calibration),
                '--candidate-dir',job['pool'],'--observation',job['observation'],
                '--source-npz',str(Path(job['representation_dir'])/'raw_camera_candidates.npz'),
                '--max-arm-plans',str(self.limits['max_arm_per_trial']),
                '--candidate-indices',*[str(i) for i in indices]]
            if job.get('collision_refinement'):
                argv += ['--collision-refinement',job['collision_refinement']]
            env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',
                MKL_NUM_THREADS='1',OMP_NUM_THREADS='4',CANONDEX_COLLISION_PARTITION_CACHE=str(self.root/'outputs/boya_collision_partition_cache'))
            with (directory/'runner.log').open('xb') as log:
                process=subprocess.Popen(argv,cwd=self.root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            info=proc_info(process.pid)
            receipt=dict(pid=process.pid,start_ticks=info['start_ticks'] if info else None,argv=argv,
                         status='running',started_at=now(),started_unix_s=time.time())
            atomic(receipt_path,receipt)
        if receipt and (process is not None or group_live(receipt)):
            last=0
            while True:
                if process is not None: process.poll()
                if not group_live(receipt): break
                if time.monotonic()-last>=5:
                    try: snapshot=self.trial_snapshot(directory)
                    except (ValueError, OSError): snapshot=dict(directory=str(directory),status='running_snapshot_unavailable')
                    self.update_round(job['id'],round_index,snapshot);last=time.monotonic()
                time.sleep(1)
            if process is not None:receipt['returncode']=process.returncode
            receipt.update(status='exited',finished_at=now());atomic(receipt_path,receipt)
        result=self.trial_snapshot(directory,terminal=True)
        result['offered_indices']=indices
        if not set(result['consumed_indices']).issubset(indices):raise ValueError('Backend attempted an index outside frozen request')
        if len(result['consumed_indices'])>self.limits['max_arm_per_trial']:raise ValueError('Per-trial arm budget exceeded')
        atomic(directory/'BATCH_RESULT.json',result);return result

    def run_job(self, job, prepared):
        jid=job['id'];self.progress['jobs'][jid]['status']='running';self.publish()
        if job['pool_status']=='no_candidate':
            self.progress['jobs'][jid].update(status='no_candidate',failure_reason='Frozen geometry pool has no candidate',training_label=None)
            self.publish();return
        consumed=[]
        for round_index in range(self.limits['max_rounds']):
            if self.stop.is_set():
                self.progress['jobs'][jid]['status']='paused_after_current_trial';self.publish();return
            if (jid,round_index) in self.inherited_rounds:
                result=copy.deepcopy(self.progress['jobs'][jid]['rounds'][round_index])
                if set(result['consumed_indices'])&set(consumed):raise ValueError('Inherited round overlaps prior consumed indices')
                consumed+=result['consumed_indices']
                if infrastructure_failure(result):
                    result['historical_infrastructure_failure_acknowledged_on_resume']=True
                    self.update_round(jid,round_index,result)
                continue
            existing=self.run/'trials'/jid/f'round_{round_index+1:02d}'
            request=read(existing/'REQUEST.json')
            indices=request['candidate_indices'] if request else pending_indices(job['selected_indices'],consumed,round_index,self.limits)
            if not indices:break
            if set(indices)&set(consumed):raise ValueError('Second round reuses an attempted pool index')
            result=self.run_round(job,round_index,indices,prepared)
            consumed+=result['consumed_indices']
            if infrastructure_failure(result):
                if (jid,round_index) in getattr(self,'acknowledged_dead_rounds',set()):
                    result['historical_infrastructure_failure_acknowledged_on_resume']=True
                else:
                    self.update_round(jid,round_index,result)
                    self.progress['jobs'][jid].update(status='infrastructure_error',
                        failure_reason=result.get('failure_reason') or result.get('status'))
                    self.request_stop('infrastructure',result.get('failure_reason') or result.get('status'),jid)
                    return
            self.update_round(jid,round_index,result)
            # Collect both outcomes; a positive label never stops round two.
        if len(set(consumed))>self.limits['max_arm_per_object_gesture']:raise ValueError('Per-gesture arm budget exceeded')
        rows=self.progress['jobs'][jid]['rounds']
        if sum(r.get('actual_grasp_attempts',0)+r.get('unconfirmed_execution_attempts',0) for r in rows)>self.limits['max_actual_per_object_gesture']:
            raise ValueError('Per-gesture actual/possibly-executed budget exceeded')
        self.progress['jobs'][jid]['status']='complete';self.publish()

    def execute(self, prepared):
        lineage=prepared.get('continuation')
        if not lineage:return self.execute_segment(prepared)
        # Lock an existing parent file read-only; never create/rewrite parent artifacts.
        parent=Path(lineage['parent_run'])
        with (parent/'.batch.lock').open('r') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if sha(parent/'PREPARED.json')!=lineage['prepared_sha256'] or sha(parent/'progress.json')!=lineage['progress_sha256']:
                raise ValueError('Parent ledger changed after continuation freeze')
            self.verify(prepared)
            return self.execute_segment(prepared)

    def execute_segment(self, prepared):
        previous=read(self.run/'progress.json')
        if previous:
            if previous['protocol_sha256']!=prepared['protocol_sha256']:raise ValueError('Resume protocol changed')
            self.progress=previous
        lineage=prepared.get('continuation')
        self.inherited_rounds={(jid,i) for jid,rows in (lineage['inherited_rounds'].items() if lineage else []) for i in range(len(rows))}
        self.stop.clear()
        if previous and previous.get('pause_reasons'):
            self.progress.setdefault('pause_history',[]).append(dict(
                acknowledged_on_resume_at=now(),reasons=previous['pause_reasons']))
        self.progress.update(phase='running',resumed_at=now(),max_cpu_workers=self.workers,pause_reasons=[])
        self.publish()
        self.acknowledged_dead_rounds=set()
        def trial_live(directory):
            receipt=read(directory/'process.json')
            if receipt:return group_live(receipt)
            return bool(find_trial_process(directory,self.first)) if (directory/'REQUEST.json').exists() else False
        def active(job):
            for i in range(self.limits['max_rounds']):
                directory=self.run/'trials'/job['id']/f'round_{i+1:02d}'
                if trial_live(directory):return True
            return False
        if previous:
            # An explicit new execute acknowledges old dead attempts. Reconcile
            # their null result, consume their indices, and never relaunch them.
            for job in prepared['jobs']:
                for i in range(self.limits['max_rounds']):
                    directory=self.run/'trials'/job['id']/f'round_{i+1:02d}'
                    if (directory/'REQUEST.json').exists() and not trial_live(directory):
                        self.acknowledged_dead_rounds.add((job['id'],i))
        jobs=sorted(prepared['jobs'],key=lambda j:not active(j))
        if sum(active(j) for j in jobs)>self.workers:raise ValueError('More surviving workers than requested; no new dispatch')
        pending=[j for j in jobs if self.progress['jobs'][j['id']]['status'] not in ['complete','no_candidate']]
        began=time.monotonic(); iterator=iter(pending)
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures={}
            def submit():
                if self.stop.is_set():return
                job=next(iterator,None)
                if job:futures[executor.submit(self.run_job,job,prepared)]=job
            for _ in range(self.workers):submit()
            while futures:
                done,_=wait(futures,return_when=FIRST_COMPLETED)
                for future in done:
                    job=futures.pop(future)
                    try:future.result()
                    except Exception:
                        error=traceback.format_exc()
                        self.progress['jobs'][job['id']].update(status='infrastructure_error',error=error)
                        self.request_stop('unhandled_exception',error,job['id'])
                # Handle every completed error before replenishing either slot.
                for _ in done:submit()
        errors=(any(j['status']=='infrastructure_error' for j in self.progress['jobs'].values()) or
                any(infrastructure_failure(r) for j in self.progress['jobs'].values() for r in j.get('rounds',[])))
        frozen_stop=any(r['kind']=='frozen_input' for r in self.progress.get('pause_reasons',[]))
        phase=('blocked_frozen_input' if frozen_stop else 'paused_infrastructure') if self.stop.is_set() else (
            'complete_with_infrastructure_errors' if errors else 'complete')
        self.progress.update(phase=phase,finished_at=now(),
            last_session_wall_s=time.monotonic()-began)
        self.publish();return self.progress


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--backend-config',type=Path);p.add_argument('--calibration',type=Path)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--stage',choices=['prepare','check','execute'],default='check')
    a=p.parse_args();pilot=Pilot(a.protocol,a.run_dir,a.backend_config,a.calibration,a.workers)
    if a.stage=='check':
        prepared=read(pilot.run/'PREPARED.json')
        if prepared:pilot.verify(prepared)
        else:prepared=pilot.freeze_inputs()
        print(json.dumps(dict(status='ready',jobs=len(prepared['jobs']),network_forward_dispatches=0,physics_steps=0),indent=2));return
    pilot.run.mkdir(parents=True,exist_ok=True)
    with (pilot.run/'.batch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepared=pilot.prepare()
        if a.stage=='prepare':print(json.dumps(dict(status='prepared',jobs=len(prepared['jobs']),physics_steps=0),indent=2))
        else:print(json.dumps(pilot.execute(prepared),indent=2))


if __name__=='__main__':main()
