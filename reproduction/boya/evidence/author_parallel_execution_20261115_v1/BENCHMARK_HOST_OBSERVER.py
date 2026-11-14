"""Independent host-only observer; never imports a simulator or model."""
import importlib.util
import json
from pathlib import Path
import time

ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
BASE=ROOT/'outputs/boya_parallel_runtime_audit_20261115_v1/observe_wave_runtime.py'
spec=importlib.util.spec_from_file_location('old_read_only_observer',BASE)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
RUN=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/benchmark_execution_20261115_v1')
OUT=Path(__file__).resolve().parent
M.RUN=RUN
PROTOCOL=RUN/'BENCHMARK_PROTOCOL.json'


def main():
    assert len(list(Path('/proc').glob('[0-9]*')))>8, 'Host process namespace required'
    deadline=time.time()+3600; cache={}; history={}; stages={}; witnesses=[]
    previous={}; maximum=0; maximum_reserved=0; samples=0
    while not PROTOCOL.exists():
        assert time.time()<deadline,'No frozen benchmark appeared'
        time.sleep(2)
    binding=dict(path=str(PROTOCOL),sha256=M.sha(PROTOCOL)); protocol=M.read(PROTOCOL)
    pins={x['path']:x['sha256'] for x in protocol['sources'].values()}
    pins[str(BASE)]=M.sha(BASE); pins[str(Path(__file__).resolve())]=M.sha(__file__)
    assert all(M.sha(p)==h for p,h in pins.items())
    events=(OUT/'OBSERVATIONS.jsonl').open('x')
    while time.time()<deadline:
        now=time.time(); loaded=[]; reserved=0
        for path in sorted((RUN/'resource_slots/kit').glob('*.json')):
            state=M.read(path); key=state['reservation_id']
            if history.get(str(path))!=state:
                history[str(path)]=state
                events.write(json.dumps(dict(event='resource_state',unix_s=now,state=state))+'\n');events.flush()
            record=stages.setdefault(key,dict(stage_dir=state['stage_dir'],slot=state['slot'],reservation_id=key))
            record['last_resource_state']=state
            if state['state'] not in ('reserved','running'):continue
            reserved+=1; sample=M.process_sample(state,path,cache)
            if not sample or not sample.get('alive_identity_match'):continue
            if not sample['kit_loaded']:continue
            job=Path(state['stage_dir']).name.removeprefix('execute_')
            execution=RUN/'jobs'/job/'execution'
            journal=execution/'MULTI_OBJECT_CONTACTS.jsonl.gz'
            sample['journal_path']=str(journal)
            sample['journal_bytes']=journal.stat().st_size if journal.exists() else 0
            progress=execution/'execution_progress.json'
            sample['stage_progress']=M.read(progress) if progress.exists() else None
            record.setdefault('first_library_verified_unix_s',now)
            record['last_library_verified_unix_s']=now
            record['last_sample']=sample
            loaded.append(dict(slot=state['slot'],stage_dir=state['stage_dir'],sample=sample))
        samples+=1;maximum_reserved=max(maximum_reserved,reserved)
        growing=[x['sample']['process']['pid'] for x in loaded
                 if x['sample']['journal_bytes']>previous.get(x['sample']['process']['pid'],0)>0]
        previous={x['sample']['process']['pid']:x['sample']['journal_bytes'] for x in loaded}
        kind=None
        if len(loaded)>maximum:maximum=len(loaded);kind='new_maximum_library_verified_kit'
        if len(growing)>=2 and not any(w['kind']=='two_concurrently_growing_journals' for w in witnesses):
            kind='two_concurrently_growing_journals'
        if kind:
            witness=dict(kind=kind,unix_s=now,kits=loaded,growing_journal_pids=growing,gpu=M.gpu_snapshot())
            witnesses.append(witness);events.write(json.dumps(witness)+'\n');events.flush()
            print(json.dumps(dict(kind=kind,unix_s=now,pids=[x['sample']['process']['pid'] for x in loaded])),flush=True)
        summary=dict(schema='anydex_boya_independent_benchmark_host_audit_v1',status='monitoring',
            protocol=binding,samples=samples,last_sample_unix_s=now,maximum_reserved_kit_slots=maximum_reserved,
            maximum_library_verified_kit_processes=maximum,simultaneous_two_kit_proven=maximum>=2,
            two_journals_growing_proven=any(w['kind']=='two_concurrently_growing_journals' for w in witnesses),
            witnesses=witnesses,stages=list(stages.values()),costs_added=dict(Kit_starts=0,physics_steps=0,planning_calls=0,CGR_forwards=0))
        M.save(OUT/'LIVE.json',summary)
        registry=M.read(M.REGISTRY) if M.REGISTRY.exists() else {}
        if registry.get('run_dir')==str(RUN) and registry.get('state') in ('released','active_review_required') and not loaded:
            summary.update(status='complete' if registry['state']=='released' else 'runtime_quarantine',registry=registry,
                changed_sources=[p for p,h in pins.items() if M.sha(p)!=h],protocol_unchanged=M.sha(PROTOCOL)==binding['sha256'])
            events.close();summary['observations']=dict(path=str(OUT/'OBSERVATIONS.jsonl'),sha256=M.sha(OUT/'OBSERVATIONS.jsonl'))
            M.save(OUT/'FINAL.json',summary);print(json.dumps(dict(status=summary['status'],final=str(OUT/'FINAL.json'))),flush=True)
            return
        time.sleep(2)
    summary['status']='observer_deadline_reached';M.save(OUT/'FINAL.json',summary)


if __name__=='__main__':main()
