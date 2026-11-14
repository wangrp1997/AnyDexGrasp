"""Freeze lightweight evidence after the completed, no-execute component queue."""
from pathlib import Path
import hashlib, json, shutil

ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
OUT=ROOT/'outputs/boya_author_pipeline_home_release_20261114_v1'
EVID=ROOT/'reproduction/boya/evidence/author_component_transport_20261114_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
checks={}
def verify(value):
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
            p=Path(value['path']);digest=value['sha256']
            if str(p) not in checks:checks[str(p)]=sha(p)
            assert checks[str(p)]==digest, p
            if 'archive' in value:
                assert sha(value['archive'])==digest,value['archive']
        for v in value.values():verify(v)
    elif isinstance(value,list):
        for v in value:verify(v)
protocols=[OUT/'COMPONENT_INTEGRATION_PROTOCOL.json']+[OUT/f'integration_g{g}/COMPONENT_INTEGRATION_PROTOCOL.json' for g in (1,6,7)]
for p in protocols:verify(read(p))
json_paths=list(OUT.rglob('*.json'))
for p in json_paths:
    assert b'\x00' not in p.read_bytes(),p
    read(p)
queue=read(OUT/'COMPONENT_INTEGRATION_RESULTS.json')
breakdown=read(OUT/'POST_GEOMETRY_BREAKDOWN.json')
counts=[read(OUT/f'PLANNING_COUNTS_g{g}.json') for g in (1,6,7)]
summary=dict(schema='anydex_boya_author_component_transport_final_v1',complete=True,
    protocol_sha256=sha(protocols[0]),candidate_order=[1,6,7],prior_g8_null_preserved=True,
    ready=0,null_geometry=3,infrastructure_failures=0,physical_execution_dispatched=False,
    new_labels=0,formal_training_updates=0,new_CGR_forwards=0,new_collection_collision_queries=0,
    new_native_approach_IK=0,new_HOME_search_calls=0,
    original_transport_IK_calls_started=sum(x['original_ik_calls_started'] for x in counts),
    original_transport_IK_calls_returned=sum(x['original_ik_calls_returned'] for x in counts),
    original_transport_IK_calls_raised=sum(x['original_ik_calls_raised'] for x in counts),
    original_transport_IK_max_nfev=300,original_transport_IK_wall_s=sum(x['original_ik_wall_s'] for x in counts),
    queue_wall_s=queue['wall_s'],component_wall_s=[x['component_wall_s'] for x in queue['jobs']],
    original3_source_object_proxy_violations=0,floor_violations=0,
    violation_phases=['transport_bin','release','release_settle'],
    violating_links=['upper_arm_link'],self_others=['wrist_1_link','wrist_2_link','wrist_3_link','fabase'],
    bin_owners=['wall_x_plus','wall_y_plus'],
    violations=[dict(gesture_id=x['gesture_id'],count=x['count'],by_kind=x['counts_by_kind'],
        by_phase=x['counts_by_phase'],by_owner=x['counts_by_object_owner'],commands=x['commands'],
        first=x['first'],deepest=x['deepest'],open_ok=x['open_ok'],full_scene_seed_parts=x['seed_full_scene_parts']) for x in breakdown['results']],
    triangle_representatives=9,triangle_surface_intersections=9,
    result_interpretation='All three complete transport preparations are null_geometry. Exact source surfaces cross at the selected deepest self and final bin states. No physical success rate or training negatives exist.',
    old_collision_queries_reused=387,physics_steps=0,Kit_launches=0,
    frozen_pipeline_sha256=sha(ROOT/'reproduction/boya/author_pipeline.py'),
    frozen_tests_sha256=sha(ROOT/'reproduction/boya/test_author_pipeline_entry.py'))
save(OUT/'FINAL_SUMMARY.json',summary)
integrity=dict(schema='anydex_boya_author_component_final_integrity_v1',status='passed',
    checked_protocols=[dict(path=str(p),sha256=sha(p)) for p in protocols],
    unique_current_dependency_files=len(checks),all_bound_source_archives_exact=True,
    checked_files=checks,json_files_valid_without_NUL=len(json_paths),physics_steps=0,new_planning_calls=0)
save(OUT/'FINAL_INTEGRITY.json',integrity)
EVID.mkdir(exist_ok=False)
copies=['COMPONENT_INTEGRATION_PROTOCOL.json','COMPONENT_INTEGRATION_RESULTS.json','FINAL_SUMMARY.json',
    'FINAL_INTEGRITY.json','POST_GEOMETRY_BREAKDOWN.json','SOURCE_TRIANGLE_DIAGNOSTIC.json',
    'POST_GEOMETRY_DIAGNOSTIC.log','PIPELINE_TESTS_V3.log','FIXED_RELEASE.json',
    'FREEZE_DEVELOPMENT_FAILURE.json','RELEASE_GENERATION_FAILURE.json','RELEASE_TOOL_FREEZE.json',
    'RELEASE_TOOL_FREEZE_V2.json','RELEASE_GENERATION.log','RELEASE_GENERATION_V2.log',
    'post_geometry_diagnostic.py','run_integrations.py','run_one_integration.py','finalize_evidence.py']
for g in (1,6,7):
    copies += [f'PLANNING_COUNTS_g{g}.json',f'DISPATCH_g{g}.json',f'integration_g{g}/COMPONENT_INTEGRATION_PROTOCOL.json',
               f'integration_g{g}/COMPONENT_INTEGRATION_RESULTS.json']
copied={}
for name in copies:
    dest=EVID/name;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(OUT/name,dest)
    copied[name]=dict(path=str((OUT/name).resolve()),sha256=sha(dest))
save(EVID/'EVIDENCE.json',dict(schema='anydex_boya_author_component_transport_evidence_v1',
    summary=summary,files=copied,large_outputs_ignored=True,immutable_source_snapshot_root=str(OUT)))
print(json.dumps(dict(evidence=str(EVID),summary=summary,integrity_files=len(checks)),indent=2))
