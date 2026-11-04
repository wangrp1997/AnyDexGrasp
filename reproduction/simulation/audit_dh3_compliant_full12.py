"""Read-only full12 accounting; delegates saved physics to the existing auditor.
Functional held lifts and legacy port diagnostics are explicitly separate.
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path
REPO=Path('/home/wangrenpeng/AnyDexGrasp')
spec=importlib.util.spec_from_file_location('saved_audit',REPO/'reproduction/simulation/audit_dh3_compliant_scenes.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
MODEL='50e02b4d8a7696c5507a27acb92a20b06c81195f75fde8322380ec7999580aec'
RUNNER='cdc86125eb944446442d396a369aef44333ca2b87d902a72bd150ccc44f96349'
ORIGINAL=REPO/'outputs/dh3_real_mesh_generalization_20261103_v1/run/RESULTS.json'
REUSED=REPO/'outputs/dh3_compliant_comparison_20261104_v3/physics'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    a.require(not args.output.exists(),'Preserve audits: use new output');root=args.run_dir;start=time.monotonic();report=a.read(root/'RESULTS.json');old=a.read(ORIGINAL)
    rows=report['scenes'];expected=[r['scene_id'] for r in old['scenes']];a.require(len(rows)==12 and len({r['scene_id'] for r in rows})==12 and {r['scene_id'] for r in rows}==set(expected),'All original 12 IDs exactly once')
    a.require(str(report['status']) in ('completed','cohort_finished','full12_finished','finished'),'Only a final completed report may be audited')
    records=[];errors=[]
    for prior in old['scenes']:
        sid=prior['scene_id'];row=next(r for r in rows if r['scene_id']==sid)
        try:
            if not prior['final_top1_selected']:
                inference=Path(prior['inference_output_dir']);selection=a.read(inference/'selection/RESULTS.json');collision=inference/'collision/RESULTS.json'
                a.require(not selection['selected'] and selection['collision_admitted_candidates']==0,'Original no-candidate source')
                a.require(a.sha(collision)==selection['collision_RESULTS_sha256'],'No-candidate collision report binding')
                a.require(a.sha(inference/'collision/collision_outputs.npz')==selection['collision_outputs_sha256'],'No-candidate original collision outputs binding')
                for key in ('physics_steps','completed_physics_steps','actual_physical_attempts','physics_invocations'):
                    if key in row:a.require(row[key]==0,'Zero-step/no-invocation for no-candidate '+key)
                a.require(not (root/'physics'/sid/'trace.npz').exists(),'No undeclared physics on original rejected case')
                records.append(dict(scene_id=sid,functional_held=False,physics_steps=0,new_physics_steps=0,reused_physics_steps=0,status='original_collision_no_candidate',source_selection_sha256=a.sha(inference/'selection/RESULTS.json')));continue
            reuse=sid in ('banana_yaw000','banana_yaw090');run=REUSED/sid if reuse else root/'physics'/sid
            if 'physics_output_dir' in row:a.require(Path(row['physics_output_dir']).resolve()==run.resolve(),'Declared physics source path')
            c=a.read(run/'CONTROLLER.json');r=a.read(run/'RESULTS.json');a.require(c['controller_script_sha256']==MODEL and c['runner_script_sha256']==RUNNER,'Same frozen model/refined runner')
            a.require(c['assumptions_SI_not_hardware_identification']['dt']==1/3840 and r['physics_steps']==39360,'Same3840Hz/39360steps')
            a.require(c['selection_sha256']==prior['selection_sha256'] and c['candidate_sha256']==prior['selected_candidate_sha256'] and c['scene_manifest_sha256']==prior['scene_manifest_sha256'],'Same original top1/scene')
            verified=a.audit_run(run,Path(prior['physics_output_dir']));outcome=verified['outcome_and_media']
            for key in ('lift_observed','object_final_lift_m','port_stability_pass','physics_steps'):
                if key in row:a.require(row[key]==r[key],'Summary matches physical source '+key)
            records.append(dict(scene_id=sid,functional_held=outcome['lift_observed'],status='held' if outcome['lift_observed'] else 'not_held',physics_steps=39360,new_physics_steps=0 if reuse else 39360,reused_physics_steps=39360 if reuse else 0,source_run=str(run),legacy_port_diagnostics=outcome['failed_original_gates'],legacy_port_stability_pass=outcome['port_stability_pass'],final_lift_m=outcome['object_final_lift_m'],source_trace_sha256=a.sha(run/'trace.npz'),independent_saved_execution=verified))
        except Exception as e:errors.append(dict(scene_id=sid,type=type(e).__name__,error=str(e)));records.append(dict(scene_id=sid,status='audit_error',functional_held=None))
    out=dict(status='full12_saved_accounting_verified' if not errors else 'audit_failed',source_report=dict(path=str(root/'RESULTS.json'),sha256=a.sha(root/'RESULTS.json')),original_source_report=dict(path=str(ORIGINAL),sha256=a.sha(ORIGINAL)),records=records,errors=errors,fixed_denominator=12,functional_held=sum(r['functional_held'] is True for r in records),no_original_candidate=sum(r['status']=='original_collision_no_candidate' for r in records),physics_steps_verified=sum(r.get('physics_steps',0) for r in records),new_physics_steps_verified=sum(r.get('new_physics_steps',0) for r in records),reused_physics_steps_verified=sum(r.get('reused_physics_steps',0) for r in records),additional_physics_steps=0,additional_network_forwards=0,wall_seconds=time.monotonic()-start,auditor_source=dict(path=str(Path(__file__).resolve()),sha256=a.sha(__file__)),reused_auditor=dict(path=str(spec.origin),sha256=a.sha(spec.origin)),boundary='Held lift is the functional result. Existing port speed/residual/tracking thresholds are diagnostics, not original-paper requirements or conditions for the functional count. No new inference, physics, training, topK trials or hardware claim.')
    if not errors:a.require(out['no_original_candidate']==3 and out['new_physics_steps_verified']==7*39360 and out['reused_physics_steps_verified']==2*39360,'Three zero-step, seven new single runs, two reused runs')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2));sys.exit(1 if errors else 0)
if __name__=='__main__':main()
