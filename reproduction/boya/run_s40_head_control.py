"""Execute the declared S40 training-loss control with the frozen v3 protocol.

The prior runner's code100 string is its compatibility format, not a claim
that the changed training loss is the author's recipe. The primary control
receipt below records the different loss before and after physical execution.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');BOYA=ROOT/'reproduction/boya'
sys.path.insert(0,str(BOYA))
from evaluate_boya_decision import Reader
from deploy_score_first import save
import run_s40_port_trials_v2 as prior
PRIOR_SHA='ff0ee0fe567e8f16f7370e6179e832a9cfe7f27252159e8b7010bf3858ff2c75'
VARIANT='preactivation_selected_mse'

def bind(a):
    r=Reader();r.freeze(prior.__file__,PRIOR_SHA);r.freeze(__file__)
    entry=r.read(a.diagnostic/'HEAD_CONTROL_RESULT.json')
    if (entry.get('schema')!='anydex_s40_head_control_deployment_result_v1'
        or entry.get('status')!='complete' or entry.get('training_variant')!=VARIANT
        or entry.get('original_training_recipe')is not False
        or entry.get('threshold_and_geometry_policy_unchanged')is not True):
        raise ValueError('Explicit complete preactivation diagnostic required')
    diag=entry['diagnostic_results'];r.read(diag['path'],diag['sha256'])
    if Path(diag['path']).resolve()!=(a.diagnostic/'RESULTS.json').resolve():
        raise ValueError('Diagnostic receipt points to another run')
    v=entry['interface_view'];view=r.read(v['path'],v['sha256'])
    control=view['head_control']
    if control.get('variant')!=VARIANT or control.get('original_training_recipe')is not False:
        raise ValueError('Explicit changed training loss required')
    c=control['control_results_ref'];report=r.read(c['path'],c['sha256'])
    if report.get('status')!='complete' or report.get('training_variant')!=VARIANT:
        raise ValueError('Complete control training required')
    r.freeze(a.queue);prior.validate_execution_inputs(a)
    return r,control

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('queue','diagnostic','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--workers',type=int,choices=range(1,5),default=4)
    p.add_argument('--render',action='store_true');a=p.parse_args(argv)
    a.queue=a.queue.resolve();a.diagnostic=a.diagnostic.resolve();a.output=a.output.resolve()
    r,control=bind(a);binding=a.output.with_name(a.output.name+'_binding')
    if binding.exists():raise ValueError('Fresh control execution binding required')
    binding.mkdir(parents=True)
    save(binding/'PLAN.json',dict(schema='anydex_s40_head_control_physical_plan_v1',
        training_variant=VARIANT,original_training_recipe=False,head_control=control,
        fixed_internal_objects=8,max_arm_candidates_per_object=4,
        physical_attempts_per_ready_object=1,outcome_dependent_retry=False,
        benchmark_only=True,training_writeback=False,input_sha256=r.files.copy()))
    receipt=prior.run(a);r.verify()
    result=dict(receipt,schema='anydex_s40_head_control_execution_results_v1',
        head_recipe=VARIANT,original_training_recipe=False,head_control=control,
        underlying_result_format_head_recipe=receipt['head_recipe'],
        underlying_result=dict(path=str(a.output/'RESULTS.json'),sha256=r.freeze(a.output/'RESULTS.json')),
        control_binding_plan=dict(path=str(binding/'PLAN.json'),sha256=r.freeze(binding/'PLAN.json')),
        explicit_control_inputs_unchanged=True)
    save(a.output/'HEAD_CONTROL_EXECUTION_RESULTS.json',result)
    return result
if __name__=='__main__':main()
