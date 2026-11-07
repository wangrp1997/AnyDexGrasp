"""Explicit compatibility adapter for a declared S40 training-loss control.

The frozen v3 hand-domain evaluator and its geometry/arm/execution contracts
are unchanged. The interface view preserves the base code100 configuration,
while binding the different loss and control report as explicit provenance.
This is a Boya training adaptation, not the author's original training recipe.
"""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
import sys
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');BOYA=ROOT/'reproduction/boya'
sys.path.insert(0,str(BOYA))
import torch
from evaluate_boya_decision import Reader,original_inference_class
from deploy_score_first import save
import evaluate_s40_hand_domain as prior

PRIOR_SHA='23a20241054e048100ed5e96880d531a2813f2a1daf823bb3bf798a062e7f58c'
SCHEMA='anydex_s40_preactivation_control_v1'
VARIANT='preactivation_selected_mse'


def validate_checkpoint(new,base,gesture,allowed,heldout):
    for key in ['schema','gesture_id','stage','epoch','input_dim','output_dim','architecture_kwargs',
                'training_config','dataset_manifest_sha256','fit_mesh_sha256','fit_trial_ids']:
        if new.get(key)!=base.get(key):raise ValueError('Control changed checkpoint identity: '+key)
    if new.get('training_variant')!=VARIANT:raise ValueError('Explicit preactivation control required')
    fit=set(new['fit_mesh_sha256'])
    if (new['gesture_id']!=gesture or new['stage']!='internal_check' or new['epoch']!=100
        or not fit or not fit<=allowed or fit&heldout):raise ValueError('Control fit split or hand identity changed')
    before,after=base['model_state_dict'],new['model_state_dict']
    if before.keys()!=after.keys():raise ValueError('Original inference state keys changed')
    if any(before[k].shape!=after[k].shape or before[k].dtype!=after[k].dtype for k in before):
        raise ValueError('Original inference state shapes/dtypes changed')
    if any(not torch.isfinite(x).all() for x in after.values()):raise ValueError('Nonfinite head state')


def create_view(control_path,collection,binding_dir):
    reader=Reader();control_path=Path(control_path).resolve();control=reader.read(control_path)
    if (control.get('schema')!=SCHEMA or control.get('status')!='complete'
        or control.get('external_validation_evaluated')is not False
        or control.get('inputs_and_sources_unchanged')is not True):raise ValueError('Complete declared internal control required')
    base_ref=control['base_training_report_ref'];base=reader.read(base_ref['path'],base_ref['sha256'])
    if base.get('schema')!='anydex_s40_code_recipe_training_v1' or base.get('status')!='complete':
        raise ValueError('Base must be original fixed code100 training')
    if control['config']!=base['config'] or control['dataset_manifest']!=base['dataset_manifest']:
        raise ValueError('Loss control must preserve base configuration and dataset')
    if set(control['internal_heads'])!={'1','6','7','8'}:raise ValueError('Exactly four controlled heads required')
    protocol=reader.read(Path(collection)/'protocol.json')
    allowed={x['mesh_sha256']for x in protocol['internal_mesh_groups']['fit']}
    heldout={x['mesh_sha256']for x in protocol['internal_mesh_groups']['internal_check']}
    if len(allowed)!=32 or len(heldout)!=8 or allowed&heldout:raise ValueError('Original32/8 required')
    for doc in (control,base):
        for field in ('source_sha256','input_sha256'):
            for path,digest in doc.get(field,{}).items():reader.freeze(path,digest)
    reader.freeze(prior.__file__,PRIOR_SHA);reader.freeze(__file__)
    source=ROOT/'models/minkowski_graspnet_single_point.py';reader.freeze(source)
    cls=original_inference_class(source,torch)
    for g in (1,6,7,8):
        entry=control['internal_heads'][str(g)];old=base['internal_heads'][str(g)]
        reader.freeze(entry['path'],entry['sha256']);reader.freeze(old['path'],old['sha256'])
        new=torch.load(entry['path'],map_location='cpu',weights_only=True)
        before=torch.load(old['path'],map_location='cpu',weights_only=True)
        validate_checkpoint(new,before,g,allowed,heldout)
        model=cls(**new['architecture_kwargs']);model.load_state_dict(new['model_state_dict'],strict=True);model.eval()
    view=copy.deepcopy(base)
    view.update(internal_heads=control['internal_heads'],heads={},
        source_sha256={**base['source_sha256'],**control.get('source_sha256',{}),str(Path(__file__).resolve()):reader.files[str(Path(__file__).resolve())]},
        input_sha256=reader.files.copy(),
        artifact_role='Compatibility interface view of base code100 configuration plus explicit training-loss control; not an original-code100 retraining claim',
        head_control=dict(schema=SCHEMA,variant=VARIANT,control_results_ref=dict(path=str(control_path),sha256=reader.files[str(control_path)]),
            original_training_recipe=False,inference_class_and_final_relu_unchanged=True,
            only_training_difference='Original selected-output MSE receives cloned BN7 preactivation instead of final ReLU output',
            base_training_report_ref=base_ref))
    binding_dir=Path(binding_dir).resolve()
    if binding_dir.exists():raise ValueError('Fresh control binding directory required')
    binding_dir.mkdir(parents=True)
    save(binding_dir/'PLAN.json',dict(schema='anydex_s40_head_control_deployment_binding_v1',
        head_control=view['head_control'],input_sha256=reader.files.copy(),new_physics_steps=0,
        input_labels_used_for_ranking=False,geometry_policy_changed=False,score_threshold_changed=False))
    path=binding_dir/'TRAINING_INTERFACE_VIEW.json';save(path,view);reader.verify()
    return path,reader


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('control-results','collection-run','output-dir','calibration','domain-policy'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--geometry-workers',type=int,choices=range(1,5),default=4)
    a=p.parse_args(argv);out=a.output_dir.resolve()
    if out.exists():raise ValueError('Fresh diagnostic output required')
    path,reader=create_view(a.control_results,a.collection_run,out.with_name(out.name+'_binding'))
    result=prior.evaluate(a.collection_run.resolve(),path,out,a.calibration.resolve(),a.domain_policy.resolve(),'code100',a.geometry_workers)
    reader.verify()
    save(out/'HEAD_CONTROL_RESULT.json',dict(schema='anydex_s40_head_control_deployment_result_v1',
        status='complete',control_schema=SCHEMA,training_variant=VARIANT,original_training_recipe=False,
        interface_view=dict(path=str(path),sha256=reader.freeze(path)),
        diagnostic_results=dict(path=str(out/'RESULTS.json'),sha256=reader.freeze(out/'RESULTS.json')),
        frozen_prior_evaluator=dict(path=str(Path(prior.__file__).resolve()),sha256=PRIOR_SHA),
        no_new_training_or_physics=True,threshold_and_geometry_policy_unchanged=True))
    return result
if __name__=='__main__':main()
