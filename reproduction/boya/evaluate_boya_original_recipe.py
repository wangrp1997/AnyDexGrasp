#!/usr/bin/env python3
"""Opt-in code100 original selected-MSE scoring control on four KNOWN EGAD scenes.

Reuses frozen score-first/domain/full-cloud geometry functions. No new CGR,
training or physics. Public20-output configuration/final100 are explicit and
are not a reconstruction of the complete original hardware training history.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

import evaluate_boya_external as pipeline

ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
RECIPE='code100_original_selected_mse'
ROLE='known_object_recipe_alignment'
MANIFEST_SCHEMA='anydex_boya_original_recipe_diagnostic_scene_v1'
PROTOCOL_SCHEMA='anydex_boya_original_recipe_diagnostic_protocol_v1'
RESULT_SCHEMA='anydex_boya_original_recipe_diagnostic_v1'
TRAINING_REPORT=ROOT/'outputs/s40_code_recipe_20261107_v1/TRAINING_RESULTS.json'
TRAINING_SHA='447302d26c1e5cccc10f23ae897105a5d845abe661640cfc4ea5fa6d912ccac0'
DATASET_SHA='c85a284be680707a2b3bcaae7950e1981bd907a9e88870c434ced1c22f6e91bb'
APPROVED=('egad_A0','egad_A3','egad_A6','egad_E0')
CONFIG=dict(epochs=100,batch_size=128,learning_rate=5e-4,weight_decay=5e-4,optimizer='Adam',scheduler='original PolyLR',poly_power=.99,seed=20261106,input_dim=480,output_dim=20)


def require(value,message):
    if not value:raise ValueError(message)


def validate_manifests(manifest,protocol):
    require(manifest.get('schema')==MANIFEST_SCHEMA and protocol.get('schema')==PROTOCOL_SCHEMA,'Explicit original-recipe diagnostic schemas required')
    rows=manifest.get('objects');ids=pipeline.ids_from_rows(rows)
    require(tuple(ids)==APPROVED and pipeline.ids_from_rows(manifest.get('approved_object_ids'))==ids,'Exactly four known EGAD A0/A3/A6/E0 in declared order required')
    require(pipeline.ids_from_rows(protocol.get('objects'))==ids and protocol.get('object_count')==4,'Diagnostic object identities/count changed')
    require(protocol.get('head_recipe')==RECIPE and protocol.get('evaluation_role')==ROLE and protocol.get('known_object_diagnostic')is True,'Known-object original selected-MSE control must be explicit')
    require(protocol.get('arm_budget_per_object')==4 and protocol.get('physical_attempts_per_object')==1 and protocol.get('outcome_dependent_retry')is False and protocol.get('benchmark_only')is True and protocol.get('training_writeback')is False,'One-attempt/no-retry/no-writeback protocol required')
    require(manifest.get('exclusion_manifest_refs'),'Source-bound training identity exclusions required')
    for row in rows:
        require(isinstance(row,dict)and {'object_id','observation_json','representation_dir','mesh_sha256'}<=set(row),'Missing frozen cached scene columns')
        require(row.get('split')==ROLE,'Known diagnostic scenes cannot be called unseen validation')
        require(not any(row.get(k)is not None for k in ['label','training_label','success','result','trial_outcome']),'No outcomes in selection manifest')
        pipeline.mesh_shas(row)
    return rows,ids


def validate_training(training):
    require(training.get('schema')=='anydex_s40_code_recipe_training_v1'and training.get('status')=='complete'and training.get('inputs_and_sources_unchanged')is True and training.get('plan_unchanged')is True,'Complete frozen code100 report required')
    require(training.get('config')==CONFIG and training.get('all40_refit')is False and training.get('selected_epoch')==100 and training.get('external_validation_evaluated')is False and training.get('code_recipe_control_not_full_paper_reproduction')is True,'Explicit code-default final100/32-fit control required')
    require(training.get('dataset_manifest',{}).get('sha256')==DATASET_SHA and set(training.get('internal_heads',{}))=={'1','6','7','8'},'Frozen435 and four code100 heads required')
    require('training_variant'not in training,'Preactivation/modified-loss training is not the original-MSE control')
    groups=training.get('mesh_groups',{})
    require(set(groups)=={'train','internal_check'}and len(groups['train'])==32 and len(groups['internal_check'])==8 and len(set(groups['train']+groups['internal_check']))==40,'Disjoint32-fit/8-internal groups required')
    return set(groups['train']),set(groups['internal_check'])


def validate_checkpoint(checkpoint,gesture,dataset_sha,allowed,internal,external):
    fit=set(checkpoint.get('fit_mesh_sha256',[]))
    require(checkpoint.get('schema')=='anydex_boya_decision_state_dict_v1'and checkpoint.get('stage')=='internal_check'and checkpoint.get('gesture_id')==gesture and checkpoint.get('epoch')==100,'Original code100 checkpoint identity required')
    require('training_variant'not in checkpoint and checkpoint.get('model_class')=='MinkowskiGraspNetMultifingerType1Inference','Modified-loss checkpoint rejected')
    require(checkpoint.get('dataset_manifest_sha256')==dataset_sha==DATASET_SHA and checkpoint.get('training_config')==CONFIG and checkpoint.get('input_dim')==480 and checkpoint.get('output_dim')==20,'Frozen original selected-MSE20-output checkpoint required')
    require(fit and fit<=allowed and not fit&internal and not fit&external,'Checkpoint trained outside32-fit groups')
    expected=dict(input_num=480,num_multifinger_type=1,num_multifinger_depth=4,two_finger_num_angle=1,two_finger_num_depth=5)
    require(checkpoint.get('architecture_kwargs')==expected,'Original20-output architecture required')
    return fit


def annotate(document):
    """Label artifacts without adding metadata to hash maps or query dictionaries."""
    markers=dict(head_recipe=RECIPE,training_variant=RECIPE,author_original_training=False,full_author_training_protocol_reproduced=False,known_object_diagnostic=True,evaluation_role=ROLE)
    def walk(value):
        if isinstance(value,list):return [walk(x)for x in value]
        if not isinstance(value,dict):return value
        result={k:walk(v)for k,v in value.items()}
        if 'evaluation_role'in result:result.update(markers)
        if 'scope'in result:result['scope']='Known-scene Boya code100 original selected-MSE control; score/geometry only, no unseen or physical-success claim.'
        if 'novelty_audit'in result:
            result['novelty_audit'].update(known_prior_evaluation_objects=True,independent_unseen_validation=False,scope='Disjoint from training meshes only; these four evaluation objects have been used before.')
        return result
    result=walk(document)
    if isinstance(result,dict):result.update(markers)
    return result


def evaluate(objects_manifest,protocol_path,training_path,output,calibration,domain_policy,geometry_workers=4):
    require(Path(training_path).resolve()==TRAINING_REPORT and hashlib.sha256(TRAINING_REPORT.read_bytes()).hexdigest()==TRAINING_SHA,'Use the untouched frozen code100 training report, not alternate heads')
    names=['SCHEMA','TRAINING_SCHEMA','VARIANT','validate_external_manifests','validate_training','validate_checkpoint','save','Reader']
    previous={name:getattr(pipeline,name)for name in names}
    original_reader=previous['Reader'];original_save=previous['save']
    class BoundReader(original_reader):
        def __init__(self):
            super().__init__();self.freeze(Path(__file__).resolve());self.freeze(TRAINING_REPORT,TRAINING_SHA)
    def save(path,value):return original_save(path,annotate(value))
    try:
        pipeline.SCHEMA=RESULT_SCHEMA;pipeline.TRAINING_SCHEMA='anydex_s40_code_recipe_training_v1';pipeline.VARIANT=RECIPE
        pipeline.validate_external_manifests=validate_manifests;pipeline.validate_training=validate_training;pipeline.validate_checkpoint=validate_checkpoint
        pipeline.save=save;pipeline.Reader=BoundReader
        return annotate(pipeline.evaluate(objects_manifest,protocol_path,training_path,output,calibration,domain_policy,geometry_workers))
    finally:
        for name,value in previous.items():setattr(pipeline,name,value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['objects-manifest','protocol','output-dir','calibration','domain-policy']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--training-results',type=Path,default=TRAINING_REPORT);p.add_argument('--geometry-workers',type=int,choices=range(1,5),default=4)
    a=p.parse_args();evaluate(a.objects_manifest,a.protocol,a.training_results,a.output_dir,a.calibration,a.domain_policy,a.geometry_workers)
if __name__=='__main__':main()
