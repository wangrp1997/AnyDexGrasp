"""Train the original g1 sample-cap100 recipe on the audited quota dataset."""
import argparse
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import author_collection_batch as B
import author_collection_wave_resources as R
import author_protocol as A
import author_training as T


def run(root,manifest_path):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    manifest_path=Path(manifest_path).resolve();manifest=B.read(manifest_path)
    if not manifest['formal_training_eligible'] or not manifest['target_met'] or manifest['fit']<100:
        raise ValueError('Audited formal fit quota required')
    B.verify({'dataset':manifest['dataset']})
    with np.load(manifest['dataset']['path'],allow_pickle=False)as z:data={k:z[k].copy()for k in z.files}
    fit,check=T.validate_arrays(data)
    selected=A.select_training_records(data['trial_id'][fit],sample_cap=100)
    if len(selected)!=100:raise ValueError('One hundred real fit labels required')
    sources={name:B.binding(B.HERE/name)for name in [Path(__file__).name,'author_training.py','author_protocol.py']}
    sources.update({name:B.binding(B.ROOT/name)for name in A.PINS})
    protocol=dict(schema='anydex_mjx_author_g1_fit100_v1',manifest=B.binding(manifest_path),dataset=manifest['dataset'],
        sources=sources,seed=2026111903,sample_cap=100,gesture_id=1,device='cpu',cpu_threads=1,
        available_fit_records=len(fit),selected_fit_trial_ids=list(map(str,selected)),internal_records=len(check),
        selected_fit_positive=int(data['labels'][fit[:100]].sum()),selected_fit_negative=int((data['labels'][fit[:100]]==0).sum()),
        internal_positive=int(data['labels'][check].sum()),internal_negative=int((data['labels'][check]==0).sum()),
        recipe=T.RECIPE,formal_sample_cap100_run=True,class_balancing=False,seed_retries=0,
        video_recording=False,new_physics_steps=0,new_CGR_forwards=0,
        automatic_collection_policy_replacement=False)
    B.save(root/'PROTOCOL.json',protocol,exclusive=True)
    archive=root/'source_archive';archive.mkdir()
    for name,binding in sources.items():
        dest=archive/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(binding['path'],dest)
    B.save(root/'RUN_STARTED.json',dict(process=R.process_identity(__import__('os').getpid()),unix_s=time.time()),exclusive=True)
    torch.set_num_threads(1);began=time.monotonic()
    result=T.fit_head(data,root/'head',gesture=1,sample_cap=100,seed=protocol['seed'],device='cpu')
    if result['fit_trial_ids']!=protocol['selected_fit_trial_ids'] or result['optimizer_updates']!=100:
        raise ValueError('Original cap100 prefix or update count changed')
    B.verify(sources);B.verify({'dataset':protocol['dataset'],'manifest':protocol['manifest']})
    inference_check=None
    chosen=result['checkpoints'].get('0.5')
    if chosen is not None:
        _,cls,_,_=T.components();model=cls().eval()
        state=torch.load(chosen['path'],map_location='cpu',weights_only=True)
        model.load_state_dict(state['model_state_dict'],strict=True)
        with torch.no_grad():scores=model(torch.from_numpy(data['features'][check].copy()))[0].numpy()
        if scores.shape!=(len(check),1,20) or not np.isfinite(scores).all():raise ValueError('Invalid held-out inference scores')
        target=data['target_index'][check];selected_scores=scores[np.arange(len(check)),0,target]
        pred=selected_scores>=.5;truth=data['labels'][check].astype(bool)
        np.savez_compressed(root/'INTERNAL_SCORES.npz',trial_id=data['trial_id'][check],labels=data['labels'][check],
            target_index=target,all_scores=scores,selected_output_score=selected_scores)
        inference_check=dict(checkpoint=B.binding(chosen['path']),score_file=B.binding(root/'INTERNAL_SCORES.npz'),
            internal_rows=len(check),true_positive=int((pred&truth).sum()),false_positive=int((pred&~truth).sum()),
            true_negative=int((~pred&~truth).sum()),false_negative=int((~pred&truth).sum()),
            validation_was_used_for_model_selection=True,new_scene_physical_success_rate_measured=False)
    summary=dict(status='formal_fit100_training_complete',source_training_results=B.binding(root/'head/TRAINING_RESULTS.json'),
        protocol=B.binding(root/'PROTOCOL.json'),fit_records=100,internal_records=len(check),
        selected_fit_positive=protocol['selected_fit_positive'],selected_fit_negative=protocol['selected_fit_negative'],
        optimizer_updates=result['optimizer_updates'],selected_checkpoint_count=result['selected_checkpoint_count'],
        strict_published_inference_reload=result['strict_published_inference_reload'],inference_check=inference_check,
        wall_s=time.monotonic()-began,training_only=True,new_physics_steps=0,new_CGR_forwards=0,video_recording=False,
        automatically_installed_in_collection=False)
    B.save(root/'RESULTS.json',summary,exclusive=True)
    print(summary,flush=True);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True);args=parser.parse_args();run(args.root,args.manifest)
