"""Read-only closure and published-inference checks for the cap100 diagnosis."""
import argparse
from pathlib import Path
import shutil

import numpy as np
import torch

import author_collection_batch as B
import author_mjx_head_diagnosis as D
import author_training as T


def audit(root):
    root = Path(root).resolve(); p = B.read(root/'PROTOCOL.json'); result = B.read(root/'RESULTS.json')
    B.verify(p['sources']); B.verify({'training': p['input_training'], 'dataset': p['dataset'], 'protocol': p['original_protocol']})
    for name, source in p['sources'].items():
        assert B.sha(root/'source_archive'/name) == source['sha256']
    with np.load(p['dataset']['path'], allow_pickle=False) as z: data = {k: z[k].copy() for k in z.files}
    fit, check = T.validate_arrays(data); fit = fit[:100]
    assert len(fit) == 100 and len(check) == 16
    assert result['optimizer_updates'] == p['optimizer_update_budget'] == 400
    assert [v['variant'] for v in result['variants']] == list(D.VARIANTS)
    original = B.read(p['input_training']['path'])
    torch.set_num_threads(1); inference = T.components()[1]; published = {}; bindings = {}
    for variant in result['variants']:
        name = variant['variant']; path = root/name/'RESULTS.json'; full = B.read(path)
        assert all(full[k] == v for k, v in variant.items())
        assert len(full['history']) == 100 and full['optimizer_updates'] == 100
        assert all(h['epoch'] == i and h['updates'] == i + 1 for i, h in enumerate(full['history']))
        checkpoint = root/name/'final_diagnostic.pt'; state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert state['fit_trial_ids'] == data['trial_id'][fit].tolist() and state['diagnostic_only']
        model = inference().eval(); model.load_state_dict(state['model_state_dict'], strict=True)
        with torch.no_grad(): pred = model(torch.from_numpy(data['features']))[0]
        assert torch.isfinite(pred).all()
        published[name] = dict(fit=D.metrics(D.selected(pred[fit], data['target_index'][fit]).numpy(), data['labels'][fit]),
            internal=D.metrics(D.selected(pred[check], data['target_index'][check]).numpy(), data['labels'][check]),
            inference_activation='original ReLU', strict_state_reload=True,
            training_activation=state['activation'], training_recipe_modified=name != 'original_relu')
        if name in ('original_relu', 'relu_positive_output_bias', 'linear_output'):
            for split, key in [('fit', 'fit_eval'), ('internal', 'internal_eval')]:
                assert all(published[name][split][k] == variant[key][k] for k in ('TP', 'FP', 'TN', 'FN'))
        bindings[name] = dict(results=B.binding(path), checkpoint=B.binding(checkpoint))
    baseline = result['variants'][0]
    assert baseline['original_replay']['all100_epoch_losses_exact']
    assert baseline['original_replay']['selected_checkpoint_all_tensors_exact']
    assert all(h['train_loss'] == old['train']['train/loss'] and h['internal_loss'] == old['internal']['test/loss']
               for h, old in zip(B.read(root/'original_relu/RESULTS.json')['history'], original['history']))
    assert len(baseline['persistent_nonpositive_training_positives_after_epoch10']) == 32
    rows = B.read(root/'ROW_AUDIT.json')
    for item in rows['record_bindings']: B.verify(item)
    assert rows['rows'] == 117 and rows['raw_to_canonical_exact'] and rows['labels_exact'] and rows['depth_output_indices_exact']
    per_target = []
    for target in range(20):
        a = fit[data['target_index'][fit] == target]; b = check[data['target_index'][check] == target]
        per_target.append(dict(target_index=target, fit=len(a), fit_positive=int(data['labels'][a].sum()),
            internal=len(b), internal_positive=int(data['labels'][b].sum())))
    prior = float(data['labels'][fit].mean())
    source = B.binding(Path(__file__)); archive = root/'review_source_archive'/Path(__file__).name
    archive.parent.mkdir(exist_ok=False); shutil.copyfile(source['path'], archive)
    review = dict(status='head_diagnosis_final_review_pass', results=B.binding(root/'RESULTS.json'),
        review_source=source, original_sources_inputs_archives_and_rows_exact=True, source_count=len(p['sources']),
        baseline_all100_losses_and_selected_checkpoint_exact=True,
        new_diagnostic_optimizer_updates=400, original_formal_optimizer_updates=100,
        published_inference=published, per_target_label_support=per_target,
        model_parameter_count=sum(v.numel() for v in model.parameters()),
        fit_source_objects=len(T.source_object_set(data, fit)), internal_source_objects=len(T.source_object_set(data, check)),
        internal_prior_constant_MSE=float(np.mean((prior - data['labels'][check])**2)), fit_positive_prior=prior,
        validation_used_for_original_selection=True, diagnostic_variants_compared_on_same_internal_split=True,
        new_physical_grasp_success_rate_measured=False, additional_physics_steps=0,
        review_optimizer_updates=0, new_CGR_forwards=0, video_recording=False, automatically_installed=False)
    B.verify({'review_source': source})
    B.save(root/'FINAL_REVIEW.json', review, exclusive=True)
    print({k: v for k, v in review.items() if k not in ('published_inference', 'per_target_label_support')}, flush=True)
    return review


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); audit(args.root)
