"""Bound CPU diagnosis of the cap100 head; never collects or installs a model."""
import argparse
import ast
import copy
import hashlib
from pathlib import Path
import random
import shutil
import time
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F

import author_collection_batch as B
import author_protocol as A
import author_training as T


VARIANTS = ('original_relu', 'relu_positive_output_bias', 'linear_output', 'softplus_output')


def forward(core, x, *, activation='relu', batch_stats=False):
    """Original seven blocks, exposing final preactivation without state writes."""
    def block(i, value):
        value = getattr(core, f'conv{i}')(value)
        bn = getattr(core, f'bn{i}')
        if batch_stats:
            return F.batch_norm(value, None, None, bn.weight, bn.bias, True, 0., bn.eps)
        return bn(value)
    y = F.relu(block(1, x.reshape(len(x), 480, 1)))
    residual = F.relu(block(2, y))
    y = F.relu(block(3, residual))
    y = F.relu(block(4, y))
    y = F.relu(block(5, y + residual))
    y = F.relu(block(6, y))
    pre = block(7, y).transpose(1, 2).contiguous()
    if activation == 'relu':
        return F.relu(pre), pre
    if activation == 'linear':
        return pre, pre
    if activation == 'softplus':
        return F.softplus(pre), pre
    raise ValueError(activation)


def metrics(scores, labels):
    scores = np.asarray(scores, dtype=float); truth = np.asarray(labels, dtype=bool)
    pred = scores >= .5
    pos, neg = scores[truth], scores[~truth]
    pairs = pos[:, None] - neg[None, :]
    return dict(records=len(scores), positive=int(truth.sum()), negative=int((~truth).sum()),
        TP=int((pred & truth).sum()), FP=int((pred & ~truth).sum()),
        TN=int((~pred & ~truth).sum()), FN=int((~pred & truth).sum()),
        MSE=float(np.mean((scores - truth)**2)),
        ROC_AUC=float(((pairs > 0) + .5*(pairs == 0)).mean()) if pairs.size else None,
        positive_exact_zero=int((pos == 0).sum()), negative_exact_zero=int((neg == 0).sum()),
        minimum=float(scores.min()), maximum=float(scores.max()))


def selected(pred, target):
    return pred.reshape(len(target), 20)[torch.arange(len(target)), torch.as_tensor(target)]


def inspect(core, data, ids, *, activation='relu', batch_stats=False):
    core.eval()
    with torch.no_grad():
        pred, pre = forward(core, torch.from_numpy(data['features'][ids]),
                            activation=activation, batch_stats=batch_stats)
    return metrics(selected(pred, data['target_index'][ids]).numpy(), data['labels'][ids]), pred, pre


def calibrate_fit_only(core, features):
    """Independent clone, seven sequential fit-only BN moment replacements."""
    result = copy.deepcopy(core).eval(); x = torch.from_numpy(features)
    with torch.no_grad():
        for i in range(1, 8):
            captured = []
            h = getattr(result, f'conv{i}').register_forward_hook(lambda m, a, y: captured.append(y.clone()))
            forward(result, x); h.remove()
            bn = getattr(result, f'bn{i}')
            bn.running_mean.copy_(captured[0].mean((0, 2)))
            bn.running_var.copy_(captured[0].var((0, 2), unbiased=True))
    return result


def verify_rows(data):
    """Re-derive canonical input from raw saved features and actual flip flag."""
    node = next(n for n in ast.parse((B.ROOT/'robot_dh3.py').read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'get_graspgroup_features')
    scope = dict(np=np, copy=copy)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'robot_dh3.py', 'exec'), scope)
    robot_sha = B.sha(B.ROOT/'robot_dh3.py')
    bindings = []
    for i, (path, digest) in enumerate(zip(data['record_json'], data['record_sha256'])):
        B.verify({'record': dict(path=str(path), sha256=str(digest))})
        r = B.read(str(path)); b = r['bindings']
        B.verify({k: b[k] for k in ('candidate', 'execution')})
        c = B.read(b['candidate']['path']); e = B.read(b['execution']['path'])
        assert r['eligible'] and r['formal_training_eligible'] and not r['exclude_training']
        assert int(data['labels'][i]) == r['label'] == int(e['evaluation']['task_success'])
        assert r['label'] is not None and e['status'] == 'complete' and e['stop'] is None
        assert e['steps'] == e['expected_steps'] and e['native_warning_count'] == 0
        assert all(v for k, v in e['gates'].items() if k != 'penetration')
        assert e['maxima']['penetration_m'] <= .005
        assert c['gesture_id'] == 1 and c['candidate_sha256'] == str(data['candidate_sha256'][i])
        assert c['canonicalization']['source_sha256'] == robot_sha
        raw = np.asarray(c['source_feature2261']); flip = c['source_if_flip']
        literal = scope['get_graspgroup_features'](np.c_[raw[None].copy(), [flip]], None)
        angle = int(raw[-2] + .1); depth = int(raw[-1]*100 + .1)
        base = raw[240:720].astype(np.float32).reshape(2, 240).copy()
        if flip:
            base = np.concatenate([base[:, 120:], base[:, :120]], axis=1)
        independent = np.roll(base, -angle*5, axis=1).ravel()
        np.testing.assert_array_equal(data['features'][i], independent)
        np.testing.assert_array_equal(data['features'][i], literal['grasp_preds_features'][0])
        np.testing.assert_array_equal(data['features'][i], np.asarray(c['canonical_cgr480'], np.float32))
        assert angle == c['raw_angle_index'] and depth == c['raw_depth_index']
        assert int(data['target_index'][i]) == depth*4 + c['extra_depth_index']
        assert ('train' if r['split'] == 'fit' else 'internal_check') == str(data['split'][i])
        bindings.append(dict(record=dict(path=str(path), sha256=str(digest)),
                             candidate=b['candidate'], execution=b['execution']))
    return dict(rows=len(bindings), raw_to_canonical_exact=True, labels_exact=True,
                depth_output_indices_exact=True, record_bindings=bindings)


def run_variant(name, data, fit, check, seed, root, components, original):
    torch.manual_seed(seed)
    fit_order = fit.tolist(); check_order = check.tolist(); rng = random.Random(seed)
    rng.shuffle(fit_order); rng.shuffle(check_order)
    train = torch.utils.data.DataLoader(T.CanonicalDataset(data, fit_order, 1), batch_size=128, shuffle=True, num_workers=0)
    validation = torch.utils.data.DataLoader(T.CanonicalDataset(data, check_order, 1), batch_size=128, shuffle=False, num_workers=0)
    cls, _, loss_cls, poly = components
    cfg = dict(num_multifinger_type=1, num_multifinger_depth=4, num_two_finger_angle=1, num_two_finger_depth=5)
    model = cls(**cfg); activation = {'linear_output': 'linear', 'softplus_output': 'softplus'}.get(name, 'relu')
    if name == 'relu_positive_output_bias':
        with torch.no_grad(): model.bn7.bias.fill_(1.)
    if activation != 'relu':
        def alternative(self, endpoints):
            pred, _ = forward(self, endpoints['grasp_preds_features'], activation=activation)
            endpoints['stage4_grasp_preds_five_hand'] = pred
            return pred, endpoints
        model.forward = MethodType(alternative, model)
    criterion = loss_cls(**cfg, train_type=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005, weight_decay=.0005)
    scheduler = poly(optimizer, max_iter=100, power=.99, last_step=-1)
    output = root/name; output.mkdir()
    selected_states = {}; history = []; cached = {}
    feature_ids = {data['features'][i].tobytes(): int(i) for i in fit}
    assert len(feature_ids) == len(fit)
    def before(m, inputs):
        if m.training:
            cached['ids'] = np.asarray([feature_ids[x.numpy().tobytes()] for x in inputs[0]['grasp_preds_features']], int)
    def after_bn(m, inputs, out):
        if model.training: cached['pre'] = out.detach().clone().squeeze(-1)
    hooks = [model.register_forward_pre_hook(before), model.bn7.register_forward_hook(after_bn)]
    def save_selected(current, path):
        selected_states[path.name.split('_', 1)[0]] = (int(scope['EPOCH_CNT']), copy.deepcopy(current.state_dict()))
    scope, writer, checker, _ = T.source_scope(model, criterion, optimizer, scheduler, train, validation, 1, output, save_selected, 'cpu')
    best = [0.]*9; updates = 0; dead_sets = []
    for epoch in range(100):
        scope['EPOCH_CNT'] = epoch; scope['train_one_epoch']()
        current = max(int(s['step']) for s in optimizer.state.values()); updates = current
        ids = cached['ids']; z = cached['pre'][torch.arange(len(ids)), torch.from_numpy(data['target_index'][ids])].numpy()
        dead = set(map(int, ids[(z <= 0) & (data['labels'][ids] == 1)])); dead_sets.append(dead)
        with torch.no_grad(): indicators = scope['eval_one_epoch']()
        scheduler.step(); best = scope['save_model'](epoch, indicators, best)
        history.append(dict(epoch=epoch, train_loss=writer.values['train/loss'], internal_loss=checker.values['test/loss'],
            train_positive_nonpositive_preactivation=len(dead), updates=current,
            internal_recall=checker.values['test/recall_0.5']))
    for h in hooks: h.remove()
    assert updates == 100
    fit_metrics, _, _ = inspect(model, data, fit, activation=activation)
    internal_metrics, _, _ = inspect(model, data, check, activation=activation)
    batch_metrics, _, _ = inspect(model, data, fit, activation=activation, batch_stats=True)
    calibrated = calibrate_fit_only(model, data['features'][fit])
    calibrated_internal, _, _ = inspect(calibrated, data, check, activation=activation)
    torch.save(dict(schema='anydex_mjx_head_diagnostic_checkpoint_v1', variant=name, activation=activation,
                    model_state_dict=model.state_dict(), epochs=100, optimizer_updates=updates,
                    fit_trial_ids=data['trial_id'][fit].tolist(), diagnostic_only=True), output/'final_diagnostic.pt')
    replay = None
    if name == 'original_relu':
        assert all(h['train_loss'] == old['train']['train/loss'] and h['internal_loss'] == old['internal']['test/loss']
                   for h, old in zip(history, original['history']))
        official = torch.load(original['checkpoints']['0.5']['path'], map_location='cpu', weights_only=True)['model_state_dict']
        epoch, replayed = selected_states['0.5']
        assert epoch == original['checkpoints']['0.5']['epoch'] and all(torch.equal(v, official[k]) for k, v in replayed.items())
        replay = dict(all100_epoch_losses_exact=True, selected_checkpoint_all_tensors_exact=True, selected_epoch=epoch)
    result = dict(variant=name, activation=activation, epochs=100, optimizer_updates=updates,
        fit_eval=fit_metrics, fit_batch_statistics=batch_metrics, internal_eval=internal_metrics,
        fit_only_BN_recalibrated_internal=calibrated_internal,
        persistent_nonpositive_training_positives_after_epoch10=sorted(set.intersection(*dead_sets[10:])),
        original_replay=replay, selected_epochs={k: v[0] for k, v in selected_states.items()},
        validation_used_for_original_selection=True, comparison_endpoint='fixed_final_epoch99',
        diagnostic_only=True, automatically_installed=False, history=history)
    B.save(output/'RESULTS.json', result, exclusive=True)
    return result


def run(root, training_root):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False)
    training_root = Path(training_root).resolve(); p = B.read(training_root/'PROTOCOL.json')
    original = B.read(training_root/'head/TRAINING_RESULTS.json')
    B.verify(p['sources']); B.verify({'dataset': p['dataset'], 'manifest': p['manifest']})
    sources = dict(p['sources']); sources[Path(__file__).name] = B.binding(Path(__file__))
    sources['robot_dh3.py'] = B.binding(B.ROOT/'robot_dh3.py')
    protocol = dict(schema='anydex_mjx_cap100_head_diagnosis_v1', sources=sources,
        input_training=B.binding(training_root/'head/TRAINING_RESULTS.json'), dataset=p['dataset'],
        original_protocol=B.binding(training_root/'PROTOCOL.json'), seed=p['seed'], variants=VARIANTS,
        optimizer_update_budget=400, fixed_final_epoch_comparison=99, class_balancing=False,
        physics_steps=0, CGR_forwards=0, video_recording=False, automatic_policy_replacement=False,
        diagnostic_only=True, independent_physical_evaluation=False)
    B.save(root/'PROTOCOL.json', protocol, exclusive=True)
    for name, b in sources.items():
        dest = root/'source_archive'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(b['path'], dest)
    torch.set_num_threads(1); started = time.monotonic()
    with np.load(p['dataset']['path'], allow_pickle=False) as z: data = {k: z[k].copy() for k in z.files}
    fit, check = T.validate_arrays(data); fit = fit[:100]
    assert data['trial_id'][fit].tolist() == p['selected_fit_trial_ids']
    rows = verify_rows(data); B.save(root/'ROW_AUDIT.json', rows, exclusive=True)
    components = T.components(); _, inference, loss_cls, _ = components
    chosen = original['checkpoints']['0.5']; B.verify({'chosen': chosen})
    core = inference().eval(); core.load_state_dict(torch.load(chosen['path'], map_location='cpu', weights_only=True)['model_state_dict'], strict=True)
    before = copy.deepcopy(core.state_dict()); all_ids = np.arange(len(data['labels']))
    _, all_pred, all_pre = inspect(core, data, all_ids)
    with torch.no_grad(): assert torch.equal(core(torch.from_numpy(data['features']))[0], all_pred)
    fit_metrics, _, _ = inspect(core, data, fit)
    batch_metrics, _, batch_pre = inspect(core, data, fit, batch_stats=True)
    check_metrics, _, _ = inspect(core, data, check)
    calibrated = calibrate_fit_only(core, data['features'][fit]); calibrated_check, _, _ = inspect(calibrated, data, check)
    assert all(torch.equal(v, core.state_dict()[k]) for k, v in before.items())
    z = selected(batch_pre, data['target_index'][fit]).detach().requires_grad_(); labels = torch.from_numpy(data['labels'][fit]).float()
    F.mse_loss(F.relu(z), labels).backward(); relu_grad = z.grad.clone(); z.grad = None
    F.mse_loss(F.softplus(z), labels).backward(); softplus_grad = z.grad.clone()
    dead = (labels == 1) & (z <= 0)
    assert (relu_grad[dead] == 0).all() and (softplus_grad[dead] != 0).all()
    # Exhaust all20 target addresses through the untouched original loss.
    criterion = loss_cls(num_multifinger_type=1, num_multifinger_depth=4, num_two_finger_angle=1, num_two_finger_depth=5, train_type=1)
    idx = torch.arange(20); prediction = torch.arange(400).float().reshape(20, 1, 20)/400
    endpoints = dict(stage4_grasp_preds_five_hand=prediction, result=idx%2,
        multifinger_pose_finger_type=torch.ones(20, 1), multifinger_pose_depth_type=(idx%4)[:, None],
        two_fingers_pose_angle_type=torch.zeros(20, 1), two_fingers_pose_depth_type=(idx//4)[:, None])
    assert torch.equal(criterion(endpoints)[0], F.mse_loss(selected(prediction, idx), (idx%2).float()))
    support = np.bincount(data['target_index'][fit], minlength=20)
    audit = dict(status='read_only_head_diagnosis_pass', records_verified=len(data['labels']),
        fit_target_counts=support.tolist(), internal_target_counts=np.bincount(data['target_index'][check], minlength=20).tolist(),
        internal_targets_absent_from_fit=int((support[data['target_index'][check]] == 0).sum()),
        selected_checkpoint_fit_eval=fit_metrics, selected_checkpoint_fit_batch_statistics=batch_metrics,
        selected_checkpoint_internal=check_metrics, fit_only_BN_recalibrated_internal=calibrated_check,
        dead_training_positive_count=int(dead.sum()), actual_selected_ReLU_MSE_grad_zero=int((relu_grad[dead] == 0).sum()),
        same_values_softplus_MSE_grad_nonzero=int((softplus_grad[dead] != 0).sum()),
        functional_forward_published_exact=True, all20_original_loss_addresses_exact=True,
        original_checkpoint_buffers_and_parameters_unchanged=True, new_physics_steps=0)
    B.save(root/'READ_ONLY_DIAGNOSIS.json', audit, exclusive=True)
    np.savez_compressed(root/'SELECTED_CHECKPOINT_SCORES.npz', trial_id=data['trial_id'],
                        scores=all_pred.numpy(), preactivation=all_pre.numpy(), target_index=data['target_index'], labels=data['labels'])
    print('READ_ONLY', audit, flush=True)
    results = []
    for name in VARIANTS:
        result = run_variant(name, data, fit, check, p['seed'], root, components, original); results.append(result)
        print('VARIANT', name, result['fit_batch_statistics'], result['internal_eval'], flush=True)
    B.verify(sources); B.verify({'dataset': p['dataset'], 'manifest': p['manifest'], 'training': protocol['input_training']})
    summary = dict(status='head_diagnosis_complete', protocol=B.binding(root/'PROTOCOL.json'),
        row_audit=B.binding(root/'ROW_AUDIT.json'), read_only_diagnosis=B.binding(root/'READ_ONLY_DIAGNOSIS.json'),
        variants=[{k: v for k, v in result.items() if k != 'history'} for result in results],
        optimizer_updates=sum(r['optimizer_updates'] for r in results), wall_s=time.monotonic()-started,
        physics_steps=0, CGR_forwards=0, video_recording=False, diagnostic_only=True, automatically_installed=False)
    B.save(root/'RESULTS.json', summary, exclusive=True)
    print('COMPLETE', summary['optimizer_updates'], summary['wall_s'], flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--training-root', type=Path, required=True)
    args = parser.parse_args(); run(args.root, args.training_root)
