"""Metrics for historical executed candidate pools, not new grasp success rates."""
from __future__ import annotations
import numpy as np

EPS = 1e-6
SCOPE = 'Development analysis of historical executed candidate pools; not full-pool or new physical grasp success rates'

def _identifiers(values, name):
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(name + ' must be one-dimensional')
    result = []
    for value in array.tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, int, np.integer)):
            raise ValueError(name + ' must contain nonempty strings or integer identifiers')
        value = str(value)
        if not value:
            raise ValueError(name + ' contains an empty identifier')
        result.append(value)
    return np.asarray(result, dtype=object)

def _auc(labels, scores):
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind='stable')
    ranked_scores, ranked_labels = scores[order], labels[order]
    positive_rank_sum = 0.0
    begin = 0
    while begin < len(labels):
        end = begin + 1
        while end < len(labels) and ranked_scores[end] == ranked_scores[begin]:
            end += 1
        positive_rank_sum += float(ranked_labels[begin:end].sum()) * (begin + end + 1) / 2.0
        begin = end
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))

def _mean(values):
    return float(np.mean(values)) if len(values) else None

def _losses(labels, scores):
    if not len(labels):
        return dict(raw_mse=None, clipped_brier=None, clipped_logloss=None)
    probability = np.clip(scores, 0.0, 1.0)
    safe_probability = np.clip(probability, EPS, 1.0 - EPS)
    with np.errstate(over='ignore', invalid='ignore'):
        raw_mse = float(np.mean(np.square(scores - labels)))
    if not np.isfinite(raw_mse):
        raise ValueError('Raw MSE is not representable as a finite float')
    return dict(raw_mse=raw_mse, clipped_brier=float(np.mean(np.square(probability - labels))),
                clipped_logloss=float(-np.mean(labels * np.log(safe_probability)
                    + (1.0 - labels) * np.log1p(-safe_probability))))

def evaluate_scores(labels, scores, object_ids, trial_ids):
    """Macro AUC uses mixed objects; Brier all objects; raw-score ties averaged."""
    try:
        labels = np.asarray(labels, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError('Numeric binary labels and numeric scores required') from error
    objects = _identifiers(object_ids, 'object_ids')
    trials = _identifiers(trial_ids, 'trial_ids')
    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError('Labels and scores must be one-dimensional')
    if not (len(labels) == len(scores) == len(objects) == len(trials)):
        raise ValueError('All four inputs must have equal lengths')
    if not np.isfinite(labels).all() or not np.isfinite(scores).all():
        raise ValueError('Nonfinite labels or scores are forbidden')
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError('Labels must be exactly zero or one')
    if len(set(trials.tolist())) != len(trials):
        raise ValueError('Trial IDs must be globally unique')
    per_object = []
    for identifier in sorted(set(objects.tolist())):
        mask = objects == identifier
        y, s, ids = labels[mask], scores[mask], trials[mask]
        top = s == s.max()
        positive = int(y.sum())
        per_object.append(dict(object_id=identifier, count=len(y), pos=positive,
            neg=len(y)-positive, pair_auc=_auc(y, s),
            top_tie_mean_label=float(y[top].mean()), top_tie_count=int(top.sum()),
            top1_selected_trial_id=min(ids[top].tolist()),
            score_min=float(s.min()), score_max=float(s.max()), **_losses(y, s)))
    mixed = [row for row in per_object if row['pos'] and row['neg']]
    positive = int(labels.sum())
    return dict(count=len(labels), pos=positive, neg=len(labels)-positive,
        **_losses(labels, scores), global_auc=_auc(labels, scores),
        macro_object_auc=_mean([row['pair_auc'] for row in mixed]),
        macro_object_brier=_mean([row['clipped_brier'] for row in per_object]),
        macro_top_tie_mean_label=_mean([row['top_tie_mean_label'] for row in per_object]),
        macro_mixed_top_tie_mean_label=_mean([row['top_tie_mean_label'] for row in mixed]),
        object_count=len(per_object), mixed_object_count=len(mixed),
        single_class_object_count=len(per_object)-len(mixed),
        positive_only_object_count=sum(row['neg'] == 0 for row in per_object),
        negative_only_object_count=sum(row['pos'] == 0 for row in per_object),
        score_min=float(scores.min()) if len(scores) else None,
        score_max=float(scores.max()) if len(scores) else None,
        per_object=per_object, logloss_clamp=EPS, scope=SCOPE)

def _mixed_auc_map(metrics):
    if not isinstance(metrics, dict) or not isinstance(metrics.get('per_object'), list):
        raise ValueError('Metrics must contain a per_object list')
    values, seen = {}, set()
    for row in metrics['per_object']:
        if not isinstance(row, dict) or not isinstance(row.get('object_id'), str):
            raise ValueError('Each object metric must have a string object_id')
        identifier = row['object_id']
        if identifier in seen:
            raise ValueError('Duplicate object metrics')
        seen.add(identifier)
        for key in ('count', 'pos', 'neg'):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError('Object counts must be nonnegative integers')
        if row['count'] != row['pos'] + row['neg'] or row['count'] == 0:
            raise ValueError('Object count must equal positive plus negative counts and be nonzero')
        auc = row.get('pair_auc')
        if row['pos'] and row['neg']:
            if isinstance(auc, bool) or not isinstance(auc, (int, float, np.number)) or not np.isfinite(auc) or not 0 <= auc <= 1:
                raise ValueError('Mixed objects require finite AUC in [0,1]')
            values[identifier] = float(auc)
        elif auc is not None:
            raise ValueError('Single-class object AUC must be None')
    return values

def paired_object_bootstrap(ref_metrics, new_metrics, seed=20261111, replicates=2000):
    """Paired object resampling of new-minus-reference AUC, for development only."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('Seed must be a nonnegative integer')
    if isinstance(replicates, bool) or not isinstance(replicates, (int, np.integer)) or replicates < 1:
        raise ValueError('Replicates must be a positive integer')
    reference, new = _mixed_auc_map(ref_metrics), _mixed_auc_map(new_metrics)
    identifiers = sorted(set(reference) & set(new))
    deltas = np.asarray([new[key]-reference[key] for key in identifiers], dtype=float)
    if len(deltas):
        rng = np.random.default_rng(int(seed))
        samples = rng.integers(0, len(deltas), size=(int(replicates), len(deltas)))
        boot = deltas[samples].mean(axis=1)
        interval = [float(value) for value in np.quantile(boot, [.025, .975])]
    else:
        interval = [None, None]
    return dict(common_mixed_object_count=len(identifiers), object_ids=identifiers,
        reference_mixed_object_count=len(reference), new_mixed_object_count=len(new),
        excluded_reference_mixed_object_count=len(set(reference)-set(new)),
        excluded_new_mixed_object_count=len(set(new)-set(reference)),
        mean_delta_macro_pair_auc=_mean(deltas), ci95=interval,
        object_wins=int(np.sum(deltas > 0)), object_ties=int(np.sum(deltas == 0)),
        object_losses=int(np.sum(deltas < 0)), seed=int(seed), replicates=int(replicates),
        per_object_deltas=[dict(object_id=key, reference_pair_auc=reference[key],
            new_pair_auc=new[key], delta=new[key]-reference[key]) for key in identifiers],
        estimand='Unweighted mean paired object AUC difference, new minus reference',
        interval_method='Paired object bootstrap percentile 2.5/97.5; NumPy linear quantiles', scope=SCOPE)
