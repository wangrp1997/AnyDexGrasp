"""Read-only grouped inputs; only pre-execution metadata enters model features."""
from pathlib import Path
import hashlib
import json
import numpy as np

FIXED_MANIFEST_SHA256 = 'c85a284be680707a2b3bcaae7950e1981bd907a9e88870c434ced1c22f6e91bb'
GESTURES = (1, 6, 7, 8)
REQUIRED = ('features', 'labels', 'target_index', 'gesture_ids', 'mesh_sha256', 'trial_id', 'object_id', 'split')
METADATA_COLUMNS = tuple([f'gesture_{g}' for g in GESTURES] + [f'raw_depth_{i}' for i in range(5)] + [f'extra_depth_{i}' for i in range(4)] + ['if_flip', 'raw_width_m', 'actual_reference_width_m'])

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _require(condition, message):
    if not condition:
        raise ValueError(message)

def _index(value, lower, upper, name):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)), name+' must be an integer')
    value = int(value)
    _require(lower <= value < upper, name+' out of range')
    return value

def metadata_features(binding, gesture_id, target_index):
    """Encode the explicit 16-column whitelist; ignore other binding fields."""
    gesture_id = _index(gesture_id, 1, 9, 'gesture_id')
    _require(gesture_id in GESTURES, 'Unsupported Boya gesture')
    target_index = _index(target_index, 0, 20, 'target_index')
    raw, extra = divmod(target_index, 4)
    _require(_index(binding['multifinger_pose_finger_type'], 1, 9, 'binding gesture') == gesture_id, 'Binding gesture mismatch')
    _require(_index(binding['two_fingers_pose_depth_type'], 0, 5, 'binding raw depth') == raw, 'Binding raw depth mismatch')
    _require(_index(binding['multifinger_pose_depth_type'], 0, 4, 'binding extra depth') == extra, 'Binding extra depth mismatch')
    _require(type(binding['if_flip']) is bool, 'if_flip must be a JSON boolean')
    widths = (binding['raw_width_m'], binding['aperture']['actual_ref_width_m'])
    for width in widths:
        _require(isinstance(width, (int, float)) and not isinstance(width, bool) and np.isfinite(width) and 0 < width <= 1., 'Width must be finite positive metres, at most 1m')
    expected_raw = (.005, .010, .020, .030, .040)[raw]
    _require(np.isfinite(binding['raw_depth_m']) and abs(binding['raw_depth_m']-expected_raw) <= 1e-8, 'Raw depth value/index mismatch')
    _require(np.isfinite(binding['extra_depth_m']) and abs(binding['extra_depth_m']-.01*extra) <= 1e-8, 'Extra depth value/index mismatch')
    out = np.zeros(16, dtype=np.float32)
    out[GESTURES.index(gesture_id)] = 1
    out[4+raw] = 1
    out[9+extra] = 1
    out[13] = binding['if_flip']
    out[14:] = widths
    _require(np.isfinite(out).all(), 'Nonfinite metadata after float32 conversion')
    return out

def load_shared_data(manifest_path):
    """Validate frozen 435 rows and return arrays in stable trial-ID order."""
    from train_s40_code_recipe import validate_fixed_dataset
    manifest_path = Path(manifest_path).resolve()
    _require(_sha(manifest_path) == FIXED_MANIFEST_SHA256, 'Only frozen 435-row manifest permitted')
    manifest, data, inputs, _counts = validate_fixed_dataset(manifest_path)
    inputs = dict(inputs)
    inputs[str(manifest_path)] = _sha(manifest_path)
    groups = [data[g] for g in GESTURES]
    for group in groups:
        _require(all(k in group for k in REQUIRED), 'Required old NPZ arrays missing')
        n = len(group['trial_id'])
        _require(all(len(group[k]) == n for k in REQUIRED), 'Old NPZ row count mismatch')
    keys = set.intersection(*(set(g) for g in groups))
    keys = sorted(k for k in keys if all(isinstance(g[k], np.ndarray) and g[k].ndim > 0 and len(g[k]) == len(g['trial_id']) for g in groups))
    arrays = {k: np.concatenate([g[k] for g in groups], axis=0) for k in keys}
    _require(all(k in arrays for k in REQUIRED), 'Required array has invalid layout')
    order = np.argsort(arrays['trial_id'].astype(str), kind='stable')
    arrays = {k: v[order].copy() for k, v in arrays.items()}
    ids = arrays['trial_id'].astype(str)
    _require(len(ids) == 435 and len(set(ids)) == 435, 'Expected 435 unique trial IDs')
    _require(arrays['features'].shape == (435, 480) and arrays['features'].dtype == np.dtype('float32'), 'Expected canonical float32 CGR480')
    _require(np.isfinite(arrays['features']).all(), 'Nonfinite canonical features')
    for k in ('labels', 'target_index', 'gesture_ids'):
        _require(np.issubdtype(arrays[k].dtype, np.integer), k+' must remain integer')
    samples = {str(s['trial_id']): s for s in manifest['samples']}
    _require(len(samples) == len(manifest['samples']) == 435 and set(samples) == set(ids), 'Manifest/NPZ trial identity mismatch')
    archives, encoded, observation_groups = {}, [], []
    for i, tid in enumerate(ids):
        sample = samples[tid]
        for key, column in [('trial_id','trial_id'), ('object_id','object_id'), ('gesture_id','gesture_ids'), ('mesh_sha256','mesh_sha256'), ('split','split'), ('label','labels'), ('target_index','target_index')]:
            _require(sample[key] == arrays[column][i], f'Sample/NPZ {key} mismatch for {tid}')
        target = _index(int(arrays['target_index'][i]), 0, 20, 'target_index')
        _require(sample['raw_depth_type'] == target//4 and sample['extra_depth_index'] == target%4, 'Sample depth/index mismatch')
        _require(int(arrays['labels'][i]) in (0,1), 'Labels must be binary')
        npz_path = str(Path(sample['npz']).resolve())
        if npz_path not in archives:
            with np.load(npz_path, allow_pickle=False) as z:
                archives[npz_path] = {k:z[k].copy() for k in REQUIRED}
            inputs[npz_path] = _sha(npz_path)
        archive = archives[npz_path]
        row = _index(sample['npz_row'], 0, len(archive['trial_id']), 'npz_row')
        for key in REQUIRED:
            _require(np.array_equal(arrays[key][i], archive[key][row]), f'Declared NPZ row mismatch: {tid}/{key}')
        feature = np.ascontiguousarray(arrays['features'][i], dtype='<f4')
        identity = sample['canonical_CGR480_identity']
        _require(identity['shape'] == [480] and identity['dtype'] == '<f4', 'Feature identity layout mismatch')
        _require(hashlib.sha256(feature.tobytes()).hexdigest() == identity['c_order_bytes_sha256'], 'Canonical feature SHA mismatch')
        record_path = Path(sample['record_json']).resolve()
        record_sha = _sha(record_path)
        _require(record_sha == sample['record_sha256'], 'Record SHA mismatch: '+tid)
        if str(record_path) in inputs:
            _require(inputs[str(record_path)] == record_sha, 'Old validation record hash mismatch')
        inputs[str(record_path)] = record_sha
        record = json.loads(record_path.read_text())
        _require(record['label_eligible'] is True and record['benchmark_only'] is False, 'Nontraining or ineligible record')
        _require(type(record['result']) is int and record['result'] in (0,1) and record['result'] == int(arrays['labels'][i]), 'Record label mismatch')
        binding = record['binding']
        encoded.append(metadata_features(binding, int(arrays['gesture_ids'][i]), target))
        observation_sha = binding['execution_spec']['source_observation_sha256']
        _require(isinstance(observation_sha,str) and len(observation_sha)==64 and all(c in '0123456789abcdef' for c in observation_sha), 'Invalid pre-execution observation SHA')
        observation_groups.append(str(arrays['object_id'][i])+'::'+observation_sha)
    arrays['metadata_features'] = np.stack(encoded).astype(np.float32)
    arrays['all_features'] = np.concatenate([arrays['features'], arrays['metadata_features']], axis=1).astype(np.float32)
    arrays['observation_group'] = np.array(observation_groups)
    _require(arrays['all_features'].shape == (435,496) and np.isfinite(arrays['all_features']).all(), 'Invalid shared model inputs')
    return manifest, arrays, inputs

def grouped_folds(mesh_sha256, split, n_folds=4, seed=20261111, declared_train_meshes=None):
    """Split train meshes without labels; internal rows never enter any fold."""
    mesh = np.asarray(mesh_sha256).astype(str)
    split = np.asarray(split).astype(str)
    _require(mesh.ndim == split.ndim == 1 and len(mesh) == len(split), 'Mesh/split vectors must align')
    _require(isinstance(n_folds,int) and n_folds >= 2, 'At least two folds required')
    train = split == 'train'
    observed = set(mesh[train])
    internal = set(mesh[~train])
    _require(not observed & internal, 'A mesh crosses training and internal split')
    declared = observed if declared_train_meshes is None else set(map(str,declared_train_meshes))
    _require(observed <= declared and not declared & internal, 'Declared train meshes omit labels or include internal meshes')
    _require(len(declared) >= n_folds, 'Too few declared training meshes for folds')
    shuffled = np.random.default_rng(seed).permutation(sorted(declared))
    output = []
    for fold in range(n_folds):
        check_meshes = sorted(shuffled[fold::n_folds].tolist())
        fit_meshes = sorted(declared-set(check_meshes))
        fit_indices = np.flatnonzero(train & np.isin(mesh,fit_meshes))
        check_indices = np.flatnonzero(train & np.isin(mesh,check_meshes))
        output.append(dict(fit_indices=fit_indices, check_indices=check_indices, fit_meshes=fit_meshes, check_meshes=check_meshes))
    return output

def fit_standardizer(x, fit_indices, continuous_columns=None):
    """Fit mean/std only on supplied fitting rows and continuous columns."""
    x = np.asarray(x)
    _require(x.ndim==2 and np.issubdtype(x.dtype,np.number) and np.isfinite(x).all(), 'Expected finite 2D numeric input')
    fit_indices = np.asarray(fit_indices)
    _require(fit_indices.ndim==1 and np.issubdtype(fit_indices.dtype,np.integer) and len(fit_indices)>0, 'Nonempty integer fit indices required')
    _require(len(np.unique(fit_indices))==len(fit_indices) and np.all((fit_indices>=0)&(fit_indices<len(x))), 'Invalid or repeated fit indices')
    columns = np.arange(x.shape[1],dtype=np.int64) if continuous_columns is None else np.asarray(continuous_columns)
    _require(columns.ndim==1 and np.issubdtype(columns.dtype,np.integer), 'Integer continuous column indices required')
    _require(len(np.unique(columns))==len(columns) and np.all((columns>=0)&(columns<x.shape[1])), 'Invalid continuous columns')
    mean=np.zeros(x.shape[1],dtype=np.float64)
    scale=np.ones(x.shape[1],dtype=np.float64)
    fitting=np.asarray(x[np.ix_(fit_indices,columns)],dtype=np.float64)
    mean[columns]=fitting.mean(axis=0)
    deviation=fitting.std(axis=0,ddof=0)
    scale[columns]=np.where(deviation>1e-12,deviation,1.)
    return dict(mean=mean,scale=scale,continuous_columns=columns.astype(np.int64),fit_count=int(len(fit_indices)))

def transform(x, stats):
    x=np.asarray(x)
    mean=np.asarray(stats['mean'],dtype=np.float64)
    scale=np.asarray(stats['scale'],dtype=np.float64)
    _require(x.ndim==2 and mean.shape==scale.shape==(x.shape[1],), 'Standardizer dimension mismatch')
    _require(np.isfinite(x).all() and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all(), 'Invalid standardizer/input values')
    out=((x.astype(np.float64)-mean)/scale).astype(np.float32)
    _require(np.isfinite(out).all(),'Nonfinite standardized features')
    return out
