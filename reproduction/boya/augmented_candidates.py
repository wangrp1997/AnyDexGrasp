"""Original 11-branch recipe without DH3 camera-rig admission filters.

Only AST-selected, hash-pinned numerical statements are executed. This module
does not import robot_dh3 (which imports hardware), load weights or run physics.
Scene reflection is distinct from the later 180-degree thumb orientation flip.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_SHA = '7d227451761a429fff87f689e5dcb9c9e215da5ec3c5c594be1e16f3333fa797'


@lru_cache(maxsize=1)
def original_numerics():
    import copy
    import torch

    path = ROOT / 'robot_dh3.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != ORIGINAL_SHA:
        raise ValueError('Original robot_dh3 source no longer matches the pinned baseline')
    tree = ast.parse(path.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    body = functions['get_grasp'].body
    # From `if flip: augment_mat[:,0] = ...` through the recovered R assignment.
    start = next(i for i, n in enumerate(body) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Name) and n.test.id == 'flip')
    stop = next(i for i in range(start, len(body)) if isinstance(body[i], ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'mask' for t in body[i].targets))
    recover = ast.parse('def recover(preds, augment_mat, flip, device):\n    pass').body[0]
    recover.body = body[start:stop] + [ast.Return(value=ast.Name(id='preds', ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[functions['augment_data'], recover], type_ignores=[]))
    scope = dict(np=np, torch=torch, copy=copy)
    exec(compile(module, str(path), 'exec'), scope)
    return scope['augment_data'], scope['recover']


def augmentation_branches():
    """Consume the original 30 NumPy draws before the identity model forward."""
    augment, _ = original_numerics()
    return [(np.eye(4), False)] + [(augment(flip=bool(i % 2)), bool(i % 2)) for i in range(10)]


def check_transform(matrix, scene_flip):
    a = np.asarray(matrix)
    if (a.shape != (4, 4) or not np.isfinite(a).all()
            or not np.allclose(a[3], [0, 0, 0, 1], atol=2e-6, rtol=0)
            or not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=2e-6, rtol=0)
            or not np.isclose(np.linalg.det(a[:3, :3]), -1 if scene_flip else 1, atol=2e-6, rtol=0)):
        raise ValueError('Augmentation must have the declared rotation/reflection parity')
    return a.copy()


def recover_candidates(candidates, matrix, scene_flip, *, device='cpu'):
    import torch

    a = check_transform(matrix, scene_flip)
    rows = np.asarray(candidates)
    if rows.shape != (len(rows), 15) or rows.dtype != np.float32 or not np.isfinite(rows).all():
        raise ValueError('Finite original float32 Nx15 parser rows required')
    if not len(rows):
        return rows.copy()
    _, recover = original_numerics()
    with torch.no_grad():
        result = recover(torch.from_numpy(rows.copy()).to(device), a, scene_flip, torch.device(device))
    out = result.cpu().numpy()
    rotations = out[:, 3:12].reshape(-1, 3, 3)
    if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=3e-6, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=3e-6, rtol=0)):
        raise ValueError('Recovered grasp orientations must remain proper rotations')
    return out


def merge_branches(branches):
    """Keep source order and paired features, including empty identity branches."""
    rows, features, ids, local_rows = [], [], [], []
    for index, (candidates, paired) in enumerate(branches):
        c, f = np.asarray(candidates), np.asarray(paired)
        if c.shape != (len(c), 15) or f.shape != (len(c), 2261):
            raise ValueError('Unpaired candidate/feature arrays')
        if c.dtype != np.float32 or f.dtype != np.float32 or not np.isfinite(c).all() or not np.isfinite(f).all():
            raise ValueError('Finite original float32 arrays required')
        rows.append(c); features.append(f)
        ids.append(np.full(len(c), index, dtype=np.int16))
        local_rows.append(np.arange(len(c), dtype=np.int32))
    return dict(candidates=np.concatenate(rows) if rows else np.empty((0, 15), np.float32),
                candidate_features=np.concatenate(features) if features else np.empty((0, 2261), np.float32),
                source_branch=np.concatenate(ids) if ids else np.empty(0, np.int16),
                source_branch_row=np.concatenate(local_rows) if local_rows else np.empty(0, np.int32))
