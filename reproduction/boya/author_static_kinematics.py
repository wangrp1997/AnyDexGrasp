"""Keep the original FK arithmetic; compute immutable URDF frames only once."""
from collections import OrderedDict
from contextlib import contextmanager
import threading
from unittest.mock import patch

import numpy as np
from author_planning_reuse import PlanningSession as OriginalSession, array_key

_OWNER = threading.Lock()


class StaticKinematics:
    def __init__(self, stats=None, *, max_rotations=65536):
        if max_rotations < 1:
            raise ValueError('Positive rotation cache capacity required')
        self.origins, self.rotations = {}, OrderedDict()
        self.max_rotations = max_rotations
        self.stats = {} if stats is None else stats
        self.stats.update(urdf_origin_calls=0, urdf_origin_builds=0,
                          joint_rotation_calls=0, joint_rotation_builds=0,
                          joint_rotation_evictions=0)

    def origin(self, original, node):
        # These are exactly the only two XML attributes read by origin_matrix.
        key = None if node is None else (node.get('xyz', '0 0 0'), node.get('rpy', '0 0 0'))
        self.stats['urdf_origin_calls'] += 1
        if key not in self.origins:
            self.origins[key] = original(node)
            self.stats['urdf_origin_builds'] += 1
        return self.origins[key].copy()

    def rotation(self, original, axis, angle):
        self.stats['joint_rotation_calls'] += 1
        axis_value = np.asarray(axis, dtype=float)
        # Preserve angle dtype: np.sin(float32) and np.sin(float64) differ.
        angle_value = np.asarray(angle)
        if not np.isfinite(axis_value).all() or not np.isfinite(angle_value).all():
            return original(axis, angle)
        key = array_key(axis_value), array_key(angle_value)
        if key not in self.rotations:
            self.rotations[key] = original(axis, angle)
            self.stats['joint_rotation_builds'] += 1
            if len(self.rotations) > self.max_rotations:
                self.rotations.popitem(last=False)
                self.stats['joint_rotation_evictions'] += 1
        else:
            self.rotations.move_to_end(key)
        return self.rotations[key].copy()

    @contextmanager
    def installed(self):
        if not _OWNER.acquire(blocking=False):
            raise RuntimeError('Static FK cache requires one serial owner')
        try:
            import canondex.robots.kinematics as K
            origin, rotation = K.origin_matrix, K.rotation
            with patch.object(K, 'origin_matrix', lambda node: self.origin(origin, node)), \
                 patch.object(K, 'rotation', lambda axis, angle: self.rotation(rotation, axis, angle)):
                yield self
        finally:
            _OWNER.release()


class PlanningSession(OriginalSession):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kinematics = StaticKinematics(self.stats)

    @contextmanager
    def installed(self, **kwargs):
        with super().installed(**kwargs), self.kinematics.installed():
            yield self

    def receipt(self):
        result = super().receipt()
        result.update(static_kinematics=True, cached_urdf_origins=len(self.kinematics.origins),
                      cached_joint_rotations=len(self.kinematics.rotations),
                      kinematics_arithmetic='Original origin/rotation functions and FK multiplication order; exact input cache, copied outputs')
        return result
