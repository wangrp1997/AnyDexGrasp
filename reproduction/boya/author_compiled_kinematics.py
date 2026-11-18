"""Preparse the immutable URDF tree; retain every original FK multiplication."""
from collections import OrderedDict
from contextlib import contextmanager
import weakref
from unittest.mock import patch

import numpy as np
from author_planning_reuse import PlanningSession as OriginalSession
from author_static_kinematics import StaticKinematics


class CompiledKinematics:
    def __init__(self, stats=None, *, max_locals=131072):
        self.stats = {} if stats is None else stats
        self.static = StaticKinematics(self.stats)
        self.models, self.locals = weakref.WeakKeyDictionary(), OrderedDict()
        self.max_locals = max_locals
        self.stats.update(fk_calls=0, fk_metadata_builds=0, local_transform_builds=0,
                          local_transform_reuses=0, local_transform_evictions=0)

    def compile(self, model, K):
        if model in self.models:
            return self.models[model]
        order, done, visiting = [], {model.root}, set()
        def visit(link):
            if link in done: return
            if link in visiting: raise ValueError('URDF cycle at '+link)
            visiting.add(link)
            node = model.by_child[link]
            parent = node.find('parent').attrib['link']
            visit(parent)
            origin = self.static.origin(K.origin_matrix, node.find('origin'))
            kind, name = node.attrib['type'], node.attrib['name']
            axis = node.find('axis')
            axis = tuple(float(x) for x in (axis.get('xyz','1 0 0') if axis is not None else '1 0 0').split())
            mimic = node.find('mimic')
            dependent = None if mimic is None else (mimic.attrib['joint'],
                float(mimic.get('multiplier',1)), float(mimic.get('offset',0)))
            if kind not in ('fixed','revolute','continuous','prismatic'):
                raise ValueError('Unsupported URDF joint type '+kind)
            # Local matrices depend only on these immutable values and q.
            prefix = kind, origin.tobytes(), axis
            order.append((link,parent,name,kind,origin,axis,dependent,prefix))
            visiting.remove(link); done.add(link)
        for link in model.links: visit(link)
        self.models[model] = order
        self.stats['fk_metadata_builds'] += 1
        return order

    def transforms(self, model, positions, root_transform=None):
        import canondex.robots.kinematics as K
        self.stats['fk_calls'] += 1
        poses = {model.root:np.eye(4) if root_transform is None else np.asarray(root_transform)}
        for link,parent,name,kind,origin,axis,mimic,prefix in self.compile(model,K):
            local = origin
            if kind != 'fixed':
                q = positions.get(name,0.0)
                if mimic is not None and name not in positions:
                    q = positions.get(mimic[0],0.0)*mimic[1]+mimic[2]
                value = np.asarray(q)
                key = prefix, value.dtype.str, value.shape, value.tobytes()
                if key in self.locals:
                    local = self.locals[key]
                    self.locals.move_to_end(key)
                    self.stats['local_transform_reuses'] += 1
                else:
                    motion = np.eye(4)
                    if kind in ('revolute','continuous'):
                        motion[:3,:3] = self.static.rotation(K.rotation,axis,q)
                    else:
                        motion[:3,3] = np.asarray(axis)*q
                    local = origin@motion
                    self.locals[key] = local
                    self.stats['local_transform_builds'] += 1
                    if len(self.locals)>self.max_locals:
                        self.locals.popitem(last=False)
                        self.stats['local_transform_evictions'] += 1
            # Identical order/operands to the source recursive FK.
            poses[link] = poses[parent]@local
        return poses

    @contextmanager
    def installed(self):
        import canondex.robots.kinematics as K
        original = K.UrdfKinematics.transforms
        with patch.object(K.UrdfKinematics,'transforms',
                          lambda model,positions,root_transform=None:self.transforms(model,positions,root_transform)):
            yield self


class PlanningSession(OriginalSession):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kinematics = CompiledKinematics(self.stats)

    @contextmanager
    def installed(self, **kwargs):
        with super().installed(**kwargs), self.kinematics.installed():
            yield self

    def receipt(self):
        result=super().receipt()
        result.update(compiled_static_kinematics=True,
            cached_urdf_origins=len(self.kinematics.static.origins),
            cached_local_transforms=len(self.kinematics.locals),
            kinematics_arithmetic='Original origin, joint rotation and parent@local multiplication; static tree metadata parsed once per owned FK model; exact local input cache')
        return result
