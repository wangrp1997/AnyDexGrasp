"""Independent NumPy/URDF audit of the frozen Boya pad-reference table.

Reads asset data only; never imports CanonDex, simulator or controller code.
Normal/plane diagnostics are hypotheses, not physical admission thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def unit(value):
    a = np.asarray(value, dtype=float)
    length = np.linalg.norm(a)
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError('Degenerate vector')
    return a / length


def axis_rotation(axis, angle):
    x, y, z = unit(axis)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross@cross)


class URDFFK:
    """URDF origin Rz*Ry*Rx followed by the joint's local motion."""

    def __init__(self, robot):
        self.links = {n.attrib['name'] for n in robot.findall('link')}
        self.joints = {}
        children = set()
        for element in robot.findall('joint'):
            name, kind = element.attrib['name'], element.attrib['type']
            if kind not in ('fixed', 'revolute', 'continuous', 'prismatic'):
                raise ValueError('Unsupported joint type: ' + kind)
            origin = element.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            if xyz.shape != (3,) or rpy.shape != (3,):
                raise ValueError('Invalid URDF origin')
            transform = np.eye(4); transform[:3, 3] = xyz
            transform[:3, :3] = (axis_rotation([0, 0, 1], rpy[2]) @
                                  axis_rotation([0, 1, 0], rpy[1]) @ axis_rotation([1, 0, 0], rpy[0]))
            axis = element.find('axis')
            mimic = element.find('mimic')
            child = element.find('child').attrib['link']
            if child in children or name in self.joints:
                raise ValueError('URDF must be a tree with unique joints')
            children.add(child)
            self.joints[name] = dict(kind=kind, origin=transform,
                parent=element.find('parent').attrib['link'], child=child,
                axis=np.array([1., 0., 0.]) if kind == 'fixed' else unit(np.fromstring(axis.get('xyz', '1 0 0') if axis is not None else '1 0 0', sep=' ')),
                mimic=None if mimic is None else (mimic.attrib['joint'], float(mimic.get('multiplier', 1)), float(mimic.get('offset', 0))))
        roots = self.links - children
        if len(roots) != 1:
            raise ValueError('Exactly one URDF root required')
        self.root = roots.pop()

    def evaluate(self, coordinates, datum='palm'):
        values = {}

        def coordinate(name, stack=()):
            if name in stack:
                raise ValueError('Cyclic mimic')
            if name not in values:
                j = self.joints[name]
                if j['kind'] == 'fixed':
                    values[name] = 0.
                elif j['mimic']:
                    target, multiplier, offset = j['mimic']
                    values[name] = multiplier*coordinate(target, stack+(name,)) + offset
                else:
                    values[name] = float(coordinates[name])
                if not np.isfinite(values[name]):
                    raise ValueError('Nonfinite joint coordinate')
            return values[name]

        poses = {self.root: np.eye(4)}
        remaining = dict(self.joints)
        while remaining:
            progress = []
            for name, j in remaining.items():
                if j['parent'] not in poses:
                    continue
                motion = np.eye(4); q = coordinate(name)
                if j['kind'] in ('revolute', 'continuous'):
                    motion[:3, :3] = axis_rotation(j['axis'], q)
                elif j['kind'] == 'prismatic':
                    motion[:3, 3] = j['axis']*q
                poses[j['child']] = poses[j['parent']] @ j['origin'] @ motion
                progress.append(name)
            if not progress:
                raise ValueError('Disconnected or cyclic URDF')
            for name in progress:
                del remaining[name]
        inverse = np.linalg.inv(poses[datum])
        return {name: inverse @ transform for name, transform in poses.items()}


def angle(a, b, unsigned=False):
    dot = float(unit(a) @ unit(b))
    return float(np.degrees(np.arccos(np.clip(abs(dot) if unsigned else dot, -1, 1))))


def summary(values):
    a = np.asarray(values, dtype=float)
    return dict(min=float(a.min()), median=float(np.median(a)), max=float(a.max()))


def audit(calibration_path):
    calibration = json.loads(calibration_path.read_text())
    sources = calibration['source_lock']['source_files']
    for path, record in sources.items():
        if sha(path) != record['sha256']:
            raise ValueError('Frozen asset changed: ' + path)
    urdf = next(Path(p) for p in sources if p.endswith('/boya.urdf'))
    pads_path = next(Path(p) for p in sources if p.endswith('/templates/five_finger_wrap.json'))
    package_path = next(Path(p) for p in sources if p.endswith('/package.json'))
    pads = {p['group']: p for p in json.loads(pads_path.read_text())['contacts']}
    package = json.loads(package_path.read_text())
    active = {e['gesture_id']: [c['group'] for c in e['template']['contacts'] if c['group'] in pads]
              for e in package['entries']}
    fk = URDFFK(ET.parse(urdf).getroot())
    names = calibration['source_lock']['independent_motor_names']

    def contacts(state, close=False):
        q = dict(zip(names, state['close_q'] if close else state['independent_q']))
        q.update(state['wrist']); transforms = fk.evaluate(q)
        points, normals = {}, {}
        for group in active[state['gesture_id']]:
            pad = pads[group]; t = transforms[pad['collision_link']]
            points[group] = t[:3, :3] @ pad['point_link_m'] + t[:3, 3]
            normals[group] = unit(t[:3, :3] @ pad['normal_link'])
        return points, normals

    cache = {(s['gesture_id'], s['sample']): contacts(s) for s in calibration['states']}
    rows = []
    for slot in calibration['slots']:
        points, normals = cache[slot['gesture_id'], slot['sample']]
        other = slot['opposing_group']; delta = points[other]-points['th']
        width = float(np.linalg.norm(delta)); x = unit(delta)
        tangent = np.array([0., 1., 0.]); tangent -= (tangent@normals['th'])*normals['th']
        if np.linalg.norm(tangent) < 1e-6:
            tangent = np.eye(3)[np.argmin(abs(normals['th']))]; tangent -= (tangent@normals['th'])*normals['th']
        z = unit(tangent-(tangent@x)*x)
        rebuilt = np.eye(4); rebuilt[:3, :3] = np.column_stack((x, np.cross(z, x), z))
        rebuilt[:3, 3] = (points['th']+points[other])/2
        saved = np.asarray(slot['palm_T_reference'])
        closed, _ = contacts(slot, close=True)
        row = dict(slot_id=slot['slot_id'], gesture_id=slot['gesture_id'], sample=slot['sample'],
                   opposing_group=other, width_m=width,
                   width_error_m=abs(width-slot['reference_width_m']),
                   transform_max_abs_error=float(np.max(abs(rebuilt-saved))),
                   midpoint_error_m=float(np.linalg.norm(rebuilt[:3, 3]-saved[:3, 3])),
                   thumb_normal_dot_closing=float(normals['th']@x),
                   opponent_normal_dot_closing=float(normals[other]@(-x)),
                   approach_dot_thumb_normal=float(z@normals['th']),
                   approach_dot_opponent_normal=float(z@normals[other]),
                   pair_midpoint_drift_to_closed_m=float(np.linalg.norm((closed['th']+closed[other])/2-rebuilt[:3, 3])),
                   palm_T_reference=rebuilt.tolist(), pad_points={g:p.tolist() for g,p in points.items()})
        opposing = [g for g in active[slot['gesture_id']] if g != 'th']
        # Jonas uses a thumb/opponent/opponent triangle. With only two contacts
        # (g6) its normal cannot be determined; no arbitrary third point is added.
        if len(opposing) >= 2:
            cross = np.cross(points[opposing[0]]-points['th'], points[opposing[1]]-points['th'])
            if np.linalg.norm(cross) > 1e-10:
                row['first_two_opponents_plane_normal_angle_deg_mod180'] = angle(z, cross, unsigned=True)
        rows.append(row)
    max_error = max(r['transform_max_abs_error'] for r in rows)
    if max_error > 1e-9:
        raise ValueError(f'Independent FK differs from frozen calibration: {max_error}')
    deployed = {}
    for gesture in (1, 6, 7, 8):
        subset = sorted((r for r in rows if r['gesture_id'] == gesture and r['opposing_group'] == 'ff'), key=lambda r:r['sample'])
        widths = [r['width_m'] for r in subset]
        deployed[str(gesture)] = dict(samples=len(subset), opening_width_m=summary(widths),
            width_increases_along_closure=int(np.count_nonzero(np.diff(widths) > 1e-9)),
            thumb_normal_dot_closing=summary([r['thumb_normal_dot_closing'] for r in subset]),
            opponent_normal_dot_closing=summary([r['opponent_normal_dot_closing'] for r in subset]),
            approach_dot_opponent_normal=summary([r['approach_dot_opponent_normal'] for r in subset]),
            midpoint_drift_to_closed_m=summary([r['pair_midpoint_drift_to_closed_m'] for r in subset]),
            triangle_normal_angle_deg_mod180=summary([r['first_two_opponents_plane_normal_angle_deg_mod180'] for r in subset])
                if 'first_two_opponents_plane_normal_angle_deg_mod180' in subset[0] else None,
            reference_axis_change_open_to_closed_deg=angle(np.asarray(subset[0]['palm_T_reference'])[:3, 2], np.asarray(subset[-1]['palm_T_reference'])[:3, 2]))
    return dict(schema='boya_independent_reference_geometry_audit_v1', status='complete',
        calibration=dict(path=str(calibration_path), sha256=sha(calibration_path)),
        source_files=sources, source_pad_status={g:p.get('status') for g,p in pads.items()},
        independent_FK='NumPy XML URDF; no CanonDex code imported', states=len(cache), slots=len(rows),
        max_transform_abs_error=max_error, max_width_error_m=max(r['width_error_m'] for r in rows),
        max_midpoint_error_m=max(r['midpoint_error_m'] for r in rows), deployed_gestures=deployed, rows=rows,
        limitations=['Pad annotations are pending human review, not measured hardware contact frames',
                     'Triangle-normal comparison uses source pad points; Jonas uses different mesh contact groups',
                     'Angles and signed normal dots are diagnostics without admission thresholds',
                     'g6 has only two contacts; triangle normal is underdetermined',
                     'No mapping, closure endpoint, label, model or deployment default changed'],
        physics_steps=0, training_steps=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, default=ROOT/'outputs/boya_adapter_preparation_20261105_v2/calibration.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve existing audit; use a new output path')
    result = audit(args.calibration.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status', 'states', 'slots', 'max_transform_abs_error', 'deployed_gestures']}, indent=2))


if __name__ == '__main__':
    main()
