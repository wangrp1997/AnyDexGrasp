"""Read-only independent audit of all newly sampled Boya hand states.

Uses XML/NumPy FK and source STL triangles; imports no CanonDex code, simulator
or neural network. This checks assets and bindings, not physical graspability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from audit_reference_geometry import URDFFK, axis_rotation, sha


def audit(manifest,output):
    manifest,output = Path(manifest).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError('Fresh audit output required')
    report = json.loads(manifest.read_text())
    if report['status']!='complete': raise ValueError('Incomplete surface library')
    for path,digest in report['source_and_input_sha256'].items():
        if sha(path)!=digest: raise ValueError('Changed source/input: '+path)
    cal = json.loads(Path(report['calibration']['path']).read_text())
    urdf = Path(next(p for p in cal['source_lock']['source_files'] if p.endswith('/boya.urdf')))
    robot = ET.parse(urdf).getroot()
    fk = URDFFK(robot)
    expected_links = cal['source_lock']['collision_links']
    local = {}
    for name in expected_links:
        element = robot.find(f"link[@name='{name}']/collision")
        geometry = element.find('geometry/mesh')
        mesh = trimesh.load((urdf.parent/geometry.get('filename')).resolve(),force='mesh')
        vertices = np.asarray(mesh.vertices)*np.fromstring(geometry.get('scale','1 1 1'),sep=' ')
        origin = element.find('origin')
        xyz = np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
        rpy = np.fromstring(origin.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ')
        rotation = axis_rotation([0,0,1],rpy[2])@axis_rotation([0,1,0],rpy[1])@axis_rotation([1,0,0],rpy[0])
        local[name] = (vertices@rotation.T+xyz,len(mesh.faces))
    states = {(s['gesture_id'],s['sample']):s for s in cal['states']}
    maximum = 0.
    total_clouds = total_points = total_bytes = 0
    files = {str(manifest):sha(manifest),str(Path(__file__).resolve()):sha(__file__)}
    parts_checked = 0
    for state in report['states']:
        bound = states[(state['gesture_id'],state['sample'])]
        if state['independent_q'] != bound['independent_q']:
            raise ValueError('Surface state and calibration differ')
        coordinates = dict(zip(cal['source_lock']['independent_motor_names'],state['independent_q']))
        coordinates.update(bound['wrist'])
        frames = fk.evaluate(coordinates)
        if [p['link'] for p in state['parts']] != expected_links:
            raise ValueError('Missing/reordered source hand links')
        source_vertices = []
        offset = 0
        for part in state['parts']:
            name = part['link'];vertices,triangles = local[name]
            transform = frames[name]
            error = float(np.max(np.abs(transform-np.asarray(part['palm_T_link']))))
            maximum = max(maximum,error)
            if error>1e-11 or part['vertices']!=len(vertices) or part['triangles']!=triangles or part['vertex_offset']!=offset:
                raise ValueError('Source triangle assembly/FK differs from independent URDF')
            offset += len(vertices)
            source_vertices.append(vertices@transform[:3,:3].T+transform[:3,3])
            parts_checked += 1
        vertices = np.concatenate(source_vertices)
        low,high = vertices.min(axis=0),vertices.max(axis=0)
        path = Path(state['path'])
        if sha(path)!=state['sha256']: raise ValueError('Cloud bytes changed')
        files[str(path)] = state['sha256'];total_bytes += path.stat().st_size
        with np.load(path,allow_pickle=False) as data:
            if set(data.files)!={f'points_{mm}mm' for mm in range(1,6)}:
                raise ValueError('Missing source cloud resolution')
            for mm in range(1,6):
                points = data[f'points_{mm}mm']
                if (points.shape!=(state['downsampled_counts'][str(mm)],3)
                        or not np.isfinite(points).all() or not len(points)
                        or (points<low-1e-10).any() or (points>high+1e-10).any()):
                    raise ValueError('Invalid source surface cloud geometry/bounds')
                total_clouds += 1;total_points += len(points)
    valid_slots = {s['slot_id']:s for s in cal['slots'] if s['gesture_id'] in report['gestures']}
    if set(report['slot_to_state'])!=set(valid_slots): raise ValueError('Incomplete slot bindings')
    for slot_id,key in report['slot_to_state'].items():
        slot = valid_slots[slot_id]
        if key != f"g{slot['gesture_id']:02d}_s{slot['sample']:03d}":
            raise ValueError('Incorrect slot state binding')
    if len(report['states'])!=132 or report['total_uniform_samples']!=39600000:
        raise ValueError('Unexpected frozen four-type asset scope')
    result = dict(schema='anydex_boya_author_surface_audit_v1',status='complete',
        states=len(report['states']),collision_links_per_state=len(expected_links),
        independent_link_transforms_checked=parts_checked,max_FK_matrix_error=maximum,
        cloud_resolutions_checked=total_clouds,downsampled_points_checked=total_points,
        source_sample_count=report['total_uniform_samples'],slot_bindings_checked=len(valid_slots),
        asset_npz_bytes=total_bytes,source_and_asset_hashes=files,
        source_inputs_unchanged=all(sha(p)==h for p,h in report['source_and_input_sha256'].items()),
        boundary='Source triangles/independent FK, finite cloud bounds and byte bindings; not a force/closure/physical grasp audit',
        calibration_semantics_validated=False,physics_steps=0,training_updates=0)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','states','max_FK_matrix_error','cloud_resolutions_checked','slot_bindings_checked')}))
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a = p.parse_args();audit(a.manifest,a.output)
