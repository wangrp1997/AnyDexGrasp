"""Bounded CPU-only mapper benchmark on saved CGR, no new collection."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import author_collection_batch as B
from author_fast_contact_alignment import FiveFingerWidthMapper,GROUPS,OPPONENTS,GRID_SIZE,REFINEMENTS,WIDTH_TOLERANCE_M
from author_action_semantics import source_flip

SCENE=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/production_tail_20261115_v1/run/windows/01/scenes/0000009_s40_lift_fit_00013_s2026111501')


def stats(values):
    a=np.asarray(values)*1000
    return dict(count=len(a),min_ms=float(a.min()),median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)),max_ms=float(a.max()))


def run(output):
    began=time.monotonic();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    example=SCENE/'g1/pool/selected_author_candidate.json';candidate=B.read(example)
    inputs=dict(candidate=B.binding(example),raw=candidate['input_bindings']['raw_candidates'],calibration=candidate['input_bindings']['calibration'])
    sources={str(Path(p).resolve()):B.binding(p)for p in (__file__,B.HERE/'author_fast_contact_alignment.py',B.HERE/'author_action_semantics.py',B.HERE/'audit_reference_geometry.py')}
    B.verify(inputs);B.verify(sources)
    B.save(output/'PROTOCOL.json',dict(inputs=inputs,sources=sources,gesture=1,groups=list(GROUPS),thumb_required=True,optional_opponents=list(OPPONENTS),
        grid_size=GRID_SIZE,fixed_refinement_passes=REFINEMENTS,numeric_width_tolerance_m=WIDTH_TOLERANCE_M,
        source_width_multiplier=1.,target_definition='Original raw CGR width; 1.7 preopening request is not the contact target',
        single_calls=129,batch_repetitions=5,independent_fk_audit_cases=64,max_branches_per_CGR=4,
        no_object_mesh_or_GT_target_in_mapper=True,source_CGR_frozen=True,
        physics_steps=0,Kit_starts=0,CGR_forwards=0,IK_calls=0,new_labels=0,training_updates=0,
        production_ready=False,whole_closure_collision_or_force_validation=False),exclusive=True)
    with np.load(inputs['raw']['path'],allow_pickle=False)as z:raw=z['candidates'].copy()
    row17=np.c_[raw[:,:2],np.full(len(raw),.03),raw[:,2:],np.full(len(raw),-1.)]
    flipped,flags=source_flip(row17,'dh3_collection');rows=np.c_[flipped[:,:2],flipped[:,3:16]]
    keep=rows[:,1]>0;rows=rows[keep];world=np.asarray(candidate['world_T_camera'])
    start=time.perf_counter();mapper=FiveFingerWidthMapper();precompute_s=time.perf_counter()-start
    first_start=time.perf_counter();first=mapper.align(np.asarray(candidate['actual_row15_camera'])[None],world,candidate['extra_depth_m']);first_call_s=time.perf_counter()-first_start
    indices=np.linspace(0,len(rows)-1,129,dtype=int);single=[]
    for i in indices:
        start=time.perf_counter();mapper.align(rows[i:i+1],world);single.append(time.perf_counter()-start)
    batch=[]
    for _ in range(5):
        start=time.perf_counter();mapped=mapper.align(rows,world);batch.append(time.perf_counter()-start)
    # Independent original XML FK, separate from timed mapping calls.
    valid_pairs=np.argwhere(mapped.valid)
    chosen=valid_pairs[np.linspace(0,len(valid_pairs)-1,min(64,len(valid_pairs)),dtype=int)]
    point_error=0.;width_error=0.;reference_error=0.
    for i,j in chosen:
        coords=dict(zip(mapper.contract.names,mapped.q_contact[i,j]));coords.update(mapper.slot['wrist']);frames=mapper.contract.fk.evaluate(coords)
        actual=[]
        for g in GROUPS:
            pad=mapper.contract.pads[g];t=frames[pad['collision_link']];actual.append(t[:3,:3]@pad['point_link_m']+t[:3,3])
        actual=np.asarray(actual);point_error=max(point_error,float(abs(actual-mapped.pad_points_palm[i,j]).max()))
        width_error=max(width_error,abs(float(np.linalg.norm(actual[j+1]-actual[0]))-rows[i,1]))
        reference_error=max(reference_error,float(abs(mapped.world_T_palm[i,j]@mapped.palm_T_contact_reference[i,j]-mapped.target_world_reference[i]).max()))
    if point_error>2e-14 or width_error>WIDTH_TOLERANCE_M or reference_error>2e-14:raise ValueError('Independent original-FK numerical audit failed')
    example_rows=[]
    for j,opponent in enumerate(OPPONENTS):
        if not first.valid[0,j]:continue
        example_rows.append(dict(opponent=opponent,progress=float(first.progress[0,j]),width_mm=float(first.width_m[0,j]*1000),
            world_T_palm=first.world_T_palm[0,j].tolist(),q_contact=first.q_contact[0,j].tolist(),
            palm_T_contact_reference=first.palm_T_contact_reference[0,j].tolist(),
            thumb_pad_normal_toward_opponent=float(first.pad_normals_palm[0,j,0]@first.palm_T_contact_reference[0,j,:3,0]),
            opponent_pad_normal_toward_thumb=float(-first.pad_normals_palm[0,j,j+1]@first.palm_T_contact_reference[0,j,:3,0]),
            predicted_geometry_only_not_physical_contact=True))
    result=dict(status='bounded_geometric_mapper_benchmark_complete',precompute_including_source_contract_wall_s=precompute_s,
        first_single_call_ms=first_call_s*1000,single_CGR_all_four_branches=stats(single),
        batch_rows=len(rows),original_saved_raw_rows=len(raw),excluded_nonpositive_width_rows=int((~keep).sum()),
        batch_all_four_branches=stats(batch),batch_amortized_ms_per_CGR=float(np.median(batch)*1000/len(rows)),
        CGRs_with_some_width_solution=int(mapped.valid.any(axis=1).sum()),CGRs_without_width_solution=int((~mapped.valid.any(axis=1)).sum()),
        geometric_branch_count=int(mapped.valid.sum()),worst_all_returned_width_residual_m=float(np.max(abs(mapped.width_m-rows[:,1,None])[mapped.valid])),
        independent_original_FK_cases=len(chosen),independent_pad_position_max_error_m=point_error,
        independent_width_max_error_m=width_error,independent_reference_max_matrix_error=reference_error,
        width_ranges_mm={g:[float(w.min()*1000),float(w.max()*1000)]for g,w in zip(OPPONENTS,mapper.width_grid)},
        monotonic_segments={g:len(s)for g,s in zip(OPPONENTS,mapper.segments)},
        source_example_raw_index=candidate['source_row_index'],source_example_target_width_mm=candidate['raw_width_m']*1000,
        example_geometry=example_rows,latency_scope='CPU geometric mapping including input checks/SO3 normalization/4 branches/2 fixed refinements; no object collision, arm IK/path, physics, renderer or disk IO',
        excluded_from_training=True,production_ready=False,whole_closure_collision_checked=False,real_contact_or_lift_validated=False,
        Kit_starts=0,physics_steps=0,CGR_forwards=0,IK_calls=0,new_labels=0,training_updates=0,
        total_function_wall_s=time.monotonic()-began)
    B.verify(inputs);B.verify(sources);B.save(output/'RESULTS.json',result,exclusive=True)
    np.savez_compressed(output/'EXAMPLE_MAPPINGS.npz',progress=first.progress,valid=first.valid,world_T_palm=first.world_T_palm,
        q_contact=first.q_contact,palm_T_contact_reference=first.palm_T_contact_reference,
        pad_points_palm=first.pad_points_palm,pad_normals_palm=first.pad_normals_palm,
        q_open=mapper.q_open,q_close=mapper.q_close,target_world_reference=first.target_world_reference)
    print(json.dumps({k:v for k,v in result.items()if k!='example_geometry'},indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args().output_dir)
