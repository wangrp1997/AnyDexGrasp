"""Independently compare saved CPU publication profiles, with zero physics."""
import argparse
import gzip
from pathlib import Path
import pickle
import numpy as np
import author_cpu_fabric_control as C

B=C.B


def array_difference(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:return dict(exact=False,shape_a=list(a.shape),shape_b=list(b.shape),maximum_absolute_difference=None)
    return dict(exact=bool(np.array_equal(a,b)),shape_a=list(a.shape),shape_b=list(b.shape),
        maximum_absolute_difference=float(np.max(abs(a.astype(float)-b.astype(float)),initial=0.)))


def audit(root,output):
    root=Path(root).resolve();p=C.verify(root)
    profiles=[];traces=[];buffers=[];sources={}
    import author_transport as T
    for mode in ('usd','fabric'):
        path=root/'jobs'/mode/'PROFILE.json';d=B.read(path)
        if (d['status']!='partial_profile'or d['complete']or d['label']is not None
                or not d['exclude_training']or d['actual_steps']!=64 or d['measured_ticks']!=48
                or not d['native']['within_20um']):raise ValueError('Invalid bounded CPU profile')
        B.verify({'trace':d['trace'],'contacts':d['contacts']})
        if len(d['native_startup_integrations'])>4 or not all(x['returned']and x['dt_s']==.004 for x in d['native_startup_integrations']):
            raise ValueError('Unconfirmed or unexpected startup cost')
        with np.load(d['trace']['path'],allow_pickle=False)as z:traces.append({k:z[k]for k in z.files})
        with gzip.open(d['contacts']['path'],'rb')as stream:buf=pickle.load(stream)
        if len(buf)!=64 or any(len(a)!=64 for a in traces[-1].values()):raise ValueError('Missing complete-prefix observations')
        for data in buf:T.decode_contacts(d['contact_roots'],d['contact_targets'],data,262144)
        sources[mode]=B.binding(path);profiles.append(d);buffers.append(buf)
    a,b=profiles
    if a['contact_roots']!=b['contact_roots']or a['contact_targets']!=b['contact_targets']:raise ValueError('Contact coverage changed')
    if a['readback']['scene_properties']!=b['readback']['scene_properties']:raise ValueError('Physics scene configuration changed')
    state={key:array_difference(traces[0][key],traces[1][key])for key in traces[0]}
    initial={key:array_difference(a['initial'][key],b['initial'][key])for key in ('joint_pos','joint_velocity','robot_body_pose')}
    for key in a['initial']['objects']:
        for field in ('pose_xyzw','velocity'):
            initial[key+':'+field]=array_difference(a['initial']['objects'][key][field],b['initial']['objects'][key][field])
    max_difference=dict(force=0.,point=0.,normal=0.,separation=0.);contact_exact=True;pair_counts_exact=True;points=[0,0]
    for pair in zip(*buffers):
        da,db=pair
        points[0]+=int(da[4].sum());points[1]+=int(db[4].sum())
        if not np.array_equal(da[4],db[4]):contact_exact=False;pair_counts_exact=False;continue
        for i,j in np.argwhere(da[4]>0):
            n=int(da[4][i,j]);sa=int(da[5][i,j]);sb=int(db[5][i,j])
            for field,x,y in zip(max_difference,da[:4],db[:4]):
                diff=array_difference(x[sa:sa+n],y[sb:sb+n])
                contact_exact=contact_exact and diff['exact']
                max_difference[field]=max(max_difference[field],diff['maximum_absolute_difference'])
    totals=[d['measured_totals']for d in profiles]
    result=dict(status='pass',scope='Full 64-tick stationary-prefix readback only; not closed-hand/lift or collection admission',
        sources=sources,profile_protocol=B.binding(root/'PROTOCOL.json'),
        same_initial_measured_state_exact=all(x['exact']for x in initial.values()),initial_differences=initial,
        full_prefix_state_exact=all(x['exact']for x in state.values()),state_differences=state,
        full_referenced_contact_fields_exact=contact_exact,all_pair_counts_exact=pair_counts_exact,
        contact_points=points,maximum_contact_differences=max_difference,
        measured_step_wall_s=[t['step_s']for t in totals],measured_fetch_wall_s=[t['fetch_s']for t in totals],
        observed_step_wall_ratio_usd_over_fabric=totals[0]['step_s']/totals[1]['step_s'],
        actual_commanded_steps=128,native_startup_steps=sum(len(d['native_startup_integrations'])for d in profiles),
        actual_integrated_steps=sum(d['actual_integrated_steps']for d in profiles),
        current_CPU_PhysX_and_original_scene_settings=True,full_lift_validated=False,
        sampling_or_training_started=False,labels=0,Kit_starts=0,physics_steps=0)
    B.save(output,result,exclusive=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.root,a.output)
    print({k:r[k]for k in ('status','same_initial_measured_state_exact','full_prefix_state_exact','full_referenced_contact_fields_exact','observed_step_wall_ratio_usd_over_fabric')})
