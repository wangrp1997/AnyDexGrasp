"""Independently audit saved five-condition DH3 fixed-force measurements.
No simulator/controller module is imported. No integration, contact query,
render, network forward, model fit or confidence interval is performed.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,time,traceback
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

CORE_SHA='8d33ba1a4ca4d33df31da9717b818505bed73554df3c171696a7a5f0140df239'
COLUMNS=['step','contactFlag','bodyA','bodyB','linkA','linkB','pointA_x_m','pointA_y_m','pointA_z_m',
 'pointB_x_m','pointB_y_m','pointB_z_m','normalOnB_x','normalOnB_y','normalOnB_z','signed_distance_m',
 'normal_force_N','lateral_force1_N','lateral_direction1_x','lateral_direction1_y','lateral_direction1_z',
 'lateral_force2_N','lateral_direction2_x','lateral_direction2_y','lateral_direction2_z']
CASES=['finger1_theta045','finger2_theta045','finger3_theta045','all3_theta045','asymmetric_finger2_045_other_080']


def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rec(p):return dict(path=str(Path(p).resolve()),bytes=Path(p).stat().st_size,sha256=sha(p))
def require(x,label):
 if not x:raise AssertionError(label)
def eq(x,y,label,atol=1e-11):
 require(np.allclose(x,y,rtol=0,atol=atol),label)
def bits(x,y):return x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
def stats(v):
 v=np.asarray(v);return dict(min=float(v.min()),time_mean=float(v.mean()),max=float(v.max()))
def archive(p):
 with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def verify(item):
 p=Path(item['path']);require(p.is_file(),'Missing '+str(p));require(sha(p)==item['sha256'],'SHA changed '+str(p))
 if 'bytes' in item:require(p.stat().st_size==item['bytes'],'Size changed '+str(p))

def Rquat(q):
 x,y,z,w=np.asarray(q,dtype=float)/np.linalg.norm(q)
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
  [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def Rrpy(v):
 x,y,z=v;cx,sx=np.cos(x),np.sin(x);cy,sy=np.cos(y),np.sin(y);cz,sz=np.cos(z),np.sin(z)
 return np.array([[cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx],[sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx],[-sy,cy*sx,cy*cx]])
def Txyzq(x,q):
 T=np.eye(4);T[:3,:3]=Rquat(q);T[:3,3]=x;return T

def urdf_info(path):
 root=ET.parse(path).getroot();joint={};links={l.attrib['name']:l for l in root.findall('link')};remaining=[]
 for j in root.findall('joint'):
  origin=j.find('origin');T=np.eye(4)
  T[:3,:3]=Rrpy(np.fromstring(origin.attrib.get('rpy','0 0 0'),sep=' '));T[:3,3]=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ')
  info=dict(name=j.attrib['name'],parent=j.find('parent').attrib['link'],child=j.find('child').attrib['link'],fixed=j.attrib['type']=='fixed',origin=T,
   axis=np.fromstring(j.find('axis').attrib['xyz'],sep=' ') if j.find('axis') is not None else np.zeros(3))
  joint[info['name']]=info;remaining.append(info)
 order=[];known={'base_Link'}
 while remaining:
  chosen=[j for j in remaining if j['parent'] in known];require(chosen,'Invalid URDF graph')
  for j in chosen:order.append(j);known.add(j['child']);remaining.remove(j)
 return links,joint,order

def world_fk(order,q,Tbase):
 frames={'base_Link':Tbase};chains={'base_Link':[]};axes={}
 for j in order:
  before=frames[j['parent']]@j['origin'];rot=np.eye(4);chain=list(chains[j['parent']])
  if not j['fixed']:
   axis=j['axis'];axis=axis/np.linalg.norm(axis);axes[j['name']]=(before[:3,:3]@axis,before[:3,3])
   angle=q[j['name']];K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
   rot[:3,:3]=np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K);chain.append(j['name'])
  frames[j['child']]=before@rot;chains[j['child']]=chain
 return frames,chains,axes

def flex_virtual_velocity(point,link,chains,axes,B):
 value=np.zeros(3)
 for name in chains[link]:
  if B.get(name,0):
   axis,origin=axes[name];value+=B[name]*np.cross(axis,np.asarray(point)-origin)
 return value

def quasistatic_virtual_work(a,co,c,urdf):
 """Project actual contact forces/gravity onto ideal common-flex virtual motion.
 Uses all final480 saved states and actual contact points/directions; no new FK
 state is written to the simulator. Convective/inertial terms are not presumed
 zero without also reporting final-window velocities and acceleration.
 """
 links,joint,order=urdf_info(urdf);names=c['joint_order'];ix={n:i for i,n in enumerate(names)}
 B={n:(1. if n=='finger1_joint' else 0.) for n in names}
 for g in c['gears']:
  if g['master']=='finger1_joint':B[g['dependent']]=g['factor']
 byindex={-1:'base_Link'}|{j['index']:joint[j['name']]['child'] for j in c['actual_joint_info']}
 mass={d['link_index']:d['mass_kg'] for d in c['all_hand_link_dynamics']};hand=c['body_ids']['hand']
 inert=c['base_link_to_COM'];T_LC=Txyzq(inert[0],inert[1]);invT=np.linalg.inv(T_LC)
 positive=co[co[:,16]>0];group={step:positive[positive[:,0]==step] for step in range(5761,6241)}
 qcontact=[];qgravity=[];motor=[];fkerror=0.
 for k in range(5760,6240):
  base=a['all_link_COM_pose'][k,0];Tbase=Txyzq(base[:3],base[3:])@invT
  frames,chains,axes=world_fk(order,dict(zip(names,a['q'][k])),Tbase)
  for link_index,link in byindex.items():
   inertial=links[link].find('inertial/origin');local=np.fromstring(inertial.attrib.get('xyz','0 0 0'),sep=' ')
   expected=frames[link][:3,:3]@local+frames[link][:3,3]
   fkerror=max(fkerror,float(np.max(abs(expected-a['all_link_COM_pose'][k,link_index+1,:3]))))
  Q=0.
  for contact in group[k+1]:
   force=contact[16]*contact[12:15]+contact[17]*contact[18:21]+contact[21]*contact[22:25]
   if int(contact[2])==hand:
    Q+=np.dot(force,flex_virtual_velocity(contact[6:9],byindex[int(contact[4])],chains,axes,B))
   if int(contact[3])==hand:
    Q-=np.dot(force,flex_virtual_velocity(contact[9:12],byindex[int(contact[5])],chains,axes,B))
  G=0.
  for link_index,link in byindex.items():
   COM=a['all_link_COM_pose'][k,link_index+1,:3]
   G+=np.dot(np.array([0.,0.,-9.81])*mass[link_index],flex_virtual_velocity(COM,link,chains,axes,B))
  qcontact.append(Q);qgravity.append(G);motor.append(sum(a['motor_torque'][k,ix[n]]*b for n,b in B.items()))
 balance=np.asarray(qcontact)+qgravity+motor
 acceleration=np.diff(a['v'][5759:6240],axis=0)*1920
 return dict(common_flex_virtual_contact_torque_Nm=stats(qcontact),common_flex_gravity_torque_Nm=stats(qgravity),
  active_motor_projected_torque_Nm=stats(motor),sum_virtual_torque_Nm=stats(balance),
  max_native_vs_independent_URDF_COM_error_m=fkerror,
  final_window_max_joint_speed_rad_s=float(abs(a['v'][-480:]).max()),
  final_window_max_finite_difference_joint_acceleration_rad_s2=float(abs(acceleration).max()),
  interpretation='Normal-force sums depend on geometry. The proper drive comparison uses contact force dot contact-point virtual displacement per common-flex radian (Nm), plus gravity and active motor torques. Small balance residual supports this recorded quasistatic load path only; omitted inertial/solver terms and finite sampling are not a hardware model fit.')


def audit_case(path,planned,plan,reference_control,baseline,controller_sha):
 r=read(path/'RESULTS.json');c=read(path/'CONTROLLER.json');a=archive(path/'trace.npz');co=np.load(path/'contacts_full.npy',allow_pickle=False)
 require(r['status']=='fixture_completed' and r['failure'] is None,'Incomplete declared fixture case')
 require(r['source_plan_sha256']==sha(path.parent/'PLAN.json')==c['fixture_plan_sha256'],'Plan binding')
 require(c['controller_script_sha256']==CORE_SHA and r['original_controller_sha256']==CORE_SHA,'Accepted core SHA')
 verify(r['trace']);verify(r['contacts']);verify(c['fixture_script'])
 require(sha(c['controller_script'])==CORE_SHA==sha(path/'controller_source_frozen.py'),'Frozen/current core mismatch')
 for pkey,hkey in [('adapter_manifest','adapter_manifest_sha256'),('derived_urdf','derived_urdf_sha256')]:
  require(sha(c[pkey])==c[hkey],pkey+' changed')
 source=read(path/'SOURCE_SNAPSHOT.json');require(source['sha256']==CORE_SHA==sha(source['path']),'Core snapshot')
 require(c['fixture_script']['sha256']==controller_sha,'Fixture wrapper SHA differs')
 require(c['contact_columns']==COLUMNS and co.ndim==2 and co.shape[1]==25,'Full contact columns')
 require(all(len(x)==6240 for x in a.values()),'Trace array length')
 require(np.array_equal(a['step'],np.arange(1,6241)),'Step sequence')
 require(np.array_equal(a['time'],a['step']/1920),'Physical clock')
 require(r['observed_cost']==dict(attempted_python_step_calls=6240,completed_physics_steps=6240,possible_in_flight_step_calls=0),'Actual step counter')
 require(co.dtype.kind=='f' and np.isfinite(co).all() and np.isfinite(a['q']).all(),'Finite states/contact')
 require(((co[:,0]>=1)&(co[:,0]<=6240)&(co[:,0]==co[:,0].astype(int))).all(),'Contact step range')
 require((co[:,16]>=0).all(),'Normal force sign/unit')
 eq(np.linalg.norm(co[:,12:15],axis=1),np.ones(len(co)),'Unit contact normals',1e-6)
 eq(np.einsum('ij,ij->i',co[:,6:9]-co[:,9:12],co[:,12:15]),co[:,15],'Distance/points/normal consistency',1e-7)
 for fcol,d0,d1 in [(17,18,21),(21,22,25)]:
  mask=abs(co[:,fcol])>1e-10
  eq(np.linalg.norm(co[mask,d0:d1],axis=1),np.ones(mask.sum()),'Unit active friction directions',1e-6)
  eq(np.einsum('ij,ij->i',co[mask,d0:d1],co[mask,12:15]),np.zeros(mask.sum()),'Tangential friction direction',1e-6)
 start=(a['step']-1)/1920;u=np.clip((start-.5)/2,0,1);register=np.column_stack((95*(1-u*u*(3-2*u)),np.full(6240,100.)))
 eq(a['register_target'],register,'Original smooth register command',1e-13)
 expected_phase=np.where(start<.5,'hold_open',np.where(start<2.5,'smooth_close','hold_closed'))
 require(np.array_equal(a['phase'],expected_phase),'Original case phase schedule')
 for key in ('step','time','phase','register_target','joint_target'):
  require(bits(a[key],reference_control[key]),'Five-case control-array bit identity: '+key)
 names=c['joint_order'];ix={n:i for i,n in enumerate(names)}
 require(c['active_motors']==['finger1_joint','finger1_base_joint'] and c['passive_motor_force']==0,'Two original active motors')
 require(c['two_register_affine_map']==baseline['two_register_affine_map'],'Original affine mapping')
 eq(a['joint_target'],np.column_stack([v['offset']+v['position_slope']*register[:,0]+v['rotation_slope']*100 for v in c['two_register_affine_map'].values()]),'Affine joint commands',1e-12)
 def gear_signature(rows):return [{k:g[k] for k in ('dependent','master','factor','offset')} for g in rows]
 require(gear_signature(c['gears'])==gear_signature(baseline['gears']),'Original nine gear model')
 require(len(c['gears'])==9,'Nine gear constraints')
 J=np.zeros((9,len(names)));res=[]
 for j,g in enumerate(c['gears']):
  J[j,ix[g['dependent']]]=1;J[j,ix[g['master']]]=-g['factor']
  res.append(a['q'][:,ix[g['dependent']]]-g['factor']*a['q'][:,ix[g['master']]]-g['offset'])
 residual=np.stack(res,axis=1);require(bits(residual,a['gear_residual']),'Independently recomputed gear-residual bits')
 require(np.linalg.matrix_rank(J)==9,'Gear Jacobian rank')
 native_byname={j['name']:j['index'] for j in c['actual_joint_info']};hand=c['body_ids']['hand']
 for info,g in zip(c['actual_constraint_info']['gears'],c['gears']):
  require(info[0]==info[2]==hand and info[1]==native_byname[g['dependent']] and info[3]==native_byname[g['master']],'Actual gear body/joint binding')
  eq([info[4],info[10],info[11],info[13],info[14]],[6,100,-g['factor'],g['offset'],.2],'Actual gear type/cap/ratio/target/ERP')
 require(len(c['actual_constraint_info']['gears'])==9 and c['actual_constraint_info']['wrist'][2]==-1,'No pad attachment constraint')
 require(c['actual_constraint_info']['wrist'][10]==30,'Original finite wrist force')
 require(c['assumptions_SI_not_hardware_identification']==baseline['assumptions_SI_not_hardware_identification'],'Original simulation SI assumptions')
 require(c['actual_physics_engine_parameters']['fixedTimeStep']==1/1920 and c['actual_physics_engine_parameters']['numSubSteps']==1,'Actual 1920Hz/no hidden substeps')
 caps=np.zeros(len(names));caps[ix['finger1_joint']]=.6;caps[ix['finger1_base_joint']]=.3
 require((abs(a['motor_torque'])<=caps+1e-8).all(),'Finite two-motor torque caps')
 passive=[ix[n] for n in names if n not in c['active_motors']]
 require(np.count_nonzero(a['motor_torque'][:,passive])==0,'Exactly zero passive motor drive readbacks')
 require((abs(a['gear_reaction'])<=100+1e-7).all(),'Original finite gear row cap')
 require(all(r['normal_force_N_including_zero_steps'] is not None for r in r['measurement']['pad_force_summary'].values()),'Saved pad summaries')
 # All fixtures must be the exact prepared fixtures plus runtime IDs/dynamics.
 require(len(c['pads'])==len(planned['pads']),'Pad count')
 for pad,padplan in zip(c['pads'],planned['pads']):
  require(all(pad[k]==v for k,v in padplan.items()),'Prepared source support/pad position changed')
  require(pad['actual_dynamics']['mass_kg']==0 and pad['actual_dynamics']['lateral_friction']==.5,'Static pad/friction changed')
 require(read(path/'INITIAL_GEOMETRY.json')['pads']==r['initial_geometry'],'Initial native geometry receipt')
 require(not any(v['negative_pairs'] for v in r['initial_geometry']),'Initial pad penetration')
 require(not r['early_wrong_link_contacts'],'Declared wrong first contact')
 all_links=[-1]+[j['index'] for j in c['actual_joint_info']];forces={};firsts={};sumforce=np.zeros(480)
 for pad in c['pads']:
  b=pad['body_id'];ab=(co[:,2]==hand)&(co[:,3]==b);ba=(co[:,3]==hand)&(co[:,2]==b);mask=ab|ba;selected=co[mask]
  links=np.where(selected[:,2]==hand,selected[:,4],selected[:,5]).astype(int)
  positive=selected[selected[:,16]>1e-8]
  if len(positive):
   first=int(positive[:,0].min());touch=positive[positive[:,0]==first];flinks=np.where(touch[:,2]==hand,touch[:,4],touch[:,5]).astype(int)
   require(set(flinks)=={pad['intended_link_index']},'Wrong link first pad contact')
   firsts[pad['pad_name']]=dict(step=first,time_s=first/1920,hand_links=sorted(set(flinks.tolist())))
   old=r['first_positive_pad_contacts'][pad['pad_name']];require(old['first_positive_contact_step']==first,'First contact step mismatch')
  else:require(pad['pad_name'] not in r['first_positive_pad_contacts'],'Invented first pad contact')
  last=(selected[:,0]>=5761);window=selected[last];window_links=links[last]
  total=np.bincount(window[:,0].astype(int)-5761,weights=window[:,16],minlength=480) if len(window) else np.zeros(480)
  sumforce+=total;old=r['measurement']['pad_force_summary'][pad['pad_name']]
  require(old['window_steps']==480 and old['window_physics_step_range']==[5761,6240],'Final .25s/480 step range')
  eq([total.min(),total.mean(),total.max()],[old['normal_force_N_including_zero_steps'][x] for x in ('min','mean','max')],'Pad mean/min/max incl zero',1e-12)
  require(int((total>1e-8).sum())==old['positive_contact_steps'],'Pad contact occupancy')
  bylink={}
  for link in all_links:
   use=window_links==link
   f=np.bincount(window[use,0].astype(int)-5761,weights=window[use,16],minlength=480) if use.any() else np.zeros(480)
   oldlink=old['per_hand_link_normal_force_N_including_zero_steps'][str(link)]
   eq([f.min(),f.mean(),f.max()],[oldlink[x] for x in ('min','mean','max')],'Per-link force including zero',1e-12)
   bylink[str(link)]=stats(f)
  forces[pad['pad_name']]=dict(normal_force_N=stats(total),positive_contact_steps=int((total>1e-8).sum()),per_link_normal_force_N=bylink)
 # All contact endpoints must be hand/self/pads, not hidden support/objects.
 bodies={hand}|{p['body_id'] for p in c['pads']}
 require(set(co[:,2].astype(int))|set(co[:,3].astype(int))<=bodies,'Undeclared contact body')
 require(((co[:,2]==hand)|(co[:,3]==hand)).all(),'Non-hand contacts in hand log')
 qstat={f'finger{i}':stats(a['q'][-480:,ix[f'finger{i}_joint']]) for i in (1,2,3)}
 virtual=quasistatic_virtual_work(a,co,c,Path(c['derived_urdf']))
 require(virtual['max_native_vs_independent_URDF_COM_error_m']<2e-6,'Independent virtual-work FK does not bind native link states')
 return dict(case_id=planned['case_id'],status='saved_fixture_case_verified',inputs=[rec(path/x) for x in ('RESULTS.json','CONTROLLER.json','trace.npz','contacts_full.npy','INITIAL_GEOMETRY.json','SOURCE_SNAPSHOT.json')],
  physics_steps=6240,physical_seconds=3.25,full_contact_rows=len(co),first_positive_pad_contacts=firsts,
  final_window=dict(start_step=5761,end_step=6240,steps=480,physical_seconds=.25,
    interpretation='One temporally correlated deterministic trajectory window, not480 independent trials; no inferential CI',confidence_interval=None),
  per_pad_forces=forces,sum_pad_normal_force_N=stats(sumforce),common_finger_angles_rad=qstat,
  position_master_final_target_error_rad=stats(a['joint_target'][-480:,ix['finger1_joint']]-a['q'][-480:,ix['finger1_joint']]),
  position_motor_final_torque_Nm=stats(a['motor_torque'][-480:,ix['finger1_joint']]),
  position_motor_final_saturation_fraction=float((abs(a['motor_torque'][-480:,ix['finger1_joint']])>=.6-1e-7).mean()),
  passive_motor_readback_exact_zero=True,gear_J_rank=9,ideal_motion_subspace_dimension=2,
  maximum_gear_position_residual_rad=float(abs(residual).max()),maximum_gear_velocity_residual_rad_s=float(abs(a['v']@J.T).max()),
  peak_gear_reaction_Nm=float(abs(a['gear_reaction']).max()),gear_cap_utilization=float(abs(a['gear_reaction']).max()/100),
  peak_joint_speed_rad_s=float(abs(a['v']).max()),maximum_wrist_COM_error_m=float(a['wrist_position_error'].max()),
  native_hand_masses_kg=sum(d['mass_kg'] for d in c['all_hand_link_dynamics']),
  native_hand_collision_margins_m={str(d['link_index']):d['collision_margin_m'] for d in c['all_hand_link_dynamics']},
  quasistatic_virtual_work=virtual,all_declared_integrity_checks_passed=True),a


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--run-dir',type=Path,default=Path('/home/wangrenpeng/AnyDexGrasp/outputs/dh3_force_fixture_20261104_v1'))
 parser.add_argument('--output',type=Path,default=Path('/tmp/dh3_force_fixture_audit.json'))
 parser.add_argument('--reference-controller',type=Path,default=Path('/home/wangrenpeng/AnyDexGrasp/outputs/dh3_real_mesh_generalization_20261103_v1/run_nonapple_v2/physics/banana_yaw000/CONTROLLER.json'))
 args=parser.parse_args();require(not args.output.exists(),'Retain audit; use a new output path');started=time.monotonic()
 report=dict(scope=__doc__,audit_script=rec(__file__),input_run=str(args.run_dir.resolve()),additional_physical_steps=0,
   additional_network_forwards=0,additional_collision_queries=0,additional_renders=0,hardware_connections=0,cases=[])
 try:
  root=args.run_dir;top=read(root/'RESULTS.json');plan=read(root/'PLAN.json');verify(top['plan']);verify(plan['input_plan']);verify(plan['fixture_script'])
  require(sha(root/'fixture_source_frozen.py')==plan['fixture_script']['sha256'],'Frozen wrapper changed')
  require(read(root/'INPUT_PLAN.json')==read(plan['input_plan']['path'])==plan['input_protocol'],'Input protocol copy changed')
  for item in plan['source_bindings']:verify(item)
  require(top['status']=='fixed_force_fixture_campaign_finished' and top['conditions']==5,'All declared cases finalized')
  require([c['case_id'] for c in top['cases']]==CASES==[c['case_id'] for c in plan['prepared_conditions']],'Five-condition order')
  require(top['completed_physics_steps']==top['attempted_python_step_calls']==plan['protocol_physics_steps_if_all_five_complete']==31200,'Full31200-step accounting')
  require(top['possible_in_flight_step_calls']==0 and top['physical_grasp_attempts']==top['new_network_forwards']==0,'Protocol accounting boundary')
  baseline=read(args.reference_controller);require(baseline['controller_script_sha256']==CORE_SHA,'Reference is not original accepted controller')
  reference=archive(root/CASES[0]/'trace.npz');arrays={}
  for case,old in zip(plan['prepared_conditions'],top['cases']):
   require(read(root/case['case_id']/'RESULTS.json')==old,'Top/per-case result diverged')
   audited,a=audit_case(root/case['case_id'],case,plan,reference,baseline,plan['fixture_script']['sha256'])
   report['cases'].append(audited);arrays[case['case_id']]=a
   print(json.dumps(dict(case_id=case['case_id'],audit=audited['status'],sum_normal_N=audited['sum_pad_normal_force_N']['time_mean'],
      virtual_torque_balance_Nm=audited['quasistatic_virtual_work']['sum_virtual_torque_Nm']['time_mean'])),flush=True)
  a=arrays['finger2_theta045'];b=arrays['asymmetric_finger2_045_other_080']
  shared_response={key:bits(a[key],b[key]) for key in ('q','v','motor_torque','gear_residual','gear_reaction','com','base_velocity','wrist_reaction','wrist_position_error','all_link_COM_pose','all_link_COM_velocity')}
  report.update(status='five_saved_force_fixtures_independently_verified',all_checks_passed=True,
    completed_physics_steps=31200,physical_seconds_total=16.25,complete_case_count=5,
    control_arrays_all_five_bit_identical=True,original_two_motors_nine_gears_preserved=True,
    asymmetric_vs_single_finger2_response_bit_identity=shared_response,
    source_results=rec(root/'RESULTS.json'),source_plan=rec(root/'PLAN.json'),original_reference_controller=rec(args.reference_controller),
    source_wrapper=rec(root/'fixture_source_frozen.py'),
    statistical_boundary='Five distinct deterministic conditions, one execution per condition. Final480 time samples are correlated; force means/min/max describe trajectories, not independent sample estimates. No hardware parameter fit, no success-conditioned tuning, and no confidence interval.',
    unit_boundary='Angles rad; rates rad/s; per-pad scalar normal sums N; torques and force-dot-virtual-lever-arm Nm. Force sums are not equated to motor torque. In this fixed geometry multiple pads share one common-flex drive; total force need not be invariant for other contact geometry.')
 except Exception as exc:
  report.update(status='saved_fixture_audit_failed',all_checks_passed=False,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
 report['audit_wall_seconds']=time.monotonic()-started;args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(status=report['status'],output=str(args.output),verified_cases=len(report['cases']),error=report.get('error'),audit_wall_seconds=report['audit_wall_seconds'])),flush=True)
 return 0 if report['all_checks_passed'] else 1

if __name__=='__main__':raise SystemExit(main())
