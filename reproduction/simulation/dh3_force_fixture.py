"""Five frozen DH3 force-fixture conditions; no object grasp or hardware claim.

Preparation uses only NumPy and the accepted URDF/author-kinematic interfaces.
--prepare-only never imports PyBullet or connects a simulator. Actual execution
must use a reviewed plan and a new output directory. The accepted two motors,
nine gear constraints and force limits are never changed by this wrapper.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

ACCEPTED_SHA='8d33ba1a4ca4d33df31da9717b818505bed73554df3c171696a7a5f0140df239'
CONTACT_COLUMNS=['step','contactFlag','bodyA','bodyB','linkA','linkB',
 'pointA_x_m','pointA_y_m','pointA_z_m','pointB_x_m','pointB_y_m','pointB_z_m',
 'normalOnB_x','normalOnB_y','normalOnB_z','signed_distance_m','normal_force_N',
 'lateral_force1_N','lateral_direction1_x','lateral_direction1_y','lateral_direction1_z',
 'lateral_force2_N','lateral_direction2_x','lateral_direction2_y','lateral_direction2_z']


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def rec(path):return dict(path=str(Path(path).resolve()),bytes=Path(path).stat().st_size,sha256=sha(path))
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def require(value,message):
 if not value:raise AssertionError(message)
def stats(values):
 values=np.asarray(values)
 return dict(min=float(values.min()),mean=float(values.mean()),max=float(values.max())) if values.size else None

def required_protocol():
 return dict(schema='DH3-fixed-force-fixture-v1',
  accepted_controller_sha256=ACCEPTED_SHA,
  initial_position_register=95.,initial_rotation_register=100.,
  baseLink_position_world_m=[0.,0.,.45],baseLink_orientation_xyzw=[0.,0.,0.,1.],
  integration_hz=1920,numSubSteps=1,solver_iterations=100,non_contact_inner_iterations=20,
  gravity_m_s2=[0.,0.,-9.81],position_motor_max_torque_Nm=.6,rotation_motor_max_torque_Nm=.3,
  position_motor_max_velocity_rad_s=1.,rotation_motor_max_velocity_rad_s=.5,
  gear_max_torque_Nm=100.,gear_erp=.2,wrist_max_force_per_row_N_or_Nm=30.,wrist_erp=.2,
  position_gain=.15,velocity_gain=1.,hand_lateral_friction=.5,contact_restitution=0.,
  sphere_radius_m=.006,pad_mass_kg=0.,pad_lateral_friction=.5,pad_restitution=0.,
  master_theta_central_difference_rad=.0001,initial_hold_s=.5,smooth_close_s=2.,final_hold_s=.75,
  total_s=3.25,steps_per_complete_case=6240,final_force_window_s=.25,
  initial_penetration_tolerance_m=1e-9,wrong_link_contact_force_threshold_N=1e-8,
  cases=[dict(case_id='finger1_theta045',stops=[dict(finger=1,theta_rad=.45)]),
   dict(case_id='finger2_theta045',stops=[dict(finger=2,theta_rad=.45)]),
   dict(case_id='finger3_theta045',stops=[dict(finger=3,theta_rad=.45)]),
   dict(case_id='all3_theta045',stops=[dict(finger=i,theta_rad=.45) for i in (1,2,3)]),
   dict(case_id='asymmetric_finger2_045_other_080',stops=[dict(finger=1,theta_rad=.8),dict(finger=2,theta_rad=.45),dict(finger=3,theta_rad=.8)])],
  case_fresh_client=True,no_table=True,no_grasped_object=True,no_scene_reset_after_integration=True,
  keep_original_two_motors_nine_gears=True,
  on_initial_penetration='zero_step_fixture_failure_keep_evidence_continue_next_declared_case',
  on_wrong_link_first_positive_contact='stop_case_keep_partial_trace_continue_next_declared_case',
  no_pad_reposition_or_case_replacement=True,
  purpose='Measure this declared rigid two-drive simulation port under fixed unilateral load; not hardware identification, grasp scoring, success-driven calibration or mechanism-topology validation')


def install_pure_paths(repo):
 sys.path[:0]=[str(repo/'reproduction/simulation'),str(repo/'reproduction/runtime')]
 geometry=importlib.import_module('dh3_geometry_adapter')
 require('pybullet' not in sys.modules,'Preparation unexpectedly imported PyBullet')
 return geometry


def origin_matrix(element,geometry):
 result=np.eye(4)
 if element is not None:
  result[:3,:3]=geometry.rpy_matrix(np.fromstring(element.attrib.get('rpy','0 0 0'),sep=' '))
  result[:3,3]=np.fromstring(element.attrib.get('xyz','0 0 0'),sep=' ')
 return result


def prepare(config,repo,manifest_path):
 require(config==required_protocol(),'Plan differs from the reviewed finite five-condition protocol')
 geometry=install_pure_paths(repo);manifest=read(manifest_path);urdf=Path(manifest['output_URDF'])
 require(sha(urdf)==manifest['output_URDF_sha256'],'Derived URDF changed')
 controller=repo/'reproduction/simulation/dh3_time_refinement_smoke.py'
 require(sha(controller)==ACCEPTED_SHA,'Accepted controller changed')
 document=ET.parse(urdf).getroot();links={link.attrib['name']:link for link in document.findall('link')}
 def targets(position,rotation):return geometry.adapter_targets(geometry.joint_angles(position,rotation),manifest)
 q0=targets(0.,100.)['finger1_joint'];q95=targets(95.,100.)['finger1_joint'];slope=(q95-q0)/95.
 require(slope<0 and abs(q95)<1e-12,'Unexpected original flex coordinate')
 def frame_at(theta):
  position=(theta-q0)/slope
  require(0<=position<=95,'Fixture reference outside original register domain')
  return geometry.fk(document,targets(position,100.)),position
 sources=[controller,manifest_path,urdf,Path(geometry.__file__),repo/'reproduction/runtime/dh3_author_kinematics.py']
 meshes={};visual_offsets={}
 for name,link in links.items():
  visual=link.find('visual');collision=link.find('collision')
  mesh=visual.find('geometry/mesh');path=Path(mesh.attrib['filename'])
  if not path.is_absolute():path=(urdf.parent/path).resolve()
  require(path.is_file(),'Source mesh missing: '+str(path))
  scale=np.fromstring(mesh.attrib.get('scale','1 1 1'),sep=' ')
  require(np.array_equal(scale,[1,1,1]),'Unexpected mesh scaling')
  off=origin_matrix(visual.find('origin'),geometry);coff=origin_matrix(collision.find('origin'),geometry)
  require(np.array_equal(off,coff),'Visual/collision mesh origins differ')
  cmesh=collision.find('geometry/mesh');require(Path(cmesh.attrib['filename']).name==path.name,'Visual/collision source differs')
  vertices=geometry.stl_vertices(path)
  meshes[name]=vertices@off[:3,:3].T+off[:3,3];visual_offsets[name]=off;sources.append(path)
 initial_frames=geometry.fk(document,targets(95.,100.));base=np.array(config['baseLink_position_world_m'])
 prepared=[];eps=config['master_theta_central_difference_rad']
 for case in config['cases']:
  pads=[]
  for stop in case['stops']:
   finger=stop['finger'];theta=stop['theta_rad'];name=f'finger{finger}_tip_Link';V=meshes[name]
   reference,position=frame_at(theta);plus,_=frame_at(theta+eps);minus,_=frame_at(theta-eps)
   def moved(matrix):return V@matrix[:3,:3].T+matrix[:3,3]+base
   points=moved(reference[name]);center=points.mean(0)
   derivative=(moved(plus[name]).mean(0)-moved(minus[name]).mean(0))/(2*eps)
   speed=float(np.linalg.norm(derivative));require(speed>1e-8,'Tip center has no closing direction')
   direction=derivative/speed;support_index=int(np.argmax(points@direction));support=points[support_index]
   sphere_center=support+direction*config['sphere_radius_m']
   pads.append(dict(pad_name=f'pad_finger{finger}',finger=finger,intended_link=name,
    master_reference_theta_rad=theta,position_register_at_reference=position,tip_vertex_count=len(V),
    center_definition='Arithmetic mean of all original STL triangle vertices, including repeated corners',
    tip_center_world_m=center.tolist(),d_tip_center_d_master_theta_m_per_rad=derivative.tolist(),
    closing_direction_world=direction.tolist(),support_vertex_index=support_index,
    support_vertex_world_m=support.tolist(),sphere_center_world_m=sphere_center.tolist(),
    sphere_radius_m=config['sphere_radius_m'],
    geometry_boundary='Exact support of original source tip vertices; native convex margin may make first contact occur before reference theta. No outcome-based position adjustment.'))
  prepared.append(dict(case_id=case['case_id'],pads=pads))
 return dict(input_protocol=config,adapter_manifest=rec(manifest_path),source_bindings=[rec(p) for p in dict.fromkeys(sources)],
  prepared_conditions=prepared,common_flex_coordinate=dict(name='finger1_joint',offset_rad=q0,position_register_slope_rad=slope),
  pure_preparation=True,prepared_with_no_PyBullet_import=True,preparation_physics_steps=0,
  initial_fk_tip_link_frames={f'finger{i}_tip_Link':initial_frames[f'finger{i}_tip_Link'].tolist() for i in (1,2,3)},
  instrumentation_boundary='Source controller remains unchanged. New wrapper freezes fixtures, logs complete contacts and per-link state, and refuses contaminated first contacts. It does not add actuators or remove constraints.')


def contact_row(step,c):
 return [step,c[0],c[1],c[2],c[3],c[4],*c[5],*c[6],*c[7],c[8],c[9],c[10],*c[11],c[12],*c[13]]


def dynamics_record(p,client,body,link):
 d=p.getDynamicsInfo(body,link,physicsClientId=client)
 return dict(link_index=link,mass_kg=d[0],lateral_friction=d[1],local_inertia_diagonal_kg_m2=d[2],
  local_inertial_position_m=d[3],local_inertial_orientation_xyzw=d[4],restitution=d[5],
  rolling_friction=d[6],spinning_friction=d[7],contact_damping=d[8],contact_stiffness=d[9],
  body_type=d[10],collision_margin_m=d[11] if len(d)>11 else None)


def end_force_summary(contacts,pads,hand,rows,hz,all_link_indices):
 window=round(.25*hz);complete=len(rows)==6240
 # The requested final window means the actual end of the full scripted case.
 # Aborted partial traces get a separately labeled last-observed window.
 first=max(1,len(rows)-window+1);N=max(0,len(rows)-first+1);result={}
 for pad in pads:
  selected=contacts[(contacts[:,0]>=first)&(((contacts[:,2]==hand)&(contacts[:,3]==pad['body_id']))|((contacts[:,3]==hand)&(contacts[:,2]==pad['body_id'])))] if len(contacts) else contacts
  total=np.zeros(N);by_link={str(i):np.zeros(N) for i in all_link_indices}
  for c in selected:
   link=int(c[4] if int(c[2])==hand else c[5]);k=int(c[0])-first
   total[k]+=c[16];by_link.setdefault(str(link),np.zeros(N))[k]+=c[16]
  result[pad['pad_name']]=dict(window_is_full_script_final_quarter_second=complete,
   window_physics_step_range=[first,len(rows)],window_steps=N,
   normal_force_N_including_zero_steps=stats(total),positive_contact_steps=int((total>1e-8).sum()),
   per_hand_link_normal_force_N_including_zero_steps={name:stats(v) for name,v in by_link.items()},
   intended_hand_link_index=pad['intended_link_index'])
 return result


def summarize_case(case,rows,contacts,hand,controller,hz):
 n=len(rows);dt=1/hz;names=controller['joint_order'];indices={name:i for i,name in enumerate(names)}
 if not rows:return dict(physics_steps=0,actual_physical_attempts=0,pad_force_summary=None)
 q=np.stack([r['q'] for r in rows]);v=np.stack([r['v'] for r in rows]);tau=np.stack([r['motor_torque'] for r in rows])
 gears=np.stack([r['gear_residual'] for r in rows]);reaction=np.stack([r['gear_reaction'] for r in rows]);target=np.stack([r['joint_target'] for r in rows])
 motor_caps=np.zeros(len(names));motor_caps[indices['finger1_joint']]=.6;motor_caps[indices['finger1_base_joint']]=.3
 C=np.zeros((len(controller['gears']),len(names)))
 for i,g in enumerate(controller['gears']):C[i,indices[g['dependent']]]=1;C[i,indices[g['master']]]=-g['factor']
 relative_v=v@C.T;motor_power=np.sum(tau*v,axis=1);gear_proxy=np.sum(reaction*relative_v,axis=1)
 passive=[i for i,nm in enumerate(names) if nm not in controller['active_motors']]
 last=slice(max(0,n-round(.25*hz)),n)
 return dict(physics_steps=n,physical_seconds=n*dt,actual_physical_attempts=int(n>0),
  finite=bool(all(np.isfinite(x).all() for x in (q,v,tau,gears,reaction))),
  motor_torque_bounds_pass=bool((abs(tau)<=motor_caps+1e-8).all()),
  passive_reported_motor_torque_max_Nm=float(abs(tau[:,passive]).max()),
  peak_joint_velocity_rad_s=float(abs(v).max()),max_gear_residual_rad=float(abs(gears).max()),
  peak_gear_reaction_Nm=float(abs(reaction).max()),gear_reaction_cap_utilization=float(abs(reaction).max()/100),
  max_wrist_COM_error_m=max(r['wrist_position_error'] for r in rows),
  observed_gear_Jacobian_rank=int(np.linalg.matrix_rank(C)),unconstrained_joint_coordinates=len(names)-int(np.linalg.matrix_rank(C)),
  per_joint_final_quarter_second={name:dict(q_rad=stats(q[last,i]),v_rad_s=stats(v[last,i]),
    target_error_rad=stats(target[last,i]-q[last,i]),motor_torque_Nm=stats(tau[last,i]),
    motor_cap_fraction=float((abs(tau[last,i])>=motor_caps[i]-1e-7).mean()) if motor_caps[i]>0 else None) for i,name in enumerate(names)},
  closing_finger_master_angles_final_rad={f'finger{i}':float(q[-1,indices[f'finger{i}_joint']]) for i in (1,2,3)},
  motor_work_endpoint_sample_proxy_J=float(motor_power.sum()*dt),
  gear_row_work_endpoint_sample_proxy_J=float(gear_proxy.sum()*dt),
  gear_velocity_residual_rad_s_max=float(abs(relative_v).max()),
  pad_force_summary=end_force_summary(contacts,case['pads'],hand,rows,hz,[-1]+[j['index'] for j in controller['actual_joint_info']]),
  energy_boundary='Endpoint-sampled torque*velocity and lambda*(Jv) are diagnostic proxies, not exact impulse-work, total mechanical energy, or hardware efficiency. Per-link COM velocities/inertia and raw contact vectors are saved for subsequent independent balance.')


def run_case(output,case,prepared,accepted,manifest_path):
 p=accepted.p;config=prepared['input_protocol'];hz=config['integration_hz'];a=accepted.PhysicsAssumptions(dt=1/hz,non_contact_inner_iterations=20)
 output.mkdir();rows=[];contacts=[];client=None;attempted=0;completed=0;start=time.monotonic();controller=None;pad_records=[];failure=None;early_wrong=[];first_contacts={};seen_intended=set();initial=[];originals={}
 try:
  client=p.connect(p.DIRECT);accepted.configure(client,a)
  hand=accepted.CoupledDH3(client,manifest_path,config['baseLink_position_world_m'],config['baseLink_orientation_xyzw'],95.,100.,assumptions=a)
  require(len(hand.gears)==9 and len(hand.movable_names)==11,'Original transmission size changed')
  require(hand.position_master=='finger1_joint' and hand.rotation_master=='finger1_base_joint','Active motors changed')
  for planned in case['pads']:
   shape=p.createCollisionShape(p.GEOM_SPHERE,radius=.006,physicsClientId=client)
   body=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,baseVisualShapeIndex=-1,
       basePosition=planned['sphere_center_world_m'],physicsClientId=client)
   p.changeDynamics(body,-1,lateralFriction=.5,restitution=0,physicsClientId=client)
   link=next(i for i in range(p.getNumJoints(hand.body,physicsClientId=client)) if p.getJointInfo(hand.body,i,physicsClientId=client)[12].decode()==planned['intended_link'])
   pad_records.append(dict(planned,body_id=body,intended_link_index=link,actual_dynamics=dynamics_record(p,client,body,-1)))
  p.performCollisionDetection(physicsClientId=client)
  controller=hand.receipt();controller.update(fixture_case=case['case_id'],pads=pad_records,
   fixture_plan_sha256=sha(output.parent/'PLAN.json'),fixture_script=rec(__file__),
   no_table=True,no_grasped_object=True,script_duration_s=3.25,
   contact_columns=CONTACT_COLUMNS,body_ids=dict(hand=hand.body,pads=[v['body_id'] for v in pad_records]),
   all_hand_link_dynamics=[dynamics_record(p,client,hand.body,i) for i in range(-1,p.getNumJoints(hand.body,physicsClientId=client))],
   actual_joint_info=[dict(index=i,name=p.getJointInfo(hand.body,i,physicsClientId=client)[1].decode(),
     type=p.getJointInfo(hand.body,i,physicsClientId=client)[2],axis=p.getJointInfo(hand.body,i,physicsClientId=client)[13],
     lower=p.getJointInfo(hand.body,i,physicsClientId=client)[8],upper=p.getJointInfo(hand.body,i,physicsClientId=client)[9],
     effort=p.getJointInfo(hand.body,i,physicsClientId=client)[10],velocity=p.getJointInfo(hand.body,i,physicsClientId=client)[11])
     for i in range(p.getNumJoints(hand.body,physicsClientId=client))])
  accepted.write_controller_receipt(output,controller)
  # Check actual native initialization and complete hand-vs-pad geometry before
  # any integration; no positioning repair is permitted.
  for pad in pad_records:
   pairs=p.getClosestPoints(hand.body,pad['body_id'],.02,physicsClientId=client)
   closest=min((c[8] for c in pairs),default=None);initial.append(dict(pad_name=pad['pad_name'],minimum_native_distance_m=closest,
     negative_pairs=[dict(hand_link=c[3],distance_m=c[8]) for c in pairs if c[8]< -config['initial_penetration_tolerance_m']]))
  save(output/'INITIAL_GEOMETRY.json',dict(physics_steps=0,pads=initial))
  require(not any(x['negative_pairs'] for x in initial),'Initial hand-pad penetration: fixture rejected without position repair')
  guarded=['resetJointState','resetBasePositionAndOrientation','resetBaseVelocity','resetSimulation','createConstraint']
  def forbidden(*args,**kwargs):raise RuntimeError('No state reset or added constraint after initialization')
  for name in guarded:originals[name]=getattr(p,name);setattr(p,name,forbidden)
  ids=list(range(p.getNumJoints(hand.body,physicsClientId=client)))
  for step in range(config['steps_per_complete_case']):
   t=step/hz;phase='hold_open' if t<.5 else ('smooth_close' if t<2.5 else 'hold_closed')
   position=95.*(1.-accepted.smooth((t-.5)/2.))
   target=hand.command(position,100.,config['baseLink_position_world_m'],config['baseLink_orientation_xyzw'])
   attempted+=1;p.stepSimulation(physicsClientId=client);completed+=1
   row=hand.observe();states=p.getLinkStates(hand.body,ids,computeLinkVelocity=1,computeForwardKinematics=1,physicsClientId=client)
   base=np.r_[p.getBasePositionAndOrientation(hand.body,physicsClientId=client)[0],p.getBasePositionAndOrientation(hand.body,physicsClientId=client)[1]]
   base_v=np.r_[p.getBaseVelocity(hand.body,physicsClientId=client)[0],p.getBaseVelocity(hand.body,physicsClientId=client)[1]]
   row.update(step=step+1,time=(step+1)/hz,phase=phase,register_target=np.array([position,100.]),joint_target=np.array([target[n] for n in hand.movable_names]),
     all_link_COM_pose=np.vstack([base]+[np.r_[s[0],s[1]] for s in states]),all_link_COM_velocity=np.vstack([base_v]+[np.r_[s[6],s[7]] for s in states]))
   rows.append(row);now=[]
   for c in p.getContactPoints(physicsClientId=client):
    if hand.body in (c[1],c[2]):contacts.append(contact_row(step+1,c));now.append(c)
   wrong=[]
   for pad in pad_records:
    active=[]
    for c in now:
     if c[9]<=config['wrong_link_contact_force_threshold_N']:continue
     if c[1]==hand.body and c[2]==pad['body_id']:active.append(c[3])
     elif c[2]==hand.body and c[1]==pad['body_id']:active.append(c[4])
    if active and pad['pad_name'] not in seen_intended:
     first=dict(pad_name=pad['pad_name'],first_positive_contact_step=step+1,first_positive_contact_time_s=(step+1)/hz,
       intended_link_index=pad['intended_link_index'],observed_links=sorted(set(active)))
     first_contacts.setdefault(pad['pad_name'],first)
     if any(link!=pad['intended_link_index'] for link in active):wrong.append(first)
     elif pad['intended_link_index'] in active:seen_intended.add(pad['pad_name'])
   if wrong:
    early_wrong+=wrong;raise RuntimeError('Non-designated hand link touched a force pad first; fixture stopped, not repositioned')
 except BaseException as exc:
  failure=dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
 finally:
  for name,original in originals.items():setattr(p,name,original)
  array=np.asarray(contacts,dtype=np.float64).reshape(-1,len(CONTACT_COLUMNS))
  np.save(output/'contacts_full.npy',array)
  if rows:
   np.savez_compressed(output/'trace.npz',**{key:np.asarray([r[key] for r in rows]) for key in rows[0]})
  else:np.savez_compressed(output/'trace.npz',step=np.empty(0,dtype=np.int64),time=np.empty(0))
  metric=summarize_case(dict(case,pads=pad_records),rows,array,hand.body if controller else -1,controller,hz) if controller else dict(physics_steps=completed,actual_physical_attempts=int(completed>0))
  result=dict(case_id=case['case_id'],status='fixture_completed' if failure is None else 'fixture_failed',
   observed_cost=dict(attempted_python_step_calls=attempted,completed_physics_steps=completed,possible_in_flight_step_calls=attempted-completed),
   measurement=metric,first_positive_pad_contacts=first_contacts,early_wrong_link_contacts=early_wrong,initial_geometry=initial,failure=failure,
   wall_seconds=time.monotonic()-start,original_controller_sha256=ACCEPTED_SHA,
   source_plan_sha256=sha(output.parent/'PLAN.json'),trace=rec(output/'trace.npz'),contacts=rec(output/'contacts_full.npy'),
   physical_grasp_attempts=0,new_network_forwards=0,rendered_frames=0,
   no_state_resets_after_first_step=True,pad_positions_never_adapted=True,
   boundary='Declared fixed rigid obstacles measuring this unchanged simulation-port response; no hardware identification or grasp-success claim')
  save(output/'RESULTS.json',result)
  if client is not None:p.disconnect(physicsClientId=client)
 return result


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--plan-file',type=Path,required=True)
 parser.add_argument('--output-dir',type=Path,required=True)
 parser.add_argument('--repo-root',type=Path,default=Path('/home/wangrenpeng/AnyDexGrasp'))
 parser.add_argument('--adapter-manifest',type=Path,default=Path('/home/wangrenpeng/AnyDexGrasp/outputs/dh3_geometry_adapter_20261103_v2/ADAPTER.json'))
 parser.add_argument('--prepare-only',action='store_true')
 args=parser.parse_args();repo=args.repo_root.resolve();out=args.output_dir.resolve()
 require(not out.exists(),'Keep all trials; output directory must not exist')
 config=read(args.plan_file);prepared=prepare(config,repo,args.adapter_manifest.resolve())
 prepared.update(input_plan=rec(args.plan_file),fixture_script=rec(__file__),
  protocol_physics_steps_if_all_five_complete=5*6240,
  identifiability=dict(can_measure=['Rigid two-drive load sharing and stall/compliance error of the declared solver/controller',
   'Per-pad normal and friction reactions with exact first-contact link and full state/source provenance',
   'Saturation, passive-zero-drive, gear residual/velocity, joint limits and dynamic collision margin metadata',
   'Endpoint-sampled actuator/transmission power proxies and per-link states for subsequent energy balance'],
   cannot_identify=['Physical DH3 force70 or speed100 to SI conversion','True passive differential topology, backlash, stiffness, damping or contact skin',
    'Hardware torque-speed/current curves or measured CAD mass/inertia accuracy','General grasp success or an appropriate adaptive mechanism just from fixed-obstacle data']),
  no_core_source_modification=True)
 out.mkdir(parents=True);shutil.copyfile(args.plan_file,out/'INPUT_PLAN.json');shutil.copyfile(__file__,out/'fixture_source_frozen.py')
 save(out/'PLAN.json',prepared)
 if args.prepare_only:
  require('pybullet' not in sys.modules,'Prepare-only imported simulator')
  save(out/'RESULTS.json',dict(status='pure_fixture_preparation_complete',plan=rec(out/'PLAN.json'),
    physics_steps=0,physics_clients_created=0,new_network_forwards=0,case_count=5,
    boundary='FK and all-source-tip-vertex fixture construction only; initial native penetration has not been checked'))
  print(json.dumps(dict(status='pure_fixture_preparation_complete',output=str(out),physics_steps=0,physics_clients_created=0)),flush=True);return
 # All plans/positions/source hashes exist before importing the simulation module.
 sys.path.insert(0,str(repo/'outputs/native_deps/python_overlay'))
 accepted=importlib.import_module('dh3_time_refinement_smoke')
 require(sha(accepted.__file__)==ACCEPTED_SHA,'Accepted controller import SHA changed')
 results=[];started=time.monotonic()
 for case in prepared['prepared_conditions']:
  for item in prepared['source_bindings']:require(sha(item['path'])==item['sha256'],'Frozen source changed: '+item['path'])
  result=run_case(out/case['case_id'],case,prepared,accepted,args.adapter_manifest.resolve());results.append(result)
  print(json.dumps(dict(case_id=case['case_id'],status=result['status'],physics_steps=result['observed_cost']['completed_physics_steps'])),flush=True)
 final=dict(status='fixed_force_fixture_campaign_finished',conditions=len(results),cases=results,
  completed_physics_steps=sum(r['observed_cost']['completed_physics_steps'] for r in results),
  attempted_python_step_calls=sum(r['observed_cost']['attempted_python_step_calls'] for r in results),
  possible_in_flight_step_calls=sum(r['observed_cost']['possible_in_flight_step_calls'] for r in results),
  original_core_source_changed=False,hardware_calibration_claim=False,physical_grasp_attempts=0,new_network_forwards=0,
  wall_seconds=time.monotonic()-started,plan=rec(out/'PLAN.json'),
  boundary='All five declared conditions retained, including geometry failures and partial traces; no replacements or outcome-driven tuning')
 save(out/'RESULTS.json',final);print(json.dumps(dict(status=final['status'],completed_physics_steps=final['completed_physics_steps'],output=str(out))),flush=True)

if __name__=='__main__':main()
