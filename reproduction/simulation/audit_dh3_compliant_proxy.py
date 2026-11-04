"""Independent source/saved-state audit. No simulator import or physical execution.

--source-review-only executes only extracted command() against a fake API.
It is a software equation check, not a physical experiment.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

OLD_FIXTURE_AUDIT = Path(__file__).resolve().parent / 'evidence/compliant_proxy_20261104/dh3_force_fixture_audit.json'
GUARDS = {'resetJointState','resetBasePositionAndOrientation','resetBaseVelocity','resetSimulation','createConstraint'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def require(ok, label):
    if not bool(ok):raise AssertionError(label)
def close(a,b,label,atol=1e-11):
    require(np.allclose(a,b,atol=atol,rtol=0),label)
def stats(a):
    a=np.asarray(a)
    return dict(min=float(a.min()),max=float(a.max()),mean=float(a.mean())) if a.size else None

def source_review(source):
    """Extract only author-proxy command; fake API has no simulator/client."""
    source=Path(source); tree=ast.parse(source.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='CompliantDH3')
    command=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='command')
    cc=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Compliance')
    constants={x.target.id:ast.literal_eval(x.value) for x in cc.body if isinstance(x,ast.AnnAssign)}
    calls=[]
    class FakeAPI:
        TORQUE_CONTROL=1;POSITION_CONTROL=2
        def getJointStates(self,*args,**kwargs):return [(q,v,(),0) for q,v in zip(self.q,self.v)]
        def setJointMotorControl2(self,*args,**kwargs):calls.append((args,kwargs))
        def changeConstraint(self,*args,**kwargs):pass
    api=FakeAPI(); env={'np':np,'p':api,'joint_angles':lambda p,r:{'finger1_joint':p,'finger1_base_joint':r},'adapter_targets':lambda x,m:x}
    module=ast.Module(body=[command],type_ignores=[]);ast.fix_missing_locations(module)
    exec(compile(module,str(source)+':isolated_command','exec'),env)
    f=env['command']; rng=np.random.default_rng(20261104); max_power=0.; tests=0
    for i in range(200):
        q=rng.uniform(.05,1.2,3);v=rng.uniform(-.3,.3,3);x=float(rng.uniform(.05,1.25));xd=float(rng.uniform(-.5,.5))
        api.q=q;api.v=v;calls.clear()
        a=SimpleNamespace(dt=1/1920,position_motor_max_velocity_rad_s=1.,position_motor_max_torque_Nm=.6,
            rotation_motor_max_torque_Nm=.3,rotation_motor_max_velocity_rad_s=.5,position_gain=.15,velocity_gain=1.,wrist_max_force_per_row_N_or_Nm=30.,wrist_erp=.2)
        s=SimpleNamespace(proxy_ready=True,compliance=SimpleNamespace(**constants),a=a,client=-1,body=-1,manifest={},
            branch_ids=[1,2,3],shaft_q=x,shaft_v=xd,position_master='finger1_joint',rotation_master='finger1_base_joint',
            names={'finger1_base_joint':0},wrist=-1,proxy_steps=0,com_target=lambda p,q:(p,q))
        target=float(rng.uniform(0,1.333));f(s,target,.2,[0,0,.45],[0,0,0,1])
        k=constants['branch_stiffness_Nm_rad'];d=constants['branch_damping_Nm_s_rad'];J=constants['shaft_inertia_kg_m2'];b=constants['shaft_viscous_Nm_s_rad'];h=a.dt
        e=x-q;r=xd-v;tau=k*e+d*r
        m=float(np.clip(constants['shaft_velocity_feedback_Nm_s_rad']*(np.clip(constants['shaft_position_to_velocity_gain_s_inv']*(target-x),-1,1)-xd),-.6,.6))
        acc=(m-tau.sum()-b*xd)/J;vp=xd+h*acc;xp=x+h*vp;stopped=xp<constants['shaft_lower_rad'] or xp>constants['shaft_upper_rad']
        close(s.last['branch_torque_Nm'],tau,'isolated branch constitutive law')
        close(s.last['shaft_motor_torque_Nm'],m,'isolated bounded common drive')
        close(s.shaft_q,np.clip(xp,0,1.333),'isolated shaft position integration')
        close(s.shaft_v,0 if stopped else vp,'isolated shaft velocity integration')
        close(s.last['shaft_stop_dissipation_J'],.5*J*vp**2 if stopped else 0,'isolated inelastic stop')
        torque_calls=[c for c in calls if c[0][2]==api.TORQUE_CONTROL]
        require(len(torque_calls)==3 and len(calls)==4,'three transmitted torque channels and one spread servo')
        close([c[1]['force'] for c in torque_calls],tau,'actual fake API torque arguments')
        # Continuous-time identity: branch torques and their shaft reaction are internal.
        Pmotor=m*xd;Pbranch=float(tau@v);dU=float(k*e@r);dK=J*xd*acc;D=float(d*(r@r)+b*xd**2)
        power=dK+dU+Pbranch+D-Pmotor;max_power=max(max_power,abs(power));close(power,0,'one shared drive continuous virtual-work identity',2e-14)
        tests+=1
    return dict(source=str(source),source_sha256=sha(source),constants=constants,isolated_method_cases=tests,
        fake_API_only=True,maximum_continuous_power_identity_residual_W=max_power,
        findings=[
          'Equation returns all three branch reactions to the same finite-inertia shaft; not three independently powered servos.',
          'Branch torque sum may transiently exceed .6Nm by releasing stored shaft/spring energy; do not misclassify as extra motor power.',
          'Shaft position clipping is an internal inelastic stop, not a hand/object reset; its numerical energy residual must be reported.',
          'This permits cross-finger elastic deflection and is explicitly an engineering proxy, not identified commercial DH3 linkage.',
          'Inherited fixture summary must map virtual shaft to finger1_joint when identifying active torque slots.',
          'Actuator power is shaft_motor_torque_Nm*shaft_v_before_step_rad plus spread motor power; physical finger1 velocity is not shaft velocity.'
        ],physical_steps=0,simulator_imported=False)

def trace_data(run):
    with np.load(run/'trace.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}

def contacts(run,c,n):
    path=run/('contacts_full.npy' if (run/'contacts_full.npy').exists() else 'contacts.npy');z=np.load(path,allow_pickle=False)
    require(z.ndim==2 and z.shape[1] in (9,25),'known contact record width');require(np.isfinite(z).all(),'finite contacts')
    if z.shape[1]==25: cols={'step':0,'bodyA':2,'bodyB':3,'linkA':4,'linkB':5,'normal_force_N':16,'signed_distance_m':15}
    else:cols={'step':0,'bodyA':1,'bodyB':2,'linkA':3,'linkB':4,'normal_force_N':6,'signed_distance_m':5}
    require(((z[:,0]>=1)&(z[:,0]<=n)).all(),'contact step range'); hand=int(c['body_ids']['hand']); ishand=(z[:,cols['bodyA']]==hand)|(z[:,cols['bodyB']]==hand)
    require(ishand.all(),'all records include hand')
    if z.shape[1]==25:
        close(np.linalg.norm(z[:,12:15],axis=1),1,'contact normal unit vectors',2e-12)
        close(np.sum((z[:,6:9]-z[:,9:12])*z[:,12:15],axis=1),z[:,15],'contact points/distance coordinate convention',2e-12)
        for fi,di in [(17,18),(21,22)]:
            active=np.abs(z[:,fi])>1e-12
            if active.any():
                close(np.linalg.norm(z[active,di:di+3],axis=1),1,'active friction direction unit',2e-12)
                close(np.sum(z[active,di:di+3]*z[active,12:15],axis=1),0,'friction tangent',2e-12)
    if 'object' in c['body_ids']: others=[('object',int(c['body_ids']['object']))]
    else:others=[(p['pad_name'],int(p['body_id'])) for p in c['pads']]
    out={};N=min(n,480)
    for label,b in others:
        keep=((z[:,cols['bodyA']]==hand)&(z[:,cols['bodyB']]==b))|((z[:,cols['bodyB']]==hand)&(z[:,cols['bodyA']]==b))
        q=z[keep];q=q[q[:,cols['normal_force_N']]>1e-8]; ids=np.where(q[:,cols['bodyA']]==hand,q[:,cols['linkA']],q[:,cols['linkB']]).astype(int)
        per={}
        for link in np.unique(ids):
            rows=q[ids==link]; total=np.bincount(rows[:,0].astype(int),weights=rows[:,cols['normal_force_N']],minlength=n+1)
            per[str(int(link))]=dict(positive_contact_rows=len(rows),positive_contact_steps=len(np.unique(rows[:,0])),first_step=int(rows[:,0].min()),last_step=int(rows[:,0].max()),last_window_normal_force_N=stats(total[n-N+1:n+1]))
        out[label]=dict(total_positive_rows=len(q),hand_link_indices=sorted(map(int,np.unique(ids))),per_link=per)
    return dict(path=str(path),sha256=sha(path),columns=z.shape[1],last_window_steps=N,last_window_s=N/1920,body_pairs=out)

def source_guards(run):
    p=run/('runner_source_frozen.py' if (run/'runner_source_frozen.py').exists() else 'fixture_source_frozen.py')
    if not p.exists():p=run.parent/'fixture_source_frozen.py'
    tree=ast.parse(p.read_text()); funcs=[x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name in ('run_scene','run_case')]
    require(len(funcs)==1,'one frozen executable loop');f=funcs[0]
    guardsets=[]
    for x in ast.walk(f):
        if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('guarded_names','guarded') for t in x.targets):guardsets.append(set(ast.literal_eval(x.value)))
    require(any(GUARDS<=g for g in guardsets),'all reset/attachment APIs guarded')
    blockers=[x for x in ast.walk(f) if isinstance(x,ast.FunctionDef) and x.name in ('forbid_reset','forbidden')]
    require(any(any(isinstance(y,ast.Raise) for y in ast.walk(x)) for x in blockers),'guard raises')
    dynamic_calls=[x.func.attr for x in ast.walk(f) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and isinstance(x.func.value,ast.Name) and x.func.value.id=='p']
    require(not GUARDS.intersection(dynamic_calls),'no direct reset/attachment call in execution wrapper')
    return dict(path=str(p),sha256=sha(p),API_guards_verified=True,source_inventory_boundary='Checks frozen code and saved initial constraints; not an external syscall monitor.')

def mechanics(t,c):
    n=len(t['step']);h=1/1920;names=c['joint_order'];ix={k:i for i,k in enumerate(names)};branch=[ix[f'finger{i}_joint'] for i in (1,2,3)]
    p=c['compliance'];k=p['branch_stiffness_Nm_rad'];d=p['branch_damping_Nm_s_rad'];J=p['shaft_inertia_kg_m2'];b=p['shaft_viscous_Nm_s_rad']
    require(k>=0 and d>=0 and J>0 and b>=0,'passive parameter signs')
    x0=t['shaft_q_before_step_rad'];v0=t['shaft_v_before_step_rad'];x=t['shaft_q_rad'];v=t['shaft_v_rad_s'];e0=t['branch_extension_before_step_rad'];q0=x0[:,None]-e0
    q=t['q'][:,branch];u=t['v'][:,branch];u0=np.vstack([np.zeros((1,3)),u[:-1]])
    close(q0[1:],q[:-1],'saved pre-step branch q continuity');close(x0[1:],x[:-1],'shaft q continuity');close(v0[1:],v[:-1],'shaft v continuity')
    rel=v0[:,None]-u0;tau=k*e0+d*rel;close(t['branch_torque_Nm'],tau,'all branch constitutive torques')
    close(t['branch_damping_power_W'],d*np.sum(rel**2,axis=1),'nonnegative branch dissipation')
    active=t['shaft_motor_torque_Nm'];target=t['joint_target'][:,ix['finger1_joint']]
    vt=np.clip(p['shaft_position_to_velocity_gain_s_inv']*(target-x0),-1,1);m=np.clip(p['shaft_velocity_feedback_Nm_s_rad']*(vt-v0),-.6,.6);close(active,m,'common motor law')
    acc=(m-tau.sum(1)-b*v0)/J;vp=v0+h*acc;xp=x0+h*vp;stopped=(xp<p['shaft_lower_rad'])|(xp>p['shaft_upper_rad']);expectedv=np.where(stopped,0,vp);expectedx=np.clip(xp,p['shaft_lower_rad'],p['shaft_upper_rad'])
    close(x,expectedx,'shaft reaction/inertia position integration');close(v,expectedv,'shaft reaction/inertia velocity integration')
    stop=np.where(stopped,.5*J*vp**2,0);close(t['shaft_stop_dissipation_J'],stop,'nonnegative inelastic stop loss')
    E=.5*k*np.sum((x[:,None]-q)**2,axis=1);K=.5*J*v**2;close(t['branch_elastic_energy_J'],E,'stored nonnegative elastic energy');close(t['shaft_kinetic_energy_J'],K,'shaft kinetic energy')
    close(t['branch_extension_rad'],x[:,None]-q,'post-step spring extension');require((E>=0).all() and (K>=0).all(),'nonnegative stored energy')
    close(t['shaft_step_count'],np.arange(1,n+1),'one shaft update per physical step',0)
    caps=np.zeros(len(names));caps[ix['finger1_joint']]=.6;caps[ix['finger1_base_joint']]=.3;require((abs(t['motor_torque'])<=caps+1e-8).all(),'two actuator caps/no other active slots')
    close(t['motor_torque'][:,ix['finger1_joint']],m,'virtual shaft explicit actuator mapping')
    close(t['motor_torque'][:,ix['finger1_base_joint']],t['joint_motor_torque_readback'][:,ix['finger1_base_joint']],'spread native readback')
    native_passive=[i for i,name in enumerate(names) if name!='finger1_base_joint']
    close(t['joint_motor_torque_readback'][:,native_passive],0,'no hidden native position/velocity servo at flex or passive joints',0)
    # Energy accounting uses finite increments, not an assertion that Euler splitting is exactly passive.
    dq=q-q0;dx=x-x0;dxp=xp-x0;de=dx[:,None]-dq;E0=.5*k*np.sum(e0**2,axis=1);K0=.5*J*v0**2
    Wm=m*dxp;Wbranch=np.sum(tau*dq,axis=1);Drelative=d*np.sum(rel*de,axis=1);Dshaft=b*v0*dxp
    measured=(E+K)-(E0+K0)+Wbranch+Drelative+Dshaft+stop-Wm
    defect=-.5*J*(vp-v0)**2+.5*k*np.sum(de**2,axis=1)+tau.sum(1)*(dx-dxp)
    close(measured,defect,'exact split-integration energy identity',2e-13)
    power_identity=J*v0*acc+np.sum(k*e0*rel,axis=1)+np.sum(tau*u0,axis=1)+d*np.sum(rel**2,axis=1)+b*v0**2-m*v0
    close(power_identity,0,'continuous internal reaction power identity',2e-13)
    C=np.zeros((len(c['gears']),len(names)))
    for i,g in enumerate(c['gears']):C[i,ix[g['dependent']]]=1;C[i,ix[g['master']]]=-g['factor']
    require(len(c['gears'])==7 and np.linalg.matrix_rank(C)==7,'seven retained independent gears')
    eq=np.column_stack([t['q'][:,ix[g['dependent']]]-g['factor']*t['q'][:,ix[g['master']]]-g['offset'] for g in c['gears']]);close(eq,t['gear_residual'],'gear residual equations',0)
    return dict(equations_verified=True,actuator_caps_Nm=[.6,.3],spring_stored_energy_J=stats(E),shaft_kinetic_energy_J=stats(K),
        shaft_motor_torque_Nm=stats(m),sum_branch_torque_Nm=stats(tau.sum(1)),branch_torque_peak_abs_Nm=abs(tau).max(0).tolist(),
        nonnegative_relative_damping_power_W=stats(d*np.sum(rel**2,axis=1)),shaft_stop_steps=int(stopped.sum()),shaft_stop_loss_total_J=float(stop.sum()),
        endpoint_sample_motor_work_J=float(h*np.sum(m*v0)),finite_increment_motor_work_J=float(Wm.sum()),branch_work_to_native_hand_J=float(Wbranch.sum()),
        split_integration_energy_defect_total_J=float(defect.sum()),split_integration_energy_defect_max_abs_per_step_J=float(abs(defect).max()),
        energy_accounting_identity_max_abs_residual_J=float(abs(measured-defect).max()),continuous_power_identity_max_abs_residual_W=float(abs(power_identity).max()),
        gear_rank=int(np.linalg.matrix_rank(C)),joint_coordinates_without_gears=len(names)-int(np.linalg.matrix_rank(C)),additional_virtual_shaft_coordinate=1,
        shaft_max_speed_rad_s=float(abs(v).max()),branch_max_speed_rad_s=float(abs(u).max()),
        boundary='Two bounded actuator channels; branch torques are passive transmission. Numerical split defects are reported, not assumed zero; no hardware parameter identification.')

def original_pose_and_wrist(t,c,baseline):
    bc=read(baseline/'CONTROLLER.json');bt=trace_data(baseline)
    for k in ['selection_sha256','candidate_sha256','scene_manifest_sha256','scene_signature_sha256','original_hand_datum_T_WH','original_depth_m','original_position_backoff_m','world_approach_axis','world_pre_baseLink_m','world_target_baseLink_m','baseLink_orientation_xyzw','wrist_script','derived_urdf_sha256']:
        require(c[k]==bc[k],'original top1/scene/wrist unchanged: '+k)
    for k in ['step','time','phase','register_target','joint_target']:
        require(t[k].dtype==bt[k].dtype and np.array_equal(t[k],bt[k]),'unchanged saved command '+k)
    h=1/1920; s=(t['step']-1)*h; smooth=lambda a:np.clip(a,0,1)**2*(3-2*np.clip(a,0,1))
    pre=np.array(c['world_pre_baseLink_m']);target=np.array(c['world_target_baseLink_m']);pos=pre[None,:]+smooth((s-.5)/3)[:,None]*(target-pre)[None,:];pos[:,2]+=.1*smooth((s-6.25)/3)
    quat=np.array(c['baseLink_orientation_xyzw']);qx,qy,qz,qw=quat;R=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],[2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],[2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])
    expected_com=pos+(R@np.asarray(c['base_link_to_COM'][0]))[None,:]
    close(np.linalg.norm(t['com'][:,:3]-expected_com,axis=1),t['wrist_position_error'],'independent original per-step wrist COM target',2e-12)
    return dict(original_top1_scene_depth_unchanged=True,commands_bit_identical=True,all_original_wrist_targets_reconstructed=True,
        baseline=str(baseline),baseline_trace_sha256=sha(baseline/'trace.npz'),actual_wrist_trajectory_allowed_to_change_under_different_load=True)

def audit_run(run,baseline=None):
    run=Path(run);c=read(run/'CONTROLLER.json');r=read(run/'RESULTS.json');t=trace_data(run);n=len(t['step']);require(n>0,'saved nonempty physical trace')
    archived={};working_source_changes=[]
    for f in [run/'controller_source_frozen.py',run/'rigid_base_source_frozen.py',run/'runner_source_frozen.py',run/'scene_source_frozen.py',run.parent/'fixture_source_frozen.py']:
        if f.exists():archived[sha(f)]=f
    def bound_file(path,expected,label):
        path=Path(path)
        if path.exists() and sha(path)==expected:return
        if path.suffix=='.py' and expected in archived:
            working_source_changes.append(dict(original_path=str(path),run_sha256=expected,current_sha256=sha(path) if path.exists() else None,frozen_copy=str(archived[expected]),run_frozen_copy_verified=True));return
        require(False,label)
    require(all(len(v)==n for v in t.values()),'all trace lengths');require(all(np.isfinite(v).all() for v in t.values() if v.dtype.kind in 'fiu'),'finite trace');close(t['step'],np.arange(1,n+1),'exact steps',0);close(t['time'],np.arange(1,n+1)/1920,'actual1920Hz clock',1e-15)
    require(n in (6240,19680),'declared complete fixture/scene length');require(r.get('physics_steps',r.get('observed_cost',{}).get('completed_physics_steps'))==n,'saved physics steps')
    for pk,sk in [('controller_script','controller_script_sha256'),('adapter_manifest','adapter_manifest_sha256'),('derived_urdf','derived_urdf_sha256'),('selection_results','selection_sha256'),('candidate_npz','candidate_sha256'),('scene_manifest','scene_manifest_sha256')]:
        if pk in c:bound_file(c[pk],c[sk],'bound file SHA '+pk)
    if 'trace_sha256' in r:require(sha(run/'trace.npz')==r['trace_sha256'],'trace SHA')
    if 'controller_sha256' in r:require(sha(run/'CONTROLLER.json')==r['controller_sha256'],'controller SHA')
    def records(x):
        if isinstance(x,dict):
            if 'path' in x and 'sha256' in x:bound_file(x['path'],x['sha256'],'source snapshot SHA')
            for v in x.values():records(v)
        elif isinstance(x,list):
            for v in x:records(v)
    records(read(run/'SOURCE_SNAPSHOT.json'))
    if 'fixture_plan_sha256' in c:
        plan_path=run.parent/'PLAN.json';plan=read(plan_path);records(plan)
        require(sha(plan_path)==c['fixture_plan_sha256']==r['source_plan_sha256'],'fixture plan source SHA')
        records(r.get('trace'));records(r.get('contacts'))
        top=read(run.parent/'RESULTS.json');require(next(x for x in top['cases'] if x['case_id']==run.name)==r,'case result identical in top-level report')
        require(top['completed_physics_steps']==top['attempted_python_step_calls']==31200 and top['possible_in_flight_step_calls']==0,'five-case full step accounting')
        require(not any(x['negative_pairs'] for x in r['initial_geometry']),'initial native hand-pad geometry clear')
        require(not r['early_wrong_link_contacts'],'no fixture early wrong-link contamination')
        require(r['no_state_resets_after_first_step'] and r['pad_positions_never_adapted'],'fixture no reset/reposition flags')
    guard=source_guards(run);ids=c['body_ids'];info=c['actual_constraint_info'];constraints=[info['wrist']]+info['gears'];require(len(constraints)==8,'one wrist/seven gear inventory')
    import xml.etree.ElementTree as ET
    joints=list(ET.parse(c['derived_urdf']).getroot().findall('joint'));native_index={j.attrib['name']:i for i,j in enumerate(joints)}
    if 'actual_joint_info' in c:
        require(all(native_index[j['name']]==j['index'] for j in c['actual_joint_info']),'URDF order independently bound to actual joint indices')
    for g,actual in zip(c['gears'],info['gears']):
        close([actual[1],actual[3]],[native_index[g['dependent']],native_index[g['master']]],'native gear endpoints',0)
        close([actual[4],actual[10],actual[11],actual[13],actual[14]],[6,100,-g['factor'],g['offset'],.2],'native gear type/cap/ratio/offset/ERP',0)
    close([info['wrist'][4],info['wrist'][10],info['wrist'][14]],[4,30,.2],'original finite wrist constraint',0)
    require(info['wrist'][0]==ids['hand'] and info['wrist'][2]==-1,'world wrist only');require(all(x[0]==ids['hand'] and x[2]==ids['hand'] for x in info['gears']),'all transmission constraints internal')
    other_ids=[ids['object']] if 'object' in ids else ids['pads'];require(all(b not in (x[0],x[2]) for b in other_ids for x in constraints),'no object/pad attachment in saved inventory')
    out=dict(run=str(run),status='saved_compliant_proxy_execution_verified',physics_steps_verified=n,source_guards=guard,mechanics=mechanics(t,c),contacts=contacts(run,c,n),artifact_SHA={name:sha(run/name) for name in ['RESULTS.json','CONTROLLER.json','SOURCE_SNAPSHOT.json','trace.npz']},physical_steps_added_by_audit=0,working_source_changes_after_saved_execution=working_source_changes)
    if baseline is not None:
        baseline=Path(baseline)
        if 'object' in ids:out['original_pose_and_wrist']=original_pose_and_wrist(t,c,baseline)
        else:
            old=read(baseline/'CONTROLLER.json');oldtrace=trace_data(baseline)
            for k in ('step','time','phase','register_target','joint_target'):require(np.array_equal(t[k],oldtrace[k]),'fixture original command '+k)
            for a,b in zip(c['pads'],old['pads']):require(a['pad_name']==b['pad_name'] and a['sphere_center_world_m']==b['sphere_center_world_m'],'frozen pad geometry')
            np0=read(run.parent/'PLAN.json')['input_protocol'];op0=read(baseline.parent/'PLAN.json')['input_protocol']
            differences={k:dict(new=np0.get(k),old=op0.get(k)) for k in np0.keys()|op0.keys() if np0.get(k)!=op0.get(k)}
            require(set(differences)=={'keep_original_two_motors_nine_gears','compliant_common_shaft_proxy'},'only declared controller/proxy protocol flags differ')
            out['fixture_protocol_declared_differences']=differences
            out['original_fixture_commands_and_pad_geometry_unchanged']=True
            out['proximal_joint_final_rad']={f'finger{i}':float(t['q'][-1,c['joint_order'].index(f'finger{i}_joint')]) for i in (1,2,3)}
            co=np.load(run/'contacts_full.npy',allow_pickle=False);firsts={}
            for pad in c['pads']:
                h=int(c['body_ids']['hand']);pid=pad['body_id'];rows=co[(((co[:,2]==h)&(co[:,3]==pid))|((co[:,3]==h)&(co[:,2]==pid)))&(co[:,16]>1e-8)]
                if len(rows):
                    step=int(rows[:,0].min());rr=rows[rows[:,0]==step];handlinks=np.where(rr[:,2]==h,rr[:,4],rr[:,5]).astype(int)
                    require(set(handlinks)=={pad['intended_link_index']},'independent first contact is designated tip')
                    require(r['first_positive_pad_contacts'][pad['pad_name']]['first_positive_contact_step']==step,'first-contact reported step')
                    firsts[pad['pad_name']]=dict(step=step,hand_link_indices=sorted(map(int,set(handlinks))))
                else:require(pad['pad_name'] not in r['first_positive_pad_contacts'],'no invented initial pad contact')
            out['first_positive_pad_contacts_independently_recomputed']=firsts
        oldcontacts=contacts(baseline,read(baseline/'CONTROLLER.json'),len(trace_data(baseline)['step']))
        out['baseline_contact_comparison']={}
        for label,now in out['contacts']['body_pairs'].items():
            old=oldcontacts['body_pairs'].get(label,{});new=set(now['hand_link_indices'])-set(old.get('hand_link_indices',[]))
            out['baseline_contact_comparison'][label]=dict(new_hand_link_indices=sorted(new),old_positive_rows=old.get('total_positive_rows',0),new_positive_rows=now['total_positive_rows'])
    out['statistics_boundary']='Each condition is one deterministic trial; time-window frames are correlated. No confidence interval or hardware-calibration claim.'
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,default=Path(__file__).with_name('dh3_compliant_proxy.py'));ap.add_argument('--source-review-only',action='store_true');ap.add_argument('--run-dir',type=Path,action='append',default=[]);ap.add_argument('--baseline-dir',type=Path,action='append',default=[]);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    require(not args.output.exists(),'Retain audit receipts; choose a new output');require(not args.baseline_dir or len(args.baseline_dir)==len(args.run_dir),'one baseline per run')
    start=time.monotonic();out=dict(audit_source=str(Path(__file__).resolve()),audit_source_sha256=sha(__file__),source_review=source_review(args.source),runs=[],errors=[],additional_physics_steps=0,additional_network_forwards=0,simulator_imported=False)
    for i,run in enumerate(args.run_dir):
        try:out['runs'].append(audit_run(run,args.baseline_dir[i] if args.baseline_dir else None))
        except Exception as e:out['errors'].append(dict(run=str(run),type=type(e).__name__,message=str(e)))
    out['total_saved_physics_steps_verified']=sum(x['physics_steps_verified'] for x in out['runs'])
    out.update(status='source_equations_checked_no_physical_results' if not args.run_dir else ('all_saved_runs_verified' if not out['errors'] else 'audit_failed'),wall_seconds=time.monotonic()-start)
    if OLD_FIXTURE_AUDIT.exists():out['prior_five_fixture_audit_preserved']=dict(path=str(OLD_FIXTURE_AUDIT),sha256=sha(OLD_FIXTURE_AUDIT))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2));raise SystemExit(1 if out['errors'] else 0)
if __name__=='__main__':main()
