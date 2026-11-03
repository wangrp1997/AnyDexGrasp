"""Bounded, two-drive DH3 physics port; not author-identified hardware dynamics.

All joint/base resets are initialization only. Original 13 entity inertials and
meshes remain in the separately validated geometry adapter. During integration,
two finite motors drive a finite gear-coupled mechanism and a finite world wrist
constraint tracks a smooth script. Every step and failed trial is retained.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import shutil
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'outputs/native_deps/python_overlay'), str(ROOT/'reproduction/runtime'), str(ROOT/'reproduction/simulation')]
import pybullet as p
from dh3_author_kinematics import joint_angles, source_receipt
from dh3_geometry_adapter import adapter_targets


@dataclass(frozen=True)
class PhysicsAssumptions:
    dt: float = 1/240
    solver_iterations: int = 100
    non_contact_inner_iterations: int = 20
    gravity_m_s2: tuple = (0., 0., -9.81)
    position_motor_max_torque_Nm: float = .6
    rotation_motor_max_torque_Nm: float = .3
    position_motor_max_velocity_rad_s: float = 1.
    rotation_motor_max_velocity_rad_s: float = .5
    gear_max_torque_Nm: float = .8
    gear_erp: float = .2
    wrist_max_force_per_row_N_or_Nm: float = 30.
    wrist_erp: float = .2
    position_gain: float = .15
    velocity_gain: float = 1.
    contact_lateral_friction: float = .5
    contact_restitution: float = 0.
    geometric_margin_m: float = .001


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smooth(t):
    t = np.clip(t, 0., 1.)
    return float(t*t*(3-2*t))


def affine_map(manifest):
    """Compose the unchanged author's two-register formula with geometry map."""
    q0 = adapter_targets(joint_angles(0, 0), manifest)
    qp = adapter_targets(joint_angles(95, 0), manifest)
    qr = adapter_targets(joint_angles(0, 95), manifest)
    rows = {name: {'offset': q0[name], 'position_slope': (qp[name]-q0[name])/95,
                   'rotation_slope': (qr[name]-q0[name])/95} for name in q0}
    for row in rows.values():
        if bool(abs(row['position_slope']) > 1e-13) == bool(abs(row['rotation_slope']) > 1e-13):
            raise ValueError('A joint must belong to exactly one released scalar drive')
    return rows


class CoupledDH3:
    """Only two active motors; all other coordinates transmit through gears."""
    def __init__(self, client, manifest_path, initial_base_link_position, initial_base_link_orientation,
                 position=94., rotation=67., assumptions=PhysicsAssumptions()):
        self.client = client
        self.a = assumptions
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.rows = affine_map(self.manifest)
        self.urdf = Path(self.manifest['output_URDF'])
        assert sha(self.urdf) == self.manifest['output_URDF_sha256']
        flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        self.body = p.loadURDF(str(self.urdf), initial_base_link_position, initial_base_link_orientation,
                               useFixedBase=False, flags=flags, physicsClientId=client)
        self.names = {p.getJointInfo(self.body, i, physicsClientId=client)[1].decode(): i
                      for i in range(p.getNumJoints(self.body, physicsClientId=client))}
        self.movable_names = list(self.rows)
        self.ids = [self.names[n] for n in self.movable_names]
        self.position_master = 'finger1_joint'
        self.rotation_master = 'finger1_base_joint'
        initial = adapter_targets(joint_angles(position, rotation), self.manifest)
        for name, value in initial.items():
            p.resetJointState(self.body, self.names[name], value, 0, physicsClientId=client)
            p.setJointMotorControl2(self.body, self.names[name], p.VELOCITY_CONTROL, targetVelocity=0, force=0, physicsClientId=client)
        for link in range(-1, p.getNumJoints(self.body, physicsClientId=client)):
            p.changeDynamics(self.body, link, lateralFriction=assumptions.contact_lateral_friction,
                             restitution=assumptions.contact_restitution, jointDamping=0,
                             physicsClientId=client)
        link_ids={-1:'base_Link'}|{i:p.getJointInfo(self.body,i,physicsClientId=client)[12].decode()
                                          for i in range(p.getNumJoints(self.body,physicsClientId=client))}
        by_link={name:index for index,name in link_ids.items()}
        self.mechanism_exclusions=[]
        p.performCollisionDetection(physicsClientId=client)
        for finger in range(1,4):
            prefix='finger'+str(finger)
            # Source CAD represents a linked mechanism as sibling bodies. The
            # external rod overlaps the convex palm housing and its own mates.
            # This explicit mechanism topology does not suppress inter-finger,
            # hand-object, hand-table, or arbitrary same-finger collisions.
            for first,second,reason in [
                ('base_Link',prefix+'_outlet_Link','palm housing / external rod mechanism mate; original dynamic convex hull overlaps the housing'),
                (prefix+'_Link',prefix+'_outlet_Link','parallel linkage members with shared end mechanism'),
                (prefix+'_tip_Link',prefix+'_outlet_Link','physical linkage end mates represented as URDF siblings')]:
                ia,ib=by_link[first],by_link[second]
                close=p.getClosestPoints(self.body,self.body,.02,linkIndexA=ia,linkIndexB=ib,physicsClientId=client)
                self.mechanism_exclusions.append({'first':first,'second':second,'reason':reason,
                    'initial_convex_closest_distance_m':min((x[8] for x in close),default=None)})
                p.setCollisionFilterPair(self.body,self.body,ia,ib,0,physicsClientId=client)
        base = p.getDynamicsInfo(self.body, -1, physicsClientId=client)
        self.base_link_to_com = (base[3], base[4])
        self.initial_base_link_pose = (tuple(initial_base_link_position), tuple(initial_base_link_orientation))
        target_com = self.com_target(*self.initial_base_link_pose)
        measured = p.getBasePositionAndOrientation(self.body, physicsClientId=client)
        assert np.linalg.norm(np.asarray(target_com[0])-measured[0]) < 1e-10
        assert abs(abs(np.dot(target_com[1], measured[1]))-1) < 1e-10
        self.wrist = p.createConstraint(self.body, -1, -1, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], target_com[0],
                                       parentFrameOrientation=[0,0,0,1], childFrameOrientation=target_com[1], physicsClientId=client)
        p.changeConstraint(self.wrist, maxForce=assumptions.wrist_max_force_per_row_N_or_Nm,
                           erp=assumptions.wrist_erp, physicsClientId=client)
        self.gears = []
        for name, row in self.rows.items():
            if name in (self.position_master, self.rotation_master):
                continue
            master = self.position_master if abs(row['position_slope']) > 1e-13 else self.rotation_master
            key = 'position_slope' if master == self.position_master else 'rotation_slope'
            factor = row[key] / self.rows[master][key]
            offset = row['offset'] - factor*self.rows[master]['offset']
            constraint = p.createConstraint(self.body, self.names[name], self.body, self.names[master], p.JOINT_GEAR,
                                            [0,1,0], [0,0,0], [0,0,0], physicsClientId=client)
            p.changeConstraint(constraint, gearRatio=-factor, relativePositionTarget=offset,
                               erp=assumptions.gear_erp, maxForce=assumptions.gear_max_torque_Nm, physicsClientId=client)
            self.gears.append({'id':constraint, 'dependent':name, 'master':master, 'factor':factor, 'offset':offset})
        self.initial_reset_count = len(initial)
        self.target_com = target_com
        self.command(position, rotation, *self.initial_base_link_pose)

    def com_target(self, base_link_position, base_link_orientation):
        # multiplyTransforms returns float32; preserve original URDF/datum doubles.
        q=np.asarray(base_link_orientation,dtype=float); r=np.asarray(self.base_link_to_com[1],dtype=float)
        matrix=np.asarray(p.getMatrixFromQuaternion(q)).reshape(3,3)
        xyz=np.asarray(base_link_position)+matrix@self.base_link_to_com[0]
        x,y,z,w=q; X,Y,Z,W=r
        quat=np.array([w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W,w*W-x*X-y*Y-z*Z])
        return xyz.tolist(),(quat/np.linalg.norm(quat)).tolist()

    def command(self, position, rotation, base_link_position, base_link_orientation):
        targets = adapter_targets(joint_angles(position, rotation), self.manifest)
        for master, force, speed in ((self.position_master, self.a.position_motor_max_torque_Nm, self.a.position_motor_max_velocity_rad_s),
                                     (self.rotation_master, self.a.rotation_motor_max_torque_Nm, self.a.rotation_motor_max_velocity_rad_s)):
            p.setJointMotorControl2(self.body, self.names[master], p.POSITION_CONTROL, targetPosition=targets[master],
                                   targetVelocity=0, force=force, maxVelocity=speed, positionGain=self.a.position_gain,
                                   velocityGain=self.a.velocity_gain, physicsClientId=self.client)
        self.target_com = self.com_target(base_link_position, base_link_orientation)
        p.changeConstraint(self.wrist, self.target_com[0], self.target_com[1], maxForce=self.a.wrist_max_force_per_row_N_or_Nm,
                           erp=self.a.wrist_erp, physicsClientId=self.client)
        return targets

    def observe(self):
        states = p.getJointStates(self.body, self.ids, physicsClientId=self.client)
        q = np.array([s[0] for s in states]); v = np.array([s[1] for s in states]); tau = np.array([s[3] for s in states])
        by_name = dict(zip(self.movable_names, q))
        residuals = np.array([by_name[g['dependent']]-g['factor']*by_name[g['master']]-g['offset'] for g in self.gears])
        reactions = np.array([p.getConstraintState(g['id'], physicsClientId=self.client)[0] for g in self.gears])
        com, orn = p.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
        linear, angular = p.getBaseVelocity(self.body, physicsClientId=self.client)
        wrist_reaction = np.array(p.getConstraintState(self.wrist, physicsClientId=self.client))
        wrist_error = np.linalg.norm(np.asarray(com)-self.target_com[0])
        wrist_rotation_error = 2*np.arccos(np.clip(abs(np.dot(orn,self.target_com[1])), -1, 1))
        return {'q':q,'v':v,'motor_torque':tau,'gear_residual':residuals,'gear_reaction':reactions,
                'com':np.r_[com,orn], 'base_velocity':np.r_[linear,angular], 'wrist_reaction':wrist_reaction,
                'wrist_position_error':float(wrist_error),'wrist_rotation_error':float(wrist_rotation_error)}

    def receipt(self):
        return {'controller_script':str(Path(__file__).resolve()),'controller_script_sha256':sha(Path(__file__)), 'adapter_manifest':str(self.manifest_path), 'adapter_manifest_sha256':sha(self.manifest_path),
                'derived_urdf':str(self.urdf),'derived_urdf_sha256':sha(self.urdf),'joint_order':self.movable_names,
                'two_register_affine_map':self.rows,'active_motors':[self.position_master,self.rotation_master],
                'passive_motor_force':0.,'gears':self.gears,'contact_record_columns':['step','bodyA','bodyB','linkA','linkB','signed_distance_m','normal_force_N','lateral_force1_N','lateral_force2_N'],'trace_metadata':{'time_unit':'s','q_unit':'rad','v_unit':'rad/s','motor_torque_unit':'Nm','gear_reaction_unit':'Nm','gear_residual_unit':'rad','com_format':['x_m','y_m','z_m','qx','qy','qz','qw'],'wrist_reaction_format':['Fx_N','Fy_N','Fz_N','Tx_Nm','Ty_Nm','Tz_Nm']},'actual_constraint_info':{'wrist':p.getConstraintInfo(self.wrist,physicsClientId=self.client),'gears':[p.getConstraintInfo(g['id'],physicsClientId=self.client) for g in self.gears]},'base_link_to_COM':self.base_link_to_com,
                'initial_resets_before_physics':self.initial_reset_count,'original_13_inertias_retained':True,
                'helper_frames':self.manifest.get('helper_frames',[]),'mechanism_self_collision_exclusions':self.mechanism_exclusions,
                'collision_flags':['URDF_USE_SELF_COLLISION','URDF_USE_SELF_COLLISION_EXCLUDE_PARENT'],
                'collision_geometry':'Original STL shapes imported by Bullet as dynamic convex hulls with engine default mesh margin; not original concave triangle contacts',
                'assumptions_SI_not_hardware_identification':asdict(self.a),
                'gear_force_boundary':'Finite ideal transmission reactions, separately bounded from two motor torques. ERP positional correction can inject energy; residual and reaction work are logged.',
                'register_boundary':'Original force70/speed100 have no known SI conversion. Chosen torque/speed/gear/wrist values are declared simulation-port assumptions.',
                'author_kinematics':source_receipt()}


def configure(client, a):
    p.setTimeStep(a.dt, physicsClientId=client)
    p.setPhysicsEngineParameter(numSolverIterations=a.solver_iterations, numSubSteps=1, numNonContactInnerIterations=a.non_contact_inner_iterations, deterministicOverlappingPairs=1, physicsClientId=client)
    p.setGravity(*a.gravity_m_s2, physicsClientId=client)


def write_controller_receipt(output_dir, receipt):
    """Freeze the exact code used before integrating the first physical step."""
    source=Path(__file__).resolve();target=Path(output_dir)/'controller_source_frozen.py'
    shutil.copyfile(source,target)
    assert sha(target)==receipt['controller_script_sha256']
    receipt['frozen_controller_source']=str(target.resolve())
    (Path(output_dir)/'CONTROLLER.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (Path(output_dir)/'SOURCE_SNAPSHOT.json').write_text(json.dumps({
        'path':str(target.resolve()),'sha256':sha(target),
        'matches_pre_existing_CONTROLLER_source_sha256':True,
        'acquisition':'Copied before first physical integration',
        'prior_receipt_unchanged':True},indent=2)+'\n')


def run_empty(output_dir, manifest_path, initial_position=94., initial_rotation=67., non_contact_inner_iterations=20):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('Keep all trial directories; choose a new --output-dir')
    output_dir.mkdir(parents=True)
    a = PhysicsAssumptions(non_contact_inner_iterations=non_contact_inner_iterations); client = p.connect(p.DIRECT); start = time.monotonic(); records=[]; contacts=[]
    try:
        configure(client,a)
        hand = CoupledDH3(client,manifest_path,[0,0,.5],[0,0,0,1],initial_position,initial_rotation,assumptions=a)
        receipt = hand.receipt()
        write_controller_receipt(output_dir,receipt)
        # Enforce no state-teleport API after initialization, including accidental future edits.
        old_reset_joint, old_reset_base = p.resetJointState,p.resetBasePositionAndOrientation
        def forbid_reset(*args,**kwargs):
            raise RuntimeError('Dynamic trace cannot reset joint or base state')
        p.resetJointState=p.resetBasePositionAndOrientation=forbid_reset
        try:
            # 0.5s hold; 1s rotation excursion and 1s return; 2s close;
            # 0.75s closed hold; 2.5s /0.1m wrist lift; 0.5s lifted hold.
            total=8.25
            for step in range(int(round(total/a.dt))):
                t=step*a.dt; position=initial_position;rotation=initial_rotation;z=.5;phase='hold_open'
                other_rotation=initial_rotation+23 if initial_rotation<=77 else initial_rotation-23
                if .5<=t<1.5:rotation=initial_rotation+(other_rotation-initial_rotation)*smooth(t-.5);phase='rotate_out'
                elif 1.5<=t<2.5:rotation=other_rotation+(initial_rotation-other_rotation)*smooth(t-1.5);phase='rotate_back'
                elif 2.5<=t<4.5:position=initial_position*(1-smooth((t-2.5)/2));phase='close'
                elif t>=4.5:position=0.;phase='hold_closed'
                if t>=5.25:z=.5+.1*smooth((t-5.25)/2.5);phase='wrist_lift' if t<7.75 else 'hold_lift'
                targets=hand.command(position,rotation,[0,0,z],[0,0,0,1])
                p.stepSimulation(physicsClientId=client)
                row=hand.observe();row.update(step=step+1,time=(step+1)*a.dt,register_target=np.array([position,rotation]),
                    joint_target=np.array([targets[n] for n in hand.movable_names]),phase=phase)
                records.append(row)
                cs=p.getContactPoints(hand.body,hand.body,physicsClientId=client)
                for c in cs:contacts.append([step+1,c[3],c[4],c[8],c[9]])
        finally:
            p.resetJointState,p.resetBasePositionAndOrientation=old_reset_joint,old_reset_base
        arrays={key:np.array([r[key] for r in records]) for key in records[0]}
        np.savez_compressed(output_dir/'trace.npz',**arrays)
        np.save(output_dir/'self_contacts.npy',np.asarray(contacts,dtype=float).reshape(-1,5))
        finite=all(np.isfinite(value).all() for key,value in arrays.items() if value.dtype.kind in 'fiu')
        gearmax=float(abs(arrays['gear_residual']).max()); wpos=float(arrays['wrist_position_error'].max())
        close_rows=np.flatnonzero(arrays['phase']=='hold_closed');close_final=int(close_rows[-1])
        final_error=abs(arrays['q'][close_final]-arrays['joint_target'][close_final])
        torque_bounds=np.zeros(len(hand.ids));torque_bounds[hand.movable_names.index(hand.position_master)]=a.position_motor_max_torque_Nm
        torque_bounds[hand.movable_names.index(hand.rotation_master)]=a.rotation_motor_max_torque_Nm
        motor_bound=bool((abs(arrays['motor_torque'])<=torque_bounds+1e-8).all())
        gear_bound=bool((abs(arrays['gear_reaction'])<=a.gear_max_torque_Nm+1e-7).all())
        gear_work=float(np.sum(abs(arrays['gear_reaction'][1:]*np.diff(arrays['gear_residual'],axis=0))))
        contact_stopped=False
        if final_error.max()>=.04:
            active_index=hand.movable_names.index(hand.position_master)
            # The empty fingers can meet before the unmeasured geometric0 end.
            # This is accepted only as a contact stop, never exact pose tracking.
            recent=[c for c in contacts if close_final-24<=c[0]<=close_final+1 and c[4]>.01]
            contact_stopped=bool(recent and abs(arrays['v'][close_final,active_index])<.05)
        tracking_or_contact_stop=bool(final_error.max()<.04 or contact_stopped)
        result={'scope':'Empty-hand dynamic mechanism and scripted finite-wrist smoke; not a grasp result',
                'status':'passed' if finite and gearmax<.01 and wpos<.002 and tracking_or_contact_stop and motor_bound and gear_bound and abs(arrays['v']).max()<1.5 else 'failed',
                'initial_registers':[initial_position,initial_rotation],'physics_steps':len(records),'physical_seconds':len(records)*a.dt,'wall_seconds':time.monotonic()-start,
                'finite':finite,'max_gear_residual_rad':gearmax,'per_gear_max_residual_rad':dict(zip([g['dependent'] for g in hand.gears],abs(arrays['gear_residual']).max(0).tolist())),
                'max_wrist_COM_error_m':wpos,'max_wrist_rotation_error_rad':float(arrays['wrist_rotation_error'].max()),
                'closed_final_contact_stopped':contact_stopped,'closed_final_tracking_or_contact_stop':tracking_or_contact_stop,'closed_final_max_target_error_rad':float(final_error.max()),'closed_final_per_joint_error_rad':dict(zip(hand.movable_names,final_error.tolist())),
                'peak_joint_velocity_rad_s':float(abs(arrays['v']).max()),'motor_torque_bounds_pass':motor_bound,'gear_reaction_bounds_pass':gear_bound,
                'peak_motor_torque_Nm':abs(arrays['motor_torque']).max(0).tolist(),'peak_gear_reaction_Nm':abs(arrays['gear_reaction']).max(0).tolist(),
                'absolute_constraint_correction_work_upper_proxy_J':gear_work,
                'self_contact_rows':len(contacts),'minimum_self_contact_distance_m':float(np.asarray(contacts)[:,3].min()) if contacts else None,
                'zero_state_resets_after_first_step':True,'object_attached':False,'gravity_enabled':True,
                'declared_acceptance':{'finite':True,'max_gear_residual_rad':.01,'max_wrist_COM_error_m':.002,'closed_final_max_target_error_rad':.04,'contact_stop_alternative':'Actual non-neighbor selfcontact plus position-master speed<.05rad/s; remaining target error reported','peak_joint_velocity_rad_s':1.5,'finite_motor_and_gear_bounds':True},
                'controller_receipt_sha256':sha(output_dir/'CONTROLLER.json'),'trace_sha256':sha(output_dir/'trace.npz'),
                'boundary':'Author geometry preserved; two ideal finite gear-coupled simulated actuators and finite script wrist are engineering assumptions, not identified DH3 hardware. No object grasp or paper success-rate claim.'}
        (output_dir/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2));return result
    finally:
        p.disconnect(client)


def run_cube(output_dir, manifest_path, selection_results, scene_manifest, non_contact_inner_iterations=20):
    """One frozen original top1, original depth/backoff, bounded scripted wrist."""
    from synthetic_scene import create_scene, assert_scene_matches_manifest
    from scipy.spatial.transform import Rotation
    from PIL import Image
    output_dir=Path(output_dir)
    if output_dir.exists():raise FileExistsError('Retain every trial; use a new directory')
    output_dir.mkdir(parents=True);(output_dir/'frames').mkdir()
    selection_results=Path(selection_results);scene_manifest=Path(scene_manifest)
    selection=json.loads(selection_results.read_text());observation=json.loads(scene_manifest.read_text())
    if not selection.get('selected'):raise ValueError('Original pipeline did not select a hand pose')
    candidate_path=Path(selection['output']['path']);assert sha(candidate_path)==selection['output']['sha256']
    selected=np.load(candidate_path,allow_pickle=False)
    hand_candidate=selected['top1_DH3_grasp_array'];two_candidate=selected['top1_two_finger_grasp_array']
    assert hand_candidate.shape==(19,) and two_candidate.shape==(17,)
    T_WC=np.array(observation['camera']['T_WC']);H=np.eye(4);H[:3,:3]=hand_candidate[3:12].reshape(3,3);H[:3,3]=hand_candidate[12:15]
    T_WH=T_WC@H;direction=T_WC[:3,:3]@two_candidate[4:13].reshape(3,3)[:,0];direction/=np.linalg.norm(direction)
    registers=hand_candidate[15:17];depth=float(hand_candidate[1]);backoff=float(registers[0]/100*.005)
    target=T_WH[:3,3]+(depth-backoff)*direction;pre=target-.07*direction
    orientation=Rotation.from_matrix(T_WH[:3,:3]).as_quat().tolist()
    a=PhysicsAssumptions(non_contact_inner_iterations=non_contact_inner_iterations);client=p.connect(p.DIRECT);start=time.monotonic();rows=[];allcontacts=[];frames=[]
    try:
        scene=create_scene(p,client);assert_scene_matches_manifest(p,client,scene,observation)
        assert abs(scene['spec']['physics']['time_step_s']-a.dt)<1e-15
        p.setPhysicsEngineParameter(numNonContactInnerIterations=a.non_contact_inner_iterations,physicsClientId=client)
        hand=CoupledDH3(client,manifest_path,pre.tolist(),orientation,*registers,assumptions=a)
        receipt=hand.receipt();receipt.update(scope='One original selected-candidate scripted-wrist simulation port',
            selection_results=str(selection_results),selection_sha256=sha(selection_results),candidate_npz=str(candidate_path),candidate_sha256=sha(candidate_path),
            scene_manifest=str(scene_manifest),scene_manifest_sha256=sha(scene_manifest),scene_signature_sha256=scene['scene_signature_sha256'],
            scene_observation_bound_before_steps=True,body_ids={'hand':hand.body,'object':scene['object_id'],'table':scene['table_id']},T_WC=T_WC.tolist(),original_hand_datum_T_WH=T_WH.tolist(),
            original_depth_m=depth,original_position_backoff_m=backoff,net_insertion_m=depth-backoff,
            world_approach_axis=direction.tolist(),world_pre_baseLink_m=pre.tolist(),world_target_baseLink_m=target.tolist(),
            baseLink_orientation_xyzw=orientation,rotation_matrix_orthogonality_error=float(abs(T_WH[:3,:3].T@T_WH[:3,:3]-np.eye(3)).max()),
            wrist_script={'approach_m':.07,'approach_s':3.,'close_s':2.,'close_register':0.,'hold_closed_s':.75,'lift_world_z_m':.1,'lift_s':3.,'hold_lift_s':1.,'initial_hold_s':.5},
            port_boundary='Original network/map/top1/depth/approach preserved. Two-actuator ideal gear model and world-Z scripted lift replace unprovided firmware/UR-arm execution; no path-planning or paper-success-rate claim.')
        write_controller_receipt(output_dir,receipt)
        view=p.computeViewMatrix([.35,-.4,.35],[0,.08,.12],[0,0,1]);projection=p.computeProjectionMatrixFOV(50,640/480,.01,2.)
        initial_position=np.array(p.getBasePositionAndOrientation(scene['object_id'],physicsClientId=client)[0])
        old_joint,old_base=p.resetJointState,p.resetBasePositionAndOrientation
        def forbid_reset(*args,**kwargs):raise RuntimeError('No dynamic state reset')
        p.resetJointState=p.resetBasePositionAndOrientation=forbid_reset
        try:
            total=10.25
            for step in range(round(total/a.dt)):
                t=step*a.dt;pos=pre.copy();position=float(registers[0]);phase='hold_open'
                if .5<=t<3.5:pos=pre+(target-pre)*smooth((t-.5)/3);phase='approach'
                elif t>=3.5:pos=target.copy();phase='close'
                if 3.5<=t<5.5:position=float(registers[0])*(1-smooth((t-3.5)/2))
                elif t>=5.5:position=0.;phase='hold_closed'
                if t>=6.25:pos=target+np.array([0,0,.1])*smooth((t-6.25)/3);phase='lift' if t<9.25 else 'hold_lift'
                targets=hand.command(position,float(registers[1]),pos.tolist(),orientation)
                p.stepSimulation(physicsClientId=client)
                row=hand.observe();object_pose=p.getBasePositionAndOrientation(scene['object_id'],physicsClientId=client)
                object_velocity=p.getBaseVelocity(scene['object_id'],physicsClientId=client)
                row.update(step=step+1,time=(step+1)*a.dt,phase=phase,register_target=np.array([position,registers[1]]),
                           joint_target=np.array([targets[n] for n in hand.movable_names]),
                           object_pose=np.r_[object_pose[0],object_pose[1]],object_velocity=np.r_[object_velocity[0],object_velocity[1]],
                           object_aabb=np.array(p.getAABB(scene['object_id'],-1,physicsClientId=client)))
                rows.append(row)
                # Contact records retain all robot/object/table/self contact costs.
                for c in p.getContactPoints(physicsClientId=client):
                    if hand.body in (c[1],c[2]):allcontacts.append([step+1,c[1],c[2],c[3],c[4],c[8],c[9],c[10],c[12]])
                if step%12==0:
                    rendered=p.getCameraImage(640,480,viewMatrix=view,projectionMatrix=projection,renderer=p.ER_TINY_RENDERER,
                                               shadow=0,lightDirection=[1,1,3],physicsClientId=client)
                    filename='frame_%05d.png'%(step//12)
                    Image.fromarray(np.asarray(rendered[2],dtype=np.uint8).reshape(480,640,4)[...,:3]).save(output_dir/'frames'/filename)
                    frames.append({'path':'frames/'+filename,'physics_step':step+1,'physical_time_s':(step+1)*a.dt})
        finally:p.resetJointState,p.resetBasePositionAndOrientation=old_joint,old_base
        arrays={key:np.array([row[key] for row in rows]) for key in rows[0]}
        contact_array=np.asarray(allcontacts,dtype=float).reshape(-1,9)
        np.savez_compressed(output_dir/'trace.npz',**arrays);np.save(output_dir/'contacts.npy',contact_array)
        (output_dir/'frames.json').write_text(json.dumps({'fps':20,'every_physics_steps':12,'actual_simultaneous_render':True,'zero_additional_physics_steps_for_media':True,'frames':frames},indent=2)+'\n')
        finite=all(np.isfinite(v).all() for v in arrays.values() if v.dtype.kind in 'fiu')
        lift=arrays['object_pose'][-1,2]-initial_position[2];final_low=arrays['object_aabb'][-1,0,2]
        is_object=(contact_array[:,1]==scene['object_id'])|(contact_array[:,2]==scene['object_id']) if len(contact_array) else np.array([],bool)
        is_table=(contact_array[:,1]==scene['table_id'])|(contact_array[:,2]==scene['table_id']) if len(contact_array) else np.array([],bool)
        object_contacts=contact_array[is_object];table_contacts=contact_array[is_table]
        last_contacts=object_contacts[(object_contacts[:,0]>len(rows)-120)&(object_contacts[:,6]>0)] if len(object_contacts) else object_contacts
        gearmax=float(abs(arrays['gear_residual']).max());maxspeed=float(abs(arrays['v']).max())
        torque_bounds=np.zeros(len(hand.ids))
        torque_bounds[hand.movable_names.index(hand.position_master)]=a.position_motor_max_torque_Nm
        torque_bounds[hand.movable_names.index(hand.rotation_master)]=a.rotation_motor_max_torque_Nm
        motor_bound=bool((abs(arrays['motor_torque'])<=torque_bounds+1e-8).all())
        gear_bound=bool((abs(arrays['gear_reaction'])<=a.gear_max_torque_Nm+1e-7).all())
        wrist_bound=bool((arrays['wrist_position_error']<.002).all())
        stable=bool(finite and gearmax<.01 and maxspeed<1.5 and motor_bound and gear_bound and wrist_bound)
        held=bool(lift>.05 and final_low>.04 and len(last_contacts)>0)
        result={'scope':'One new synthetic cube original-top1 physical smoke; not original paper benchmark',
            'status':'object_lifted' if held and stable else 'not_lifted_or_port_invalid','lift_observed':bool(held),'port_stability_pass':stable,
            'physics_steps':len(rows),'physical_seconds':len(rows)*a.dt,'wall_seconds_including_actual_render':time.monotonic()-start,
            'object_final_lift_m':float(lift),'object_final_lowest_world_z_m':float(final_low),'object_max_lift_m':float(arrays['object_pose'][:,2].max()-initial_position[2]),
            'finite':finite,'motor_torque_bounds_pass':motor_bound,'gear_reaction_bounds_pass':gear_bound,'wrist_tracking_pass':wrist_bound,'max_gear_residual_rad':gearmax,'per_gear_max_residual_rad':dict(zip([g['dependent'] for g in hand.gears],abs(arrays['gear_residual']).max(0).tolist())),
            'peak_joint_velocity_rad_s':maxspeed,'max_wrist_COM_error_m':float(arrays['wrist_position_error'].max()),
            'peak_motor_torque_Nm':abs(arrays['motor_torque']).max(0).tolist(),'peak_gear_reaction_Nm':abs(arrays['gear_reaction']).max(0).tolist(),
            'hand_object_contact_rows':len(object_contacts),'minimum_hand_object_contact_distance_m':float(object_contacts[:,5].min()) if len(object_contacts) else None,
            'peak_hand_object_normal_force_N':float(object_contacts[:,6].max()) if len(object_contacts) else None,
            'hand_table_contact_rows':len(table_contacts),'minimum_hand_table_contact_distance_m':float(table_contacts[:,5].min()) if len(table_contacts) else None,
            'final_half_second_hand_object_contact_rows':len(last_contacts),'state_resets_after_first_step':0,'object_attachment_constraints':0,'object_gravity_retained':True,
            'continuous_same_scene':True,'actual_rendered_frames':len(frames),'normal_speed_video_fps':20,
            'controller_sha256':sha(output_dir/'CONTROLLER.json'),'trace_sha256':sha(output_dir/'trace.npz'),
            'boundary':'Declared simulation-port force/speed/friction/convex mesh dynamics, original13 entity inertias and explicit internal adjacency exclusions. A single primitive result does not establish hardware accuracy or general grasp success.'}
        (output_dir/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
    finally:p.disconnect(client)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['empty','cube'],default='empty')
    ap.add_argument('--adapter-manifest',type=Path,default=ROOT/'outputs/dh3_geometry_adapter_20261103_v2/ADAPTER.json')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'outputs/dh3_empty_physics_20261103_v1')
    ap.add_argument('--non-contact-inner-iterations',type=int,choices=range(1,101),default=20)
    ap.add_argument('--position',type=float,default=94.)
    ap.add_argument('--rotation',type=float,default=67.)
    ap.add_argument('--selection-results',type=Path)
    ap.add_argument('--scene-manifest',type=Path,default=ROOT/'outputs/synthetic_cube_scene_20261103_v2/scene_manifest.json')
    args=ap.parse_args()
    try:
        if args.mode=='empty':run_empty(args.output_dir,args.adapter_manifest,args.position,args.rotation,args.non_contact_inner_iterations)
        else:run_cube(args.output_dir,args.adapter_manifest,args.selection_results,args.scene_manifest,args.non_contact_inner_iterations)
    except Exception as exc:
        if args.output_dir.exists() and not (args.output_dir/'RESULTS.json').exists():
            (args.output_dir/'FAILED.json').write_text(json.dumps({'status':'execution_error','type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'results_not_completed':True},indent=2)+'\n')
        raise


if __name__=='__main__':main()
