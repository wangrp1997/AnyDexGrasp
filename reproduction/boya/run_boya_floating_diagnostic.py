"""Diagnostic Isaac Boya hand on three physical translation drives.

No arm, IK or per-step state teleport. The original palm subtree, native mimic
constraints, finger PD parameters and object-only weight are retained. Virtual
non-colliding gantry links move the wrist with a separately declared drive.
Import after AppLauncher before calling build_scene(). This is a diagnostic,
not an AnyDex paper result or the frozen external12 benchmark executor.
"""
from pathlib import Path
import argparse
from copy import deepcopy
import hashlib
import json
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

CANON = Path('/home/wangrenpeng/CanonDex')
ANYDEX = Path('/home/wangrenpeng/AnyDexGrasp')
sys.path[:0] = [str(CANON/'scripts/research'), str(CANON), str(ANYDEX/'reproduction/boya')]
GANTRY_NAMES = ['diagnostic_slide_x', 'diagnostic_slide_y', 'diagnostic_slide_z']
GANTRY = dict(stiffness_n_m=100000., damping_n_s_m=1000., effort_limit_n=1000.,
              velocity_limit_m_s=1., virtual_link_mass_kg=.001,
              virtual_link_diagonal_inertia_kg_m2=1e-6)
PHASE_LENGTHS = [('prehold', 800), ('approach', 500), ('close', 250), ('hold', 50), ('lift', 500)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_inputs(job):
    import numpy as np
    from canondex.grasp.template_gripper_adapter import TemplateFamily
    from canondex.grasp.template_gripper_closure import TwoSidedClosure
    pool = Path(job.get('candidate_dir', job.get('pool', '')))
    generation = read(pool/'generation.json')
    if not generation['complete'] or sha(pool/'candidates.npz') != generation['candidates_sha256']:
        raise ValueError('Frozen candidate pool changed or incomplete')
    for p, h in generation.get('input_sha256', {}).items():
        if sha(p) != h:
            raise ValueError('Candidate source changed: '+p)
    if job.get('protocol_path') and sha(job['protocol_path']) != job['protocol_sha256']:
        raise ValueError('Protocol changed')
    family = TemplateFamily.load(Path(job['package']), Path(job['hand_urdf']), Path(job['hand_scene']), generation['gesture_id'])
    with np.load(pool/'candidates.npz') as z:
        if list(z['independent_motor_names']) != list(family.hand.names):
            raise ValueError('Original motor order changed')
        i = int(job.get('candidate_index', 0))
        opened = z['proposal_q'][i].copy()
        closed = z['proposal_q_close'][i].copy()
        palm = z['proposal_palm'][i].copy()
        world_object = z['world_T_object'].copy()
        direction = z['proposal_approach_direction'][i].copy()
    if not np.array_equal(closed, family.nominal):
        raise ValueError('Frozen AnyDex closure differs from original company template')
    row = job['object_row']
    if not np.allclose(world_object, row['world_T_object'], atol=2e-8, rtol=0):
        raise ValueError('Pool and settled object pose differ')
    if not np.isclose(np.linalg.norm(direction), 1., atol=1e-8):
        raise ValueError('Unit approach direction required')
    closure = TwoSidedClosure.from_template(family, opened, closed)
    return dict(family=family, hand=family.hand, opened=opened, closed=closed,
                palm=palm, world_object=world_object, direction=direction,
                closure=closure, generation=generation, pool=pool)


def command_sequence(inputs):
    import numpy as np
    from canondex.grasp.template_arm_bridge import smooth_fraction
    palm = inputs['palm']
    pre = palm.copy()
    pre[:3, 3] -= .1*inputs['direction']
    commands, desired, phases = [], [], []
    for phase, count in PHASE_LENGTHS:
        for k in range(count):
            u = float(smooth_fraction((k+1)/count))
            pose = pre.copy() if phase in ('prehold', 'approach') else palm.copy()
            if phase == 'approach':
                pose[:3, 3] += u*.1*inputs['direction']
            elif phase == 'lift':
                pose[:3, 3] += [0., 0., u*.1]
            q = (inputs['closure'].command((k+1)/count, (k+1)/count) if phase == 'close'
                 else inputs['closed'] if phase in ('hold', 'lift') else inputs['opened'])
            commands.append(q.copy())
            desired.append(pose)
            phases.append(phase)
    return dict(motor_target=np.asarray(commands), desired_palm=np.asarray(desired),
                phase=np.asarray(phases), pregrasp_palm=pre)


def write_gantry_urdf(output, hand):
    from canondex.grasp.baselines.isaac_assets import prepare_physics_hand
    source_path = prepare_physics_hand(output, hand=hand)
    source = ET.parse(source_path).getroot()
    root = ET.Element('robot', name='boya_diagnostic_translation_gantry')
    carrier_names = ['diagnostic_gantry_base', 'diagnostic_x_carriage', 'diagnostic_y_carriage', 'diagnostic_z_carriage']
    for name in carrier_names:
        link = ET.SubElement(root, 'link', name=name)
        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'mass', value=str(GANTRY['virtual_link_mass_kg']))
        ii = str(GANTRY['virtual_link_diagonal_inertia_kg_m2'])
        ET.SubElement(inertial, 'inertia', ixx=ii, iyy=ii, izz=ii, ixy='0', ixz='0', iyz='0')
    for i, name in enumerate(GANTRY_NAMES):
        joint = ET.SubElement(root, 'joint', name=name, type='prismatic')
        ET.SubElement(joint, 'parent', link=carrier_names[i])
        ET.SubElement(joint, 'child', link=carrier_names[i+1])
        ET.SubElement(joint, 'origin', xyz='0 0 0', rpy='0 0 0')
        ET.SubElement(joint, 'axis', xyz=['1 0 0', '0 1 0', '0 0 1'][i])
        ET.SubElement(joint, 'limit', lower='-1', upper='1', effort=str(GANTRY['effort_limit_n']), velocity=str(GANTRY['velocity_limit_m_s']))
    joint = ET.SubElement(root, 'joint', name='diagnostic_mount_to_palm', type='fixed')
    ET.SubElement(joint, 'parent', link=carrier_names[-1])
    ET.SubElement(joint, 'child', link='palm')
    ET.SubElement(joint, 'origin', xyz='0 0 0', rpy='0 0 0')
    for node in source:
        root.append(deepcopy(node))
    target = Path(output)/'boya_translation_gantry.urdf'
    ET.ElementTree(root).write(target, encoding='unicode')
    return target


def anchor_gantry(stage):
    """Installed Lab3 world-joint topology, adapted only for the carrier name."""
    from pxr import Usd, UsdPhysics
    from isaaclab.sim.schemas.schemas import create_world_fixed_joint
    roots = [p for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot')) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1 or roots[0].GetName() != 'diagnostic_gantry_base':
        raise RuntimeError('Expected the imported diagnostic gantry articulation root')
    root = roots[0]
    create_world_fixed_joint(root, stage)
    parent = root.GetParent()
    UsdPhysics.ArticulationRootAPI.Apply(parent)
    for schema in ('PhysxArticulationAPI', 'NewtonArticulationRootAPI'):
        if schema in root.GetAppliedSchemas():
            parent.AddAppliedSchema(schema)
            root.RemoveAppliedSchema(schema)
    for attr in root.GetAttributes():
        if attr.GetName().startswith(('physxArticulation:', 'newton:selfCollision')):
            parent.CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr.Get())
    root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    return dict(root=str(root.GetPath()), articulation_root=str(parent.GetPath()), root_stationary=True)


def build_scene(job, output, enable_cameras=False):
    """Return a dictionary of scene assets/inputs for execution or zero-step replay.

    Caller must create AppLauncher first. enable_cameras is accepted to make the
    same build function useful to a renderer; the renderer creates its cameras.
    """
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from scipy.spatial.transform import Rotation
    from pxr import Usd, UsdPhysics
    from carb.settings import get_settings
    from canondex.grasp.physics import load_physics_config
    from canondex.grasp.template_gripper_drive import apply_thumb_root_effort_diagnostic, thumb_root_effort_overrides
    from canondex.robots.ur16e_boya import _hand_actuators
    from canondex.robots.hand_actuation import audit_hand_actuation
    from canondex.robots.kinematics import UrdfKinematics
    from canondex.scene.robot_collision import preserve_hand_collision
    from canondex.scene.gesture_collision import apply_hand_collision_filters
    from canondex.scene.floor import spawn_floor_and_light
    from canondex.scene.grasp_object import spawn_object, configure_contact_materials
    from canondex.scene.execute_motion import enable_robot_contacts
    from canondex.scene.runtime import forward_scene
    from run_template_arm_positive import object_asset
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(job)
    seq = command_sequence(inputs)
    hand = inputs['hand']
    profile = load_physics_config(Path(job['physics_config']))
    diagnostic = read(job['diagnostic_config'])
    urdf = write_gantry_urdf(output, hand)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.004, device='cpu', gravity=(0., 0., 0.),
        physics=profile.scene_cfg(), use_fabric=False, visualizer_cfgs=[]))
    spawn_floor_and_light()
    actuators = _hand_actuators(profile.hand_damping_mode, scene_xml=Path(job['hand_scene']))
    expected_hand = set(hand.names)|set(hand.mimics)
    for cfg in actuators.values():
        cfg.joint_names_expr = [n for n in cfg.joint_names_expr if n in expected_hand]
    actuators = {k:v for k,v in actuators.items() if v.joint_names_expr}
    actuators = apply_thumb_root_effort_diagnostic(actuators, diagnostic['thumb_root_effort_limit_nm'])
    actuators['diagnostic_gantry'] = ImplicitActuatorCfg(joint_names_expr=GANTRY_NAMES,
        stiffness=GANTRY['stiffness_n_m'], damping=GANTRY['damping_n_s_m'], joint_effort_limit=GANTRY['effort_limit_n'], armature=0.)
    pre = seq['pregrasp_palm']
    robot = Articulation(ArticulationCfg(prim_path='/World/Robot',
        spawn=sim_utils.UrdfFileCfg(asset_path=str(urdf), fix_base=False, merge_fixed_joints=True,
            merge_mesh=False, collision_from_visuals=False, self_collision=True,
            convert_mimic_joints_to_normal_joints=False, make_instanceable=False,
            usd_dir=str(output/'usd'), usd_file_name='boya_translation_gantry.usd',
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True,
                max_depenetration_velocity=profile.robot_max_depenetration_velocity_m_s),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                solver_position_iteration_count=diagnostic['position_iterations'], solver_velocity_iteration_count=1),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(drive_type='force',
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.))),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(pre[:3,3]), rot=tuple(Rotation.from_matrix(pre[:3,:3]).as_quat())),
        actuators=actuators))
    anchor = anchor_gantry(sim.stage)
    get_settings().set_bool('/physics/disableContactProcessing', False)
    filters = apply_hand_collision_filters(sim.stage, collision_links=hand.collision_links, scene=Path(job['hand_scene']))
    model = SimpleNamespace(fk=UrdfKinematics(urdf), collision_mesh=hand.mesh)
    collision = preserve_hand_collision(sim, model, physics_profile=profile,
        rest_offset=profile.hand_rest_offset(0.), collision_links=hand.collision_links)
    row = job['object_row']
    asset = object_asset(SimpleNamespace(object_root=Path(row['object_root'])), inputs['world_object'], row['scale'])
    refinement = read(job['collision_refinement']) if job.get('collision_refinement') else None
    obj = spawn_object(sim, asset, physics_profile=profile, collision_refinement=refinement)
    sim_utils.bind_physics_material('/World/ground', '/World/ObjectMaterial')
    enable_robot_contacts(sim)
    sim.reset()
    dt = sim.get_physics_dt()
    robot.update(dt)
    obj.update(dt)
    names = list(robot.joint_names)
    if set(names) != expected_hand|set(GANTRY_NAMES) or not robot.is_fixed_base:
        raise RuntimeError('Unexpected gantry-hand joint structure: '+str(names))
    expanded = hand.full_state(inputs['opened'], inputs['family'].wrist)
    q = np.array([0. if n in GANTRY_NAMES else expanded[n] for n in names])
    tensor = lambda a:torch.tensor(a, dtype=torch.float32, device=robot.device)
    # SimulationContext.reset starts play; InitialStateCfg is not a state write.
    root_pose = np.r_[pre[:3,3], Rotation.from_matrix(pre[:3,:3]).as_quat()]
    robot.write_root_link_pose_to_sim_index(root_pose=tensor(root_pose[None]))
    robot.write_root_link_velocity_to_sim_index(root_velocity=tensor(np.zeros((1,6))))
    robot.write_joint_state_to_sim(tensor(q[None]), tensor(np.zeros((1,len(q)))))
    robot.actuators.target_command.set_position_index(value=tensor(q[None]))
    robot.actuators.target_command.set_velocity_index(value=tensor(np.zeros((1,len(q)))))
    # Cold scene initialization only; no pose or velocity writes in trial loop.
    obj.write_root_link_pose_to_sim_index(root_pose=tensor(asset.pose_xyzw[None]))
    obj.write_root_link_velocity_to_sim_index(root_velocity=tensor(np.zeros((1,6))))
    robot.permanent_wrench_composer.reset()
    obj.permanent_wrench_composer.reset()
    obj.permanent_wrench_composer.add_forces_and_torques_index(forces=tensor([[[0.,0.,-.98]]]), is_global=True)
    forward_scene(sim)
    robot.update(dt)
    obj.update(dt)
    from canondex.scene.runtime import snapshot
    from canondex.robots.kinematics import pose_matrix
    from canondex.grasp.template_arm_bridge import palm_error
    measured = snapshot(robot)
    expected = model.fk.transforms(expanded|dict.fromkeys(GANTRY_NAMES,0.), pre)
    fk_errors = {n:dict(zip(('position_m','orientation_rad'),palm_error(pose_matrix(p),expected[n])))
                 for n,p in zip(robot.body_names,measured['body_pose'])}
    owner, offset = model.fk.fixed_attachment('palm',np.eye(4))
    palm_error_values = palm_error(pose_matrix(measured['body_pose'][list(robot.body_names).index(owner)])@offset,pre)
    measured_object = obj.data.root_link_pose_w.torch[0].cpu().numpy().copy()
    object_error_values = palm_error(pose_matrix(measured_object),pose_matrix(asset.pose_xyzw))
    initialization = dict(body_fk_errors=fk_errors,
        palm_error=dict(zip(('position_m','orientation_rad'),palm_error_values)),
        object_error=dict(zip(('position_m','orientation_rad'),object_error_values)),
        max_joint_position_error_rad=float(np.max(np.abs(measured['joint_pos']-q))),
        max_joint_speed_rad_s=float(np.max(np.abs(measured['joint_vel']))),
        max_object_speed=float(np.max(np.abs(obj.data.root_link_vel_w.torch[0].cpu().numpy()))),
        expected_root_pose_xyzw=root_pose.tolist(),expected_palm=pre.tolist(),
        expected_object_pose_xyzw=asset.pose_xyzw.tolist(),measured_object_pose_xyzw=measured_object.tolist(),
        articulation_root_initial_pose_writes=1,object_initial_pose_writes=1,physics_steps=0)
    save(output/'initialization_audit.json',initialization)
    errors=[v for row in fk_errors.values()for v in row.values()]+list(palm_error_values)+list(object_error_values)+[initialization[k]for k in ('max_joint_position_error_rad','max_joint_speed_rad_s','max_object_speed')]
    if max(errors)>2e-5:
        raise RuntimeError('Controlled wrist cold initialization differs from declared FK/object pose')
    actuation = audit_hand_actuation(robot, profile.hand_damping_mode, joint_names=sorted(expected_hand),
        intentional_effort_overrides=thumb_root_effort_overrides(diagnostic['thumb_root_effort_limit_nm']), scene=Path(job['hand_scene']))
    materials = configure_contact_materials(robot, obj, .6)
    joints = {p.GetName():p for p in Usd.PrimRange(sim.stage.GetPrimAtPath('/World/Robot')) if p.GetTypeName()=='PhysicsRevoluteJoint'}
    mimic = {}
    for name, (driver, coefficient, offset) in hand.mimics.items():
        prim = joints[name]
        ok = (prim.HasAPI('NewtonMimicAPI') and prim.GetAttribute('newton:mimicEnabled').Get() is True
            and list(prim.GetRelationship('newton:mimicJoint').GetTargets())==[joints[driver].GetPath()]
            and np.isclose(prim.GetAttribute('newton:mimicCoef1').Get(),coefficient)
            and np.isclose(np.deg2rad(prim.GetAttribute('newton:mimicCoef0').Get()),offset))
        mimic[name] = bool(ok)
        if not ok:
            raise RuntimeError('Original native mimic constraint changed: '+name)
    mass = float(obj.root_view.get_masses().numpy().ravel()[0])
    if mass != float(np.float32(.1)):
        raise RuntimeError('Native object mass changed')
    audit = dict(anchor=anchor, collision=collision, filters=filters, actuation=actuation, materials=materials,
                 native_mimics=mimic, gantry=GANTRY, mass_kg=mass, hand_urdf=str(urdf), hand_urdf_sha256=sha(urdf),
                 solver_position_iteration_count=diagnostic['position_iterations'],initialization=initialization)
    return dict(sim=sim, robot=robot, obj=obj, asset=asset, hand=hand, model=model, inputs=inputs,
                sequence=seq, audit=audit, urdf=urdf, names=names, tensor=tensor)


def run(job):
    import numpy as np
    from canondex.robots.kinematics import pose_matrix
    from canondex.grasp.template_arm_bridge import palm_error
    from canondex.scene.runtime import snapshot
    from canondex.scene.execute_motion import RobotContactReports
    from canondex.scene.object_contacts import ObjectContactReports
    from canondex.grasp.tabletop_pickup import observe_tabletop_pickup
    from template_gripper_native_audit import audit_scene
    out = Path(job['output_dir'])
    if (out/'evaluation.json').exists():
        raise ValueError('Retain previous trial; use new output directory')
    rt = build_scene(job, out)
    sim, robot, obj = rt['sim'], rt['robot'], rt['obj']
    hand, seq, names = rt['hand'], rt['sequence'], rt['names']
    native = audit_scene(sim, out, ['all','object'], roots=('/World/Robot','/World/Object'), fast=True)
    if not native['within_20um']:
        raise RuntimeError('Native hand/object cooking gate rejected')
    own = RobotContactReports(sim, capacity=262144, compact=True)
    contacts = ObjectContactReports(sim, own.bodies, capacity=262144, compact=True)
    initial = snapshot(robot)
    palm_owner, palm_offset = rt['model'].fk.fixed_attachment('palm', np.eye(4))
    palm_index = list(robot.body_names).index(palm_owner)
    hand_owners = {rt['model'].fk.fixed_attachment(n,np.eye(4))[0] for n in hand.links}
    root_pose = pose_matrix(initial['body_pose'][0])
    maxima = dict(joint_limit_violation_rad=0., mimic_error_rad=0., penetration_m=0.,
        palm_position_error_m=0., palm_orientation_error_rad=0., illegal_object_contact_force_n=0.,
        base_position_drift_m=0., base_orientation_drift_rad=0.)
    commands = []
    for motors, desired in zip(seq['motor_target'], seq['desired_palm']):
        xyz = seq['pregrasp_palm'][:3,:3].T @ (desired[:3,3]-seq['pregrasp_palm'][:3,3])
        state = hand.full_state(motors,rt['inputs']['family'].wrist)|dict(zip(GANTRY_NAMES,xyz))
        commands.append([state[n] for n in names])
    commands = np.array(commands,dtype=np.float32)
    velocity = np.zeros_like(commands)
    axes = [names.index(n) for n in GANTRY_NAMES]
    velocity[1:,axes] = np.diff(commands[:,axes],axis=0)/.004
    np.savez_compressed(out/'commands.npz',joint_names=np.array(names),joint_target=commands,
        velocity_target=velocity,desired_palm=seq['desired_palm'],phase=seq['phase'],time_s=np.arange(1,2101)*.004)
    report = dict(schema='anydex_boya_controlled_wrist_diagnostic_v1',complete=False,task_success=False,
        object_id=job['object_id'],arm_present=False,engine='PhysX',device='cpu',dt_s=.004,
        declared_command_steps=2100,total_physics_steps=0,state_writes_during_execution=0,
        root_pose_writes_during_execution=0,object_state_writes_during_execution=0,
        object_weight_world_n=[0.,0.,-.98],global_gravity_m_s2=[0.,0.,0.],new_CGR_forwards=0,
        new_training_updates=0,outcome_dependent_retry=False,benchmark_only=True,training_writeback=False,
        commanded_gantry=GANTRY,phase_lengths=dict(PHASE_LENGTHS),audit=rt['audit'],native_within_20um=True,
        native_hulls=native['hulls'],joint_names=names,body_names=list(robot.body_names),
        hand_candidate_sha256=sha(rt['inputs']['pool']/'candidates.npz'),commands_sha256=sha(out/'commands.npz'),
        protocol_ref=dict(path=job.get('protocol_path'),sha256=job.get('protocol_sha256')),
        failure_reason=None,scope='Controlled physical wrist translation; no arm, no IK, no candidate reranking')
    save(out/'evaluation.json',report)
    rows=[]
    checkpoints=[]
    start=time.monotonic()
    limits={n:(float(j.find('limit').get('lower')),float(j.find('limit').get('upper')))
            for j in rt['model'].fk.joints for n in [j.get('name')] if j.get('type') in ('revolute','prismatic')}
    for i,(phase,command,desired) in enumerate(zip(seq['phase'],commands,seq['desired_palm'])):
        robot.actuators.target_command.set_position_index(value=rt['tensor'](command[None]))
        robot.actuators.target_command.set_velocity_index(value=rt['tensor'](velocity[i][None]))
        robot.write_data_to_sim()
        obj.write_data_to_sim()
        sim.step(render=False)
        robot.update(.004)
        obj.update(.004)
        state=snapshot(robot)
        pose=obj.data.root_link_pose_w.torch[0].cpu().numpy().copy()
        contact=contacts.read()
        rc=own.read()
        loads=np.array([sum(abs(c['normal_force_n'])for c in contact if c['body'].startswith(f))for f in('th','ff','mf','rf','lf')])
        od=max([0.]+[-c['separation_m']for c in contact])
        rd=max([0.]+[-c['separation_m']for c in rc])
        limit=max([0.]+[max(limits[n][0]-state['joint_pos'][j],state['joint_pos'][j]-limits[n][1],0.)for j,n in enumerate(names)if n in limits])
        mimic=max(abs(state['joint_pos'][names.index(n)]-state['joint_pos'][names.index(d)]*m-o)for n,(d,m,o)in hand.mimics.items())
        palm=pose_matrix(state['body_pose'][palm_index])@palm_offset
        position,angle=palm_error(palm,desired)
        drift,rotation=palm_error(pose_matrix(state['body_pose'][0]),root_pose)
        illegal=max([0.]+[abs(c['normal_force_n'])for c in contact if c['body']not in hand_owners])
        for k,v in [('joint_limit_violation_rad',limit),('mimic_error_rad',mimic),('penetration_m',max(od,rd)),
                    ('palm_position_error_m',position),('palm_orientation_error_rad',angle),
                    ('illegal_object_contact_force_n',illegal),('base_position_drift_m',drift),('base_orientation_drift_rad',rotation)]:
            maxima[k]=max(maxima[k],float(v))
        rows.append((pose,state['joint_pos'],state['joint_vel'],state['body_pose'],palm,loads,od,rd))
        boundary=i==2099 or seq['phase'][i+1]!=phase
        if boundary:
            checkpoints.append(dict(phase=str(phase),step=i+1,object_pose_xyzw=pose.tolist(),finger_loads_n=loads.tolist(),object_contacts=contact,robot_contacts=rc))
            print('CONTROLLED_WRIST_PHASE '+json.dumps(dict(phase=str(phase),step=i+1,object_z=float(pose[2]),loads_n=loads.tolist(),maxima=maxima)),flush=True)
        if (i+1)%50==0 or boundary:
            save(out/'execution_progress.json',dict(step=i+1,phase=str(phase),wall_s=time.monotonic()-start,maxima=maxima))
        if max(od,rd)>.005:
            report['failure_reason']='deep_collision_stop'
            break
        if phase=='approach' and np.linalg.norm(pose[:2]-rows[799][0][:2])>.03:
            report['failure_reason']='object_pushed_before_closure'
            break
    n=len(rows)
    trace=dict(time_s=np.arange(1,n+1)*.004,phase=seq['phase'][:n],object_pose_xyzw=np.array([r[0]for r in rows]),
        joint_pos=np.array([r[1]for r in rows]),joint_vel=np.array([r[2]for r in rows]),body_pose=np.array([r[3]for r in rows]),
        palm_pose=np.array([r[4]for r in rows]),finger_loads_n=np.array([r[5]for r in rows]),
        object_penetration_m=np.array([r[6]for r in rows]),robot_penetration_m=np.array([r[7]for r in rows]),
        joint_target=commands[:n],velocity_target=velocity[:n],desired_palm=seq['desired_palm'][:n],
        joint_names=np.array(names),body_names=np.array(robot.body_names))
    np.savez_compressed(out/'trace.npz',**trace)
    gates=dict(joint_limits=maxima['joint_limit_violation_rad']<=.002,mimics=maxima['mimic_error_rad']<=.002,
        penetration=maxima['penetration_m']<=.001,palm_tracking=maxima['palm_position_error_m']<=.03 and maxima['palm_orientation_error_rad']<=.2617993877991494,
        no_nonhand_object_support=maxima['illegal_object_contact_force_n']<=.001,
        fixed_base=maxima['base_position_drift_m']<=2e-5 and maxima['base_orientation_drift_rad']<=2e-5,complete_execution=n==2100)
    gates={k:bool(v)for k,v in gates.items()}
    pickup=observe_tabletop_pickup(dict(gates=gates),trace,rt['asset'].mesh.vertices)if np.any(trace['phase']=='lift')else None
    success=bool(pickup and all(gates.values())and pickup['final_state_still_lifted_and_bearing']and pickup['final_reference_rise_m']>=.05)
    report.update(complete=True,task_success=success,total_physics_steps=n,phases=checkpoints,maxima=maxima,gates=gates,
        tabletop_pickup=pickup,final_object_pose_xyzw=trace['object_pose_xyzw'][-1].tolist(),
        trace_sha256=sha(out/'trace.npz'),execution_wall_s=time.monotonic()-start)
    save(out/'evaluation.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,required=True)
    a=p.parse_args()
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(headless=True,device='cpu',visualizer=None,kit_args='--/log/file=/dev/null')
    code=1
    try:
        result=run(read(a.job))
        print(json.dumps(dict(object_id=result['object_id'],task_success=result['task_success'],steps=result['total_physics_steps'])),flush=True)
        code=0
    finally:
        launcher.app.close(exit_code=code)

if __name__=='__main__':
    main()
