"""Declared compliant proxy, not an identified DH3 mechanism.

A single finite-torque virtual shaft drives three elastic branches. Branch
reactions oppose that SAME shaft; blocked fingers cannot receive unlimited or
independent servo power. Local tip/outlet kinematics and all source entities are
retained. Branch elasticity is an engineering approximation, not a claim that
commercial DH3 has a free inter-finger differential or these spring constants.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import shutil
import numpy as np
import dh3_time_refinement_smoke as rigid
from dh3_time_refinement_smoke import p, sha, adapter_targets, joint_angles


@dataclass(frozen=True)
class Compliance:
    branch_stiffness_Nm_rad: float = .8
    branch_damping_Nm_s_rad: float = .02
    shaft_inertia_kg_m2: float = .001
    shaft_viscous_Nm_s_rad: float = .005
    shaft_position_to_velocity_gain_s_inv: float = 6.
    shaft_velocity_feedback_Nm_s_rad: float = 2.
    shaft_lower_rad: float = 0.
    shaft_upper_rad: float = 1.333


class CompliantDH3(rigid.CoupledDH3):
    """One bounded flexion drive plus the unchanged bounded spread motor."""
    def __init__(self, *args, compliance=None, **kwargs):
        self.proxy_ready = False
        super().__init__(*args, **kwargs)
        self.compliance = Compliance() if compliance is None else compliance
        kept = []
        for gear in self.gears:
            if gear['master'] == self.position_master:
                p.removeConstraint(gear['id'], physicsClientId=self.client)
            else:
                kept.append(gear)
        self.gears = kept
        # Each finger keeps the original local affine tip/outlet relation.
        # Only the two cross-finger rigid relations are replaced by elasticity.
        for finger in (1, 2, 3):
            master = f'finger{finger}_joint'
            for suffix in ('outlet', 'tip'):
                name = f'finger{finger}_{suffix}_joint'
                factor = self.rows[name]['position_slope'] / self.rows[master]['position_slope']
                offset = self.rows[name]['offset'] - factor*self.rows[master]['offset']
                constraint = p.createConstraint(self.body, self.names[name], self.body,
                    self.names[master], p.JOINT_GEAR, [0,1,0], [0,0,0], [0,0,0], physicsClientId=self.client)
                p.changeConstraint(constraint, gearRatio=-factor, relativePositionTarget=offset,
                    erp=self.a.gear_erp, maxForce=self.a.gear_max_torque_Nm, physicsClientId=self.client)
                self.gears.append(dict(id=constraint, dependent=name, master=master, factor=factor, offset=offset))
        self.branch_names = [f'finger{i}_joint' for i in (1,2,3)]
        self.branch_ids = [self.names[name] for name in self.branch_names]
        for index in self.branch_ids:
            p.setJointMotorControl2(self.body,index,p.VELOCITY_CONTROL,targetVelocity=0,force=0,physicsClientId=self.client)
        self.shaft_q = float(p.getJointState(self.body,self.names[self.position_master],physicsClientId=self.client)[0])
        self.shaft_v = 0.
        self.last = dict(shaft_motor_torque_Nm=0., branch_torque_Nm=np.zeros(3),
            branch_extension_before_step_rad=np.zeros(3),branch_damping_power_W=0.,
            shaft_stop_dissipation_J=0.,shaft_q_before_step_rad=self.shaft_q,shaft_v_before_step_rad=0.)
        self.proxy_steps = 0
        self.proxy_ready = True

    def command(self, position, rotation, base_link_position, base_link_orientation):
        if not self.proxy_ready:
            return super().command(position,rotation,base_link_position,base_link_orientation)
        targets = adapter_targets(joint_angles(position,rotation),self.manifest)
        c = self.compliance
        state = p.getJointStates(self.body,self.branch_ids,physicsClientId=self.client)
        q = np.array([s[0] for s in state]); v = np.array([s[1] for s in state])
        extension = self.shaft_q-q
        relative_v = self.shaft_v-v
        torque = c.branch_stiffness_Nm_rad*extension+c.branch_damping_Nm_s_rad*relative_v
        v_target = float(np.clip(c.shaft_position_to_velocity_gain_s_inv*(targets[self.position_master]-self.shaft_q),
            -self.a.position_motor_max_velocity_rad_s,self.a.position_motor_max_velocity_rad_s))
        motor = float(np.clip(c.shaft_velocity_feedback_Nm_s_rad*(v_target-self.shaft_v),
            -self.a.position_motor_max_torque_Nm,self.a.position_motor_max_torque_Nm))
        acceleration = (motor-float(torque.sum())-c.shaft_viscous_Nm_s_rad*self.shaft_v)/c.shaft_inertia_kg_m2
        velocity = self.shaft_v+self.a.dt*acceleration
        proposed = self.shaft_q+self.a.dt*velocity
        stopped = proposed < c.shaft_lower_rad or proposed > c.shaft_upper_rad
        dissipation = .5*c.shaft_inertia_kg_m2*velocity**2 if stopped else 0.
        self.last = dict(shaft_motor_torque_Nm=motor,branch_torque_Nm=torque,
            branch_extension_before_step_rad=extension,
            branch_damping_power_W=float(c.branch_damping_Nm_s_rad*np.dot(relative_v,relative_v)),
            shaft_stop_dissipation_J=float(dissipation),shaft_q_before_step_rad=self.shaft_q,
            shaft_v_before_step_rad=self.shaft_v)
        self.shaft_q = float(np.clip(proposed,c.shaft_lower_rad,c.shaft_upper_rad))
        self.shaft_v = 0. if stopped else float(velocity)
        for index, value in zip(self.branch_ids,torque):
            p.setJointMotorControl2(self.body,index,p.TORQUE_CONTROL,force=float(value),physicsClientId=self.client)
        p.setJointMotorControl2(self.body,self.names[self.rotation_master],p.POSITION_CONTROL,
            targetPosition=targets[self.rotation_master],targetVelocity=0,
            force=self.a.rotation_motor_max_torque_Nm,maxVelocity=self.a.rotation_motor_max_velocity_rad_s,
            positionGain=self.a.position_gain,velocityGain=self.a.velocity_gain,physicsClientId=self.client)
        self.target_com = self.com_target(base_link_position,base_link_orientation)
        p.changeConstraint(self.wrist,self.target_com[0],self.target_com[1],
            maxForce=self.a.wrist_max_force_per_row_N_or_Nm,erp=self.a.wrist_erp,physicsClientId=self.client)
        self.proxy_steps += 1
        return targets

    def observe(self):
        row = super().observe()
        raw = row['motor_torque'].copy()
        # This field is explicitly ACTUATOR torque: virtual shaft at master slot,
        # physical spread motor at its slot. Passive branch torques are separate.
        actuator = np.zeros_like(raw)
        actuator[self.movable_names.index(self.rotation_master)] = raw[self.movable_names.index(self.rotation_master)]
        actuator[self.movable_names.index(self.position_master)] = self.last['shaft_motor_torque_Nm']
        row['motor_torque'] = actuator
        row['joint_motor_torque_readback'] = raw
        branch_q = row['q'][[self.movable_names.index(n) for n in self.branch_names]]
        extension = self.shaft_q-branch_q
        row.update(self.last)
        row.update(shaft_q_rad=self.shaft_q,shaft_v_rad_s=self.shaft_v,
            branch_extension_rad=extension,
            branch_elastic_energy_J=.5*self.compliance.branch_stiffness_Nm_rad*float(np.dot(extension,extension)),
            shaft_kinetic_energy_J=.5*self.compliance.shaft_inertia_kg_m2*self.shaft_v**2,
            shaft_step_count=self.proxy_steps)
        return row

    def receipt(self):
        result = super().receipt()
        result.update(controller_script=str(Path(__file__).resolve()),controller_script_sha256=sha(__file__),
            controller_kind='declared_common_shaft_elastic_branch_proxy_v1',
            inherited_rigid_source=dict(path=str(Path(rigid.__file__).resolve()),sha256=sha(rigid.__file__)),
            active_motors=['virtual_common_flexion_shaft',self.rotation_master],
            passive_motor_force=None,
            compliance=asdict(self.compliance),
            common_drive_equations={
                'branch_torque':'k*(x-q_i)+d*(xdot-qdot_i)',
                'shaft':'J*xddot = bounded_motor_torque - sum(branch_torque) - b*xdot',
                'stored_energy':'0.5*k*sum((x-q_i)^2) + 0.5*J*xdot^2',
                'damping':'d*sum((xdot-qdot_i)^2) >= 0',
                'shaft_end_stops':'Finite-domain inelastic stop; removed kinetic energy logged, no hand/object reset'},
            actuator_trace_mapping={'motor_torque':'Virtual shaft at finger1_joint slot and real spread at finger1_base_joint; all other entries zero',
                'branch_torque_Nm':'Passive transmitted torque, one per proximal joint; reaction returned to common shaft',
                'joint_motor_torque_readback':'Unmodified native joint readback, not virtual actuator torque'},
            removed_cross_finger_rigid_gears=2,retained_local_affine_gears=6,
            independent_finger_servos=False,original_13_entities_and_inertias_retained=True,
            source_supported='Common powered input and contact-adaptive behavior; exact commercial internal topology/parameters unavailable',
            approximation_boundary='Elastic branch compliance is an explicit engineering proxy, not identified DH3 springs, verified inter-finger differential, original four-DoF mechanism, or paper physical benchmark',
            register_boundary='No force70-to-SI calibration; original declared .6Nm common input retained')
        return result


def write_controller_receipt(output_dir, receipt):
    output_dir = Path(output_dir)
    source = Path(__file__).resolve(); target = output_dir/'controller_source_frozen.py'
    shutil.copyfile(source,target)
    inherited = output_dir/'rigid_base_source_frozen.py'
    shutil.copyfile(rigid.__file__,inherited)
    receipt['frozen_controller_source'] = str(target.resolve())
    receipt['frozen_inherited_source'] = str(inherited.resolve())
    (output_dir/'CONTROLLER.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (output_dir/'SOURCE_SNAPSHOT.json').write_text(json.dumps(dict(path=str(target.resolve()),sha256=sha(target),
        acquisition='Copied before first physical integration',
        inherited_dependency=dict(path=str(inherited.resolve()),sha256=sha(inherited))),indent=2)+'\n')
