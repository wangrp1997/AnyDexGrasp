"""Full CPU backend/label/cost admission checks without a simulator."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import author_cpu_full_control as C
import author_cpu_full_audit as A


def receipt():
    value=dict(status='diagnostic_full',complete=True,full_trial=True,prefix_ticks=0,
        expected_command_steps=3024,mode='cpu',device='cpu',lanes=1,label=None,raw_label=0,
        exclude_training=True,formal_training_eligible=False,observations_completed_steps_per_lane=[3024],
        physical_steps_unknown=False,inflight_step_unconfirmed=False,original_gates={'original':True},
        prepared_template={'ready':'same','commands':'same'},
        group_receipt={'path':'/tmp/result/GPU_VECTOR_EXECUTION.json','sha256':'fixture'},
        profile={'path':'/tmp/result/GPU_VECTOR_PROFILE.json','sha256':'fixture'},source_sha256={})
    for key in ('physical_clock_steps','returned_physical_steps','actual_simulation_step_counter',
        'physical_steps_lower_bound','physical_steps_upper_bound','actual_integrated_environment_steps',
        'recorded_command_environment_steps','integrated_environment_steps_lower_bound','integrated_environment_steps_upper_bound'):
        value[key]=3024
    return value


class FullTests(unittest.TestCase):
    def test_valid_negative_stays_diagnostic_and_is_not_rejected_for_success(self):
        r=receipt();p=dict(command_steps=3024,prepared_template=r['prepared_template'])
        with patch.object(C.B,'verify'):
            self.assertIs(C.admit(r,{'output_dir':'/tmp/result'},p),r)
        self.assertIsNone(r['label']);self.assertEqual(r['raw_label'],0)

    def test_unknown_inflight_early_stop_or_training_label_is_rejected(self):
        original=receipt();p=dict(command_steps=3024,prepared_template=original['prepared_template'])
        changes=[('physical_steps_unknown',True),('actual_simulation_step_counter',3025),
            ('returned_physical_steps',3023),('label',1),('formal_training_eligible',True),
            ('original_gates',{'original':False}),('raw_label',True)]
        for key,value in changes:
            r=copy.deepcopy(original);r[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError),patch.object(C.B,'verify'):
                C.admit(r,{'output_dir':'/tmp/result'},p)

    def test_output_escape_rejected(self):
        r=receipt();p=dict(command_steps=3024,prepared_template=r['prepared_template'])
        with patch.object(C.B,'verify'),self.assertRaisesRegex(ValueError,'another output'):
            C.admit(r,{'output_dir':'/tmp/other'},p)

    def test_native_audit_compiles_original_checks_with_cpu_only_guard(self):
        class Bindings:
            def add(self,*args):pass
        def read(path):
            if Path(path).name=='evaluation.json':return native
            if Path(path).name=='native_structure.json':return {'is_fixed_base':True,'num_base_dofs':0}
            return {}
        native=dict(device='cpu',dt_s=.004,fixed_base=True,independent_coordinates=24,passive_mimics=4)
        auditor=A.native_state_auditor()
        with patch.object(A.A,'read',side_effect=read):
            # Passing the CPU-only check reaches the next unmodified material gate.
            with self.assertRaisesRegex(ValueError,'primary material'):
                auditor(Path('/tmp'),{}, {},A.S.np.zeros(3),Bindings())
            native['device']='cuda:0'
            with self.assertRaisesRegex(ValueError,'structure/device'):
                auditor(Path('/tmp'),{}, {},A.S.np.zeros(3),Bindings())

    def test_changed_native_auditor_pin_fails_closed(self):
        with patch.object(A.A,'sha',return_value='changed'),self.assertRaisesRegex(ValueError,'auditor changed'):
            A.native_state_auditor()

    def test_failed_prefix_cannot_freeze_full_followup(self):
        with patch.object(C.B,'read',return_value={'schema':C.PA.SCHEMA,'status':'failed'}),\
             self.assertRaisesRegex(ValueError,'prefix audit'):
            C.freeze('/tmp/never-written','/tmp/failed-prefix')


if __name__=='__main__':unittest.main()
