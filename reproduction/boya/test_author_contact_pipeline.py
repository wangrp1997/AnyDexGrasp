"""Contract regressions for the opt-in mapping; no simulator or training."""
import copy
import unittest
import numpy as np
import author_contact_pipeline as P


class ContactPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old=P.BASE_CONTRACT();m=P.FiveFingerWidthMapper(old)
        row=np.r_[.5,.08386754989624023,.03,np.eye(3).ravel(),[0,0,.2]]
        mapped=m.align(row[None],np.eye(4));opening=old.slots['g1:distal_000:ff']
        cls.cal=copy.deepcopy(old.calibration);cls.ids=[];cls.row=row
        for j,opponent in enumerate(P.OPPONENTS):
            slot=copy.deepcopy(opening);slot.update(slot_id='g1:test:'+opponent,mapping_variant=P.SCHEMA,
                source_surface_slot_id=opening['slot_id'],source_opening_palm_T_reference=opening['palm_T_reference'],
                source_opening_reference_width_m=opening['reference_width_m'],global_preclosure_progress=float(mapped.progress[0,j]),
                preclosure_q=mapped.q_contact[0,j].tolist(),palm_T_reference=mapped.palm_T_contact_reference[0,j].tolist(),
                reference_width_m=float(mapped.width_m[0,j]),opposing_group=opponent)
            cls.cal['slots'].append(slot);cls.ids.append(slot['slot_id'])
        cls.cal['contact_mapping']={'schema':P.SCHEMA};cls.contract=P.contract_factory(cls.cal)

    def test_all_opponents_bind_closed_reference_and_raw_width(self):
        for sid in self.ids:
            c=self.contract;s=c.slots[sid];r,w=P.reference_from_pads(c,s,s['preclosure_q'])
            np.testing.assert_allclose(r,s['palm_T_reference'],atol=2e-14)
            self.assertAlmostEqual(w,self.row[1],places=8)
            self.assertGreater(np.max(abs(r-s['source_opening_palm_T_reference'])),.001)

    def test_execution_continues_to_company_endpoint(self):
        c=self.contract;s=c.slots[self.ids[0]];cmd=c.closure_targets(self.ids[0],[0.,1.])
        np.testing.assert_array_equal(cmd['independent_target'][1],s['close_q'])
        self.assertGreater(np.max(abs(cmd['independent_target'][1]-s['preclosure_q'])),.1)
        action=c.for_slot(self.ids[0]);self.assertFalse(action['reference_is_execution_stop'])
        self.assertTrue(action['thumb_required']);self.assertTrue(action['other_fingers_optional'])

    def test_wrong_reference_and_hand_state_rejected(self):
        for key in ('palm_T_reference','preclosure_q'):
            cal=copy.deepcopy(self.cal);s=cal['slots'][-1]
            if key=='palm_T_reference':s[key][0][3]+=.002
            else:s[key][0]+=.002
            with self.assertRaises(ValueError):P.contract_factory(cal)

    def test_opt_in_scope_restores_original_collector(self):
        old=P.C.detect
        try:
            with P.installed():
                self.assertIs(P.A.ActionContract,P.contract_factory)
                raise RuntimeError('intentional')
        except RuntimeError:pass
        self.assertIs(P.A.ActionContract,P.BASE_CONTRACT);self.assertIs(P.C.detect,old)
        self.assertIs(type(P.contract_factory()),P.BASE_CONTRACT)

    def test_new_reference_is_used_to_compute_palm(self):
        c=self.contract
        for sid in self.ids:
            palm=c.mapped_palm(sid,self.row,np.eye(4),extra_depth_m=.01)
            target=palm@np.asarray(c.slots[sid]['palm_T_reference'])
            expected=np.eye(4);expected[:3,:3]=P.CGR_R_REFERENCE;expected[:3,3]=[.04,0,.2]
            np.testing.assert_allclose(target,expected,atol=2e-14)


if __name__=='__main__':unittest.main()
