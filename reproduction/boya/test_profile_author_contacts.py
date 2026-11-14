"""Offline codec invariants; no simulator, planner, dataset or model calls."""
import copy
import unittest

try:
    from .profile_author_contacts import pack_columns, unpack_columns, exact_rows
except ImportError:
    from profile_author_contacts import pack_columns, unpack_columns, exact_rows


class ContactColumnTests(unittest.TestCase):
    def rows(self):
        a = dict(bodies=['/object', '/ground'], position_world_m=[-0., 1.25, 0.],
            raw_normal_world=[0., 0., 1.], separation_m=.000125, normal_force_n=0.)
        b = dict(bodies=['/robot/a', '/robot/b'], position_world_m=[1., 2., 3.],
            separation_m=-.00025, normal_force_n=1.2345678901234567)
        return [dict(step=1, object_contacts=[a, copy.deepcopy(a)], robot_contacts=[b]),
                dict(step=2, object_contacts=[], robot_contacts=[dict(b,bodies=list(reversed(b['bodies'])))])]

    def test_all_fields_order_optional_normals_and_signed_zero(self):
        rows = self.rows(); restored = unpack_columns(pack_columns(rows))
        self.assertEqual(len(restored[0]['object_contacts']), 2)
        self.assertNotIn('raw_normal_world', restored[0]['robot_contacts'][0])
        self.assertEqual(exact_rows(rows, restored), exact_rows(rows, rows))

    def test_unknown_field_is_rejected(self):
        rows = self.rows(); rows[0]['object_contacts'][0]['new_native_field'] = 4.
        with self.assertRaises(ValueError): pack_columns(rows)

    def test_roundtrip_checker_rejects_changed_contact_or_zero_sign(self):
        rows = self.rows(); changed = copy.deepcopy(rows)
        changed[0]['object_contacts'][0]['position_world_m'][0] = 0.
        with self.assertRaises(AssertionError): exact_rows(rows, changed)
        changed = copy.deepcopy(rows); changed[1]['robot_contacts'][0]['normal_force_n'] += .01
        with self.assertRaises(AssertionError): exact_rows(rows, changed)

    def test_all_empty_ticks_preserved(self):
        rows = [dict(step=1, object_contacts=[], robot_contacts=[]),
                dict(step=2, object_contacts=[], robot_contacts=[])]
        self.assertEqual(exact_rows(rows, rows), exact_rows(rows, unpack_columns(pack_columns(rows))))


if __name__ == '__main__': unittest.main()
