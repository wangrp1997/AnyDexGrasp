import unittest
import numpy as np
from types import SimpleNamespace
import test_author_mjx_live_batch as fixture
from author_mjx_live_batch_v2 import decode_contacts


class AllocationTests(unittest.TestCase):
    def test_unallocated_optional_rigid_model_fields_are_explicit_and_not_fabricated(self):
        p=fixture.NativePacketTests().packet();p['contact_elem']=np.empty((0,2),dtype=np.int32)
        m=SimpleNamespace(ngeom=2,geom_bodyid=np.array([0,1]))
        rows=decode_contacts(p,m,{0:'a',1:'b'},2)
        self.assertIsNone(rows[0][0]['native_fields']['elem'])
        self.assertEqual(rows[0][0]['native_fields_unallocated'],['elem'])
        self.assertEqual(rows[1][0]['normal_force_n'],5.)

    def test_required_fields_cannot_be_silently_marked_unallocated(self):
        p=fixture.NativePacketTests().packet();p['contact_geom']=np.empty((0,2),dtype=np.int32)
        with self.assertRaises(RuntimeError):decode_contacts(p,SimpleNamespace(ngeom=2),{},2)


if __name__=='__main__':unittest.main()
