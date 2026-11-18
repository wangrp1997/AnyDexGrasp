import unittest
from types import SimpleNamespace
import numpy as np
from author_mjx_live_batch import CONTACT_FIELDS,clock_table,decode_contacts


class NativePacketTests(unittest.TestCase):
    def packet(self):
        p={}
        for key in CONTACT_FIELDS:p['contact_'+key]=np.zeros((2,),dtype=np.int32)
        p.update(nacon=np.array([2]),overflow=np.zeros((2,1)),efc_force=np.array([[2.,.1,.2,.3],[5.,.4,.5,.6]]),
            contact_geom=np.array([[0,1],[0,1]]),contact_worldid=np.array([0,1]),contact_dim=np.array([4,4]),
            contact_efc_address=np.array([[0,-1],[0,-1]]),contact_dist=np.array([-.0001,-.0002]))
        for key,size in (('pos',3),('frame',9),('friction',5),('solref',2),('solimp',5),('solreffriction',2)):
            p['contact_'+key]=np.ones((2,size))
        return p

    def test_native_force_uses_its_world_and_every_contact_field_is_preserved(self):
        p=self.packet();m=SimpleNamespace(ngeom=2,geom_bodyid=np.array([0,1]))
        rows=decode_contacts(p,m,{0:'/World/ground',1:'/World/item'},2)
        self.assertEqual([x[0]['normal_force_n']for x in rows],[2.,5.])
        self.assertEqual(rows[1][0]['wrench_contact_frame'],[5.,.4,.5,.6,0.,0.])
        self.assertEqual(set(rows[0][0]['native_fields']),set(CONTACT_FIELDS))

    def test_overflow_or_wrong_world_cannot_become_a_valid_negative(self):
        p=self.packet();p['overflow'][1,0]=1;m=SimpleNamespace(ngeom=2,geom_bodyid=np.array([0,1]))
        with self.assertRaises(RuntimeError):decode_contacts(p,m,{0:'a',1:'b'},2)
        p=self.packet();p['contact_worldid'][0]=2
        with self.assertRaises(ValueError):decode_contacts(p,m,{0:'a',1:'b'},2)

    def test_float32_native_clock_counts_each_substep_and_detects_a_missing_one(self):
        table=clock_table(10000);actual=np.float32(0)
        for _ in range(8*10000):actual=np.float32(actual+np.float32(.0005))
        self.assertEqual(table[-1],actual)
        self.assertNotEqual(table[-1],np.float32(actual-np.float32(.0005)))
        self.assertNotEqual(float(table[-1]),40.)


if __name__=='__main__':unittest.main()
