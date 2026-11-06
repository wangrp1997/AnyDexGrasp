"""Meaningful format conversion and source-space support invariants, no physics."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import trimesh
source=Path(__file__).with_name('prepare_s40_scenes.py')
spec=importlib.util.spec_from_file_location('s40_scene_builder',source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SceneChecks(unittest.TestCase):
    def test_source_triangle_roundtrip_without_dgn_scale_or_simplification(self):
        a=trimesh.creation.icosphere(subdivisions=1,radius=.073)
        a.vertices += [.012,.023,-.027]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'simplified.obj'
            p.write_text(trimesh.exchange.obj.export_obj(a,include_normals=False,include_color=False,include_texture=False,digits=16))
            b=trimesh.load_mesh(p,process=False)
            self.assertLess(m.verify_mesh_conversion(a,b),1e-12)
            wrong=b.copy();wrong.apply_scale(.1)
            with self.assertRaises(ValueError):m.verify_mesh_conversion(a,wrong)
            wrong=b.copy();wrong.faces=wrong.faces[:-1]
            with self.assertRaises(ValueError):m.verify_mesh_conversion(a,wrong)

    def test_collision_proxy_supported_without_visual_or_collision_table_penetration(self):
        a=np.array([[-.03,-.04,-.08],[.04,.07,.12],[.02,-.04,.04]])
        b=np.array([[-.031,-.041,-.082],[.041,.071,.122],[.02,-.04,.04]])
        T=m.placement(a,b)
        np.testing.assert_array_equal(T[:3,:3],np.eye(3))
        w=a+T[:3,3];c=b+T[:3,3]
        self.assertGreater(w[:,2].min(),0.)
        self.assertAlmostEqual(c[:,2].min(),0.)
        np.testing.assert_allclose((w[:,:2].min(0)+w[:,:2].max(0))/2,[-.75,.1])
        np.testing.assert_allclose(w.ptp(axis=0) if hasattr(w,'ptp') else np.ptp(w,axis=0),np.ptp(a,axis=0))

    def test_visual_supported_if_convex_proxy_is_smaller(self):
        a=np.array([[-1.,-2.,-.1],[1.,2.,.2]])
        b=np.array([[-.9,-1.9,-.09],[.9,1.9,.19]])
        T=m.placement(a,b)
        self.assertAlmostEqual((a+T[:3,3])[:,2].min(),0.)
        self.assertGreater((b+T[:3,3])[:,2].min(),0.)

    def test_nonfinite_empty_or_unaligned_id_inventory_rejected(self):
        for bad in [[],[[float('nan'),0.,0.]],[[1.,2.]]]:
            with self.assertRaises(ValueError):m.placement(bad,[[0.,0.,0.]])
        with self.assertRaises(ValueError):m.validate_inventory({'schema':'old_DGN_4330'})
        with self.assertRaises(ValueError):m.validate_inventory(dict(schema='anydex_graspnet_s40_asset_inventory_v1',object_ids=[0]*40,objects=[dict(graspnet_object_id=0)]*40))

if __name__=='__main__':unittest.main(verbosity=2)
