"""Two focused contracts for uniform offline V-HACD preparation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import trimesh
import numpy as np
SOURCE=Path(__file__).with_name('prepare_egad_external_scenes_v2.py')
spec=importlib.util.spec_from_file_location('egad_vhacd_asset_builder',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class VhacdChecks(unittest.TestCase):
    def test_grouped_vhacd_obj_is_split_without_filling_with_one_hull(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'parts.obj'
            p.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 2\nv 1 0 2\nv 0 1 2\no first\nf 1 2 3\no second\nf 4 5 6\n')
            parts=m.fallback.split_obj(p)
            self.assertEqual(len(parts),2)
            np.testing.assert_array_equal(parts[0][1],[[0,1,2]])
            np.testing.assert_array_equal(parts[1][0],[[0,0,2],[1,0,2],[0,1,2]])

    def test_new_metadata_records_actual_vhacd_and_descriptive_errors(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);source=root/'native.obj'
            mesh=trimesh.creation.box(extents=[80.,60.,40.]);mesh.vertices += [40.,30.,20.]
            source.write_text(trimesh.exchange.obj.export_obj(mesh,digits=16))
            job=dict(object_id='egad_A0',source=dict(object_id='A0',path=str(source),sha256=m.base.sha(source)),
                object_root=str(root/'assets'/'egad_A0'),inventory_sha256='a'*64,run_dir=str(root/'run'),
                exclusion=dict(source_hashes=[],meter_geometry_hashes=[]))
            jobfile=root/'JOB.json';m.base.save(jobfile,job)
            sys.path.insert(0,str(m.ROOT/'outputs/native_deps/python_overlay'));import pybullet
            def passthrough(src,out,log,**kw):
                self.assertEqual(kw,m.PARAMETERS)
                Path(out).write_bytes(Path(src).read_bytes());Path(log).write_text('mock decomposition only')
            with mock.patch.object(pybullet,'vhacd',side_effect=passthrough):
                self.assertEqual(m.worker(jobfile),0)
            result=json.loads((root/'RESULT.json').read_text());geo=result['record']['geometry_approximation']
            self.assertEqual(geo['decomposition_method'],'PyBullet standalone V-HACD')
            self.assertNotIn('coacd_parameters',geo)
            self.assertNotIn('coacd_runtime',geo)
            self.assertEqual(geo['decomposition_parameters'],m.PARAMETERS)
            self.assertTrue(geo['approximation_checks']['AABB_pass'])
            self.assertEqual(geo['approximation_checks']['concavity_observation']['ray_count'],54)
            self.assertFalse(geo['approximation_checks']['one_convex_hull_substitution'])
            self.assertEqual((result['physics_steps'],result['network_forwards'],result['training_steps']),(0,0,0))

if __name__=='__main__':unittest.main(verbosity=2)
