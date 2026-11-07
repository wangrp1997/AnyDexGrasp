from pathlib import Path
import importlib.util,unittest,tempfile
from types import SimpleNamespace
import numpy as np
p=Path(__file__).with_name('settle_s40_scenes.py');spec=importlib.util.spec_from_file_location('settle_s40',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class StabilityTests(unittest.TestCase):
 def fixture(self):
  p=np.zeros((300,7));p[:,6]=1;v=np.zeros((300,6));return p,v
 def test_rest_accepts_and_sign_invariance(self):
  p,v=self.fixture();p[::2,3:]*=-1;self.assertTrue(m.window_metrics(p,v)['stable'])
 def test_short_window_does_not_accept(self):
  p,v=self.fixture();self.assertFalse(m.window_metrics(p[:100],v[:100])['stable'])
 def test_rolling_endpoint_or_speed_cannot_hide(self):
  p,v=self.fixture();p[:,0]=.005*np.sin(np.linspace(0,2*np.pi,len(p)));self.assertFalse(m.window_metrics(p,v)['stable'])
  p,v=self.fixture();v[-100,5]=.1;self.assertTrue(m.window_metrics(p,v)['stable'])
  v[-100,5]=1.;self.assertFalse(m.window_metrics(p,v)['stable'])
 def test_slow_drift_or_rotation_rejected(self):
  p,v=self.fixture();p[:,0]=np.linspace(0,.002,len(p));self.assertFalse(m.window_metrics(p,v)['stable'])
  p,v=self.fixture();a=np.linspace(0,.01,len(p));p[:,5]=np.sin(a/2);p[:,6]=np.cos(a/2);self.assertFalse(m.window_metrics(p,v)['stable'])
 def test_nonfinite_and_bad_quaternion_fail(self):
  p,v=self.fixture();p[-1,0]=np.nan
  with self.assertRaises(ValueError):m.window_metrics(p,v)
  p,v=self.fixture();p[-1,6]=2
  with self.assertRaises(ValueError):m.window_metrics(p,v)
 def test_namespacing_only_changes_exact_object_prefix(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'source.py';s="def spawn_mesh_object():\n return ('/World/Object/visual','/World/ObjectMaterial',.98)\ndef spawn_object():\n return ('/World/Object',spawn_mesh_object(),'/World/Robot')\n";p.write_text(s)
   f=m.namespaced_spawn(SimpleNamespace(__file__=str(p)),'/World/Settle/test')
   self.assertEqual(f(),('/World/Settle/test',('/World/Settle/test/visual','/World/Settle/testMaterial',.98),'/World/Robot'));self.assertEqual(p.read_text(),s)
if __name__=='__main__':unittest.main()
