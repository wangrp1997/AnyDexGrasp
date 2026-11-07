"""Meaningful contracts for the new explicitly arm-free diagnostic."""
from pathlib import Path
import ast,importlib.util,tempfile,unittest,xml.etree.ElementTree as ET
import numpy as np
B=Path('/home/wangrenpeng/AnyDexGrasp/reproduction/boya')
R=B.parent.parent/'outputs/egad_external12_20261107_v1'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
runner=load('floating_runner',Path('/tmp/anydex_float_boya_runner.py'))
coordinator=load('floating_three',Path('/tmp/run_egad_floating_three.py'))
class Contracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not (R/'objects.json').exists():raise unittest.SkipTest('Local source-bound experiment inputs absent')
  cls.cases=coordinator.source_cases();cls.backend=coordinator.read(B/'backend.example.json')
 def inputs(self,case):return runner.load_inputs(dict(case,**{k:self.backend[k]for k in ['hand_urdf','hand_scene','package']}))
 def test_selected_candidates_remain_exact(self):
  self.assertEqual([(c['object_id'],c['candidate_rank'])for c in self.cases],[('egad_A6',17),('egad_A3',58),('egad_E0',4)])
 def test_hand_closure_matches_original_float32_arrays(self):
  for case in self.cases[1:]:
   seq=runner.command_sequence(self.inputs(case))
   with np.load(Path(case['old_selected_plan'])/'commands.npz')as old:
    names=list(old['joint_names']);handnames=list(self.inputs(case)['family'].hand.names)
    for ph in ['approach','close','hold','lift']:
     np.testing.assert_array_equal(seq['motor_target'][seq['phase']==ph].astype(np.float32),old['joint_target'][old['phase']==ph][:,[names.index(n)for n in handnames]].astype(np.float32))
 def test_palm_approach_lift_match_original_targets(self):
  for case in self.cases[1:]:
   seq=runner.command_sequence(self.inputs(case))
   with np.load(Path(case['old_selected_plan'])/'commands.npz')as old:
    for ph in ['approach','close','hold','lift']:
     np.testing.assert_allclose(seq['desired_palm'][seq['phase']==ph],old['desired_palm'][old['phase']==ph],atol=1e-14,rtol=0)
 def test_gantry_world_coordinates_and_2100_steps(self):
  for case in self.cases:
   seq=runner.command_sequence(self.inputs(case));self.assertEqual(len(seq['phase']),2100)
   pre=seq['pregrasp_palm'];xyz=(seq['desired_palm'][:,:3,3]-pre[:3,3])@pre[:3,:3]
   reconstructed=xyz@pre[:3,:3].T+pre[:3,3]
   np.testing.assert_allclose(reconstructed,seq['desired_palm'][:,:3,3],atol=1e-14,rtol=0)
   endhold=seq['desired_palm'][seq['phase']=='hold'][-1];end=seq['desired_palm'][-1]
   np.testing.assert_allclose(end[:3,3]-endhold[:3,3],[0,0,.1],atol=1e-14,rtol=0)
 def test_original_link_inertias_joints_and_meshes_preserved(self):
  hand=self.inputs(self.cases[0])['hand'];source=ET.parse(hand.urdf).getroot()
  with tempfile.TemporaryDirectory(prefix='boya_gantry_')as tmp:
   p=runner.write_gantry_urdf(tmp,hand);new=ET.parse(p).getroot()
   for name in hand.links:
    a=source.find(f"link[@name='{name}']");b=new.find(f"link[@name='{name}']")
    self.assertEqual(None if a.find('inertial') is None else ET.tostring(a.find('inertial')),None if b.find('inertial') is None else ET.tostring(b.find('inertial')))
    for mesh in b.findall('.//mesh'):self.assertTrue(Path(mesh.get('filename')).is_file())
   for j in source.findall('joint'):
    if j.find('parent').get('link')in hand.links:self.assertEqual(ET.tostring(j),ET.tostring(new.find(f"joint[@name='{j.get('name')}']")))
   self.assertEqual([j.get('name')for j in new.findall('joint')if j.get('type')=='prismatic'],runner.GANTRY_NAMES)
 def test_no_state_writes_in_physical_loop(self):
  tree=ast.parse((Path('/tmp/anydex_float_boya_runner.py')).read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run')
  loop=next(n for n in ast.walk(run)if isinstance(n,ast.For)and any(isinstance(c,ast.Call)and isinstance(c.func,ast.Attribute)and c.func.attr=='step'for c in ast.walk(n)))
  attrs={n.func.attr for n in ast.walk(loop)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)}
  self.assertFalse(attrs&{'write_root_link_pose_to_sim_index','write_root_link_velocity_to_sim_index','write_joint_state_to_sim'})
if __name__=='__main__':unittest.main()
