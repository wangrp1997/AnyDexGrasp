import copy,importlib.util,unittest
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('control_entry',Path(__file__).with_name('evaluate_s40_head_control.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ControlEntryTests(unittest.TestCase):
 def pair(self):
  b=dict(schema='s',gesture_id=6,stage='internal_check',epoch=100,input_dim=480,output_dim=20,
   architecture_kwargs={},training_config={},dataset_manifest_sha256='dataset',fit_mesh_sha256=['fit'],fit_trial_ids=['a'],
   model_state_dict={'conv.weight':torch.ones(2,3),'bn.running_var':torch.ones(2)})
  n=copy.deepcopy(b);n['training_variant']=m.VARIANT;n['model_state_dict']['conv.weight']+=1
  return n,b
 def test_weight_changes_allowed_with_same_structure_and_declared_loss(self):
  n,b=self.pair();m.validate_checkpoint(n,b,6,{'fit'},{'check'})
 def test_fit_or_trial_identity_change_rejected(self):
  for key,value in [('fit_mesh_sha256',['check']),('fit_trial_ids',['internal'])]:
   n,b=self.pair();n[key]=value
   with self.assertRaises(ValueError):m.validate_checkpoint(n,b,6,{'fit'},{'check'})
 def test_missing_variant_or_shape_change_rejected(self):
  n,b=self.pair();n['training_variant']='old'
  with self.assertRaises(ValueError):m.validate_checkpoint(n,b,6,{'fit'},{'check'})
  n,b=self.pair();n['model_state_dict']['conv.weight']=torch.ones(3,3)
  with self.assertRaises(ValueError):m.validate_checkpoint(n,b,6,{'fit'},{'check'})
 def test_nonfinite_or_internal_fit_rejected(self):
  n,b=self.pair();n['model_state_dict']['conv.weight'][0,0]=float('nan')
  with self.assertRaises(ValueError):m.validate_checkpoint(n,b,6,{'fit'},{'check'})
  n,b=self.pair()
  with self.assertRaises(ValueError):m.validate_checkpoint(n,b,6,{'fit'},{'fit'})
if __name__=='__main__':unittest.main()
