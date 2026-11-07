"""Targeted draft regression checks; read-only saved pilot artifact reuse only."""
import ast,copy,hashlib,importlib.util,json,sys,types,unittest
from pathlib import Path
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');BOYA=ROOT/'reproduction/boya';sys.path.insert(0,str(BOYA))
PILOT=ROOT/'outputs/s40_alignment_pilot_20261106_v1';SOURCE=Path(__file__).with_name('run_s40_collection.py')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(mapping):
 for path,digest in mapping.items():
  if sha(path)!=digest:raise ValueError('artifact SHA differs')
node=next(n for n in ast.walk(ast.parse(SOURCE.read_text()))if isinstance(n,ast.FunctionDef)and n.name=='add_record')
def get_fixture():
 rows=[];values=[]
 for oid in ['graspnet_000','graspnet_002']:
  d=read(PILOT/'execution'/oid/'RESULTS.json')
  for row in d['trials']:
   rows.append(row);values.append(read(PILOT/'execution'/oid/'trials'/row['output_identity']/'training_record.json'))
 return rows,values
def export_function(rows):
 out=[];byid={r['trial_id']:dict(object_id=r['object_id'],gesture_id=r['gesture_id'],extra_depth_index=r['extra_depth_index'])for r in rows}
 context=dict(copy=copy,rows=rows,records=out,byid=byid,verify=verify,sha=lambda p:'test-new-protocol',self=types.SimpleNamespace(run=Path('/tmp'),inherited={'inherited_original_protocol_sha256':sha(PILOT/'PILOT_PROTOCOL.json')}))
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),context)
 return context['add_record'],out
spec=importlib.util.spec_from_file_location('draft_s40_outcomes',Path(__file__).with_name('s40_execution_outcomes.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def row(tid,n=2100,label=1):return dict(trial_id=tid,object_id='o',physics_steps=n,label=label,label_eligible=True)
def doc(rows):return dict(status='complete',physical_steps=sum(r['physics_steps']for r in rows),cohorts=[dict(environment_steps=r['physics_steps'])for r in rows],input_sha256_unchanged=True,source_sha256_unchanged=True,trials=rows)
class DraftFixTests(unittest.TestCase):
 def test_five_saved_pilot_records_reexport_without_physics(self):
  rows,values=get_fixture();add,out=export_function(rows)
  for r,v in zip(rows,values):add(v,r['trial_id'],True)
  self.assertEqual(len(out),5);self.assertEqual([v['result']for v in out],[r['label']for r in rows]);self.assertTrue(all(v['collection_context']['inherited_from_pilot']for v in out))
 def test_wrong_binding_label_steps_bool_and_sha_rejected(self):
  rows,values=get_fixture();base=values[0]
  mutations=[lambda v:v['binding'].update(trial_binding_sha256='foreign'),lambda v:v.update(result=0),lambda v:v.update(total_physics_steps=19),lambda v:v.update(result=True),lambda v:v['execution_artifacts'].update(commands_sha256='bad')]
  for mutate in mutations:
   with self.subTest(mutation=mutate):
    value=copy.deepcopy(base);mutate(value);add,out=export_function(rows)
    with self.assertRaises(ValueError):add(value,rows[0]['trial_id'],True)
    self.assertEqual(out,[])
 def test_same_gesture_other_insertion_record_rejected(self):
  rows,values=get_fixture();r=copy.deepcopy(rows[0]);r['extra_depth_index']=1;add,out=export_function([r])
  with self.assertRaises(ValueError):add(values[0],r['trial_id'],True)
 def inspect(self,value):
  # Isolate wrapper accounting from its original artifact verifier for unit cases.
  from unittest.mock import patch
  fake_v4=types.SimpleNamespace(inspect_rows=lambda rows,ids:{})
  fake_terminal=types.SimpleNamespace(certify_known_task_stop=lambda r:(_ for _ in()).throw(ValueError('unexpected terminal')))
  with patch.dict(sys.modules,{'resume_collection_execution_v4':fake_v4,'terminal_collection_outcome':fake_terminal}):return mod.inspect_execution(value,[r['trial_id']for r in value['trials']],0)
 def test_aggregate_ledger_mismatch_rejected(self):
  d=doc([row('a')]);d['physical_steps']=999
  with self.assertRaises(ValueError):self.inspect(d)
 def test_cohort_ledger_mismatch_rejected(self):
  d=doc([row('a')]);d['cohorts'][0]['environment_steps']=999
  with self.assertRaises(ValueError):self.inspect(d)
 def test_multiple_cohorts_and_extended_plan_length_retained(self):
  d=doc([row('a'),row('b',2350,0)]);r=self.inspect(d);self.assertEqual(r['known_recorded_physics_steps'],4450);self.assertEqual(r['negatives'],1);self.assertTrue(r['complete'])
 def test_internal32_8_all_original_training_meshes_disjoint(self):
  rows=read(PILOT/'scenes/objects.json')['objects'];order=sorted(rows,key=lambda r:hashlib.sha256(('s40-internal-20261107:'+r['source_visual_mesh_sha256']).encode()).hexdigest());fit=order[:32];internal=order[32:]
  self.assertEqual((len(fit),len(internal)),(32,8));self.assertEqual(len({r['object_id']for r in fit+internal}),40);self.assertFalse({r['source_visual_mesh_sha256']for r in fit}&{r['source_visual_mesh_sha256']for r in internal});self.assertTrue(all(r['split']=='train'for r in fit+internal))
if __name__=='__main__':unittest.main(verbosity=2)
