import contextlib,importlib.util,io,json,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('hand_top1_cli',Path(__file__).with_name('prepare_s40_hand_top1.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class HandTop1CLI(unittest.TestCase):
    def args(self):
        return ['--diagnostic','/tmp/diagnostic','--backend-config','/tmp/backend.json',
            '--calibration','/tmp/calibration.json','--native-preflight','/tmp/native.json',
            '--output-dir','/tmp/not_created_by_mock','--workers','4']
    def test_mock_run_prints_summary_and_returns_without_preparation_or_physics(self):
        result=dict(complete=True,objects=[{}]*8,trials=[{}]*5,nulls=[{}]*3,physical_steps=0)
        stream=io.StringIO()
        with patch.object(m.helper,'run',return_value=result)as run,contextlib.redirect_stdout(stream):
            returned=m.main(self.args())
        self.assertIs(returned,result);run.assert_called_once()
        args=run.call_args.args[0]
        self.assertEqual(args.workers,4);self.assertEqual(args.native_preflight,Path('/tmp/native.json'))
        self.assertEqual(json.loads(stream.getvalue()),dict(complete=True,objects=8,top1=5,nulls=3,physical_steps=0))
    def test_changed_helper_rejected_before_run(self):
        with patch.object(m,'HELPER_SHA256','0'*64),patch.object(m.helper,'run')as run:
            with self.assertRaisesRegex(ValueError,'helper changed'):m.main(self.args())
            run.assert_not_called()
    def test_invalid_worker_budget_rejected_before_run(self):
        with patch.object(m.helper,'run')as run,contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):m.main(self.args()[:-1]+['5'])
            run.assert_not_called()
if __name__=='__main__':unittest.main(verbosity=2)
