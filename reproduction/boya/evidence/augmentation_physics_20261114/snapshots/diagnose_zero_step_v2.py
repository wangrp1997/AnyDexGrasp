from pathlib import Path
import sys,json,traceback,os
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');sys.path[:0]=[str(ROOT/'reproduction/boya'),'/home/wangrenpeng/CanonDex','/home/wangrenpeng/CanonDex/scripts/research']
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True,device='cpu',visualizer=None,kit_args='--/log/file=/dev/null')
code=1
try:
 import run_boya_floating_diagnostic as e
 out=Path(__file__).resolve().parent/'zero_step_diagnostic_v2';out.mkdir()
 job=e.read(Path(__file__).resolve().parent/'physical_history_v2/jobs/egad_A3_augmented11_original_mse.json');job['output_dir']=str(out)
 try:
  rt=e.build_scene(job,out)
  def block_step(*args,**kwargs):raise RuntimeError('ZERO_STEP_BOUNDARY: integration prohibited by diagnostic')
  rt['sim'].step=block_step
  e.build_scene=lambda *args,**kwargs:rt
  e.run(job)
 except BaseException as exc:
  tb=traceback.format_exc();print(tb,flush=True);e.save(out/'FAILURE.json',dict(error=repr(exc),traceback=tb,physics_steps=0,scope='build_scene only; executor run never called'))
finally:launcher.app.close(exit_code=code)
