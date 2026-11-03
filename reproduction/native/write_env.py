"""Write a process-local loader for the existing Python/Torch runtime."""
from pathlib import Path
import shlex,torch
repo=Path(__file__).resolve().parents[2];native=repo/'outputs/native_deps';native.mkdir(parents=True,exist_ok=True)
torch_lib=Path(torch.__file__).resolve().parent/'lib'
text='''# Source only in this independent reproduction process.
ANYDEX_NATIVE={native}
export CUDA_HOME="$ANYDEX_NATIVE/cuda-13.0"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$ANYDEX_NATIVE/openblas/lib:$CUDA_HOME/lib64:{torch_lib}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
export PYTHONPATH="$ANYDEX_NATIVE/build_src/MinkowskiEngine:$ANYDEX_NATIVE/pointnet2/lib${{PYTHONPATH:+:$PYTHONPATH}}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST=12.0
'''.format(native=shlex.quote(str(native)),torch_lib=str(torch_lib))
(native/'env.sh').write_text(text)
print(native/'env.sh')
