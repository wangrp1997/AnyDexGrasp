import hashlib,itertools,json,pathlib,time,traceback,torch
root=pathlib.Path(__file__).resolve().parents[2]/'outputs/native_deps'
root.mkdir(parents=True,exist_ok=True)
result={'torch':torch.__version__,'torch_cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),'checks':{},'physics_steps':0,'global_install':False}
try:
 import MinkowskiEngine as ME
 import pointnet2._ext as pointnet
 result['me_version']=ME.__version__;result['me_file']=ME.__file__;result['pointnet_file']=pointnet.__file__
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 coords=torch.tensor([[0,*x] for x in itertools.product(range(2),repeat=3)],dtype=torch.int32)
 features=torch.arange(16,dtype=torch.float32).reshape(8,2)/10
 cpu=ME.MinkowskiConvolution(2,3,kernel_size=3,stride=1,bias=True,dimension=3).eval()
 with torch.no_grad():
  cpu.kernel.copy_(torch.arange(cpu.kernel.numel(),dtype=torch.float32).reshape_as(cpu.kernel)/200)
  cpu.bias.copy_(torch.tensor([[0.1,0.2,0.3]]).reshape_as(cpu.bias))
  cpu_out=cpu(ME.SparseTensor(features,coordinates=coords))
  gpu=ME.MinkowskiConvolution(2,3,kernel_size=3,stride=1,bias=True,dimension=3).cuda().eval();gpu.load_state_dict(cpu.state_dict())
  t=time.perf_counter();gpu_out=gpu(ME.SparseTensor(features.cuda(),coordinates=coords,device='cuda'));torch.cuda.synchronize();elapsed=time.perf_counter()-t
  cpu_rows={tuple(c):f for c,f in zip(cpu_out.C.tolist(),cpu_out.F)};gpu_rows={tuple(c):f for c,f in zip(gpu_out.C.cpu().tolist(),gpu_out.F.cpu())}
  assert set(cpu_rows)==set(gpu_rows)
  error=max(float((cpu_rows[c]-gpu_rows[c]).abs().max()) for c in cpu_rows)
  assert error<2e-5 and torch.isfinite(gpu_out.F).all()
 result['checks']['sparse_conv_3x3_cpu_gpu']={'passed':True,'rows':len(gpu_rows),'max_abs_error':error,'cuda_seconds':elapsed}
 xyz=torch.tensor([[[0,0,0],[1,0,0],[0,2,0],[0,0,3]]],dtype=torch.float32,device='cuda')
 fps=pointnet.furthest_point_sampling(xyz,3).cpu().tolist();assert fps==[[0,3,2]],fps
 result['checks']['fps']={'passed':True,'indices':fps}
 xyz=torch.tensor([[[0,0,0],[0.01,0.02,0],[0.02,0,0],[0,0.08,0],[-0.03,0,0]]],dtype=torch.float32,device='cuda')
 center=torch.zeros((1,1,3),dtype=torch.float32,device='cuda');rot=torch.eye(3,device='cuda').reshape(1,1,9)
 idx=pointnet.cylinder_query(center,xyz,rot,0.05,-0.02,0.04,4)
 got=idx.cpu().tolist();assert got==[[[0,1,2,0]]],got
 values=torch.arange(10,dtype=torch.float32,device='cuda').reshape(1,2,5)
 grouped=pointnet.group_points(values,idx);expected=values[:,:,idx[0,0].long()].reshape_as(grouped)
 assert torch.equal(grouped,expected)
 result['checks']['cylinder_and_grouping']={'passed':True,'indices':got,'grouped':grouped.cpu().tolist()}
 result['all_passed']=True
except Exception as e:
 result.update(all_passed=False,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
(root/'native_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if not result['all_passed']:raise SystemExit(1)
