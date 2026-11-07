"""Paired contact-geometry alignment at frozen Boya candidates.

Four objects, one fixed higher-resolution V-HACD recipe selected before results.
A0 coarse+fine is an executor positive control; A3/A6/E0 fine compare with the
already saved coarse gantry trials. No changed hand, pose, model, mass or force.
This is a simulation adapter diagnosis, never an AnyDex paper success rate.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, json, os, shutil, subprocess, sys, time
ROOT=Path('/home/wangrenpeng/AnyDexGrasp'); B=ROOT/'reproduction/boya'
sys.path.insert(0,str(B))
import numpy as np
import trimesh
import run_egad_floating_three as original
from prepare_egad_external_scenes import export_urdf
from prepare_egad_external_scenes_v2 import fallback
from prepare_egad_external_scenes_v3 import normalize_component
read=original.read;save=original.save;sha=original.sha
BASE=original.BASE
IDS=('egad_A0','egad_A3','egad_A6','egad_E0')
PARAMETERS=dict(fallback.PARAMETERS, resolution=2000000, concavity=.0005,
    maxNumVerticesPerCH=128,planeDownsampling=2,convexhullDownsampling=2)
TIMEOUT_S=180

def metric(source,parts):
    compound=trimesh.util.concatenate(parts)
    sv=np.asarray(source.vertices);pv=np.asarray(compound.vertices)
    # Finite deterministic measurements, not a union-Hausdorff certificate.
    sv=sv[np.linspace(0,len(sv)-1,min(1024,len(sv)),dtype=int)]
    pv=pv[np.linspace(0,len(pv)-1,min(1024,len(pv)),dtype=int)]
    a=trimesh.proximity.closest_point(compound,sv)[1]
    b=trimesh.proximity.closest_point(source,pv)[1]
    return dict(source_to_component_surface_max_m=float(a.max()),source_to_component_surface_p95_m=float(np.quantile(a,.95)),
        component_vertices_to_source_max_m=float(b.max()),component_vertices_to_source_p95_m=float(np.quantile(b,.95)),
        source_samples=len(sv),component_vertex_samples=len(pv),parts=len(parts),
        AABB_max_error_m=float(np.abs(source.bounds-compound.bounds).max()),
        limitation='Finite unsigned nearest-triangle samples, includes internal component surfaces; not signed union or global Hausdorff bound')

def decompose(jobpath):
    sys.path.insert(0,str(ROOT/'outputs/native_deps/python_overlay'))
    import pybullet as p
    def forbid(*a,**kw):raise RuntimeError('Offline decomposition forbids physics/client')
    p.connect=forbid;p.stepSimulation=forbid
    job=read(jobpath);root=Path(job['asset_root']);start=time.monotonic()
    for sub in ('mesh','urdf/meshes','info'):(root/sub).mkdir(parents=True,exist_ok=False)
    source=Path(job['source_row']['mesh_path'])
    shutil.copyfile(source,root/'mesh/simplified.obj')
    assert sha(source)==sha(root/'mesh/simplified.obj')
    raw=root/'vhacd_raw.obj';log=root/'vhacd.log'
    p.vhacd(str(source),str(raw),str(log),**PARAMETERS)
    parts=[];records=[];normalization=[]
    for i,(v,f) in enumerate(fallback.split_obj(raw)):
        (v,f),note=normalize_component(v,f)
        mesh=trimesh.Trimesh(v,f,process=False);path=root/'urdf/meshes'/f'part_{i:04d}.obj'
        path.write_text(trimesh.exchange.obj.export_obj(mesh,include_normals=False,include_color=False,include_texture=False,digits=16))
        parts.append(mesh);normalization.append(note)
        records.append(dict(path=str(path),sha256=sha(path),scale='1 1 1',vertices=len(v),triangles=len(f),volume_m3=float(abs(mesh.volume)),convex=True,watertight=True))
    if not parts:raise RuntimeError('No convex parts')
    urdf=export_urdf(root,records)
    shutil.copyfile(job['source_row']['pose_path'],root/'info/tabletop_pose.json')
    visual=trimesh.load_mesh(source,process=False)
    old=[trimesh.load_mesh(x['path'],process=False)for x in job['source_row']['convex_parts']]
    measurements=dict(old=metric(visual,old),fine=metric(visual,parts))
    row=dict(job['source_row'],object_root=str(root),mesh_path=str(root/'mesh/simplified.obj'),urdf_path=str(urdf),urdf_sha256=sha(urdf),
        pose_path=str(root/'info/tabletop_pose.json'),pose_sha256=sha(root/'info/tabletop_pose.json'),convex_parts=records)
    report=dict(status='ready',object_id=row['object_id'],row=row,parameters=PARAMETERS,
        measurements=measurements,normalization=normalization,raw_sha256=sha(raw),source_visual_sha256=sha(source),
        visual_bytes_unchanged=True,world_T_object_unchanged=True,mass_kg=.1,friction=.6,physics_steps=0,wall_s=time.monotonic()-start)
    save(root/'PREPARATION.json',report);return report

def prepare(out):
    if out.exists():raise ValueError('Use fresh output directory')
    out.mkdir(parents=True);(out/'jobs').mkdir();(out/'asset_jobs').mkdir()
    oldrows={r['object_id']:r for r in read(BASE/'objects.json')['objects']}
    selection=dict(object_ids=IDS,parameters=PARAMETERS,timeout_s=TIMEOUT_S,source_visuals={o:oldrows[o]['mesh_sha256']for o in IDS},
        hypothesis='Changing only approximate collision surfaces may alter frozen hand grasp outcomes',
        selection='A0 previous success control; all3 previous controlled-wrist failures',
        actual_attempts=5,extra_retries=False,model_updates=0,independent_generalization=False,
        no_candidate_pose_or_control_changes=True,coarse_baselines='A3/A6/E0 prior gantry traces, A0 newly run under same executor',
        recipe_selected_before_decomposition=True)
    save(out/'SELECTION.json',selection)
    jobs=[]
    for oid in IDS:
        jp=out/'asset_jobs'/(oid+'.json');save(jp,dict(source_row=oldrows[oid],asset_root=str(out/'fine_assets'/oid)))
        jobs.append((oid,jp))
    began=time.monotonic();prepared={};errors=[]
    def worker(oid,jp):
        log=out/'asset_jobs'/(oid+'.log')
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with log.open('xb')as stream:
            r=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--stage','asset','--output',str(jp)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=TIMEOUT_S)
        if r.returncode:raise RuntimeError(f'{oid} asset rc={r.returncode}')
        return read(out/'fine_assets'/oid/'PREPARATION.json')
    with ThreadPoolExecutor(max_workers=4)as pool:
        tasks={pool.submit(worker,*j):j[0]for j in jobs}
        for f in as_completed(tasks):
            oid=tasks[f]
            try:prepared[oid]=f.result()
            except Exception as e:errors.append(dict(object_id=oid,error=repr(e)))
            print('ALIGNED_ASSET',oid,'ready'if oid in prepared else errors[-1],flush=True)
    save(out/'ASSET_RESULTS.json',dict(status='complete'if not errors else'failed',objects=prepared,errors=errors,wall_s=time.monotonic()-began))
    if errors:raise RuntimeError('Asset preparation incomplete; no physical trials started')
    cases={c['object_id']:c for c in original.source_cases()}
    q=read(BASE/'planning/queue_manifest.json');a0=next(x for x in q['objects']if x['object_id']=='egad_A0')
    selected=a0['selected'];pool=selected['candidate_dir'] if 'candidate_dir'in selected else None
    if pool is None:
        # Queue selected stores full trial spec, inspected explicitly through its immutable plan.
        plan=read(Path(selected['plans'][0]['plan_dir'])/'plan.json')
        pool=plan.get('candidate_dir') or plan.get('candidates') or plan['arguments'].get('candidates')
    if pool is None:
        rank=selected['global_score_rank'];pool=BASE/'diagnostic/egad_A0'/f'pool_{rank:05d}'
    gen=read(Path(pool)/'generation.json');rank=gen['records'][0]['global_score_rank']
    cases['egad_A0']=dict(object_id='egad_A0',candidate_rank=rank,candidate_dir=str(pool),object_row=oldrows['egad_A0'],old_selected_plan=selected['plans'][0]['plan_dir'],old_execution=None)
    backend=read(B/'backend.example.json');docs=[]
    for oid,variant in [('egad_A0','coarse')]+[(x,'fine')for x in IDS]:
        case=dict(cases[oid]);case['source_object_id']=oid;case['object_id']=oid+'_'+variant
        case['object_row']=oldrows[oid]if variant=='coarse'else prepared[oid]['row']
        case['variant']=variant;case['old_execution']=None
        docs.append(case)
    sources={};inputs={}
    oldprotocol=read(ROOT/'outputs/egad_floating_three_20261108_v1/protocol.json')
    for section,target in [('source_sha256',sources),('input_sha256',inputs)]:
        for p,h in oldprotocol[section].items():
            if sha(p)!=h:raise ValueError('Original dependency changed '+p)
            target[p]=h
    for p in (Path(__file__).resolve(),B/'run_egad_floating_three.py',B/'run_boya_floating_diagnostic.py',B/'prepare_egad_external_scenes.py',B/'prepare_egad_external_scenes_v2.py',B/'prepare_egad_external_scenes_v3.py'):
        sources[str(p)]=sha(p)
    for p in out.rglob('*'):
        if p.is_file()and p.suffix not in ('.log',):inputs[str(p)]=sha(p)
    # Bind the A0 candidate independently; prior3 protocol does not include it.
    for case in docs:
        p=Path(case['candidate_dir']);g=read(p/'generation.json')
        for file in [p/'generation.json',p/'candidates.npz',p/'world_object_pose.json']:
            inputs[str(file)]=sha(file)
        for file,h in g['input_sha256'].items():
            if sha(file)!=h:raise ValueError('Candidate dependency changed '+file)
            inputs[file]=h
    protocol=dict(schema='anydex_boya_contact_proxy_alignment_protocol_v1',object_ids=[d['object_id']for d in docs],cases=docs,
        source_sha256=sources,input_sha256=inputs,selection=selection,
        allowed_change='Only object convex collision proxy; original visual, settled initial pose, hand-object transform, CGR/head ranking, drive and nominal commands frozen',
        executor=str(B/'run_boya_floating_diagnostic.py'),physics_attempts=5,models_updated=False,training_writeback=False,prior_external_success_rate_unchanged='1/12')
    save(out/'protocol.json',protocol)
    for case in docs:
        job=dict(case,**{k:backend[k]for k in ['hand_urdf','hand_scene','package','physics_config','diagnostic_config']},
            output_dir=str(out/'execution'/case['object_id']),protocol_path=str(out/'protocol.json'),protocol_sha256=sha(out/'protocol.json'))
        save(out/'jobs'/(case['object_id']+'.json'),job)
    save(out/'PROGRESS.json',dict(status='prepared',actual_grasps=0,protocol_sha256=sha(out/'protocol.json')))
    return protocol

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['asset','prepare','execute'],required=True);a=p.parse_args()
    if a.stage=='asset':r=decompose(a.output)
    elif a.stage=='prepare':r=prepare(a.output.resolve())
    else:r=original.execute(a.output.resolve(),workers=4)
    print(json.dumps(dict(stage=a.stage,status=r.get('status','prepared'))),flush=True)
if __name__=='__main__':main()
