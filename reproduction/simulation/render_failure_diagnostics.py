"""Saved-artifact diagnostics: zero-step rejected-hand FK and actual banana traces.

No network invocation, candidate admission change, controller edit or integration.
Static closest distances refer to the declared convex simulation geometry only.
"""
from __future__ import annotations
import argparse, hashlib, html, json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text())


def static_hand(repo, base, out, name, choice):
    import pybullet as p
    from scipy.spatial.transform import Rotation
    import mesh_scene
    from dh3_author_kinematics import joint_angles
    from dh3_geometry_adapter import adapter_targets
    inf=base/'run_nonapple_v2/inference'/name
    manifest_path=base/'input_nonapple_v2/scenes'/name/'scene_manifest.json'
    manifest=json_read(manifest_path)
    with np.load(inf/'collision/collision_outputs.npz',allow_pickle=False) as f:
        arrays={k:f[k].copy() for k in f.files}
    index=int(np.argmax(arrays['DH3_grasp_group_array'][:,0]) if choice=='highest_score' else np.argmin(arrays['overlap_point_counts']))
    assert not arrays['admission_mask'][index]
    hand=arrays['DH3_grasp_group_array'][index];two=arrays['two_finger_grasp_group_array'][index]
    points=np.load(inf/'frame/scene_points_down.npy',allow_pickle=False)
    mask=np.unpackbits(arrays['scene_inclusion_masks_packed'][index])[:len(points)].astype(bool)
    assert int(mask.sum())==int(arrays['overlap_point_counts'][index])
    cam=np.asarray(manifest['camera']['T_WC']);H=np.eye(4);H[:3,:3]=hand[3:12].reshape(3,3);H[:3,3]=hand[12:15]
    world=cam@H;direction=cam[:3,:3]@two[4:13].reshape(3,3)[:,0];direction/=np.linalg.norm(direction)
    backoff=float(hand[15]/100*.005);target=world[:3,3]+(hand[1]-backoff)*direction
    markers=points[mask]@cam[:3,:3].T+cam[:3,3]
    adapter_path=repo/'outputs/dh3_geometry_adapter_20261103_v2/ADAPTER.json';adapter=json_read(adapter_path)
    assert sha(adapter['output_URDF'])==adapter['output_URDF_sha256']
    cid=p.connect(p.DIRECT)
    try:
        scene=mesh_scene.create_scene(p,cid,manifest)
        body=p.loadURDF(adapter['output_URDF'],target.tolist(),Rotation.from_matrix(world[:3,:3]).as_quat().tolist(),useFixedBase=True,flags=p.URDF_USE_INERTIA_FROM_FILE,physicsClientId=cid)
        joints={p.getJointInfo(body,j,physicsClientId=cid)[1].decode():j for j in range(p.getNumJoints(body,physicsClientId=cid))}
        for joint,value in adapter_targets(joint_angles(*hand[15:17]),adapter).items():p.resetJointState(body,joints[joint],value,physicsClientId=cid)
        links={-1:'base_Link'}|{j:p.getJointInfo(body,j,physicsClientId=cid)[12].decode() for j in range(p.getNumJoints(body,physicsClientId=cid))}
        distances={}
        for target_kind,target_body in [('object',scene['object_id']),('table',scene['table_id'])]:
            pairs=p.getClosestPoints(body,target_body,.03,physicsClientId=cid)
            minima={}
            for row in pairs:
                key=links[row[3]];minima[key]=min(minima.get(key,float('inf')),float(row[8]))
            distances[target_kind]={'query_radius_m':.03,'min_distance_m':min(minima.values()) if minima else None,'per_link_min_m':minima,'penetrating_links':[key for key,d in minima.items() if d<0]}
        sphere=p.createVisualShape(p.GEOM_SPHERE,radius=.0016,rgbaColor=[1,.015,.015,1],physicsClientId=cid)
        for point in markers:p.createMultiBody(baseMass=0,baseVisualShapeIndex=sphere,basePosition=point.tolist(),physicsClientId=cid)
        paths=[];raw=[]
        for label,eye in [('front',[.28,-.25,.22]),('rear',[-.27,.36,.22])]:
            view=p.computeViewMatrix(eye,[0,.08,.08],[0,0,1]);proj=p.computeProjectionMatrixFOV(45,640/480,.01,2)
            rendered=p.getCameraImage(640,480,view,proj,renderer=p.ER_TINY_RENDERER,physicsClientId=cid)
            im=Image.fromarray(np.asarray(rendered[2],np.uint8).reshape(480,640,4)[...,:3]);raw.append(im)
            path=out/(name+'_'+choice+'_'+label+'.png');im.save(path);paths.append(str(path))
        card=Image.new('RGB',(1280,540),'white');draw=ImageDraw.Draw(card)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
        draw.text((10,5),name+' / '+choice+' / pre-collision index '+str(int(arrays['source_indices'][index]))+' / pose'+str(int(hand[2])),fill='black',font=font)
        draw.text((10,28),'STATIC open hand at port target. RED: original gate-overlap scene points (including its swept volume). No execution.',fill=(150,0,0),font=font)
        for j,im in enumerate(raw):card.paste(im,(j*640,60))
        sheet=out/(name+'_'+choice+'.jpg');card.save(sheet,quality=93)
        return dict(scene_id=name,choice=choice,scene_manifest=str(manifest_path),scene_manifest_sha256=sha(manifest_path),collision_archive_sha256=sha(inf/'collision/collision_outputs.npz'),camera_points_sha256=sha(inf/'frame/scene_points_down.npy'),adapter_sha256=sha(adapter_path),pre_collision_index=int(arrays['source_indices'][index]),mapped_index=index,score=float(hand[0]),hand_type=int(hand[2]),width_m=float(hand[18]),registers=hand[15:17].tolist(),original_depth_m=float(hand[1]),port_backoff_m=backoff,approach_world=direction.tolist(),port_target_world_m=target.tolist(),gate_overlap_count=int(mask.sum()),static_port_geometry_distances=distances,images=paths,card=str(sheet),physical_steps=0,boundary='Static original-register FK in frozen observed scene; red points belong to original swept/infilled voxel gate, not necessarily the hand surface shown. Convex closest distances are not source-triangle contact or proof of safety/lift.')
    finally:p.disconnect(cid)


def banana_plots(base,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,8),sharex=True)
    provenance=[]
    for col,name in enumerate(['banana_yaw000','banana_yaw090']):
        root=base/'run_nonapple_v2/physics'/name
        with np.load(root/'trace.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
        c=json_read(root/'CONTROLLER.json');j=c['joint_order'].index('finger1_joint');stride=8;t=d['time'][::stride]
        axes[0,col].plot(t,d['joint_target'][::stride,j],label='flexion command',color='black',linestyle='--');axes[0,col].plot(t,d['q'][::stride,j],label='measured shared flexion')
        axes[1,col].plot(t,d['motor_torque'][::stride,j],label='active flexion torque');axes[1,col].axhline(.6,color='red',linestyle=':',label='declared 0.6 Nm limit');axes[1,col].axhline(-.6,color='red',linestyle=':')
        initial=np.asarray(json_read(c['scene_manifest'])['actual_state']['object']['position_world_m'])
        axes[2,col].plot(t,1000*(d['object_pose'][::stride,2]-initial[2]),label='object COM lift')
        for ax in axes[:,col]:
            for x in [3.5,5.5,6.25,9.25]:ax.axvline(x,color='.7',linewidth=.6)
            ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=8)
        axes[0,col].set_title(name+' / original saved top1');axes[2,col].set_xlabel('Actual recorded time (s)')
        provenance.append(dict(scene_id=name,trace_sha256=sha(root/'trace.npz'),controller_sha256=sha(root/'CONTROLLER.json'),display_stride_steps=stride))
    for ax,label in zip(axes[:,0],['Flexion (rad)','Motor torque (Nm)','COM lift (mm)']):ax.set_ylabel(label)
    fig.suptitle('Saved physical traces: different original hand modes/poses; no new execution or causal yaw test')
    fig.tight_layout();path=out/'banana_recorded_traces.png';fig.savefig(path,dpi=150);plt.close(fig)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    card=Image.new('RGB',(1280,560),'white');draw=ImageDraw.Draw(card)
    frames=[]
    for row,name in enumerate(['banana_yaw000','banana_yaw090']):
        root=base/'run_nonapple_v2/physics'/name
        frame_rows=json_read(root/'frames.json')['frames']
        # Each recorded frame has its actual post-step time; choose nearest, never replay.
        for col,t in enumerate([3.5,5.5,6.75,7.3]):
            fr=min(frame_rows,key=lambda r:abs(r['physical_time_s']-t))
            source=root/fr['path'];actual=fr['physical_time_s']
            im=Image.open(source).convert('RGB');im.thumbnail((320,240));card.paste(im,(col*320,row*280+36))
            draw.text((col*320+5,row*280+8),name+' t=%.2fs'%actual,font=font,fill='black')
            frames.append(dict(scene_id=name,path=str(source),sha256=sha(source),time_s=actual))
    card_path=out/'banana_recorded_frames.jpg';card.save(card_path,quality=93)
    return dict(plot=str(path),recorded_frames_card=str(card_path),sources=provenance,frame_sources=frames,new_physics_steps=0)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,default=Path('/home/wangrenpeng/AnyDexGrasp'));ap.add_argument('--cohort',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    repo=args.repo.absolute();base=args.cohort.absolute();out=args.output_dir.absolute()
    if out.exists():raise FileExistsError('Retain each diagnostic; choose a new output directory')
    out.mkdir(parents=True);os.environ.setdefault('MPLCONFIGDIR','/tmp/anydex_diag_matplotlib')
    sys.path[:0]=[str(repo/'outputs/native_deps/python_overlay'),str(repo/'reproduction/runtime'),str(repo/'reproduction/simulation')]
    import pybullet as p
    def no_step(*args,**kwargs):raise RuntimeError('This diagnostic permits zero integration steps')
    p.stepSimulation=no_step
    start=time.monotonic();cases=[]
    for name in ['mug_yaw090','bowl_yaw000','bowl_yaw090']:
        for choice in ['highest_score','fewest_overlaps']:cases.append(static_hand(repo,base,out,name,choice))
    bananas=banana_plots(base,out)
    result=dict(scope='Targeted four non-success scenes of frozen 12-scene original-top1 cohort',source_script_sha256=sha(__file__),cohort_summary_sha256=sha(base/'run/RESULTS.json'),static_cases=cases,banana=bananas,new_network_forwards=0,new_collision_admission_calls=0,new_physics_steps=0,static_native_distance_queries=12,wall_seconds=time.monotonic()-start,baseline_results_modified=False)
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    doc='<!doctype html><html lang="zh"><meta charset="utf-8"><title>AnyDex 失败定位</title><style>body{font-family:system-ui;max-width:1400px;margin:25px auto}img{max-width:100%}article{margin:25px 0;padding:12px;background:#f3f4f6}</style><h1>AnyDex：四个未通过场景的定位</h1><p>原12场景结果保持8/12。以下拒绝候选仅做静态FK与凸几何距离查询，未执行抓取。红点为原碰撞门的重合场景点；原门含膨胀及接近扫掠，因此不一定在图中手表面上。</p>'
    for c in cases:doc+='<article><h2>'+html.escape(c['scene_id']+' / '+c['choice'])+'</h2><p>原碰撞重合点 '+str(c['gate_overlap_count'])+'；静态凸几何距离见 RESULTS.json。原门未更改。</p><img src="'+Path(c['card']).name+'"></article>'
    doc+='<h2>香蕉：已有真实执行帧及记录曲线</h2><p>两个方向选择了不同手型和抓姿，不能作为只改变yaw的对照。</p><img src="banana_recorded_frames.jpg"><img src="banana_recorded_traces.png"><p><a href="RESULTS.json">结构化测量与来源SHA</a></p></html>'
    (out/'index.html').write_text(doc)
    print(json.dumps(dict(status='diagnostics_rendered',static_cases=len(cases),new_physics_steps=0,output=str(out),wall_seconds=result['wall_seconds']),indent=2))

if __name__=='__main__':main()
