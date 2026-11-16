"""Render saved actual link/body poses, without replay or physics."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
import trimesh
import author_collection_batch as B
import render_author_boya_candidate_comparison as R


def pose(value):
    p=np.asarray(value);t=np.eye(4);t[:3,:3]=Rotation.from_quat(p[3:]).as_matrix();t[:3,3]=p[:3];return t


def run(prepared,output):
    prepared=Path(prepared).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    ready=B.read(prepared/'READY.json');scene=B.read(ready['scene']['path'])
    paths=[prepared/'RUNTIME_AUDIT.json',prepared/'COLLECTION_RECORD.json',prepared/'execution/trace.npz',
        prepared/'execution/MULTI_OBJECT_TRACE.json',prepared/'execution/LIFT_EVALUATION.json',Path(ready['scene']['path']),
        prepared/'execution/ur16e_boya.urdf']
    bindings={str(p):B.binding(p)for p in paths};B.verify(bindings)
    audit=B.read(paths[0])
    if audit['status']!='pass':raise ValueError('Only independently audited actual execution may be shown')
    with np.load(paths[2],allow_pickle=False)as z:arrays={k:z[k]for k in z.files}
    rows=B.read(paths[3])['rows'];evaluation=B.read(paths[4]);assets=R.BoyaAssets()
    robot=ET.parse(paths[-1]).getroot();body_names=list(arrays['body_names'])
    def actual_link(step,name):
        if name in body_names:return pose(arrays['body_pose_xyzw'][step,body_names.index(name)])
        joint=next(j for j in robot.findall('joint')if j.find('child').get('link')==name)
        if joint.get('type')!='fixed':raise ValueError('Missing measured moving link: '+name)
        origin=joint.find('origin');t=np.eye(4)
        if origin is not None:
            t[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
        return actual_link(step,joint.find('parent').get('link'))@t
    steps=[int(np.flatnonzero(arrays['phase']==phase)[-1])for phase in ('approach','hold','lift')]
    frames=[]
    for step in steps:
        if rows[step]['step']!=step+1:raise ValueError('Actual body/robot time indices differ')
        meshes=[]
        for name,mesh in assets.meshes.items():
            transform=actual_link(step,name)
            color=next((v for k,v in R.COLORS.items()if name.startswith(k)),(78,87,103))
            meshes.append(R.mesh_record(name,R.transform(mesh.vertices,transform),mesh.faces,color))
        for j,item in enumerate(scene['objects']):
            mesh=trimesh.load_mesh(item['source_mesh'],process=False)
            transform=pose(rows[step]['objects'][item['object_id']]['pose_xyzw'])
            meshes.append(R.mesh_record(item['object_id'],R.transform(mesh.vertices*item['scale'],transform),mesh.faces,R.OBJECT_COLORS[j]))
        frames.append(meshes)
    points=np.concatenate([m['vertices']for meshes in frames for m in meshes]);center=(points.min(0)+points.max(0))/2
    span=float(np.max(points.max(0)-points.min(0))*1.25)
    floor=trimesh.creation.box(extents=[.8,.7,.001]);floor.apply_translation([center[0],center[1],-.0005])
    sheet=Image.new('RGB',(3060,1010),'white');draw=ImageDraw.Draw(sheet)
    draw.text((20,15),'新预闭合映射：第一场实际执行记录',font=R.font(36),fill=(25,32,44))
    draw.text((20,70),'保存的真实仿真 link / 物体姿态 + 源网格；CPU静态绘图，无回放积分；非相机照片',font=R.font(27),fill=(40,47,58))
    for col,(step,meshes,title)in enumerate(zip(steps,frames,('接近结束，仍为预张开','继续闭合后的保持末态','提起完成'))):
        project,w,h=R.ortho(center,span,[.3,-1,.6])
        image=R.image_view([R.mesh_record('floor',floor.vertices,floor.faces,(236,239,242))]+meshes,project,w,h)
        sheet.paste(image,(20+1020*col,200));draw.text((20+1020*col,142),title,font=R.font(28),fill=(25,32,44))
        draw.text((20+1020*col,925),f'记录步 {step+1}；仿真时刻 {arrays["time_s"][step]:.3f}s',font=R.font(24),fill=(40,47,58))
    successful=[o for o in evaluation['object_outcomes']if o['task_success']]
    text='；'.join(f'{o["object_id"]} 源底离桌 {o["final_source_bottom_m"]*1000:.1f}mm'for o in successful)
    draw.text((20,968),'独立审核通过；'+(text or '本次未提起；负标签'),font=R.font(24),fill=(35,95,65))
    target=out/'actual_execution.png';sheet.save(target)
    with Image.open(target)as check:check.load();assert check.size==(3060,1010)
    B.verify(bindings)
    B.save(out/'RENDER_AUDIT.json',dict(status='saved_actual_link_and_body_pose_render',sources=bindings,
        renderer=B.binding(__file__),image=B.binding(target),steps=[s+1 for s in steps],
        frames_use_measured_link_poses=True,source_visual_geometry_not_cooked_collision=True,
        additional_physics_steps=0,Kit_starts=0,new_labels=0,training_updates=0),exclusive=True)
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();print(run(a.prepared_dir,a.output_dir))
