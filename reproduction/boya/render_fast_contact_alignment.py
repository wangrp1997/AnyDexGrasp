"""Static old/new contact-configuration comparison, without physics or IK."""
import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image,ImageDraw

import author_collection_batch as B
import render_author_boya_candidate_comparison as R


def run(benchmark,output):
    benchmark=Path(benchmark);out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    result=B.read(benchmark/'RESULTS.json');protocol=B.read(benchmark/'PROTOCOL.json');B.verify(protocol['inputs']);B.verify(protocol['sources'])
    c=B.read(protocol['inputs']['candidate']['path']);scene=B.read(c['scene_binding']['scene_json'])
    with np.load(benchmark/'EXAMPLE_MAPPINGS.npz',allow_pickle=False)as z:
        if not z['valid'][0,0]:raise ValueError('Fixed diagnostic ff branch unavailable; do not choose by old outcome')
        palm=z['world_T_palm'][0,0].copy();q=z['q_contact'][0,0].copy();local=z['palm_T_contact_reference'][0,0].copy()
        progress=float(z['progress'][0,0]);target=z['target_world_reference'][0].copy();opened=z['q_open'].copy()
    assets=R.BoyaAssets();objects=[];centers={}
    for i,row in enumerate(scene['objects']):
        mesh=trimesh.load_mesh(row['source_mesh'],process=False);vertices=R.transform(mesh.vertices*row['scale'],np.asarray(row['world_T_object']))
        objects.append(R.mesh_record(row['object_id'],vertices,mesh.faces,R.OBJECT_COLORS[i]));centers[row['object_id']]=vertices.mean(0)
    directory=Path(protocol['inputs']['candidate']['path']).parents[1]
    with np.load(directory/'pool/candidates.npz',allow_pickle=False)as z:old_palm=z['proposal_palm'][0].copy()
    old_hand=R.hand_meshes(assets,q,old_palm);new_hand=R.hand_meshes(assets,q,palm)
    new_open=R.hand_meshes(assets,opened,palm)
    points=np.concatenate([m['vertices']for m in objects+old_hand+new_hand+new_open]);low,high=points.min(0),points.max(0)
    center=(low+high)/2;span=float(np.max(high-low)*1.47)
    floor=trimesh.creation.box(extents=[.8,.7,.001]);floor.apply_translation([center[0],center[1],-.0005])
    objects.insert(0,R.mesh_record('floor',floor.vertices,floor.faces,(236,239,242)))
    raw_reference=np.eye(4);raw_reference[:3,:3]=np.asarray(c['actual_row15_camera'][3:12]).reshape(3,3);raw_reference[:3,3]=c['actual_row15_camera'][12:15]
    raw_reference=np.asarray(c['world_T_camera'])@raw_reference
    old_ref=old_palm@local
    midpoint_drift=float(np.linalg.norm(old_ref[:3,3]-target[:3,3]))
    angle=float(np.degrees(np.arccos(np.clip((np.trace(old_ref[:3,:3].T@target[:3,:3])-1)/2,-1,1))))
    prepared=dict(c,_slot_reference=local.tolist(),calibrated_reference_width_m=c['raw_width_m'])
    sheet=Image.new('RGB',(3060,1770),(255,255,255));draw=ImageDraw.Draw(sheet)
    draw.text((20,12),'g1五指握 / raw #1322：同一个接触构型，仅改变手腕对齐基准',font=R.font(35),fill=(25,32,44))
    draw.text((20,65),f'目标接触宽度 {c["raw_width_m"]*1000:.3f}mm；闭合进度 {progress*100:.2f}%；拇指+食指仅用于此图的受控对照，其余三分支保留',font=R.font(26),fill=(42,49,62))
    draw.text((20,108),f'旧手腕下接触中点偏移 {midpoint_drift*1000:.3f}mm、参考方向差 {angle:.3f}°；新手腕下宽度/中点/参考方向数值对齐',font=R.font(25),fill=(42,49,62))
    headings=['原始CGR参考（没有实体手）','旧手腕 + 闭合到目标宽度的手','新手腕 + 同一接触构型（未物理验证）']
    for i,title in enumerate(headings):draw.text((20+1020*i,160),title,font=R.font(26),fill=(35,55,90))
    for j,direction in enumerate(([0,0,1],[.3,-1,.6])):
        project,w,h=R.ortho(center,span,direction)
        for i,hand,pose in [(0,[],None),(1,old_hand,old_palm),(2,new_hand,palm)]:
            image=R.image_view(objects+hand,project,w,h)
            cc=dict(prepared,_rendered_palm=pose.tolist()if pose is not None else old_palm.tolist())
            image=R.overlay(image,project,cc,raw_reference,mapped=i>0,object_centers=centers)
            sheet.paste(image,(20+i*1020,210+j*750))
    draw.text((20,1710),'仅静态FK：参考点对齐不等于已经与物体真实接触；仍待闭合全路径、自碰撞、桌面/邻物与受载验证。',font=R.font(24),fill=(40,48,62))
    sheet.save(out/'closed_configuration_alignment.png')
    image=Image.new('RGB',(2040,950),(255,255,255));d=ImageDraw.Draw(image)
    d.text((20,15),'同一新手腕：先张开接近 → 再闭合至预计接触构型（两帧静态示意）',font=R.font(32),fill=(25,32,44))
    project,w,h=R.ortho(center,span,[.3,-1,.6])
    for i,hand in enumerate([new_open,new_hand]):
        image.paste(R.image_view(objects+hand,project,w,h),(20+1020*i,110))
    d.text((20,840),'左：原张开关节目标；右：预计接触关节目标。没有运动积分、碰撞准入或实际抓取。',font=R.font(26),fill=(40,48,62))
    image.save(out/'open_then_contact.png')
    B.save(out/'VISUAL_COMPARISON.json',dict(status='static_geometry_only',benchmark_results=B.binding(benchmark/'RESULTS.json'),source=B.binding(__file__),
        reference_renderer=B.binding(R.__file__),source_candidate=protocol['inputs']['candidate'],
        display_opponent='ff',selection='Fixed ff branch to isolate old/new wrist effect; no historical success used',
        closure_progress=progress,target_width_mm=c['raw_width_m']*1000,old_midpoint_drift_mm=midpoint_drift*1000,
        old_reference_orientation_difference_deg=angle,new_reference_max_matrix_error=float(abs(palm@local-target).max()),
        old_contact_pose_source_surface_min_z_m=min(float(m['vertices'][:,2].min())for m in old_hand),
        new_contact_pose_source_surface_min_z_m=min(float(m['vertices'][:,2].min())for m in new_hand),
        new_open_pose_source_surface_min_z_m=min(float(m['vertices'][:,2].min())for m in new_open),
        entire_closure_collision_checked=False,real_thumb_contact_or_force_validated=False,
        Kit_starts=0,physics_steps=0,IK_calls=0,new_labels=0,training_updates=0,production_ready=False),exclusive=True)
    print('Static contact-alignment images saved; physics=0')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();run(a.benchmark_dir,a.output_dir)
