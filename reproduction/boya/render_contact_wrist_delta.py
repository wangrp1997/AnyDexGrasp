"""Show the actual saved target-palm change with identical finger coordinates.

No optimization, IK, collision test, simulator or new trial. Blue/orange ghost
layers deliberately expose both hands; they are not photoreal contact evidence.
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation

import author_collection_batch as B
import render_author_boya_candidate_comparison as R


def run(benchmark,output):
    benchmark=Path(benchmark).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    protocol=B.read(benchmark/'PROTOCOL.json');B.verify(protocol['inputs']);B.verify(protocol['sources'])
    c=B.read(protocol['inputs']['candidate']['path']);directory=Path(protocol['inputs']['candidate']['path']).parents[1]
    with np.load(directory/'pool/candidates.npz')as z:old=z['proposal_palm'][0].copy()
    with np.load(benchmark/'EXAMPLE_MAPPINGS.npz')as z:
        new=z['world_T_palm'][0,0].copy();q=z['q_contact'][0,0].copy()
    delta=new@np.linalg.inv(old);rotation=Rotation.from_matrix(delta[:3,:3]);angle=float(rotation.magnitude()*180/np.pi)
    displacement=(new[:3,3]-old[:3,3])*1000
    assets=R.BoyaAssets();old_hand=R.hand_meshes(assets,q,old);new_hand=R.hand_meshes(assets,q,new)
    error=0.;motion=[]
    for a,b in zip(old_hand,new_hand):
        if a['name']!=b['name']:raise ValueError('Hand link order changed')
        error=max(error,float(abs(R.transform(a['vertices'],delta)-b['vertices']).max()))
        motion.extend(np.linalg.norm(b['vertices']-a['vertices'],axis=1).tolist())
    if error>2e-14 or angle<1e-4:raise ValueError('No consistent actual geometry transformation')
    scene=B.read(c['scene_binding']['scene_json']);objects=[]
    for i,row in enumerate(scene['objects']):
        mesh=trimesh.load_mesh(row['source_mesh'],process=False)
        objects.append(R.mesh_record(row['object_id'],R.transform(mesh.vertices*row['scale'],np.asarray(row['world_T_object'])),mesh.faces,R.OBJECT_COLORS[i]))
    points=np.concatenate([m['vertices']for m in objects+old_hand+new_hand]);center=(points.min(0)+points.max(0))/2
    span=float(np.max(points.max(0)-points.min(0))*1.4)
    floor=trimesh.creation.box(extents=[.8,.7,.001]);floor.apply_translation([center[0],center[1],-.0005])
    objects.insert(0,R.mesh_record('floor',floor.vertices,floor.faces,(236,239,242)))
    for mesh in old_hand:mesh['color']=(45,108,209)
    for mesh in new_hand:mesh['color']=(237,132,33)
    sheet=Image.new('RGB',(2040,1150),(255,255,255));draw=ImageDraw.Draw(sheet)
    draw.text((20,15),'只比较掌姿：手指保持完全相同的闭合构型',font=R.font(38),fill=(25,32,44))
    draw.text((20,72),f'蓝色=旧目标掌姿；橙色=新目标掌姿；旋转差 {angle:.3f}°；掌原点移动 {np.linalg.norm(displacement):.3f}mm',font=R.font(28),fill=(35,42,54))
    draw.text((20,117),f'世界坐标平移 ΔX={displacement[0]:+.3f}mm，ΔY={displacement[1]:+.3f}mm，ΔZ={displacement[2]:+.3f}mm',font=R.font(28),fill=(35,42,54))
    for col,direction in enumerate(([0,0,1],[.3,-1,.6])):
        project,w,h=R.ortho(center,span,direction)
        base=np.asarray(R.image_view(objects,project,w,h)).astype(float)
        for hand,alpha in [(old_hand,.42),(new_hand,.62)]:
            layer=np.asarray(R.image_view(hand,project,w,h,background=np.zeros((h,w,3),dtype=np.uint8)))
            mask=layer.max(axis=-1)>0;base[mask]=(1-alpha)*base[mask]+alpha*layer[mask]
        image=Image.fromarray(np.asarray(np.clip(base,0,255),dtype=np.uint8));d=ImageDraw.Draw(image)
        for pose,color,prefix in [(old,(30,85,190),'旧'),(new,(204,94,7),'新')]:
            p=pose[:3,3];u,v=project(p[None])[0,:2]
            d.ellipse((u-7,v-7,u+7,v+7),fill=color,outline=(255,255,255),width=2)
            for i,axis in enumerate('XYZ'):
                end=p+pose[:3,i]*.05;R.line3(d,project,[p,end],color,4,dashed=prefix=='旧')
                a,b=project(end[None])[0,:2]
                R.label(d,(a+(8 if prefix=='新'else -55),b+(8 if prefix=='新'else -28)),prefix+axis,fill=color,size=19)
        R.line3(d,project,[old[:3,3],new[:3,3]],(35,40,48),3,dashed=True)
        sheet.paste(image,(20+1020*col,230));draw.text((20+1020*col,190),'俯视'if col==0 else'侧上方；坐标轴长50mm',font=R.font(26),fill=(35,42,54))
    draw.text((20,985),'这是目标掌姿的静态变化，尚未求机械臂/腕关节IK，也未执行任何动作。',font=R.font(27),fill=(35,42,54))
    draw.text((20,1030),'参考帧数值对齐 ≠ 无碰撞的有效包握；该示例新构型仍有约2.95mm源手网格触桌。',font=R.font(27),fill=(155,45,37))
    sheet.save(out/'target_palm_pose_delta.png')
    receipt=dict(status='saved_target_pose_and_geometry_change_verified',source=B.binding(__file__),
        source_protocol=B.binding(benchmark/'PROTOCOL.json'),source_mappings=B.binding(benchmark/'EXAMPLE_MAPPINGS.npz'),
        source_candidate=protocol['inputs']['candidate'],world_T_palm_old=old.tolist(),world_T_palm_new=new.tolist(),
        old_to_new_world_rigid_transform=delta.tolist(),world_palm_origin_translation_delta_mm=displacement.tolist(),
        palm_origin_translation_norm_mm=float(np.linalg.norm(displacement)),rotation_difference_deg=angle,
        world_rotation_axis=(rotation.as_rotvec()/rotation.magnitude()).tolist(),
        full_source_mesh_old_to_new_transform_max_error_m=error,hand_vertex_motion_rms_mm=float(np.sqrt(np.mean(np.square(motion)))*1000),
        identical_finger_joint_configuration=True,palm_goal_changed=True,robot_joint_IK_solved=False,robot_executed=False,
        valid_collision_free_grasp_certified=False,physics_steps=0,IK_calls=0,new_labels=0,training_updates=0)
    B.save(out/'POSE_DELTA_AUDIT.json',receipt,exclusive=True)
    B.verify(protocol['inputs']);B.verify(protocol['sources']);print(receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();run(a.benchmark_dir,a.output_dir)
