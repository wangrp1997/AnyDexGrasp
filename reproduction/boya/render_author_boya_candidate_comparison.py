"""Read-only, source-mesh comparison of original CGR and saved Boya poses.

CPU rasterization only. No simulator, collision search, CGR forward, IK,
physics integration, labels or training. CGR ruler is not a DH3 hand mesh or
an author-verified successful grasp. Open-hand pictures are saved candidates,
not photographs of the historical executed hand.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import trimesh
from numba import njit
from PIL import Image, ImageDraw, ImageFont

import author_collection_batch as B
from asset_adapter import BoyaAssets
from author_action_semantics import ActionContract, CGR_R_REFERENCE

SCENE=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/production_tail_20261115_v1/run/windows/01/scenes/0000009_s40_lift_fit_00013_s2026111501')
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
COLORS={'th':(202,60,134),'ff':(40,112,204),'mf':(106,132,151),'rf':(106,132,151),'lf':(106,132,151)}
OBJECT_COLORS=[(149,184,171),(205,175,137),(146,175,202)]


@njit(cache=True)
def raster(projected,faces,colors,depth,image,perspective):
    h,w=depth.shape
    for f in range(len(faces)):
        ia,ib,ic=faces[f];a=projected[ia];b=projected[ib];c=projected[ic]
        if min(a[2],b[2],c[2])<=0:continue
        den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-10:continue
        x0=max(0,int(np.floor(min(a[0],b[0],c[0]))));x1=min(w-1,int(np.ceil(max(a[0],b[0],c[0]))))
        y0=max(0,int(np.floor(min(a[1],b[1],c[1]))));y1=min(h-1,int(np.ceil(max(a[1],b[1],c[1]))))
        for y in range(y0,y1+1):
            for x in range(x0,x1+1):
                u=((b[1]-c[1])*(x+.5-c[0])+(c[0]-b[0])*(y+.5-c[1]))/den
                v=((c[1]-a[1])*(x+.5-c[0])+(a[0]-c[0])*(y+.5-c[1]))/den
                t=1-u-v
                if u<0 or v<0 or t<0:continue
                z=1/(u/a[2]+v/b[2]+t/c[2])if perspective else u*a[2]+v*b[2]+t*c[2]
                if z<depth[y,x]:
                    depth[y,x]=z
                    for k in range(3):image[y,x,k]=colors[f,k]


def transform(vertices,matrix):
    return np.asarray(vertices)@matrix[:3,:3].T+matrix[:3,3]


def mesh_record(name,vertices,faces,color):
    return dict(name=name,vertices=np.asarray(vertices,dtype=np.float64),faces=np.asarray(faces,dtype=np.int32),color=color)


def hand_meshes(assets,q,palm,highlight=False):
    frames,_=assets.hand.frames(np.asarray(q))
    records=[]
    for link,mesh in assets.meshes.items():
        color=next((v for k,v in COLORS.items()if link.startswith(k)),(78,87,103))
        if highlight and link=='lfdistal':color=(226,47,51)
        records.append(mesh_record(link,transform(mesh.vertices,palm@frames[link]),mesh.faces,color))
    return records


def ortho(center,span,direction,w=1000,h=720):
    look=np.asarray(direction,dtype=float);look/=np.linalg.norm(look);forward=-look
    right=np.cross(forward,[0.,0.,1.])
    if np.linalg.norm(right)<1e-8:right=np.array([1.,0.,0.])
    right/=np.linalg.norm(right);up=np.cross(right,forward)
    eye=np.asarray(center)+look*1.5
    def project(points):
        a=np.asarray(points)-center
        return np.c_[w/2+a@right*w/span,h/2-a@up*w/span,(np.asarray(points)-eye)@forward]
    return project,w,h


def image_view(meshes,project,w,h,background=None,depth=None,perspective=False):
    pixels=np.full((h,w,3),247,dtype=np.uint8)if background is None else np.asarray(background).copy()
    zbuffer=np.full((h,w),np.inf)if depth is None else np.where(depth>0,depth,np.inf).copy()
    light=np.array([.3,-.4,.85]);light/=np.linalg.norm(light)
    for mesh in meshes:
        vertices,faces=mesh['vertices'],mesh['faces'];tri=vertices[faces]
        normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm=np.linalg.norm(normal,axis=1)
        normal/=np.maximum(norm,1e-30)[:,None]
        shade=.48+.52*np.abs(normal@light)
        colors=np.asarray(np.clip(np.asarray(mesh['color'])[None]*shade[:,None],0,255),dtype=np.uint8)
        raster(np.ascontiguousarray(project(vertices)),faces,colors,zbuffer,pixels,perspective)
    return Image.fromarray(pixels)


def font(size):return ImageFont.truetype(FONT,size)


def label(draw,xy,text,fill=(35,42,54),size=21):
    box=draw.textbbox(xy,text,font=font(size));draw.rectangle((box[0]-3,box[1]-2,box[2]+3,box[3]+2),fill=(255,255,255))
    draw.text(xy,text,font=font(size),fill=fill)


def line3(draw,project,points,color,width=4,dashed=False):
    q=project(np.asarray(points))[:,:2]
    if not np.isfinite(q).all():raise ValueError('Nonfinite projected line')
    for a,b in zip(q[:-1],q[1:]):
        if dashed:
            for t in np.arange(0.,1.,.1):
                aa=a+(b-a)*t;bb=a+(b-a)*min(t+.055,1.)
                draw.line([tuple(aa),tuple(bb)],fill=color,width=width)
        else:draw.line([tuple(a),tuple(b)],fill=color,width=width)


def overlay(image,project,candidate,reference,mapped=False,object_centers=None):
    d=ImageDraw.Draw(image);p=reference[:3,3];x,y=reference[:3,0],reference[:3,1]
    raw=float(candidate['raw_width_m']);rawdepth=float(candidate['raw_depth_m']);total=float(candidate['total_depth_m'])
    rawcenter=p+rawdepth*x;mappedcenter=p+total*x
    # Explicit geometric ruler, not a synthetic physical hand or contact claim.
    for center,width,color,dashed in [(rawcenter,raw,(229,139,26),False),
        (mappedcenter,candidate['expanded_width_m'],(229,139,26),True)]:
        ends=np.array([center-y*width/2,center+y*width/2])
        line3(d,project,ends,color,5,dashed)
        for end in ends:line3(d,project,[end-x*.025,end],color,5,dashed)
    line3(d,project,[p-x*.055,p+x*.015],(26,142,83),5)
    line3(d,project,[p+x*.015,p+x*.005+y*.006],(26,142,83),4)
    line3(d,project,[p+x*.015,p+x*.005-y*.006],(26,142,83),4)
    origin=project(p[None])[0,:2];d.ellipse((origin[0]-5,origin[1]-5,origin[0]+5,origin[1]+5),fill=(20,35,50))
    if mapped:
        ref=np.asarray(candidate['_rendered_palm'])@np.asarray(candidate['_slot_reference'])
        points=[ref[:3,3]-ref[:3,0]*candidate['calibrated_reference_width_m']/2,
                ref[:3,3]+ref[:3,0]*candidate['calibrated_reference_width_m']/2]
        line3(d,project,points,(102,62,184),3,True)
        for xyz,color in zip(points,[(202,60,134),(40,112,204)]):
            u,v=project(np.asarray(xyz)[None])[0,:2];d.ellipse((u-7,v-7,u+7,v+7),fill=color,outline=(255,255,255),width=2)
    if object_centers:
        for name,pos in object_centers.items():
            u,v=project(pos[None])[0,:2]
            if 0<u<image.width-70 and 0<v<image.height-30:label(d,(u+8,v+8),name.replace('graspnet_',''),size=20)
    return image


def build_case(g,assets,contract,objects,centers,out,center,span,observation):
    directory=SCENE/f'g{g}';candidate_path=directory/'pool/selected_author_candidate.json';c=B.read(candidate_path)
    B.verify(c['input_bindings'])
    with np.load(c['input_bindings']['raw_candidates']['path'],allow_pickle=False)as z:
        if not np.array_equal(z['candidates'][c['source_row_index']],c['source_row15_camera']):raise ValueError('Raw selected row changed')
    slot=contract.slots[c['slot_id']]
    with np.load(directory/'pool/candidates.npz',allow_pickle=False)as z:
        q=z['proposal_q'][0].copy();palm=z['proposal_palm'][0].copy()
    pose_roundtrip_error=float(abs(palm-c['world_T_palm']).max())
    if not np.array_equal(q,slot['independent_q'])or pose_roundtrip_error>1e-7:raise ValueError('Saved mapping differs from candidate beyond numeric roundtrip')
    reference=np.eye(4);reference[:3,:3]=np.asarray(c['actual_row15_camera'][3:12]).reshape(3,3);reference[:3,3]=c['actual_row15_camera'][12:15]
    reference=np.asarray(c['world_T_camera'])@reference
    expected=reference.copy();expected[:3,:3]=reference[:3,:3]@CGR_R_REFERENCE
    expected[:3,3]+=c['total_depth_m']*reference[:3,0]
    error=float(abs(palm@np.asarray(slot['palm_T_reference'])-expected).max())
    if error>2e-6:raise ValueError('Mapped reference does not recover original CGR/depth')
    c['_slot_reference']=slot['palm_T_reference']
    c['_rendered_palm']=palm.tolist()
    hands=hand_meshes(assets,q,palm,highlight=g==7)
    meshes=objects+hands
    prep=B.read(directory/'LIFT_PIPELINE_RESULTS.json')
    evaluation=directory/'execution/LIFT_EVALUATION.json'
    success=B.read(evaluation)['task_success']if evaluation.exists()else None
    status='历史实际成功'if success is True else('历史实际抓取失败'if success is False else'规划拒绝，未执行')
    views=[('俯视：世界XY，同一比例',ortho(center,span,[0,0,1])),('侧上方：同一相机、同一比例',ortho(center,span,[.3,-1.,.60]))]
    sheet=Image.new('RGB',(2040,1740),(255,255,255));draw=ImageDraw.Draw(sheet)
    draw.text((24,15),f'g{g} | 原始候选 #{c["source_row_index"]} | {status}',font=font(34),fill=(25,32,44))
    draw.text((24,67),f'原始开口 {c["raw_width_m"]*1000:.1f}mm → 1.7倍请求 {c["expanded_width_m"]*1000:.1f}mm → Boya参考开口 {c["calibrated_reference_width_m"]*1000:.1f}mm',font=font(25),fill=(42,49,62))
    draw.text((24,105),f'原始深入 {c["raw_depth_m"]*1000:.1f}mm + 额外 {c["extra_depth_m"]*1000:.1f}mm；{c["slot_id"]}',font=font(24),fill=(42,49,62))
    draw.text((24,150),'左：原始CGR／两指标尺（不是DH3实体手）',font=font(27),fill=(147,88,10))
    draw.text((1044,150),'右：转换后的Boya候选姿态（真实源网格）',font=font(27),fill=(40,70,130))
    for i,(title,(project,w,h))in enumerate(views):
        y=205+i*750
        left=overlay(image_view(objects,project,w,h),project,c,reference,False,centers)
        right=overlay(image_view(meshes,project,w,h),project,c,reference,True,centers)
        sheet.paste(left,(20,y));sheet.paste(right,(1040,y));draw.text((24,y-28),title,font=font(22),fill=(65,73,86))
    draw.text((24,1690),'绿箭头=接近方向；橙实线=原始开口；橙虚线=1.7倍请求；紫虚线/彩点=Boya计算接触参考',font=font(22),fill=(35,42,54))
    filename=f'g{g}_source_vs_boya.png';sheet.save(out/filename)
    # Saved actual RGB-D camera overlay, retaining source-scene occlusion.
    rgb=Image.open(observation['files']['color.png']['path']).convert('RGB');depth=np.asarray(Image.open(observation['files']['depth.png']['path']),dtype=float)/observation['depth_factor']
    K=np.asarray(observation['intrinsic_matrix']);camera_inverse=np.linalg.inv(np.asarray(c['world_T_camera']))
    def perspective(points):
        a=transform(points,camera_inverse)
        return np.c_[K[0,0]*a[:,0]/a[:,2]+K[0,2],K[1,1]*a[:,1]/a[:,2]+K[1,2],a[:,2]]
    original=overlay(rgb.copy(),perspective,c,reference)
    after=overlay(image_view(hands,perspective,rgb.width,rgb.height,background=rgb,depth=depth,perspective=True),perspective,c,reference,True)
    composite=Image.new('RGB',(rgb.width*2,rgb.height+130),(255,255,255));composite.paste(original,(0,90));composite.paste(after,(rgb.width,90));d=ImageDraw.Draw(composite)
    d.text((20,10),f'g{g} / #{c["source_row_index"]}：原始RGB + CGR标尺',font=font(30),fill=(25,32,44));d.text((rgb.width+20,10),'原始RGB + Boya候选投影（静态合成，非实拍）',font=font(29),fill=(25,32,44))
    d.text((20,rgb.height+96),'按原深度图遮挡；仅呈现保存的候选，未重新执行仿真。',font=font(23),fill=(45,53,64));composite.save(out/f'g{g}_rgb_overlay.png')
    return dict(gesture=g,source_row_index=c['source_row_index'],source_top500_rank=c['source_top500_rank'],status=status,
        historical_task_success=success,pipeline_stage=prep['stage'],raw_width_mm=c['raw_width_m']*1000,
        expanded_width_mm=c['expanded_width_m']*1000,reference_width_mm=c['calibrated_reference_width_m']*1000,
        raw_depth_mm=c['raw_depth_m']*1000,extra_depth_mm=c['extra_depth_m']*1000,slot_id=c['slot_id'],
        raw_to_mapped_reference_matrix_max_error=error,saved_pool_vs_candidate_pose_roundtrip_max_error=pose_roundtrip_error,palm=palm.tolist(),reference=reference.tolist(),q=q.tolist(),
        image=filename,rgb_image=f'g{g}_rgb_overlay.png',candidate=B.binding(candidate_path),preparation=B.binding(directory/'LIFT_PIPELINE_RESULTS.json'),
        evaluation=B.binding(evaluation)if evaluation.exists()else None,hand_meshes=hands)


def run(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    assets=BoyaAssets();contract=ActionContract();first=B.read(SCENE/'g7/pool/selected_author_candidate.json')
    scene=B.read(first['scene_binding']['scene_json']);observation=B.read(first['input_bindings']['observation']['path']);B.verify(observation['files'])
    objects=[];centers={};bindings={}
    for i,row in enumerate(scene['objects']):
        path=Path(row['source_mesh']);mesh=trimesh.load_mesh(path,process=False)
        vertices=transform(np.asarray(mesh.vertices)*row['scale'],np.asarray(row['world_T_object']))
        objects.append(mesh_record(row['object_id'],vertices,mesh.faces,OBJECT_COLORS[i]));centers[row['object_id']]=vertices.mean(axis=0);bindings[str(path)]=B.binding(path)
    all_points=[m['vertices']for m in objects]
    for g in (6,1,7):
        c=B.read(SCENE/f'g{g}/pool/selected_author_candidate.json');slot=contract.slots[c['slot_id']]
        all_points.extend(m['vertices']for m in hand_meshes(assets,slot['independent_q'],np.asarray(c['world_T_palm'])))
    points=np.concatenate(all_points);low,high=points.min(axis=0),points.max(axis=0)
    center=(low+high)/2;span=float(np.max(high-low)*1.47)
    floor=trimesh.creation.box(extents=[.7,.6,.001]);floor.apply_translation([center[0],center[1],-.0005])
    objects.insert(0,mesh_record('floor',floor.vertices,floor.faces,(236,239,242)))
    cases=[]
    for g in (6,1,7):
        case=build_case(g,assets,contract,objects,centers,out,center,span,observation);cases.append(case)
        print(json.dumps(dict(gesture=g,status=case['status'],reference_matrix_error=case['raw_to_mapped_reference_matrix_max_error'])),flush=True)
    geometry_path=SCENE/'g7/home/rrt/fixed_segments_geometry.json';geometry=B.read(geometry_path)
    violations=geometry['violations'];rejection=dict(binding=B.binding(geometry_path),first=violations[0],last=violations[-1],
        recorded_violations=len(violations),maximum_recorded_overlap_mm=max(-r['distance_m']for r in violations)*1000,
        red_link='lfdistal',scope='Saved source-convex geometry report, not a new physics or triangle-intersection measurement')
    metadata=dict(status='complete',scene=B.binding(first['scene_binding']['scene_json']),observation=first['input_bindings']['observation'],
        renderer='CPU orthographic/perspective z-buffer, full original triangle meshes; no decimation',
        source=B.binding(__file__),object_sources=bindings,cases=[{k:v for k,v in c.items()if k!='hand_meshes'}for c in cases],
        g7_rejection=rejection,all_cases_share_world_frame=True,all_panels_share_scale_and_camera=True,
        source_CGR_is_author_hardware_verified_success=False,DH3_physical_mesh_displayed=False,
        physical_semantics_of_computed_pad_reference_validated=False,hand_triangles=sum(len(m.faces)for m in assets.meshes.values()),
        Kit_starts=0,physics_steps=0,CGR_forwards=0,IK_calls=0,new_labels=0,training_updates=0,wall_s=time.monotonic()-began)
    B.save(out/'COMPARISON.json',metadata,exclusive=True)
    # Standalone diagnostic view, full geometry and named links, freely rotatable.
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    for case in cases:
        figure=make_subplots(rows=1,cols=2,specs=[[{'type':'scene'},{'type':'scene'}]],subplot_titles=['原始CGR方向/宽度标尺；未实机验证','转换后的Boya源网格；候选姿态'])
        for col,items in [(1,objects),(2,objects+case['hand_meshes'])]:
            for m in items:
                v=m['vertices'];f=m['faces'];color='rgb('+','.join(str(x)for x in m['color'])+')'
                figure.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color=color,name=m['name'],legendgroup=m['name'],showlegend=col==2,opacity=1.,flatshading=True),row=1,col=col)
            ref=np.asarray(case['reference']);p=ref[:3,3];x,y=ref[:3,0],ref[:3,1]
            for name,pts,color in [('接近方向',[p-x*.06,p+x*.02],'green'),('原始开口',[p+x*case['raw_depth_mm']/1000-y*case['raw_width_mm']/2000,p+x*case['raw_depth_mm']/1000+y*case['raw_width_mm']/2000],'orange')]:
                q=np.asarray(pts);figure.add_trace(go.Scatter3d(x=q[:,0],y=q[:,1],z=q[:,2],mode='lines+markers',line=dict(color=color,width=7),marker=dict(size=3),name=name,showlegend=col==2),row=1,col=col)
        scene_config=dict(aspectmode='cube',xaxis=dict(range=[center[0]-span/2,center[0]+span/2],title='世界X / m'),yaxis=dict(range=[center[1]-span/2,center[1]+span/2],title='世界Y / m'),zaxis=dict(range=[center[2]-span/2,center[2]+span/2],title='世界Z / m'),camera=dict(eye=dict(x=1.1,y=-1.7,z=1.1)),uirevision='fixed')
        figure.update_layout(title=f'g{case["gesture"]} / raw #{case["source_row_index"]} / {case["status"]}',scene=scene_config,scene2=scene_config,height=850,margin=dict(l=0,r=0,b=0,t=70))
        script="""var p=document.getElementById('{plot_id}'),busy=false;p.on('plotly_relayout',function(e){if(busy)return;var u={};if(e['scene.camera'])u['scene2.camera']=e['scene.camera'];else if(e['scene2.camera'])u['scene.camera']=e['scene2.camera'];else return;busy=true;Plotly.relayout(p,u).then(function(){busy=false;});});"""
        figure.write_html(out/f'g{case["gesture"]}_interactive.html',include_plotlyjs=True,post_script=script)
    page=['<!doctype html><meta charset="utf-8"><title>原始CGR与Boya候选对照</title><style>body{font:18px system-ui;max-width:1500px;margin:24px auto;color:#202938}img{width:100%}section{margin:36px 0}a{margin-right:20px}p{line-height:1.6}</style>',
        '<h1>原始CGR → Boya候选：同场景、同视角对照</h1><p>左侧是作者模型输出的CGR及宽度标尺，并非DH3实体手或作者实机成功证明。右侧为保存的Boya候选经FK放置的完整源网格。历史标签仅用于说明该样例结局，图中未重新执行抓取。</p>',
        '<p>橙实线：原始宽度；橙虚线：1.7倍请求；绿箭头：实际接近方向；紫虚线/紫蓝点：Boya计算接触参考。计算参考点尚未经过硬件接触标定。物体015/029/005均为原观察姿态。</p>',
        f'<p>g7拒绝记录：接近末段lfdistal与物体005的源凸代理相交，最大记录{rejection["maximum_recorded_overlap_mm"]:.6f}mm；RRT未启动。红色仅标出涉及的小指link，不是精确接触点。</p>']
    for c in cases:
        g=c['gesture'];page.extend([f'<section><h2>g{g} / #{c["source_row_index"]} / {c["status"]}</h2>',f'<p><a href="g{g}_interactive.html">旋转三维对照（左右相机联动；点击图例隐藏部件）</a><a href="{c["rgb_image"]}">原RGB上的投影对照</a><a href="{c["image"]}">查看原尺寸图片</a></p>',f'<img src="{c["image"]}" alt="g{g}原始CGR与Boya候选俯视侧视对照"></section>'])
    (out/'index.html').write_text('\n'.join(page))
    B.verify(bindings)
    print(json.dumps(dict(output=str(out),status='complete',physics_steps=0,wall_s=time.monotonic()-began)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args().output_dir)
