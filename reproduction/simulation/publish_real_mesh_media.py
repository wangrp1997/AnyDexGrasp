from __future__ import annotations
import argparse,collections,html,json,shutil,sys,time
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
sys.path[:0]=[str(ROOT/'reproduction/simulation'),str(ROOT/'outputs/native_deps/python_overlay')]
from encode_recorded_video import encode

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();root=a.root.resolve();report=json.loads((root/'run/RESULTS.json').read_text());video=root/'videos';video.mkdir(exist_ok=True)
    cards=[];records=[];st=time.monotonic();rows=report['scenes'];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    tilew=640;tileh=220;sheet=Image.new('RGB',(tilew*2,tileh*((len(rows)+1)//2)),(245,245,245));draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(rows):
        name=row['scene_id'];physics=Path(row.get('physics_output_dir',root/'run/physics'/name));scene=Path(row['scene_manifest']).parent
        item={k:row.get(k) for k in ['scene_id','status','physics_invocations','actual_physical_attempts','completed_physics_steps','possible_in_flight_steps','object_final_lift_m','object_final_lowest_world_z_m','port_stability_pass','not_entered_physics_reason']}
        manifest=json.loads((scene/'scene_manifest.json').read_text());item['input_image']=str(scene/'color.png');item['scene_manifest_sha256']=row['scene_manifest_sha256']
        frames=sorted((physics/'frames').glob('frame_*.png'))
        if frames and (physics/'RESULTS.json').exists():
            if not (physics/'VIDEO.json').exists():encode(physics)
            receipt=json.loads((physics/'VIDEO.json').read_text());dst=video/(name+'.mp4');shutil.copy2(receipt['video'],dst);item.update(video=str(dst),video_receipt=str(physics/'VIDEO.json'),video_duration_s=receipt['video_duration_s'],video_sha256=receipt['video_sha256'])
            result=json.loads((physics/'RESULTS.json').read_text());item['physics_metrics']={k:result[k] for k in ['lift_observed','object_max_lift_m','max_gear_residual_rad','peak_joint_velocity_rad_s','max_wrist_COM_error_m','minimum_hand_object_contact_distance_m','minimum_hand_table_contact_distance_m','hand_table_contact_rows']}
            images=[frames[j] for j in [0,len(frames)//2,len(frames)-1]]
            content='<video controls preload="metadata" src="videos/'+name+'.mp4" poster="'+html.escape(str(images[0].relative_to(root)))+'"></video>'
        else:
            images=[scene/'color.png']*3;content='<img src="'+html.escape(str((scene/'color.png').relative_to(root)))+'"><p>未进入实际执行；没有抓取视频。</p>'
        x=(i%2)*tilew;y=(i//2)*tileh
        lift='n/a' if row.get('object_final_lift_m') is None else '%.2f mm'%(row['object_final_lift_m']*1000)
        draw.text((x+8,y+6),name+' | '+row['status'],font=font,fill=(0,90,25) if row['status']=='object_lifted' else (170,30,20))
        draw.text((x+8,y+27),'final lift '+lift+' | steps '+str(row['completed_physics_steps']),font=font,fill=(30,30,30))
        for j,path in enumerate(images):
            im=Image.open(path).convert('RGB');im.thumbnail((208,156));sheet.paste(im,(x+j*212+3,y+54))
        cards.append('<article><h2>'+html.escape(name)+'</h2><p>'+html.escape(row['status']+' | final lift '+lift)+'</p>'+content+'</article>');records.append(item)
    sheet.save(root/'contact_sheet.jpg',quality=92)
    summary=dict(scope='Six original real YCB meshes, yaws0/90, independent hand-only top1 simulation; no arm or paper benchmark reproduction',source_cohort=str(root/'run/RESULTS.json'),count=len(rows),counts=dict(collections.Counter(r['status'] for r in rows)),aggregate=report['aggregate'],timing_seconds=report['timing_seconds'],media_wall_seconds=time.monotonic()-st,additional_physics_steps=0,scenes=records)
    (root/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    doc='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AnyDex DH3 真实网格验证</title><style>body{font-family:system-ui;margin:24px;background:#f4f5f7;color:#19202b}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}article{background:white;border-radius:10px;padding:16px}video,img{width:100%}h2{font-size:19px}</style><h1>AnyDex DH3：六类真实网格 × 两个方向</h1><p>每场景只执行官方原 top1 一次；所有失败保留。独立手部仿真，没有机械臂，不是论文实机成功率。</p><p>'+html.escape(str(summary['counts']))+'</p><p><a href="SUMMARY.json">结构化结果</a> | <a href="contact_sheet.jpg">过程对照图</a></p><main>'+''.join(cards)+'</main></html>'
    (root/'index.html').write_text(doc)
    print(json.dumps(dict(status='media_ready',scenes=len(rows),counts=summary['counts'],media_wall_seconds=summary['media_wall_seconds'],index=str(root/'index.html'),contact_sheet=str(root/'contact_sheet.jpg')),indent=2))
if __name__=='__main__':main()
