"""Show two actual RGB captures and verify shared native world/camera handles."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import author_collection_batch as B
from render_author_boya_candidate_comparison import font


def render(root,output):
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    data=[];bindings={}
    for i in range(2):
        capture=root/'scenes'/f'{i:02d}'/'capture'
        scene=B.read(capture/'observed_scene.json');obs=B.read(capture/'observation.json');image=Image.open(capture/'color.png').convert('RGB');image.load()
        data.append((scene,obs,image))
        for name in ('observed_scene.json','observation.json','color.png'):bindings[f'{i}:{name}']=B.binding(capture/name)
    for key in ('robot_handle','camera_handle','sim_handle'):
        if data[0][0]['persistent_world'][key]!=data[1][0]['persistent_world'][key]:raise ValueError('Native world/camera/robot was replaced')
    for key in ('T_WC','intrinsic_matrix'):np.testing.assert_array_equal(data[0][1][key],data[1][1][key])
    if any(d[0]['persistent_world']['world_builds']!=1 for d in data):raise ValueError('World was rebuilt')
    w,h=data[0][2].size;sheet=Image.new('RGB',(2*w+60,h+225),'white');draw=ImageDraw.Draw(sheet)
    draw.text((20,12),'固定场景连续采集：同一相机、机器人、物理世界，只在轮次间换物体',font=font(32),fill=(25,32,44))
    for i,(scene,obs,image)in enumerate(data):
        ids='/'.join(x['object_id'].split('_')[-1]for x in scene['objects'])
        draw.text((20+i*(w+20),75),f'第{i+1}轮：{ids}；实际RGB图像',font=font(27),fill=(25,32,44));sheet.paste(image,(20+i*(w+20),120))
    draw.text((20,h+145),'句柄与相机标定逐项核对一致；这两张图来自保存的RTX采集，不重跑物理。',font=font(25),fill=(35,95,65))
    image_path=output/'same_world_two_rounds.png';sheet.save(image_path)
    with Image.open(image_path)as check:check.load()
    B.verify(bindings);B.save(output/'RENDER_AUDIT.json',dict(status='pass',sources=bindings,renderer=B.binding(__file__),image=B.binding(image_path),
        same_robot_camera_sim_handles=True,same_camera_intrinsics_and_pose=True,world_builds=1,new_physics_steps=0),exclusive=True)
    return image_path


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();print(render(a.root,a.output_dir))
