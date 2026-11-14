"""Decode every new full-arm video and publish the fixed twelve-condition table.

Read-only source measurements and media delivery only; never imports a simulator.
"""
from pathlib import Path
import hashlib
import html
import json
import cv2

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'outputs/boya_augmented_arm_external6_20261114_v1'
AUDIT=ROOT/'outputs/boya_augmented_arm_audit_20261114_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    physics=read(AUDIT/'INDEPENDENT_PHYSICS_AUDIT.json')
    result=read(RUN/'execution/RESULTS.json');media=read(RUN/'videos/RESULTS.json')
    assert physics['status']==result['status']==media['status']=='complete'
    assert physics['metrics']==result['metrics'] and media['replay_physics_steps']==0
    rows={(v['object_id'],v['recipe']):v for v in physics['rows']}
    videos={(v['object_id'],v['recipe']):v for v in media['videos']}
    assert len(videos)==len(media['videos'])==len(rows)
    decoded=[]
    for key,v in videos.items():
        p=Path(v['path']);assert sha(p)==v['sha256']
        capture=cv2.VideoCapture(str(p));assert capture.isOpened(),str(p)
        fps=capture.get(cv2.CAP_PROP_FPS);declared=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));count=0
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while True:
            ok,frame=capture.read()
            if not ok:break
            assert frame.shape[:2]==(height,width);count+=1
        capture.release();assert count==declared and count>0 and fps>0
        # Existing renderer binds all original trace/palm/mesh inputs and records
        # expected playback durations. Retain its complete receipt by SHA.
        check_path=p.parent/'render_check.json';check=read(check_path)
        source=rows[key]
        assert check['complete'] and check['physics_steps_during_replay']==0
        assert check['original_robot_visual_meshes'] and not check['collision_visualization']
        assert check['source_trace_sha256']==source['source_trace_sha256']
        assert check['source_evaluation_sha256']==source['source_evaluation_sha256']
        assert check['source_task_success']==source['task_success']
        assert check['video_sha256']==v['sha256'] and fps==25.
        assert abs(check['physical_time_s']-source['steps']*.004)<1e-10
        assert count==check['frames']+round(check['final_pause_s']*fps)
        assert abs(count/fps-(check['physical_time_s']+check['final_pause_s']))<=1/fps+1e-9
        decoded.append(dict(object_id=key[0],recipe=key[1],path=str(p),sha256=v['sha256'],
            frames=count,fps=fps,duration_s=count/fps,width=width,height=height,
            all_frames_decoded=True,renderer_receipt_path=str(check_path),renderer_receipt_sha256=sha(check_path),
            renderer_receipt=check))
    styles='body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 16px;color:#202733}table{border-collapse:collapse;width:100%}th,td{padding:12px;border:1px solid #d8dde5;text-align:left}video{width:100%;max-width:960px}img{max-width:100%}.good{color:#08764b}.bad{color:#a23434}small{color:#596576}'
    page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Boya 完整机械臂新六件测试</title><style>'+styles+'</style>',
        '<h1>Boya 完整机械臂：新六件测试</h1>',
        '<p>同一组六件物体、同一张输入，使用原始候选增强。两套评分各自选姿，每件每套最多抓一次；无计划也计入分母。</p>',
        '<p>本轮是独立仿真测试，使用凸体碰撞近似；不是原论文实机成绩，也不是与旧批次的成功率提升对照。</p>',
        '<table><tr><th>评分版本</th><th>成功</th><th>实际抓取失败</th><th>无计划</th></tr>']
    labels={'original_mse':'Boya 按原配方训练','preactivation':'Boya 修改训练版'}
    for recipe in ('original_mse','preactivation'):
        m=result['metrics'][recipe]
        page.append('<tr><td>'+labels[recipe]+'</td><td>'+str(m['successes'])+'/6</td><td>'+str(m['actual_failures'])+'</td><td>'+str(m['no_plan'])+'</td></tr>')
    page.extend(['</table><p><a href="../../reproduction/boya/AUGMENTED_ARM_EXTERNAL6_RESULTS_20261114.md">完整报告</a></p>',
        '<details><summary>六件新输入</summary><img src="observation_contact_sheet.jpg" alt="六件物体的固定输入"></details>'])
    conditions=result['conditions']
    for oid in ('egad_'+r+str(c) for r in 'ACE' for c in (2,5)):
        page.append('<h2>'+oid+'</h2>')
        for recipe in ('original_mse','preactivation'):
            key=(oid,recipe);case=next(v for v in conditions if (v['object_id'],v['recipe'])==key)
            if key not in rows:
                assert case['selected'] is None
                page.append('<p>'+labels[recipe]+'：无可执行计划（计失败）。</p>')
                continue
            row=rows[key];v=videos[key];status='成功' if row['task_success'] else '失败'
            page.append('<h3>'+labels[recipe]+'：'+status+'</h3>')
            if row['failure_reason']:page.append('<small>'+html.escape(str(row['failure_reason']))+'</small>')
            page.append('<video controls preload="metadata" src="'+Path(v['path']).relative_to(RUN).as_posix()+'"></video>')
    (RUN/'index.html').write_text('\n'.join(page)+'\n</html>')
    report=dict(schema='boya_augmented_arm6_media_audit_v1',status='complete',videos=decoded,
        unique_videos=len(decoded),total_decoded_frames=sum(v['frames'] for v in decoded),
        all_frames_decoded=True,replay_physics_steps=0,source_sha256={str(Path(__file__).resolve()):sha(__file__)})
    (AUDIT/'MEDIA_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete',videos=len(decoded),frames=report['total_decoded_frames'])))


if __name__=='__main__':main()
