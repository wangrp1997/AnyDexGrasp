"""Publish paired recorded-state videos and verify their media bindings."""
from pathlib import Path
import argparse
import hashlib
import html
import json


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def publish(run, audit_path, media_dir=None):
    import imageio_ffmpeg
    result = read(run/'RESULTS.json')
    audit = read(audit_path)
    media_dir = media_dir or run/'videos'
    media = read(media_dir/'RESULTS.json')
    assert result['status'] == 'complete' and media['complete']
    assert audit['complete'] and audit['results']['sha256'] == sha(run/'RESULTS.json')
    assert audit['protocol']['sha256'] == result['protocol_sha256'] == sha(run/'protocol.json')
    assert media['physics_steps_during_replay'] == 0
    checks = []
    for row in result['results']:
        cid = row['object_id']
        folder = media_dir/cid
        report = read(folder/'render_check.json')
        evaluation = read(row['evaluation'])
        video = folder/'hand_only.mp4'
        assert report['source_trace_sha256'] == evaluation['trace_sha256']
        assert report['source_evaluation_sha256'] == row['evaluation_sha256'] == sha(row['evaluation'])
        assert report['task_success'] == evaluation['task_success']
        assert report['physics_steps_during_replay'] == 0
        assert report['video_sha256'] == sha(video)
        frames = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
        metadata = next(frames)
        fps = float(metadata['fps'])
        assert tuple(metadata['size']) == (1280,560)
        decoded = 0
        for frame in frames:
            assert len(frame) == 560*1280*3
            decoded += 1
        assert abs(fps-25.) < 1e-10
        assert decoded == report['frames'] + 25
        probe = dict(width=1280, height=560, fps=fps,
                     decoded_frames=decoded, duration_s=decoded/fps,
                     verifier='Existing imageio-ffmpeg full-frame decode')
        assert max(report['max_render_matrix_errors']) <= 1e-6
        assert min(report['pixel_motion'].values()) > .1
        checks.append(dict(object_id=cid, video_sha256=sha(video), probe=probe,
            trace_sha256=evaluation['trace_sha256'], evaluation_sha256=row['evaluation_sha256'],
            physics_steps_during_replay=0, render_check_sha256=sha(folder/'render_check.json')))
    title = 'Boya 闭合几何修正：四组固定姿态对照'
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>'+title+'</title><style>body{font:17px/1.6 system-ui;max-width:1400px;margin:30px auto;padding:0 20px;background:#eef2f6;color:#173047}section{background:white;padding:20px;border-radius:12px;margin:24px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%;background:#111}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #ddd;padding:8px}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>',
        '<h1>'+title+'</h1>',
        '<p>原候选、评分、手指闭合、物体及碰撞代理固定；仅改变手掌平移。已知物体诊断，每组各一次，非独立泛化评测。原外部 1/12、新六件 3/6 不变。</p>',
        f'<p><b>基线 {result["paired_success"]["baseline"]}/4，修正 {result["paired_success"]["refined"]}/4；{result["actual_grasps"]} 次执行，{result["physical_steps"]} 步。</b></p>',
        '<p>下方视频为真实保存状态的正常速度 RTX 回放；回放零积分。视频字幕 Object rise 是物体参考点位移，不能当作底面离桌高度。结果由源网格底面、末态双侧承载及原执行门判定。</p>']
    for oid in ('egad_A0','egad_A3','egad_A6','egad_E0'):
        parts.append('<section><h2>'+oid+'</h2><div class="pair">')
        for variant, label in (('baseline','原姿态'),('refined','闭合几何平移')):
            cid=oid+'_'+variant
            row=next(r for r in result['results'] if r['object_id']==cid)
            e=read(row['evaluation'])
            pickup=e['tabletop_pickup']
            outcome='成功' if row['task_success'] else '失败'
            delta=[round(v*1000,3) for v in row['delta_world_m']]
            relative_video=(media_dir/cid/'hand_only.mp4').relative_to(run)
            parts.append(f'<div><h3>{label}：{outcome}</h3><video controls preload="metadata" src="{html.escape(str(relative_video))}"></video><p>世界 XYZ 平移 {delta} mm。')
            if pickup:
                parts.append(f'末态参考点升高 {pickup["final_reference_rise_m"]*1000:.3f} mm；源底面高度 {pickup["final_bottom_clearance_m"]*1000:.3f} mm。')
            parts.append('</p></div>')
        parts.append('</div></section>')
    parts.append('<p>粗凸碰撞代理、单帧可见表面和受控腕部均为诊断边界；没有完整机械臂验证或训练更新。详细数字见 RESULTS.json 与独立审计。</p></html>')
    (run/'index.html').write_text('\n'.join(parts))
    receipt=dict(complete=True, protocol_sha256=result['protocol_sha256'], results_sha256=sha(run/'RESULTS.json'),
        independent_audit_sha256=sha(audit_path), renderer_results_sha256=sha(media_dir/'RESULTS.json'),
        index_sha256=sha(run/'index.html'), cases=checks, physics_steps_during_replay=0)
    (run/'MEDIA_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--media-dir',type=Path)
    a=p.parse_args()
    r=publish(a.run.resolve(),a.audit.resolve(),a.media_dir.absolute() if a.media_dir else None)
    print(json.dumps(dict(complete=r['complete'], videos=len(r['cases']), physics_steps_during_replay=0)))
