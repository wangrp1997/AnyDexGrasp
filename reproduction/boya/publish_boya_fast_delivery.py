"""Publish local evidence/media for the bounded Boya study; no simulation."""
import argparse
import html
import json
import os
from pathlib import Path
from run_boya_fast6 import ROOT, read, save, sha


def publish(output):
    output = output.resolve()
    output.mkdir(exist_ok=False)
    planroot = ROOT/'outputs/boya_fast_planning_20261111_v1'
    diagnostic = ROOT/'outputs/boya_candidate_diagnostic_20261111_v2'
    external = ROOT/'outputs/boya_fast6_20261111_v1'
    planning = read(planroot/'RESULTS.json')
    cases = read(diagnostic/'RESULTS.json')
    execution = read(external/'execution/RESULTS.json')
    queue = read(external/'planning/queue_manifest.json')
    scored = read(external/'diagnostic/RESULTS.json')
    if not planning['complete'] or cases['status']!='complete' or execution['status']!='complete':
        raise ValueError('Completed results and media required')
    if execution.get('media_errors') or execution['objects'] != 6:
        raise ValueError('Incomplete media or changed denominator')
    input_paths = [planroot/'RESULTS.json', diagnostic/'RESULTS.json',
                  diagnostic/'INDEPENDENT_AUDIT.json', diagnostic/'MEDIA_VERIFICATION.json',
                  external/'execution/RESULTS.json', external/'planning/queue_manifest.json',
                  external/'protocol.json', external/'diagnostic/RESULTS.json']
    inputs = {str(p):sha(p) for p in input_paths}
    bytrial = {r['object_id']:r['trial'] for r in execution['results']}
    geometry = {r['object_id']:r['geometry_admitted'] for r in scored['objects']}
    rows = []
    for obj in queue['objects']:
        oid = obj['object_id']
        trial = bytrial.get(oid)
        if trial is None:
            outcome = '无完整臂计划' if geometry[oid] else '无手部几何候选'
        else:
            outcome = '成功' if trial['label']==1 else '实际失败' if trial['label']==0 else '任务早停'
        rows.append(dict(object_id=oid, geometry_admitted=geometry[oid],
                         arm_candidates=obj['arm_preparations'], outcome=outcome,
                         trial_id=trial['trial_id'] if trial else None))
    summary = dict(schema='anydex_boya_fast_delivery_v1', complete=True,
        planning=dict(old_null_objects=8, original_candidates=29,
                      ready_objects=planning['ready_objects'], ready_candidates=planning['ready_candidates'],
                      physics_steps=0),
        known_candidate_diagnostic=dict(objects=3, attempts=cases['actual_grasps'],
             successes=cases['success'], physical_steps=cases['physical_steps'],
             not_top1_or_external_success_rate=True),
        new_external6=dict(objects=6, successes=execution['successes'],
             actual_attempts=execution['actual_attempts'], physical_steps=execution['physical_steps'],
             no_plan_objects=execution['no_plan_objects'], rows=rows),
        initialization_object_steps=read(external/'settling/RESULTS.json')['object_steps'],
        unchanged_training_records=435, new_training_updates=0, new_CGR_forwards=6,
        old_external12_result='1/12 unchanged',
        limitations=['Six fixed single-object scenes are small-scale validation',
                     'Original convex proxies remain approximate; not source CAD dynamics',
                     'Known-object multi-candidate diagnosis is not top1 evaluation',
                     'Paper reproduction and reliable Boya adaptation remain unaccepted'],
        source_sha256={str(Path(__file__).resolve()):sha(__file__)}, input_sha256=inputs)
    save(output/'SUMMARY.json',summary)
    def rel(p):
        p=Path(p)
        if not p.is_file():raise FileNotFoundError(p)
        return html.escape(os.path.relpath(p,output))
    text = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Boya 有限诊断与六件验收</title>',
            '<style>body{font:16px/1.65 system-ui;max-width:1100px;margin:36px auto;padding:0 20px;color:#17252e}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd4d8;padding:8px;text-align:left}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}video,img{max-width:100%}small{color:#596975}</style>',
            '<h1>Boya 有限诊断与新六件验收 · 2026-11-11</h1>',
            f'<p>新六件：<b>{execution["successes"]}/6 成功</b>，{execution["actual_attempts"]} 次实际执行，{len(execution["no_plan_objects"])} 件无计划。每件最多一次，无结果驱动重抓。</p>',
            '<p>模型及435条数据未改。唯一部署改动是有预算的额外IK分支检查与条件中转路径。旧外部1/12保持。整体论文复现未验收。</p>',
            '<p>碰撞代理为统一V-HACD近似；六件采样源表面差最大约33.48mm。原生20µm门只验证代理导入，不代表原CAD精确。</p>',
            '<p>原录像字幕“物体升高”表示物体参考原点位移，不是底面离桌高度；E1虽显示约8.81cm仍最终贴桌失载，正确结果为失败。</p>',
            '<h2>新六件完整结果</h2><table><tr><th>物体</th><th>手几何候选</th><th>臂候选次数</th><th>结果</th></tr>']
    for row in rows:
        text.append('<tr>'+''.join('<td>'+html.escape(str(row[k]))+'</td>' for k in ('object_id','geometry_admitted','arm_candidates','outcome'))+'</tr>')
    text.append('</table><h2>六件实际输入</h2><div class="grid">')
    for row in rows:
        oid=row['object_id']
        text.append(f'<figure><img src="{rel(external/"observations"/oid/"color.png")}"><figcaption>{oid}</figcaption></figure>')
    text.append('</div><h2>新六件实际动作</h2>')
    if not execution['videos']:
        text.append('<p>没有获得实际执行计划，因此没有动作录像。</p>')
    for video in execution['videos']:
        text.append(f'<p>{html.escape(video["object_id"])}，label={video["label"]}</p><video controls preload="metadata" src="{rel(video["path"])}"></video>')
    text.append('<h2>三个已知失败物体：九个新候选诊断</h2><p>1/9成功，仅A6 rank33。它证明有限候选池中存在可抓姿态；这是受控腕多候选诊断，不是完整臂top1泛化成绩。</p>')
    text.append(f'<p><a href="{rel(diagnostic/"videos/index.html")}">查看全部9段成功与失败视频</a> · <a href="{rel(planroot/"SUMMARY.json")}">查看旧29候选纯规划对照</a></p>')
    text.append('<p>旧8个无计划对象恢复2个完整计划（E3/G0），未实际执行。没有发生条件中转路径尝试，不能称绕障已验证。</p>')
    text.append(f'<p><a href="{rel(output/"SUMMARY.json")}">机器可读汇总及来源SHA</a></p></html>')
    (output/'index.html').write_text('\n'.join(text))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    publish(p.parse_args().output)
