#!/usr/bin/env python3
"""Publish local offline-score diagnostics; no network or simulator actions."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def publish(root):
    result = json.loads((root / "RESULTS.json").read_text())
    protocol = json.loads((root / "PROTOCOL.json").read_text())
    if result["status"] != "complete":
        raise ValueError("Completed offline control required")
    out = root / "review"
    out.mkdir(exist_ok=False)
    names = ["object_prior", "condition_linear", "shared_linear", "shared_mlp64",
             "original_mse", "preactivation_mse"]
    labels = {"object_prior": "物体等权常数", "condition_linear": "仅手型等条件的线性模型",
              "shared_linear": "CGR＋条件共享线性模型", "shared_mlp64": "CGR＋条件64单元网络（三种子平均）",
              "original_mse": "旧原配方评分头", "preactivation_mse": "旧预激活损失评分头"}
    tables = []
    for title, key in [("训练物体四折验证（364条，29件有标签，22件有正负样本）", "cv_metrics"),
                       ("旧内部集开发诊断（71条，6件有标签，5件有正负样本）", "internal_metrics")]:
        rows = []
        for name in names:
            if name not in result[key]:
                continue
            m = result[key][name]
            rows.append("<tr><td>" + labels[name] + "</td>" + "".join(
                f"<td>{v:.4f}</td>" for v in [m["object"]["macro_object_auc"],
                    m["same_observation"]["macro_object_auc"], m["object"]["macro_object_brier"],
                    m["object"]["macro_mixed_top_tie_mean_label"]]) + "</tr>")
        tables.append(f"<h2>{title}</h2><table><thead><tr><th>模型</th><th>物体内成对排序AUC ↑</th>"
                      "<th>同观测内排序AUC ↑</th><th>物体等权Brier ↓</th><th>混合物体池首选标签均值 ↑</th>"
                      "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    colors = ["#737373", "#ce8c30", "#2b6a99", "#228269", "#9b5c79", "#735da5"]
    for ax, title, key in zip(axes, ["Grouped OOF: 22 mixed objects", "Reused internal: 5 mixed objects"],
                              ["cv_metrics", "internal_metrics"]):
        present = [n for n in names if n in result[key]]
        xs = np.arange(len(present))
        values = [result[key][n]["object"]["macro_object_auc"] for n in present]
        ax.bar(xs, values, color=[colors[names.index(n)] for n in present])
        ax.axhline(.5, color="black", linewidth=.8, linestyle="--")
        ax.set_ylim(0, 1)
        ax.set_xticks(xs, [n.replace("_", "\n") for n in present], fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Macro within-object pair AUC")
        for x, y in zip(xs, values):
            ax.text(x, y+.015, f"{y:.3f}", ha="center", fontsize=9)
    fig.suptitle("Historical executed pools: ranking diagnostics, not physical success rates")
    fig.savefig(out / "ranking_comparison.png", dpi=160)
    plt.close(fig)
    gate = result["gate"]
    decision = "达到预设离线筛选线；仍需另立外部协议" if gate["passed"] else "未达到预设筛选线；停止本轮，不晋级物理部署"
    check_rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{'通过' if v else '未过'}</td></tr>"
                         for k, v in gate["checks"].items())
    bootstrap = result["cv_bootstrap"]
    text = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Boya共享评分器离线对照</title><style>body{{font:16px/1.7 sans-serif;max-width:1180px;margin:35px auto;padding:0 20px;color:#203040}}table{{border-collapse:collapse;width:100%;margin:15px 0}}td,th{{border:1px solid #ccd4dc;padding:7px;text-align:left}}th{{background:#eef3f7}}img{{max-width:100%}}.status{{padding:15px;background:#eef3f7;border-left:4px solid #426984}}code{{overflow-wrap:anywhere}}h2{{margin-top:30px}}</style>
<h1>Boya共享评分器离线对照 · 2026-11-11</h1>
<p class="status"><strong>{decision}</strong><br>按训练物体交叉验证选中：{labels[result['selected']]}。</p>
<p>原435条标签、抓姿映射及旧模型保持。仅训练25个离线开发模型，共14,500次优化更新；新增CGR前向、物理积分和外部物体均为0。</p>
<p>32/8是原声明划分；有标签对象实际为29/6。没有标签的5件不补负例、不复制样本。内部集已多轮用于开发，不能称独立外部测试。</p>
<img src="ranking_comparison.png" alt="离线排序指标对比">
{''.join(tables)}
<p>AUC在同时有成功和失败的组内计算，并列计0.5。同观测内比较的混合组数是训练31组、内部4组。Brier统一把分数裁剪到[0,1]后计算，旧头原始MSE另存JSON。
池首选标签均值仅从历史已执行候选中重排，最高分并列时对标签取均值，不是新策略真实试抓成功率。</p>
<h2>预先冻结的停止条件</h2><table>{check_rows}</table>
<p>所选模型相对条件线性模型，四折物体排序AUC差为{bootstrap['gain']:.4f}；按物体配对重采样的95%开发区间为[{bootstrap['gain_ci95'][0]:.4f}, {bootstrap['gain_ci95'][1]:.4f}]。该区间没有消除开发集复用和模型选择的乐观偏差。</p>
<p>两个线性模型固定1000轮；64单元网络固定300轮、三个种子平均；无按内部集挑轮数、种子、阈值或更换映射。原评分器只在原内部集上比较，未把它训练过的物体当交叉验证。</p>
<p>协议SHA：<code>{result['protocol_sha256']}</code></p>
<p><a href="../RESULTS.json">完整结果</a> · <a href="../PROTOCOL.json">冻结协议</a> · <a href="../SELECTION.json">验证前选型记录</a></p></html>"""
    (out / "index.html").write_text(text)
    return out / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    print(publish(parser.parse_args().output_dir.resolve()))
