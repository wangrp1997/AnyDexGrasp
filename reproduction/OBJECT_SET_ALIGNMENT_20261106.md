# AnyDexGrasp原物体集合：身份、资产和缺失项

2026-11-06。此处为物体来源对照，不是新的抓取评测。此前DGN40/10及其4/10、6/10保持原记录，不能改名为论文物体集。

## 已实际取得的资产

| 论文集合 | 本轮结果 | 身份与资产边界 |
| --- | --- | --- |
| S40：GraspNet原40训练对象 | 从本机GraspNet scene0000–0099的100份object_id_list重建40个编号；40件非纹理原网格、简化网格及URDF均存在，已逐文件SHA | 位于DexGraspNet2发布物的GraspNet派生副本；尚未与GraspNet官方models.zip或AnyDex作者CAD逐字节比较 |
| L144：AnyGrasp144训练对象 | 论文给出集合说明和图 | 尚未取得完整机器可读ID表及对应CAD清单 |
| T30：L中的30件子集 | 论文说明随机选取及图 | 尚未取得精确子集ID或可重建随机种子 |
| 对抗测试中的EGAD49 | 官方QUT评测包已完整下载、ZIP校验并解压；A0–G6共49个OBJ均有SHA/边界 | 可核验发布网格身份；AnyDex实际打印尺度/质量/材料尚未核验 |
| 对抗测试中的Dex-Net13 | 论文说明人工选择13件 | 尚未取得原13件选择ID，不能随选13件替代 |
| 约150件实物测试集合 | 原论文说明各类随机杂乱摆放、清桌两轮 | 尚未取得完整逐件身份表；不能用GraspNet test_seen/test_novel划分冒充 |

S40编号为：

```text
000 002 005 007 008 009 011 014 015 017
018 020 021 022 026 027 029 030 034 036
037 038 040 041 043 044 046 048 051 052
056 057 058 060 061 062 063 066 069 070
```

推导按官方GraspNet API固定版本eb57dd2：train为scene0–99，getObjIds读取这些scene的object_id_list合集。本机上述100份文件恰得40件；证据保存每份场景表SHA及每个模型SHA，不凭物体名字或数量匹配。DGN2官方README说明发布数据包含meshdata/scenes并在预处理使用GraspNet；本机副本尚缺与GraspNet原归档的逐字节身份校验，故不能升级为完全已验收的原CAD。

原网格位于`/mnt/public/datasets/DexGraspNet2.0/extracted/meshdata/NNN/`，只读使用。记录网格原坐标包围盒及URDF的scale1，**不能继承旧DGN_5k的0.1缩放，也不能继承25mm/106mm/130mm物体筛选。** 如果使用简化网格做仿真，其几何近似须另记，不称原物理资产。

EGAD官方压缩包5,747,116B，SHA256为`f3fff4c7f46331b87dbd916a437c95d330a53dc5dbf11c391c50f09b07773137`。public新目录为`/mnt/public/datasets/AnyDexGrasp/object_alignment_20261106_v1/`，49模型在`egad_eval_set/`；完整清单`graspnet_s40_inventory.json`和`egad_eval_inventory.json`。首次60秒下载超时保留在source_receipt，续传成功记在egad_resume_receipt，不把部分ZIP当完整。EGAD README明确单位取决于生成时的夹爪宽度参数，不能随意把OBJ坐标当成原AnyDex实物尺寸。

## 论文与官方来源

本机论文`outputs/paper_code_audit/source/sections/methods.tex:88,93,99-111`说明S40/L144/T30、表示训练场景和真实试错采数；`results.tex:113-145`说明约150实物清桌及13+49对抗集；`supp.tex:78`将三套CAD标为available upon request。这里“尚未取得”限于已检查的论文、固定官方源码和当前本地资产，不推断作者没有任何其它分享渠道。

官方GraspNet API、EGAD README固定版本，以及DGN2官方README的下载来源、实际文件SHA均在`evidence/object_alignment_20261106/SOURCES.json`。Git只提交轻量清单，网格/压缩包/原论文不提交。

## 当前结论

已建立可用的S40派生资产清单和EGAD49官方资产清单；原数据整体对齐仍未完成。Boya四个现有head仍用旧DGN采数训练，未因取得这些模型而变成原S40训练模型。下一阶段应基于确认的身份及显式尺度建立独立场景/采集协议；L144、T30和全测试集缺失项保持独立，不用自建集合填补后称同数据集复现。本轮没有新增训练或物理抓取。
