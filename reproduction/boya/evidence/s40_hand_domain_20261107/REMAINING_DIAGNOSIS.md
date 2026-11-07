# S40 v3剩余009/048只读分诊（2026-11-07）

全部零物理、零新网络前向、零旧标签/EGAD读取。冻结运行入口、policy与分门均未修改；真值网格只用于诊断，不接入选姿。

## 结论

当前冻结的 raw 候选、四型标定slot、双flip与四档额外深入范围内，两件都没有通过完整原几何门的组合。因此不能把剩余失败归因为head评分不足；也不能据此判手硬件绝对抓不了。证据指向“二指候选→固定开手标定”的几何覆盖不足。009实际非常薄，048有主体厚度但开手落位与可见表面冲突。相机没有把对象裁出画面或网络深度范围。

## 原域全部少量组合绕过评分复核

### graspnet_009

- 原support域 1个组合，完整原geometry通过0，全部visible_scene_point_in_hand_hull。评分门以上0个；门以下组合也没有漏掉的几何可行解。
- 唯一组合为raw1499 / g7 / flip=True / extra0；score=0，原geometry在预抓退距20mm处thdistal包住1个可见点而拒绝。该点z=14.633mm，位于源物体world AABB内（AABB只是归属代理，不是语义分割）。
- 源local AABB尺寸（mm）：[96.047, 201.499, 15.663]；实测pose下world尺寸（mm）：[96.048, 201.496, 15.386]。这里只报AABB轴向厚度，不伪称旋转不变最小几何厚度。
- 所有262140个源顶点都在1280×720画面及网络depth(.20,.65)m内；光学z范围[0.513068575890178, 0.632174983893479]；可见sparse cloud中427点落在对象world AABB内。投影bbox uv=[583.1652084571755, 304.0150545343626]到[688.769540756357, 499.6791837435827]。完整可见面/真实分割未重算，但可排除对象整体在相机外或被depth裁掉。
- 最接近原table门的失败：raw1499 / g6 / flip=True / extra0；最低点0.283301mm，距离原0.5mm门差0.216699mm。这不是可直接修正证书：平移后还要原cloud/arm检查，当前不改深度或门。

- 总raw2048，raw≥.099排935；源对象AABB内raw中心39个，**其中宽度≥.099为0**，都已通过raw宽度门。恰好float32 .099的总raw为0。
- 被strictmax排除且仍在冻结workspace的215个raw，诊断性复用同样per-type×1.7/clip/nearest-slot、双flip、四extra：1720个映射query、6880次support检查，**support通过0、完整geometry通过0**。没有head新增评分。最好的这类组合最低点-14.682mm，仍未过table。

### graspnet_048

- 原support域 10个组合，完整原geometry通过0，全部visible_scene_point_in_hand_hull。评分门以上3个；门以下组合也没有漏掉的几何可行解。
- 10个组合来自raw85、239、1993，均extra0/flip=True；原geometry均在终点distance0已有可见点进入开手凸包。raw85主要为thdistal，raw239为ffproximal，raw1993为thdistal。点的world z约9.59–39.18mm；首个冲突点集合均落在源物体world AABB内，非仅桌面点冲突。
- 源local AABB尺寸（mm）：[137.494, 98.65, 53.709]；实测pose下world尺寸（mm）：[139.81, 98.73, 50.458]。这里只报AABB轴向厚度，不伪称旋转不变最小几何厚度。
- 所有213402个源顶点都在1280×720画面及网络depth(.20,.65)m内；光学z范围[0.5395560104400988, 0.6083647269083866]；可见sparse cloud中453点落在对象world AABB内。投影bbox uv=[561.3953471107476, 325.8205745262553]到[720.3236943348898, 439.2514671709356]。完整可见面/真实分割未重算，但可排除对象整体在相机外或被depth裁掉。
- 最接近原table门的失败：raw239 / g7 / flip=True / extra1；最低点0.186452mm，距离原0.5mm门差0.313548mm。这不是可直接修正证书：平移后还要原cloud/arm检查，当前不改深度或门。

- 总raw2048，raw≥.099排944；源对象AABB内raw中心49个，**其中宽度≥.099为0**，都已通过raw宽度门。恰好float32 .099的总raw为0。
- 被strictmax排除且仍在冻结workspace的219个raw，诊断性复用同样per-type×1.7/clip/nearest-slot、双flip、四extra：1752个映射query、7008次support检查，**support通过0、完整geometry通过0**。没有head新增评分。最好的这类组合最低点-14.682mm，仍未过table。

## .099饱和解释纠正

不能把约900条被排除raw统称“恰好 .099 饱和”。原robot_dh3.py第76–77行对每侧宽度作MAX_GRASP_WIDTH裁剪，117–136行选出两侧再相加，最终raw总宽可以大于.099。这两帧raw总宽最大约.175m；>=.099分别935/944条，恰好float32 .099都是0。原strictmax对Boya总体仍是需要显式讨论的train/deploy域差异，但本轮对这两帧的零物理counterfactual没有发现它藏着可行Boya姿态。

## 边界与后续含义

没有改分门、工作区、手桌裕量、碰撞谓词、深入档位或slot。没有将AABB/真值输入评分或排序。现有证据只确定冻结候选域内没有完整静态几何解；不能分离归因到CGR候选网络本身、参考标定本身或手形硬件限制，也不能证明放宽碰撞能安全抓起。若后续改进，应作为独立预声明的候选/适配覆盖研究，不按这两件ID增补规则。

证据：`REMAINING_DIAGNOSIS.json`、`EXCLUDED_MAXWIDTH_DIAGNOSIS.json`；原只读分诊脚本在本机`/tmp/diagnose_s40_*_20261107.py`，未纳入冻结部署或改变结果。
