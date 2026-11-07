# 单一Boya评分前域配置：来源与边界

配置文件：`/tmp/boya_prehead_policy_20261107_v1.json`，顶层`config`可直接传入`apply_boya_domain`。SHA256：`c679780aee4c6b4294fb993ad70d4d70682dbdbc4e04f92f8d799b500ef1b67a`。

- 世界工作区：`[[-1.4,-0.18028318377553387,0.0005],[-0.10000000000000009,0.4682583491909351,0.398129951499991]]`m。
- 接近锥：world轴`[0,0.5633369342972592,-0.8262272680422532]`，点积严格`>.9`。
- raw宽度：严格`(.022018084578542558,.07677524081551822)`m，即约22.018–76.775mm。

工作区来源是冻结相机1280×720、K=(fx=fy=640,cx=640,cy=360)、网络深度[.20,.65]的8个near/far角点。使用完整viewport边界u=[0,W]、v=[0,H]，是包含所有实际像素的保守外包边界，非按某个物体投影裁剪。将角点通过原生核验T_WC变换，先求world AABB，再与原`build_collection_pool.geometry`的z>.0005台面裕量求交。相机统一瞄准[-.75,.10,.06]，覆盖原声明放置中心[-.75,.10]。原源码仅声明中心与z=0平面，没有有限工作区边长；本次明确新声明为**传感器域AABB包络**，不伪称既有实体桌边、机械臂可达区域或语义前景。

方向轴取同一原生核验T_WC的第三列，严格保留作者camera+Z语义：a_W·(R_WC e_z)=a_C,z。纯代数独立检查1000方向，等式最大误差<1e-14。它是原相机方向锥的世界表示，不能称纯world竖直上抓规则。

宽度只依据原标定中ff且self_collision_clear的33槽/型，采用既有raw×1.7。四型允许raw开口包络分别为：g1[.017184884180849704,.07990584305691881]；g6[.005896223078994175,.07677524081551822]；g7[.007845264757557844,.07768421652100185]；g8[.022018084578542558,.08285765985028104]。本次global mask后仍给全部四head评分，因此预先固定共同交集，避免某型包络外输入依赖clamp。此范围是保守的共同参考点开口域，不是手硬件最大开口；离散槽间隙和原每型2mm nearest-slot/clamp仍走后续原映射，本策略不改其函数、不保证每个中间宽度均有2mm内的槽。

绑定的直接源文件均在policy的source_sha256中：
1. `outputs/s40_stable_supplement_20261107_v2/settling/objects.json`（只取camera字段）；
2. `collection/observations/graspnet_000/observation.json`（只取原生T_WC、K、相机spec；不是物体bbox或标签）；
3. `outputs/boya_adapter_preparation_20261105_v2/calibration.json`；
4. `robot_dh3.py`（原>.9方向语义）；
5. `reproduction/boya/build_collection_pool.py`（既有1.7及.0005几何裕量）；
6. `reproduction/boya/boya_prehead_domain.py`。

纯检查回执：`/tmp/boya_prehead_policy_check_20261107.json`。没有读取EGAD、内部标签、诊断结果或训练曲线，未使用物体bbox来选择参数；没有多配置试优或修改已冻结源码/数据，没有物理/模型/训练执行。
