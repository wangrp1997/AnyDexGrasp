# 实际试抓记录视频（2026-11-17）

用户要求先看视频自行分析。本轮只把当前v3的3次实际记录重渲染，不改抓姿映射、轴、角速度控制、物理参数或标签，也没有启动新试抓。

- [物体014单段，26.04秒](../../outputs/boya_mujoco_failure_videos_20261117_v2/trial_01_slow.mp4)
- [三次合集，78.12秒](../../outputs/boya_mujoco_failure_videos_20261117_v2/three_grasps_slow.mp4)
- [可暂停逐帧、切换原速的本地播放器](../../outputs/boya_mujoco_failure_videos_20261117_v2/index.html)

顺序为原场景2/raw1722/014、场景1/raw1412/011、场景6/raw1450/021；场景编号不是累计实际抓取次数。每段包含完整approach/close/hold/lift，默认五倍慢放。播放器选择“原速”即5倍播放文件，左右按钮按显示帧前后移动。

左侧近景，右侧俯斜；为看清接触，前臂仅在画面隐藏、右侧掌壳半透明。红/蓝/绿/紫/青依次为拇/食/中/环/小指，橙色物体按原日志接触最多的物体选定，只用于观察，不改变原采样。彩色三轴是URDF的`palm`链接坐标系，不是虚拟CGR或pad接触参考中心。画面载荷、物体世界XYZ角速度与源底高度均直接取原日志。

它是原实际关节q与物体pose的状态重建，不是同步相机录像或按目标指令做理想动画。独立MjData只调用`mj_kinematics/mj_camlight`更新几何，无`mj_step/mj_forward`或新的动力学。每个渲染采样使用原250Hz日志隔2步的实际状态，无姿态插值；末帧另保留。三段各651帧/25fps，均完整解码；合集1953帧/78.12秒、约4.61MB。实际物体姿态重建误差0，掌姿与归档ASCII模型的微小误差有数值回执。

三段预览已目视检查。[轻证据](evidence/author_mujoco_failure_videos_20261117_v1/RESULTS.json)与[解码检查](evidence/author_mujoco_failure_videos_20261117_v1/DECODE_REVIEW.json)保留绑定，大视频/帧图继续忽略。初版侧面被前臂遮挡，主动SIGINT仅中断所属渲染组，失败/源码归档保留；修正版RESULTS complete、owner结束且逐帧解码通过。修正版工具外层最终返回143，原因未证明，不能改称正常退出0；两个host渲染PGID都已不存在、3锁可取、registry released。0Kit/物理积分/CGR/新标签/训练，两个静态渲染模型不计采集建场或吞吐。当前等用户目视分析，不自行改轴或启用新闭合控制。
