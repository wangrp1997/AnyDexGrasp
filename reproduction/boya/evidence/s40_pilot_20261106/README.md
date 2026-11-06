# S40 Boya小批证据

本轮原尺度40件训练资产与000/002两物体采集链路。8槽=5条合格实际正例+3准备null；10500实际步，零新训练。不是原论文泛化成功率，原CAD/真实物理参数及完整测试身份仍未对齐。

- result_receipt.json：完整成本、5条实际、3null、限制和运行绑定。
- asset_audit.json：3195项独立保形/来源核验摘要。
- physical_audit.json：2444项实际轨迹/数值/索引核验；非法臂支撑接触流未全部留存的边界明示。
- renderer_audit.json：原生相机、源射线和RGBD重建CGR输入字节核验。
- media_audit.json：两录像容器/源GIF与零步回放核验。
- scale_sources.json、scale_summary.json：固定官方源码来源及米制/scale1依据。

完整源码来源副本/原始审计在忽略运行目录audit，原模型在public；网格、权重、轨迹、PDF和MP4不提交。
