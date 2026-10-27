# G5：官方优化检测结构与权重 + 共享 OCC 分支

配置：`bevdet-b0-det-occ-g5-official.py`。工作目录：`work_dirs/ep1_g5_official`。
本次在 AutoDL `/root/autodl-tmp/bevdet_flashocc` 添加；没有修改 G1/G2/G3/G4 或运行中的 QAT。

## 官方对齐范围与训练边界

依据 SDK《LQ560V200 NPU工具使用指南》BEVDet 优化节（PDF 第122页给出指标），
SDK `EfficientNetB0HZ`/`efficientnet_no_se.py` 源码，以及随附的
`data/model/bevdet/bevdet-efficientnetHz-cbgs-relu6.onnx`。

- 原生无 SE、ReLU6 的 B0 block 定义，不修改 torchvision 的 ImageNet 网络冒充专用预训练。
- 图像 backbone 输出 `[6,112,16,44]`、`[6,1280,8,22]`。
- 图像 FPN 接口 `[112,1280]`，bilinear / align_corners=False。
- BEV Neck 内部/输出128通道、channels_factor=1、align_corners=False，检测头输入128通道。
- 从 SDK **已经训练好的 ONNX** 移植整条检测链路的85个卷积，包括图像 backbone/FPN、
  LSS depth/context、BEV backbone/neck 和检测头，而不只是加载形状匹配的 ImageNet 权重。
- 使用现有 LSS/BEVPool 几何计算；不移植 ONNX 中固定样本的坐标表。

**不能声称完整复现厂商训练配方。**指南没有提供联合检测/OCC的训练超参；
所提 `efficient-no-se-0304.pth` 也未随当前 SDK 找到。ONNX 已融合 Conv/BN，
无法唯一恢复原始 BN gamma/beta/运行均值方差；转换采用推理等价的 BN 重参数化，
训练时固定运行统计，保留 affine 参数可训练。每个 iteration 的专用 hook 处理这一点。
没有将融合权重作为普通训练 BN 直接加载。

OCC 是本项目的额外分支，不属于指南的纯检测模型。默认用 G3 的 OCC Adapter/Head 权重初始化，
保持 128 通道和最终200×200×16/18类；恢复0.4米特征卷积，不沿用G2的0.8米后再上采样。
由于共享特征来自官方模型，OCC 初始化后还需要联合微调，不能直接当成训练完成的联合模型。
默认不加额外深度监督，也不读LiDAR生成深度标签；这与官方BEVDet目标保持一致。

## 对比

| 项目 | G1 | G3 | G5 |
|---|---|---|---|
| 图像 backbone | ResNet50 | ResNet50 | SDK B0-noSE/ReLU6 |
| 图像特征接口 | 1024/2048 | 1024/2048 | 112/1280 |
| 共享 BEV Neck 输出 | 256 | 128 | 128 |
| BEV Neck 内部倍率 | 默认2 | 1 | 1 |
| BEV Neck align_corners | True | False | False |
| 检测头输入 | 256 | 128 | 128 |
| OCC 特征通道 | 128 | 128 | 128 |
| OCC卷积网格 | 0.4m，200×200 | 0.8m，100×100 | 0.4m，200×200 |
| OCC特征上采样 | 无 | ×2 | 无 |
| 额外深度 loss | 无 | 0.1 | 无 |
| 检测链路初始化 | 原ep0联合权重 | 从前组继续微调 | 官方优化ONNX全部检测权重 |
| OCC初始化 | 缩通道后部分重新初始化 | G2匹配权重 | G3 OCC匹配权重 |
| BN运行统计 | 原训练方式 | 原训练方式 | 固定（ONNX折叠恢复限制） |

同 G1/G3：六路256×704、检测范围±51.2m、OCC范围±40m、四卡每卡2样本、AdamW，
学习率5e-5、20 epoch、warmup500、16/19衰减、EMA；这些是本项目联合微调设置。

与G4不同：不再使用原版ImageNet B0权重删除SE/更换激活，不重新随机初始化图像FPN，
而是成套移植官方检测网络的已训练参数，并先验证数值一致。

## 初始化生成与检查

```bash
conda activate bevdet
cd /root/autodl-tmp/bevdet_flashocc
python tools/convert_g5_official_onnx.py \
  --source-onnx ckpt/g5_official_bevdet_source.onnx \
  --occ-checkpoint work_dirs/ep1_g3_bev128/epoch_20_ema.pth
```

输出 `ckpt/g5_official_det_g3_occ_init.pth`。转换程序拒绝覆盖，验证失败不写checkpoint。
数值对照比较官方 ONNX Runtime 与恢复后的 PyTorch：
112/1280路backbone特征、图像FPN、LSS logits，以及相同BEV输入下的六路检测输出。
分离图像前端/BEV后端验证，避免将官方固定样本几何表误当作通用几何模型。
参考子图/逐层映射/源模型SHA256/误差在 `work_dirs/g5_official_conversion/`。
原 SDK ONNX 节点顺序不完全满足 ONNX checker，验证时仅对提取的等价子图进行拓扑排序。

转换依赖 `onnxruntime`，只用于离线比对，不是训练 runtime。服务器已安装到 bevdet 环境，
未修改 torch/numpy 或 bevdet_qat 环境。

## 训练

```bash
bash tools/dist_train.sh \
  'configs/bevdet/ep1_bevdet+flashocc/bevdet-b0-det-occ-g5-official.py' \
  4
```

不带验证，不恢复G4，不自动启动训练。`load_from` 已指向转换后的初始化，不使用`resume_from`。
导出仍可使用原 `tools/export_bev_occ_onnx.py`，但必须传G5配置和G5训练后的权重。
不要只覆盖`depth_loss_weight`启用深度监督；要做该对照必须另外加入深度标签流水线。
不要删除冻结BN hook，否则改变了本配置的权重恢复前提。


## 本次已完成验证

- 4项回归测试通过：backbone输出/激活、BN冻结与梯度、配置隔离、85卷积覆盖和错误属性拒绝。
- 两组随机输入的20项ONNX Runtime/PyTorch数值对照通过，最大绝对误差约1.83e-4。
- 真实增强样本B=1完成一次CUDA前向/反向/AdamW更新，loss与梯度有限，BN运行统计未改变。
  详情为`work_dirs/g5_official_conversion/smoke.json`。不代表四卡训练收敛或精度结论。
- 初始化checkpoint已生成，可直接使用配置训练；没有启动正式G5训练，没有修改正在运行的QAT。


## 较大 batch 的独立配置

- `bevdet-b0-det-occ-g5-official-b4.py`：每卡4，总batch16，warmup250步。
- `bevdet-b0-det-occ-g5-official-b8.py`：每卡8，总batch32，warmup125步。
- 两者保持学习率5e-5与网络/监督不变，使用各自独立work_dir。总batch变化会影响优化，需验证精度。
- 原G5的每卡2配置及正在运行的训练没有修改；切换配置必须重启。
- 单GPU真实样本测试中B4/B8均完成前向、反向、AdamW更新；测试与正式四卡训练共用GPU，
  且关闭cuDNN算法搜索以限制测试峰值，因此不作为隔离的四卡速度结论，也不代表正式训练峰值。
  记录为`work_dirs/g5_official_conversion/batch_probe.json`。


## 2026-10-27 四卡吞吐优化（已启动）

当前正式运行 `bevdet-b0-det-occ-g5-official-fast.py`，输出目录
`work_dirs/ep1_g5_official_fast`，终端输出为该目录下`launcher.log`。
旧 B2/B4 训练经用户授权停止，保留日志，从同一官方初始化重新开始，不使用基准测试权重。

- 每卡16样本、总batch64，每卡8个DataLoader worker。
- 混合精度、动态loss scaling，初始scale512；权重/EMA仍以FP32保存。
- LR仍为5e-5，20 epoch，16/19衰减；warmup63步，约4032个样本。
- 训练标签由DataLoader worker使用原CenterHead算法生成，批量传到GPU。
  `CenterHeadCPUTargets`和`BEVDetDetOccPrecomputedTargets`仅改变训练标签的生成/传输位置。
  模型参数名、网络前向、损失权重、推理输出保持原样，BEVPool仍在GPU。
- 8个真实样本CPU/GPU标签对照：整数索引、mask完全相同，浮点最大误差约7.63e-6。
  测试还包含空GT、批量collate、参数键名一致性。
- 独立四卡40步对照，排除前10步预热：原B2约24.4样本/秒，新方案约93.1样本/秒，约3.82倍。
  新方案采样GPU利用率均值约65%，运行有阶段性等待，不表示持续100%。
  四卡短测不能代替整轮性能和精度验证；总batch和计算精度变化可能影响收敛。
- 测试期间其他训练已停止，没有混用并发单卡测试数字。数据、EMA、DDP均保留。
  详细结果为`work_dirs/g5_throughput_benchmark/summary.json`与`selected.json`。
- 测试过FP32 B2/B4/B8/B12、混合精度、CPU标签和worker标签等配置；没有选用更大显存却无收益的版本。

启动/重启命令（当前已在运行，不要重复执行）：

```bash
bash tools/dist_train.sh \
  'configs/bevdet/ep1_bevdet+flashocc/bevdet-b0-det-occ-g5-official-fast.py' \
  4
```

推理结构与原G5相同。验证/导出使用本fast配置时需要一并同步它的worker-targets-base配置、
CPU target head、detector和pipeline模块。也可以用原G5配置对同结构checkpoint做纯推理。
