# EP1：共享 BEVDet + FlashOCC 轻量化实验

在 3090 的 `/home/fjm/tong/BEVDet` 根目录、`bevdet` 环境执行。
四组是**累计结构改动**，但默认都从同一个 `ckpt/epoch_20_ema.pth` 初始化，
分别训练，便于与原始模型和相邻组比较。没有自动启用教师蒸馏。

| 组 | 配置文件 | 相比上一组新增改动 |
|---|---|---|
| G1 | `bevdet-r50-det-occ-g1-occ128.py` | OCC Adapter / Head 256→128；共享 BEV 仍为256 |
| G2 | `bevdet-r50-det-occ-g2-occ08.py` | OCC Adapter 输出100×100、0.8米；3×3卷积在粗网格上运行，特征双线性上采样到200×200后再做原有MLP预测 |
| G3 | `bevdet-r50-det-occ-g3-bev128.py` | 共享BEV neck内部/输出均为128；检测头和OCC Adapter输入适配；neck两处Resize采用align_corners=False |
| G4 | `bevdet-b0-det-occ-g4-nose.py` | 图像backbone换为EfficientNetB0无SE、ReLU6；img_neck输入为[112,1280]，融合Resize采用bilinear/align_corners=False |

所有组保持6路256×704图像、检测范围±51.2米、检测特征128×128，
OCC范围±40米、输出200×200×16、18类和原有GT/mask、评估规则。
G2只是把OCC的主要3×3卷积移到粗网格；MLP仍然在200×200上运行，
不是在粗网格上生成类别后插值，也不是缩小OCC感知范围。

## 实现与兼容性

- `mmdet3d/models/necks/lss_fpn.py`：新增`channels_factor`与`align_corners`。
  旧配置默认行为不变（有extra_upsample时内部倍率2、align_corners=True）。
  G3/G4明确设置`out_channels=128, channels_factor=1, align_corners=False`。
- `mmdet3d/models/dense_heads/bev_occ_head.py`：新增可选特征上采样，默认倍率1。
  G2/G3/G4使用倍率2，原始loss和输出XYZ顺序不变。
- `mmdet3d/models/backbones/efficientnet_b0_nose.py`：基于本环境torchvision 0.14的B0，
  移除SE，用ReLU6替换SiLU，返回112/1280通道的stride16/32特征。
  这是对SDK方案的独立实现，不是`EfficientNetB0HZ`，不能直接加载SDK格式权重。
  G4使用torchvision ImageNet B0权重中保留的Conv/BN层初始化；SE移除及激活替换后仍需任务训练。
  已下载官方权重并核对SHA256前缀7f5810bc，放入3090的torch缓存，无需再次联网。
  在其他机器首次初始化会下载配置内的官方权重URL；离线时可将`model.img_backbone.pretrained`
  改为已下载的同格式权重路径。新模块不增加timm等依赖。

## 初始化和训练

默认`load_from='ckpt/epoch_20_ema.pth'`，`resume_from=None`。
MMCV使用非严格初始化：加载名称及shape相同的参数，报告并跳过形状变化/缺失参数；
新层按初始化值开始训练。G4的新backbone先加载ImageNet权重，再由load_from恢复其他匹配层。
因此启动时的shape mismatch / missing / unexpected key提示是预期的初始化提示，
不能把这些配置直接当作原始epoch20模型评估，也不要用`--resume-from`恢复旧架构。

训练继承原2卡配方：每卡4样本、20 epoch、warmup及[16,19]学习率衰减，
初始学习率改为5e-5作为微调起点；G4更换了backbone，学习率和训练时长需根据收敛情况调整。
不自动开始训练，不覆盖ep0配置或checkpoint，每组有独立work_dir。

```bash
cd /home/fjm/tong/BEVDet
conda activate bevdet

# G1；依次更换为G2/G3/G4配置即可。四组应分别运行。
bash tools/dist_train.sh \
  'configs/bevdet/ep1_bevdet+flashocc/bevdet-r50-det-occ-g1-occ128.py' \
  2 --validate
```

如果要接着上一组的已训练权重做进一步微调，可显式覆盖初始化来源：

```bash
bash tools/dist_train.sh \
  'configs/bevdet/ep1_bevdet+flashocc/bevdet-r50-det-occ-g2-occ08.py' \
  2 --validate --cfg-options load_from=work_dirs/ep1_g1_occ128/epoch_20_ema.pth
```

如需隔离某项改动而不累计，可通过`--cfg-options`覆盖继承项，或复制配置后更改基类；
例如G3同时改变neck通道和Resize方式，单独消融Resize时需另跑一组。

## 验证与可视化

使用**对应结构训练后**的checkpoint：

```bash
python tools/validate_bev_occ.py \
  'configs/bevdet/ep1_bevdet+flashocc/bevdet-r50-det-occ-g1-occ128.py' \
  work_dirs/ep1_g1_occ128/epoch_20_ema.pth \
  --scene scene-0268 --vis-every 1 --fps 2 --workers 0 \
  --out-dir work_dirs/ep1_g1_occ128/eval_vis
```

不加`--vis-only`，完整验证集计算Det/OCC指标，选定场景另生成图片和视频。
最终加速收益需在LQ560上以相同精度/量化条件实测；本次没有开始训练、计算优化后精度或上板测速。
固定几何ONNX导出继续使用`tools/export_bev_occ_onnx.py`及对应新配置和训练后的权重。

## 实现检查

```bash
python -m pytest -q \
  tests/test_models/test_ep1_optimization.py \
  tests/test_models/test_shared_det_occ.py \
  tests/test_tools/test_bev_occ_visualization.py
```

这些是开发回归测试，不是模型指标评估。另已对四组逐个进行模型构建、六路图像编码、
共享BEV编码及双头随机输入前向检查，并验证原始checkpoint的匹配参数能按MMCV路径恢复。

## 评估结果持久保存

`tools/validate_bev_occ.py`正式评估会自动保存：

- `metrics.txt`：便于阅读的Det总指标、逐类别AP/误差、OCC逐类别IoU和mIoU。
- `metrics.json`：原始指标数值，便于程序比较。
- `run_metadata.json`：配置/权重路径和SHA256、命令参数、样本数及运行状态。
- `reports/<UTC时间_唯一ID>/`：每次运行独立归档上述文件和展开后的`config.py`。

输出目录根部保存最近一次结果；历史指标归档不会被下一次运行覆盖。
检测评估开始前先保存OCC报告，并将状态标为`occ_complete_detection_pending`；
检测评估成功后状态为`complete`。`--skip-det-eval`时状态为
`occ_complete_detection_skipped`，检测指标显示N/A，避免误当作完整验证。
`--vis-only` / `--smoke-test`不计算正式指标，也不生成这些评估报告。
历史归档保存指标及配置/权重身份信息，不复制checkpoint、图片、视频或detections.pkl。

建议每组仍使用独立的`--out-dir`；日常直接查看对应目录的`metrics.txt`即可，
不需要重跑验证或依赖终端记录。mAP/NDS使用0–1数值，OCC IoU/mIoU使用百分数。
