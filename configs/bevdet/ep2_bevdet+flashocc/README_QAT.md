# BEVDet + FlashOCC 的 Oritek QAT 训练说明

以下命令均在服务器的 `/root/autodl-tmp/bevdet_flashocc` 目录下执行，使用 `bevdet_qat` 环境。

## 实验配置

`bevdet-r50-det-occ-bev128-depth-qat.py` 继承 `bevdet-r50-det-occ-bev128-depth-qat-base.py`，包含以下设置：

- 保留 G1 的 OCC 通道调整：128 通道。
- 保留 G3 的共享 BEV neck 调整：128 通道。
- OCC 计算网格使用 0.4 米、200 × 200，不采用 G2 的粗网格和二倍上采样。
- 视角变换原有的 BEV 网格仍为 0.8 米、128 × 128；它与 OCC 计算网格是不同的设置。
- 保留 LiDAR 深度监督，损失权重为 0.1；LiDAR 仅用于训练时生成深度标签。
- OCC predictor 在训练和导出时统一使用 `1×1 Conv2d → Softplus → 1×1 Conv2d`。加载原浮点模型时，会自动将 Linear 权重转换为等价的卷积权重形状。

## 量化范围与训练策略

将 93 个 Conv2d 层分别转换为 SDK 的 INT8 模拟量化模块，避免直接追踪整个模型中的训练损失、几何条件分支和索引计算。

- 权重采用对称、逐通道（per-channel）INT8 量化。
- 卷积输入采用非对称、逐张量（per-tensor）INT8 量化。
- Softmax、Softplus、BN、插值、网格采样、池化、几何计算和损失计算均在上述量化模块之外。

这是混合精度方案。SDK 编译器最终使用的计算精度仍需通过配置和实际转换结果确认。

继续支持现有的 `--runtime-mapping` 导出方式，即外参等几何参数通过更新 mapping 输入适配。在 SDK 的 `quant_tensors` 配置中，需要保留 base-1024 编码的 mapping 输入及索引解码路径的精度，不能将 mapping 索引统一量化为 INT8。更新几何关系时，仍需满足现有 layout 容量、网格、相机数量及图像形状等约束。

训练策略如下：

- 从第 0 个 epoch 开始冻结预训练 BN。
- 完成三个完整 epoch 后冻结量化统计收集器（observer）。配置中的 epoch 编号从 0 开始。
- 调用 `model.eval()` 时关闭 observer 更新，包括 MMDet 标准评估 hook 触发的评估。
- 禁用 EMA，避免对量化统计量进行指数移动平均；同时禁用 FP16。
- 默认训练 10 轮，学习率为 `1e-5`。这是初始实验设置，尚未验证为最优参数，应根据检测和 OCC 验证指标调整。

## 启动训练

```bash
conda activate bevdet_qat
bash tools/dist_train.sh \
  'configs/bevdet/ep2_bevdet+flashocc/bevdet-r50-det-occ-bev128-depth-qat.py' \
  4 --validate
```

浮点初始化权重使用 G3 的 `epoch_20_ema.pth`。程序先严格加载浮点权重，再执行 QAT 转换；转换后不会由 runner 重复加载浮点权重。优化器和 DDP 均在 QAT 转换完成后创建。

各进程使用真实训练数据初始化量化参数，并在初始化和每次迭代结束时从 rank 0 同步量化状态。SDK 的 `scale`、`zero_point` 张量被注册为持久化 buffer，使常规 runner checkpoint 能够保存和恢复这些值。

输出目录为：

```text
work_dirs/ep2_bev128_occ04_depth_qat
```

## 断点续训

续训必须使用与配置匹配的 QAT checkpoint，不能将原浮点 EMA 权重作为 QAT 续训断点。

```bash
bash tools/dist_train.sh \
  'configs/bevdet/ep2_bevdet+flashocc/bevdet-r50-det-occ-bev128-depth-qat.py' \
  4 --validate \
  --resume-from work_dirs/ep2_bev128_occ04_depth_qat/latest.pth
```

需要通过 `--resume-from` 显式指定断点，当前实现不允许自动续训。checkpoint 元数据会记录 QAT 格式和量化配置。

恢复模型时，先重建量化计算图并通过校准分配 buffer 形状，再严格加载 checkpoint，恢复已训练的权重和量化参数。

## 导出 ONNX

```bash
python tools/export_bev_occ_onnx.py \
  --qat \
  --config 'configs/bevdet/ep2_bevdet+flashocc/bevdet-r50-det-occ-bev128-depth-qat.py' \
  --checkpoint work_dirs/ep2_bev128_occ04_depth_qat/latest.pth \
  --runtime-mapping \
  --output onnx/ep2_bev128_occ04_depth_qat.onnx
```

导出时使用 SDK 接口，并恢复训练得到的量化 buffer。导出封装兼容 SDK 2.1.6 先调用 `model(args)`、随后调用 `model(*args)` 的行为。

QAT 的 OCC 导出路径直接调用训练时使用的 Conv2d 模块，保留其量化操作，不再取出原始权重另行调用卷积函数。

浮点实验继续使用不带 `--qat` 的导出命令。仅更新 mapping 几何输入时，无需执行依赖模型权重的 QAT 转换，使用现有 mapping-only 流程以及匹配的配置和 layout 即可。

## 完整训练前的小规模检查

```bash
python tools/check_bev_occ_qat.py
```

该脚本使用一个真实训练 batch，检查：

- 原 Linear 与替换后 Conv2d 的输出等价性。
- 浮点模型的各任务损失。
- QAT 模型的前向、反向传播及优化器更新。
- 评估时 observer 是否停止更新。
- checkpoint 中量化 scale 和状态是否能够准确保存、恢复。

默认输出 `/tmp/bevdet_qat_smoke.pth` 只是单步测试文件，不能作为训练完成的模型使用。

完整验证集精度、OMG 转换、NPU 执行结果和性能仍需分别测试。仅通过 ONNX 检查不能证明板端可部署或精度达标。另外，OCC 从 G3 的 0.8 米计算网格改为 0.4 米后，计算过程也发生变化，需要单独评估该结构的浮点基线，以区分结构变化和量化各自的影响。

## 已完成的集成验证（2026-10-26）

- 使用真实 G3 checkpoint，在矩形输入上验证了 OCC 的 Linear 到 Conv2d 转换，输出完全一致，包含 X/Y 轴和高度、类别维度排列。
- 使用一个真实训练 batch，浮点及 QAT 的检测、OCC、深度损失均为有限值；QAT 反向传播及优化器更新通过，261 个参数张量具有梯度。
- 372 个 `scale`、`zero_point` 张量能够准确保存和恢复；评估时 observer 停止更新。SDK 加载状态后，会将量化 scale 张量重新移动到 GPU。
- 两卡 runner 完成两次迭代并保存 checkpoint；QAT 断点续训恢复了优化器状态，并完成后续迭代。
- 使用实际的 `tools/dist_train.sh` 和四卡 EpochBasedRunner，在四个样本上完成一轮训练（每卡一个样本），包含优化器更新和 checkpoint 保存。
- 使用真实 runner checkpoint 导出 runtime mapping ONNX，通过严格形状推导和完整模型检查。图中包含 93 个 Conv、186 个 QuantizeLinear 和 186 个 DequantizeLinear；mapping 索引解码路径中没有 Q/DQ 量化节点。
- 已修复校准提前消耗训练采样器的问题，并统一 PyTorch 与 MMCV 对 OCC `sample_grid` 的保存方式；校准不再推进训练采样器。
- 仅在 `bevdet_qat` 环境中通过 `--no-deps` 补装了 `onnx==1.14.1`。Torch 仍为 `1.13.0+cu116`，原 `bevdet` 环境未修改。

服务器上的验证日志：

```text
/tmp/bevdet_qat_integration_check.log
/tmp/bevdet_qat_runner_check/20261026_164042.log
/tmp/bevdet_qat_resume_check2.log
/tmp/bevdet_qat_4gpu_check.log
/tmp/bevdet_qat_runner_export.log
```

首个单步测试 checkpoint 生成于 `sample_grid` 保存格式调整之前；如需使用该临时测试文件，请重新运行 `tools/check_bev_occ_qat.py` 生成。

已验证成功的 runner 导出文件为 `/tmp/bevdet_qat_runner.onnx`。以上文件均为小规模验证产物，尚未完成完整精度评估。
