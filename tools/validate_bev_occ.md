# Det + OCC：验证与可视化独立入口

在项目根目录、已安装本项目依赖的 `bevdet` 环境执行。两种入口均用
模型文件后缀自动选择推理方式：`.pth` 使用 PyTorch，`.onnx` 使用 ONNX Runtime，`*_fake_quant.ortm` 使用 SDK `oritek_graph_runner`（后缀不区分大小写）。不再需要 `--backend` 参数。

- `tools/validate_bev_occ.py`：仅验证、保存指标与检测结果，不绘图。
- `tools/visualize_bev_occ.py`：仅推理选定场景、输出图像/视频，不启动指标评估。
- 共享推理实现：`tools/bev_occ_inference.py`。
- 绘图实现和详细用法：[visualize_bev_occ.md](visualize_bev_occ.md)。

原来的 `--vis-only`、`--vis-count` 等绘图参数已移至可视化入口。

## PTH 验证

```bash
python tools/validate_bev_occ.py \
  'configs/bevdet/ep1_bevdet+flashocc/bevdet-b0-det-occ-g6-occ128-nearfield.py' \
  work_dirs/ep1_g6_occ128_nearfield/epoch_20_ema.pth \
  --workers 2 --out-dir work_dirs/g6_eval_pth
```

## ONNX 验证

```bash
python tools/validate_bev_occ.py \
  'configs/bevdet/ep1_bevdet+flashocc/bevdet-b0-det-occ-g6-occ128-nearfield.py' \
  onnx/g6.onnx --onnx-provider cuda \
  --workers 2 --out-dir work_dirs/g6_eval_onnx
```

ONNX 仅替换网络前向，检测解码、阈值和 NMS 复用配置对应的原项目检测头。
OCC 依照导出约定将 `[1,Dz*C,Y,X]` 转为 `[X,Y,Z]`，统计相同 GT/camera mask。
需要本项目的 MMCV/MMDet3D 环境，不是独立 ONNX-only 运行器。

`--onnx-provider cpu|cuda` 选择 ONNX Runtime 的执行后端，默认 CPU；CUDA 需要兼容的
`onnxruntime-gpu`。`--device cuda|cpu` 控制 PTH 和检测后处理，默认 CUDA，GPU 由
`--gpu-id` 选择。纯 CPU 测试可使用 `--device cpu --workers 0 --threads 4`。
PTH CPU 路径使用同一 checkpoint 和逐帧几何下的 PyTorch Pool 分解，避开 CUDA-only BEVPool；
CUDA PTH 路径使用原模型原生推理。checkpoint 带本项目 QAT 元数据时自动按匹配配置恢复量化层。

## 快速检查、部分验证

```bash
# 上面任一命令追加这些参数，仅验证前两帧；不会产生官方全量检测分数。
--limit 2 --device cpu --workers 0
```

`--smoke-test N` 是 `--limit N` 的别名。部分数据的报告显式标记 PARTIAL，计算该子集的 OCC IoU，
保存检测结果但跳过官方 nuScenes 检测评估。正式验证时不要使用 `--limit`。

`--skip-det-eval` 用于全量统计 OCC 并保存检测结果，跳过官方检测指标；可避免额外评估内存开销。
mAP/NDS 为 0–1；OCC IoU/mIoU 为百分数，mIoU 排除 free 类。

## 结果

- `metrics.txt/json`：最新指标；`occ_metrics.json`、`occ_confusion.npy`：OCC 统计。
- `detections.pkl`：检测预测。
- `run_metadata.json`：模型/配置 SHA256、backend、参数、样本索引和运行状态。
- `reports/<UTC时间_唯一ID>/`：每次运行独立归档指标、元数据和展开的配置。

检测评估开始前先保存 OCC 报告；失败时保存错误状态，不把缺失检测结果当作完整验证。

## ONNX 几何约定

单输入 fixed-geometry ONNX 始终使用导出时固化的相机映射，不会随验证样本重新计算；
其全数据集指标衡量的是这个固定几何部署模型，不能将其与逐帧动态几何的 PTH 视为天然等价。
当前 g6 单帧测试中，两者的 OCC 混淆矩阵一致，不代表全验证集完全一致。

多输入 runtime-mapping ONNX 会逐帧生成 v3 映射，自动加载模型同名 `.mapping.json`，
也可通过 `--mapping-layout FILE` 指定；容量不足会报错，不截断投影点。

ONNX 验证是原始导出网络的验证，不是编译后 ORTM/板端量化精度验证。


## QAT 伪量化开关（诊断实验）

QAT PTH 默认开启伪量化：启用 SDK `enable_quantize`，同时 `disable_observer` 冻结量化参数。
不加参数就是原来的 QAT 评估。`--disable-fake-quant` 仅关闭量化模拟，保留同一份训练后的权重，
不会修改或保存 checkpoint。它不代表真实 INT8 部署精度，也不等于原始浮点 g6。

```bash
# 开启：默认行为
python tools/validate_bev_occ.py CONFIG ckpt/g6_qat.pth --out-dir work_dirs/g6_qat_fake_on
# 关闭：同一 checkpoint，单独目录，便于对照
python tools/validate_bev_occ.py CONFIG ckpt/g6_qat.pth --disable-fake-quant --out-dir work_dirs/g6_qat_fake_off
```

输出日志和 `metrics.txt` 的 Note 会显示 ON/OFF；参数也记录在 run_metadata.json。
ONNX 的 Q/DQ 已固化在图中，不接受该开关；普通非 QAT PTH 也会拒绝此参数。
可视化入口使用同一共享参数，支持对 QAT PTH 做相同诊断。


## 编译后 fake-quant ORTM 验证（PC）

共用 `tools/validate_bev_occ.py`，仅网络前向由 `tools/bev_occ_ortm.py` 调用 SDK 工具，
预处理、检测解码/NMS、nuScenes mAP/NDS 和 OCC camera-mask mIoU 统计复用原评测。
只支持单输入、固定几何的 `*_fake_quant.ortm`；正式部署 `.ortm` 需要仿真器或板端运行器。
这对应《NPU工具使用指南》第4章的量化损失评估，不是板端性能测试。

在 3090 上：

```bash
source /home/fjm/miniconda3/bin/activate bevdet
cd /home/fjm/tong/BEVDet

# 先验证前两帧：OCC 子集指标、检测预测，不计算官方 mAP。
python tools/validate_bev_occ.py \
  'configs/bevdet/ep1_bevdet+flashocc/bevdet-b0-det-occ-g6-occ128-nearfield.py' \
  ortm/g6_fake_quant.ortm \
  --ortm-onnx /home/fjm/tong/Compilation/app/bev_occ/tmp_models/g6.onnx \
  --ortm-provider cuda --workers 0 --limit 2 \
  --out-dir work_dirs/g6_eval_ortm_smoke

# 全验证集：计算 mAP、NDS 和 mIoU。
python tools/validate_bev_occ.py \
  'configs/bevdet/ep1_bevdet+flashocc/bevdet-b0-det-occ-g6-occ128-nearfield.py' \
  ortm/g6_fake_quant.ortm \
  --ortm-onnx /home/fjm/tong/Compilation/app/bev_occ/tmp_models/g6.onnx \
  --ortm-provider cuda --workers 2 \
  --out-dir work_dirs/g6_eval_ortm
```

QAT 编译产物可将模型改为 `ortm/g6_qat_fake_quant.ortm`，保持与导出模型匹配的架构配置。
如有多份同名 ONNX，建议显式指定编译时使用的那一份：

```bash
--ortm-onnx /home/fjm/tong/Compilation/app/bev_occ/tmp_models/g6_qat.onnx
```

参数：

- `--npu-toolchain`：工具链根目录。默认读取 `NPU_TOOLCHAIN_PATH`，未设置时使用项目旁的
  `/home/fjm/tong/npu_toolchain`，调用 `linux/simulator/bin/oritek_graph_runner`。
- `--ortm-provider cuda|cpu`：ORTM 前向设备，默认 CUDA，GPU 编号使用 `--gpu-id`。
  `--device` 独立控制检测后处理；全 CPU 可用 `--ortm-provider cpu --device cpu`。
- `--ortm-onnx`：与编译模型对应的源 ONNX，仅用于读取输入/输出名称、静态形状和 FP32 类型，
  不执行 ONNX 前向。不指定时按去掉 `_fake_quant` 的名称依次查找 ORTM 同目录和项目 `onnx/`。
- `--ortm-timeout`：每帧 SDK 进程超时秒数，默认 180。

SDK 子进程自动添加 simulator 动态库路径；机器存在 CUDA 11.8/cuDNN 8 时优先使用该环境。
每帧使用独立临时目录，正常运行后自动清理输入/输出，无逐层 dump。
当前每帧启动一次 SDK 进程，有模型加载开销；实际处理速度以评测进度条为准。
输出按 tensor 名称匹配，检查 FP32 字节数、shape 和有限值，失败时停止并报告 SDK 错误。
报告记录 ORTM、源 ONNX、SDK runner 的 SHA256 和张量形状。
源 ONNX 必须与 ORTM 对应；形状检查无法证明两份文件的权重来自同一次编译。

正式结果在 `metrics.txt/json`、`occ_metrics.json`、`detections.pkl` 及 `reports/` 中。
使用同一源 ONNX 和同一验证集的结果作对照，可观察编译量化带来的 mAP/mIoU 变化。
