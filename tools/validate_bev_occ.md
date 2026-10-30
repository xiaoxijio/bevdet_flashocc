# Det + OCC：验证与可视化独立入口

在项目根目录、已安装本项目依赖的 `bevdet` 环境执行。两种入口均用
`--backend pth|onnx` 选择模型格式，默认 `pth`。第二个位置参数是对应模型文件。

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
  --backend pth --workers 2 --out-dir work_dirs/g6_eval_pth
```

## ONNX 验证

```bash
python tools/validate_bev_occ.py \
  'configs/bevdet/ep1_bevdet+flashocc/bevdet-b0-det-occ-g6-occ128-nearfield.py' \
  onnx/g6_pool.onnx --backend onnx --onnx-provider cpu \
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
