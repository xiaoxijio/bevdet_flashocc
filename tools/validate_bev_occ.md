# epoch_20_ema 联合验证与可视化

在3090的BEVDet仓库和bevdet环境执行。验证直接使用PyTorch原模型及CUDA BEVPoolv2，不经过ONNX，与导出期的prim::Constant提示无关。

```bash
cd /home/fjm/tong/BEVDet
conda activate bevdet
mkdir -p work_dirs/eval_epoch20
set -o pipefail
python tools/validate_bev_occ.py \
  configs/bevdet/ep0_bevdet+flashocc/bevdet-r50-det-occ-shared-2gpu-32gb-pretrained.py \
  ckpt/epoch_20_ema.pth \
  --out-dir work_dirs/eval_epoch20 \
  --vis-count 20 --vis-every 50 --workers 2 \
  2>&1 | tee work_dirs/eval_epoch20/eval.log
```

默认评估完整验证集。Det调用项目原NuScenes评估器（mAP、NDS、各类AP及误差指标）；OCC使用camera mask、忽略255、18类混淆矩阵、前17类mIoU（排除free），与现有NuScenesDatasetDetOcc._evaluate_occ一致。逐帧累计OCC统计，不保存全部体素预测；仅保留检测结果和选定帧可视化。

产物：
- `metrics.json`：完整Det+OCC指标。
- `occ_metrics.json`、`occ_confusion.npy`：OCC指标及混淆矩阵。
- `detections.pkl`：检测结果，可用于重新评估。
- `detection/`：nuScenes检测格式结果和评估产物。
- `visualizations/`：每隔50帧保存一组，共20组。
  - `det_cameras.jpg`：六路原图上的预测3D框，默认置信度>=0.3。
  - `bev_comparison.png`：检测BEV（彩色预测/灰色虚线GT）与OCC预测/GT俯视图。
  - `occ_3d.png`：占用体素中心的3D预测/GT视图，最多显示50000点，并非体素网格实体渲染。
  - `occ_legend.png`：语义色表。
  - `occupancy.npz`：完整该帧预测/GT及camera mask，未做可视化降采样。

OCC可视化仅显示camera-visible范围，俯视图取最高非free体素。检测范围为[-51.2,51.2]米，OCC为[-40,40]米。

快速试跑而非正式评估：在命令中添加 `--smoke-test 2 --vis-count 2 --vis-every 1`，输出目录改为单独目录。该模式只检查前两帧和可视化，不输出完整数据集指标。已用epoch_20_ema进行单帧试跑并检查可视化。

已有项目命令也支持 `python tools/test.py CONFIG CHECKPOINT --eval bbox occ`，但会将所有OCC预测暂存在内存，且其--show接口不适配联合可视化，建议使用上述入口。
