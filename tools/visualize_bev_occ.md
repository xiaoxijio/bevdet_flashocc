# Det + FlashOCC 场景可视化

在项目根目录、`bevdet` 环境执行。此入口只绘图/生成视频，不跑全量指标。

```bash
python tools/visualize_bev_occ.py \
  'configs/bevdet/ep1_bevdet+flashocc/bevdet-b0-det-occ-g6-occ128-nearfield.py' \
  work_dirs/ep1_g6_occ128_nearfield/epoch_20_ema.pth \
  --backend pth --scene scene-0268 --vis-count 10 --workers 0 \
  --out-dir work_dirs/g6_vis_pth
```

ONNX 只需将模型文件和 backend 换成：

```bash
python tools/visualize_bev_occ.py \
  'configs/bevdet/ep1_bevdet+flashocc/bevdet-b0-det-occ-g6-occ128-nearfield.py' \
  onnx/g6_pool.onnx --backend onnx --scene scene-0268 --vis-count 10 \
  --device cpu --workers 0 --out-dir work_dirs/g6_vis_onnx
```

相同配置下，两种模型共用检测后处理、相机投影、语义颜色和拼图布局。
推理选项和固定几何的限制见 [validate_bev_occ.md](validate_bev_occ.md)。

## 图像布局

每帧一张 1920×1526 合成图：

1. 标题：模型 backend、场景、帧号、阈值和检测数量。
2. 六路原始相机图像，预测 3D 框叠加、统一相机标签与留白。
3. 三个面板：检测 BEV（LiDAR 上下文、米制距离圈、预测框/GT）＋OCC 三维体素预测＋OCC 三维体素 GT。
4. 统一 FlashOCC/Occ3D 语义图例。

设计参考官方 [BEVDet](https://github.com/HuangJunJie2017/BEVDet/blob/dev3.0/tools/analysis_tools/vis.py)
的相机框/BEV 上下文与官方 [FlashOCC](https://github.com/Yzichen/FlashOCC/blob/master/tools/analysis_tools/vis_occ.py)
的三维体素和语义配色。渲染在本地实现，不依赖 OpenGL、显示器、Open3D 或 Mayavi。
三维面板使用真正的表面体素和固定正交视角，保留高度，不是俯视语义图的倾斜贴图。

检测 BEV 覆盖 ±40m；OCC 根据 `occ_crop` 显示真实范围，g6 为 ±25.6m，面板有标注。
两者显示范围不同，不强行拉成相同米制比例。BEV 车头 +X 朝上、车左 +Y 朝左；三维图显示 X/Y/Z 坐标轴。
默认预测/GT 共用 camera-visible mask；未知/未观测/空闲体素不会画成实心障碍。
三维图中 class 0（others/undefined）不显示，二维 BEV 模式可显示该类。

## 常用参数

- `--scene scene-0268 [scene-...]`：按场景名或 token 选择。
- `--scene-count 3`：前三个场景；0 表示全部；默认 1。
- `--vis-count N` / `--limit N`：最多 N 帧；0 表示所选场景全部帧。
- `--vis-every 1`：逐帧；2 表示隔帧采样。
- `--score-thr 0.3`：仅影响显示，不改变正式验证配置。
- `--fps 2`：视频播放速度；不同场景独立生成视频。
- `--no-video`：只保存 JPEG；否则生成 MP4（ffmpeg 可用时 H.264）。
- `--occ-view 3d|bev`：默认 3D；可选传统占用俯视图。
- `--occ-mask camera|all`：默认 camera；all 显示未被 camera mask 筛选的预测/GT。
- `--save-occupancy`：额外保存预测、GT 和实际渲染掩码 `mask_render` NPZ。

所有 JPEG、场景 MP4 和 `index.json` 平铺在 `--out-dir`，不为每帧建立文件夹。
`visualization_metadata.json` 记录模型/配置身份、backend、显示参数和帧数。
原有模型和导出的 ONNX 不会被绘图脚本修改。
