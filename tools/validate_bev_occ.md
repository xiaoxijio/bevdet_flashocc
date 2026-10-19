# 单帧 Det + FlashOCC：连续场景合成图与视频

所有命令在3090的 `/home/fjm/tong/BEVDet` 根目录、`bevdet` 环境执行。
可视化直接使用PyTorch模型，不经过ONNX。

## 只看连续场景，不跑指标

```bash
python tools/validate_bev_occ.py \
  'configs/bevdet/ep0_bevdet+flashocc/bevdet-r50-det-occ-shared-2gpu-32gb-pretrained.py' \
  ckpt/epoch_20_ema.pth \
  --vis-only --scene scene-0268 --vis-every 1 --fps 2 --workers 0 \
  --out-dir work_dirs/vis_epoch20_video
```

- `--scene scene-0268`：按场景名选择，也支持scene token及多个场景名。
- 不指定 `--scene` 时，默认选择按时间排序的第一个场景。
- `--scene-count 3`：前三个场景；`--scene-count 0`：全部场景。
- `--vis-count 10`：总共最多10帧；默认0表示所选场景的全部帧。
- `--vis-every 1`：每帧都画，保证连续。不要沿用旧版每50帧抽样的设置。
- `--fps 2`：nuScenes关键帧约2Hz，正常速度；设为5等可加速播放。
- `--no-video`：只存合成JPG。默认生成MP4，ffmpeg可用时使用H.264。
- `--save-occupancy`：另外保存完整体素预测/GT/mask NPZ；默认不额外存。
- `--smoke-test 2`：只检查选中场景的前两帧，也不算官方指标。

只可视化模式只推理选定帧，不先跑完整6019帧，不加载nuScenes官方评估器，
也不在内存中保存所有体素预测。

## 输出：同一个目录，一帧一图，一个场景一个视频

```text
work_dirs/vis_epoch20_video/visualizations/
├── scene-0268_00000.jpg
├── scene-0268_00001.jpg
├── ...
├── scene-0268.mp4
└── index.json
```

多个场景也在这个目录中，用场景名区分，不再每帧新建文件夹。
index.json记录token、数据集索引、时间戳及场景内帧号。
只在场景内部按timestamp连续播放，不把不同场景拼成一次行驶。

## 每帧排版与坐标

- 上方：六路图像+预测3D框，前排为左前/前/右前，后排为右后/后/左后；原图不镜像。
- 下方：检测BEV、OCC预测俯视、OCC GT俯视，最后是统一语义图例。
- 三个BEV统一覆盖[-40,40]米，车头+x向上，车左+y向左；检测范围外的框不显示。
- 三个BEV以自车原点为圆心绘制10/20/30/40米距离圈及十字参考轴，圈上标半径；不绘制自车车身。
- 检测彩色实线为预测，灰线为GT；类别颜色与占用语义一致。
- OCC只展示camera-visible体素，每个水平位置取最高非free体素，并非地面语义投影。
- 自车坐标使用数据集的ego_cam参考；不会误用相机自身坐标作为自车位置。

布局参考Sparse4D官方的六视角相机标签、BEV上下文及连续图像/视频输出，
根据当前双任务输出实现了相机在上、三张BEV在下的排版：
https://github.com/HorizonRobotics/Sparse4D/blob/main/projects/mmdet3d_plugin/datasets/nuscenes_3d_det_track_dataset.py

实现位于 tools/bev_occ_visualization.py；不依赖Sparse4D运行环境。

## 全量验证（与仅可视化区分）

不加 `--vis-only` 时仍会推理完整验证集、累计OCC混淆矩阵并评估检测。
`--scene` 等选择参数只决定哪些帧绘图，不影响正式指标的数据范围。

```bash
python tools/validate_bev_occ.py \
  'configs/bevdet/ep0_bevdet+flashocc/bevdet-r50-det-occ-shared-2gpu-32gb-pretrained.py' \
  ckpt/epoch_20_ema.pth \
  --scene-count 1 --workers 0 --skip-det-eval \
  --out-dir work_dirs/eval_epoch20_new
```

3090系统内存有限，上例加 `--skip-det-eval`：保存detections.pkl和OCC指标，
跳过内存开销较大的nuScenes检测指标计算。去掉该参数可计算完整Det+OCC指标。
没有该参数时，OCC指标和检测预测也会先落盘，再开始检测评估。

## 验证

- 场景选择与排序、检测/占用朝向、图像合成、米制距离圈和平铺目录测试。
- epoch_20_ema真实模型已跑通scene-0268连续10帧，生成合成图及H.264 MP4。
- 完整场景视频使用同一模型再次生成；index.json保留对应时间戳。

```bash
python -m pytest -q tests/test_tools/test_bev_occ_visualization.py
```
