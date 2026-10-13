# 单帧 BEVDet + FlashOCC：共享 BEV Encoder

## 架构

六路图像 -> ResNet50 -> CustomFPN -> LSS -> CustomResNet/FPN_LSS
-> 共享特征 [B,256,128,128]。

- 检测：CenterHead，原始 0.8m 网格和 10 类检测。
- 占用：BEVGridAdapter 按真实坐标重采样并做 1x1 通道投影，
  -> [B,128,200,200] -> 官方 FlashOCC BEVOCCHead2D
  -> logits [B,200,200,16,18]。

没有第二套 BEV Encoder；两个 loss 都回传到共享图像、LSS、BEV 网络。
插值不产生真实的高分辨率细节，此实现是双任务基线，不是官方 FlashOCC
完整训练配方复现。深度范围和共享 BEV 网格保留原检测配置。

## 文件

- mmdet3d/models/detectors/bevdet_det_occ.py：双头模型与双输出。
- mmdet3d/models/necks/bev_grid_adapter.py：米制坐标/格子中心对齐。
- mmdet3d/models/dense_heads/bev_occ_head.py：从官方迁入的2D占用头。
- mmdet3d/datasets/nuscenes_det_occ_dataset.py：标签路径和双任务评估。
- mmdet3d/datasets/pipelines/det_occ.py：占用标签批处理和增强保护。

## 配置与运行

在 /home/fjm/tong/BEVDet 根目录，激活 bevdet 环境后运行：

```bash
# 20步调试，batch=1，workers=0，FP32，无EMA
python tools/train.py configs/bevdet/bevdet-r50-det-occ-shared-debug.py --gpu-id 0 --seed 0

# 正式单卡起始配方，batch=2，CBGS，20 epochs；加 --validate 启用双任务验证
python tools/train.py configs/bevdet/bevdet-r50-det-occ-shared.py --gpu-id 0 --seed 0 --validate

# 测试已有联合权重
python tools/test.py configs/bevdet/bevdet-r50-det-occ-shared.py CHECKPOINT.pth --eval bbox occ
```

现有 tools/train.py 保留了用户添加的 debugpy.wait_for_client()：
启动后需要通过 VS Code 的 sh_file_debug（9501）连接。
正式无人值守或多卡训练前需要关闭或条件化这段调试等待。

正式配置使用 torchvision ResNet50 ImageNet 初始化；调试配置使用3090本机缓存路径。
已有检测训练权重可用 --cfg-options load_from=/path/to/det_checkpoint.pth 加载；
新增的 occ_adapter/occ_head 参数没有对应权重，会正常重新初始化。
迁移到其它机器时要调整 data_root、ann_file、权重路径以及显式 occ_gt_root（如果使用）。

正式配方 lr=5e-5 仅作为单卡 batch=2 起点，多卡时应按总 batch 和实验目标重新设置。
日志同时含 task0.loss_* 和 loss_occ；两个任务的损失/梯度尺度需要观察和调参。
目前没有完整训练精度或板端部署验证结果。

## 标签和增强

使用已有 bevdetv3-nuscenes_infos_{train,val}.pkl 和 data/nuscenes/gts。
get_data_info 从PKL的 occ_path 获取场景名，再用数据集根目录拼接 scene/token/labels.npz。
不依赖PKL中原机器的绝对 gts 路径。

占用标签先加载，再进入 BEVAugDetOcc。BEV 只允许 x/y 翻转：
检测框、voxel_semantics、mask_camera、mask_lidar 同步翻转。
BEV旋转、缩放和平移被显式拒绝，避免标签错位；图像增强仍保留。
体素类别轴为 [X,Y,Z]，二维特征为 [C,Y,X]。

评估：检测继续用官方 nuScenes mAP/NDS；占用使用 camera visibility mask，
报告18类IoU（含free），mIoU只平均前17类，不包含free；单位为百分数。
format_results 只导出检测JSON；需要保存占用输出时使用 tools/test.py --out results.pkl，
每个样本为 {'pts_bbox':检测结果, 'occ':uint8 [200,200,16]}。

## 已执行验证

- 四项测试：坐标重采样/XY轴顺序、占用头高度维和全不可见mask、
  双任务评估路由及mIoU、GT和mask同步翻转。
- 真实nuScenes样本：联合loss、分别对共享BEV的梯度、反向/参数更新、双输出推理。
- forward hook 确认共享的图像backbone、LSS、BEV backbone/neck各执行一次。
- batch=1单样本前向/反向峰值约2.47 GiB；完整MMCV训练器两步测试日志峰值约3 GiB。
- MMCV IterBasedRunner 实际完成两步 CBGS 联合训练，loss和梯度有限。

```bash
python -m pytest -q tests/test_models/test_shared_det_occ.py
```
