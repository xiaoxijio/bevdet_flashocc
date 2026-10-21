# Det + OCC ONNX导出：固定几何与运行时几何

在3090的`/home/fjm/tong/BEVDet`根目录、`bevdet`环境执行。
此脚本当前使用`--config`指定配置（可省略，默认ep0预训练配置），使用`--checkpoint`指定权重（可省略，默认`ckpt/epoch_20_ema.pth`），唯一位置参数为output。

```bash
# 固定几何，默认模式：ONNX只有img输入
python tools/export_bev_occ_onnx.py \
  --config 'configs/bevdet/ep0_bevdet+flashocc/bevdet-r50-det-occ-shared-2gpu-32gb-pretrained.py' \
  --checkpoint ckpt/epoch_20_ema.pth onnx/bev_occ/bev_occ_fixed.onnx

# 不固定几何：外参/内参/图像与BEV增强变换在推理时提供
python tools/export_bev_occ_onnx.py \
  --config 'configs/bevdet/ep0_bevdet+flashocc/bevdet-r50-det-occ-shared-2gpu-32gb-pretrained.py' \
  --checkpoint ckpt/epoch_20_ema.pth onnx/bev_occ/bev_occ_dynamic.onnx \
  --dynamic-geometry
```

可加`--device cpu`在CPU上导出，输入接口不受导出设备影响。
`--sample-index`在固定模式决定冻结的几何，在动态模式只决定示例输入，不会冻结几何。
`--geometry-npz`仍支持原有sensor2egos/ego2globals/intrins/post_rots/post_trans/bda数据；
导出前会先调用`prepare_inputs`对齐参考帧。

## 动态模式的六个输入

| 名称 | shape（默认配置） | dtype | 含义 |
|---|---|---|---|
| img | [6,3,256,704] | float32 | 同一帧六路预处理图像 |
| sensor2keyegos | [1,6,4,4] | float32 | 各相机sensor到参考key ego坐标系的变换 |
| intrins_inv | [1,6,3,3] | float32 | 原始相机内参的逆矩阵 |
| post_rots_inv | [1,6,3,3] | float32 | 图像旋转/缩放变换的逆矩阵 |
| post_trans | [1,6,3] | float32 | 图像变换的平移部分 |
| bda | [1,4,4] | float32 | BEV增强变换；普通部署一般为单位矩阵 |

原始intrins和post_rots必须可逆；在CPU预处理计算其逆矩阵，作为运行时输入。
图像、标定、相机顺序、参考帧必须对应。
` sensor2keyegos = inverse(keyego2global) @ ego2globals @ sensor2egos `。
nuScenes六路图像可能不是同一采集时刻，不可直接把原始sensor2egos当作sensor2keyegos。
对于同步、固定机架且以当前自车为参考的相机，可按实际坐标约定提供相应外参。

不固定几何不代表动态分辨率/动态batch：batch固定为1，camera数、图像尺寸、depth bins和BEV网格仍由配置决定。
改动几何后会在图内重新计算射线坐标、体素索引及累加，无冻结索引、Sort或NonZero。
本实现不在ONNX内手写矩阵求逆，以避免额外的逆矩阵舍入误差。示例真实样本的PyTorch坐标与体素分配已对齐原模型；
不同执行后端仍可能产生float32舍入差异，评估时需与原模型比较。

## 输出与示例

两种模式均输出6个检测张量及`occ_logits`：`[1,288,200,200]`。
OCC通道`z*18+c`对应高度z、类别c；恢复原始logits布局：

```python
legacy = raw.reshape(1, 16, 18, 200, 200).transpose(0, 4, 3, 1, 2)
labels = raw.reshape(1, 16, 18, 200, 200).argmax(2).transpose(0, 3, 2, 1)
```

动态模式另保存`<output>.inputs.npz`（例如`bev_occ_dynamic.inputs.npz`），
包含六个输入的示例值，可用`np.load`读取。它是NPZ容器，不是SDK运行器可直接读取的输入binary；
SDK运行与量化需要将每个张量单独保存为连续binary，并按模型输入名/顺序组织多输入数据。
固定模式仍只有一个图像输入。

## SDK部署边界

动态模式使用ONNX opset16的标准`ScatterND(reduction="add")`：同一体素的多个射线必须相加，
不能改成默认覆盖写入，否则结果错误。该实现避免冻结投影，不使用CUDA/MMDeploy自定义节点。

已验证原视角变换参考值、同一ONNX更换外参后的结果、重复索引累加、全投影越界情况，以及完整模型导出和ONNX checker。
**不表示当前OMG版本已支持ScatterND-add的编译或量化执行。** 必须使用六输入配置重新执行OMG检查，
必要时确认厂商是否支持该reduction；不应通过忽略reduction或固定索引来强行编译。
原有SDK `input_cfg.yml`只声明img，现有`05_inference.sh`的infer也只支持单图像输入，不能原样运行这个六输入模型。
本次没有修改量化校准清单、SDK运行脚本或原来的ORTM。

测试：

```bash
python -m pytest -q tests/test_tools/test_export_bev_occ_dynamic.py tests/test_tools/test_export_bev_occ_nchw.py
```

## SDK高维广播兼容

动态投影已改为展平射线后的逐分量Mul/Add，避免原来的7维矩阵广播在SDK内部产生8维view，同时避免动态batch.gemm内核限制。
Lift特征乘法也改成低维布局，中间几何/池化张量不超过4维；六输入接口和外部校准binary保持不变。
遇到旧模型的EXPAND_BROADCAST维数越界，需要用当前脚本重新导出、复制到SDK模型源目录，再执行00/01/03。
算子类型检查通过不等于后端完整编译成功。

2026-10-20：逐分量低维版本已使用epoch20权重和一帧校准示例，在SDK 2.1.6/LQ560ProV200上完成OMG完整编译并生成ORTM（验证在PC临时目录进行）。
这确认后端能生成模型，不代表完整数值仿真、量化精度或其他几何范围已经验证。旧ONNX不会自动更新，需重新导出并复制后再编译。
