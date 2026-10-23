# BEVDet + FlashOCC：ONNX 导出与缓存映射

在 3090 的 `/home/fjm/tong/BEVDet` 根目录激活 `bevdet` 环境后执行。
脚本保留两种模式：固定映射、运行时加载缓存映射。
默认配置为 `configs/bevdet/ep0_bevdet+flashocc/bevdet-r50-det-occ-shared-2gpu-32gb-pretrained.py`，
默认权重为 `ckpt/epoch_20_ema.pth`；可以通过 `--config` 和 `--checkpoint` 修改。
输出路径使用 `--output` 指定，不是位置参数。可加 `--device cpu` 在 CPU 上导出。

## 固定映射

```bash
python tools/export_bev_occ_onnx.py --output onnx/bev_occ.onnx
```

不加模式参数时默认使用固定映射，模型只有 `img` 一个输入。
映射由 `--sample-index` 指定的验证样本生成，默认第 0 个样本；
也可以通过 `--geometry-npz` 提供实际车辆的标定数据。
此模式把映射写入模型常量，修改映射需要重新导出和编译。

## 缓存映射

```bash
python tools/export_bev_occ_onnx.py \
  --runtime-mapping --output onnx/bev_occ_mapping.onnx
```

此模式在导出前计算投影映射，推理时接收图像和缓存映射，
图内不再计算相机投影，也不使用 ScatterND 聚合。
当前仅支持版本 3 的零深度填充映射：深度数组前添加一个零元素，有效深度索引加 1，补零位置索引为 0。
映射输入仅包含深度索引、特征索引和 BEV 格子索引，无需掩码。
生成三个文件：

- `bev_occ_mapping.onnx`：待编译的模型。
- `bev_occ_mapping.mapping.json`：共享映射规格，记录各组容量和网格、图像等约束。
- `bev_occ_mapping.inputs.npz`：示例图像及全部映射输入。

各组容量在导出时固定，默认 `--mapping-reserve 1.5` 为示例分组数量预留余量。
这不保证所有车辆都能容纳：应先用车队的实际标定检查映射是否符合共享规格，再确定最终模型。
容量不足时程序报错，不会截断投影点；扩大容量会增加计算量和内存开销，并需要重新导出、编译。

### 为另一辆车生成映射

保留首次导出的共享规格，使用新车标定生成新映射：

```bash
python tools/export_bev_occ_onnx.py \
  --runtime-mapping --mapping-only \
  --mapping-layout onnx/bev_occ_mapping.mapping.json \
  --geometry-npz car002_geometry.npz \
  --output onnx/car002.onnx --device cpu
```

这条命令只生成 `onnx/car002.inputs.npz`，不会创建 `car002.onnx`，也不会修改原 ONNX 或 ORTM。
相同模型规格、共享容量足够时，不同车辆可共用一份编译后的模型，只需加载各自的映射。
修改相机数量、图像尺寸、深度划分、BEV 网格或映射容量，需要重新导出和编译。

当前 `--mapping-only` 是离线准备命令，仍加载配置、权重和验证数据，但跳过网络前向和 ONNX 导出。
NPZ 内的图像是验证示例；实际部署应传入每帧真实图像，并复用该车的缓存映射。
如果运动补偿使相机到参考车体的有效变换逐帧变化，映射也需相应更新。

### 标定文件要求

`--geometry-npz` 包含以下 float32 数组（默认六相机、batch=1）：

| 名称 | 形状 | 含义 |
|---|---|---|
| sensor2egos | [1,6,4,4] | 相机到各自采集时刻车体的变换 |
| ego2globals | [1,6,4,4] | 各相机时刻车体到全局坐标系的变换 |
| intrins | [1,6,3,3] | 相机内参 |
| post_rots | [1,6,3,3] | 图像旋转、缩放等预处理变换 |
| post_trans | [1,6,3] | 图像预处理平移 |
| bda | [1,4,4] | BEV 增强变换，普通部署通常为单位矩阵 |

脚本沿用原模型的 `prepare_inputs`，将各相机对齐到参考车体坐标系：
`sensor2keyegos = inverse(keyego2global) @ ego2globals @ sensor2egos`。
图像预处理、相机顺序、内外参和参考时刻必须对应。
nuScenes 六路图像可能不同步，不能直接用安装外参替代完整的参考帧变换。

### 映射输入与精度

所有映射输入以 float32 保存。索引采用 1024 进制的高、低两部分：

- `ranks_depth_<bound>`、`ranks_feat_<bound>`：形状 `[2,slots,bound]`。
- `bev_slots`：形状 `[2,H*W]`。

索引分量 0 为低位、分量 1 为高位，各分量范围为 `[0,1023]`，FP16 可精确表示。
图内恢复整数索引：`low + high * 1024`。
SDK 输入配置不支持直接使用 I32 二进制输入，因此采用上述编码。
不要将索引分量量化为 S8，否则索引舍入会改变聚合位置。
恢复整数前保留分组的二维结构，避免长度超过 65535 的扁平向量触发 SDK 无法打包的软件转换算子。

## SDK 编译与性能仿真

将 ONNX 和示例 NPZ 复制到 PC SDK 的
`npu_toolchain/common/samples/data/model/bev_occ/`，
然后进入 `npu_toolchain/common/samples/app/tda4/mod/bev_occ/`，激活 `sdk` 环境执行：

```bash
bash 00_init.sh g1
python prepare_mapping_calibration.py   --onnx ../../../../data/model/bev_occ/g1.onnx   --npz ../../../../data/model/bev_occ/g1.inputs.npz   --out-dir calibration_g1   --config input_cfg_g1.yml
bash 01_check.sh g1 --config input_cfg_g1.yml
bash 03_omg.sh g1 --config input_cfg_g1.yml
# 仿真程序尚未构建，或 build_sim 已删除时执行
bash 04_build.sh
bash 05_inference.sh --model-name g1 --tag g1
```

准备脚本仅接受版本 3 的模型和映射，NPZ 字段必须与模型输入及版本标记完全匹配。
重新导出后须配套生成输入配置和校准数据，再编译 ORTM。
NPZ 内额外的标量 `mapping_version=3` 用于校验，不是 ONNX 输入。
默认 10 个分组时，模型共有 22 个输入：图像、20 个分组索引和 1 个 BEV 格子索引。
准备脚本生成各输入二进制、校准清单和 YAML，并在 `quant.quant_tensors` 中请求映射输入使用 FP16；
SDK 可能在整数索引路径保留 FP32，这仍可精确表示这些分量。
已有输出不会被覆盖；重复准备时通过 `--out-dir` 和 `--config` 指定新目录、配置。
可重复传入 `--npz`，使用多个真实样本准备校准集。单个示例只用于跑通流程，不代表量化精度通过。

NPZ 是容器，不是 SDK 运行器可以直接读取的二进制输入文件。
00–03 的用途不变，但缓存映射模型必须使用匹配的多输入配置。
04 构建通用仿真程序，更换模型通常无需重跑。
当前 05 的 `perf` 可估算模型性能；`infer` 仍只支持单图像输入，多输入数值推理需要另行适配运行器。
仿真估算不代表板端实测，也不包含完整应用的图像预处理和检测后处理。

## 输出布局

默认 ep0 配置输出 6 个检测张量及 `occ_logits`，后者为 `[1,288,200,200]`。
OCC 使用 NCHW，通道 `z*18+c` 对应高度 z、类别 c；其他配置以实际导出形状为准。
恢复原始布局和标签：

```python
legacy = raw.reshape(1, 16, 18, 200, 200).transpose(0, 4, 3, 1, 2)
labels = raw.reshape(1, 16, 18, 200, 200).argmax(2).transpose(0, 3, 2, 1)
```

## 当前版本验证

测试覆盖同一 ONNX 更换映射、空映射、容量不足、非法版本拒绝，以及较大索引经 FP16 编码后精确恢复。
非零首个深度概率测试验证补零索引确实指向添加的零元素；ONNX 检查确认没有掩码输入和掩码乘法。
10 项相关测试通过，epoch20 完整模型通过 ONNX 检查，并使用 OMG 2.1.6 / LQ560ProV200 成功生成 ORTM。
真实样本的 179,520 个有效投影点逐格集合一致；CPU/GPU 的同格排序可能造成微小累加舍入差异。
PC 性能仿真估算为 193.3563 ms、5.17180 FPS。这不是板端实测，量化后的数值精度仍需验收。

```bash
python -m pytest -q \
  tests/test_tools/test_export_bev_occ_mapping.py \
  tests/test_tools/test_export_bev_occ_nchw.py
```
