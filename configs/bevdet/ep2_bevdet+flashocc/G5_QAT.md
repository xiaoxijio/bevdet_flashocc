# G5 QAT 四卡训练

配置：`bevdet-b0-det-occ-g5-official-qat.py`。

- 保留 G5 EfficientNetB0HZ、检测/OCC 分支、数据增强和 worker CPU 检测标签生成。
- 从 `work_dirs/ep1_g5_official_fast/epoch_20_ema.pth` 初始化，而非恢复旧 QAT。
- FP32 fake-quant INT8 QAT，不启用 AMP、普通 EMA；BN 从开始冻结，observer 在第 3 个完整 epoch 后冻结。
- 每卡 batch 12（全局 48），每卡 8 workers。AdamW lr=1e-5 保持不变；warmup 84 steps 对应约 4000 样本。
- 20 epoch，step=[16,19]，不带验证，最近三个 epoch checkpoint 保留。
- `qat.coalesced_sync=True` 将量化 collector 小 buffer 合并广播；仍每步使用 rank 0 的值，未改变同步频率或量化算法。默认关闭以兼容其他配置。

## 2026-10-28 实测

四卡、真实 NuScenes 训练数据、同一 G5 最终权重和 seed=0。每组 70 iterations，统计 iter 30～70 的平均耗时，启动/首批时间不计入吞吐；峰值显存包含启动阶段。

| 每卡 batch | 同步方式 | 四卡样本/秒（约） |
|---|---|---|
| 4 | 逐个广播 | 37.7 |
| 8 | 逐个广播 | 52.9 |
| 12 | 逐个广播 | 60.6 |
| 8 | 合并广播 | 54.6 |
| 12 | 合并广播 | 62 |

选用 batch 12 + 合并广播，峰值 allocated 29287 MiB；70-step 测试不等同于整轮准确率验证，也不保证 GPU 利用率持续 100%。
两卡同步测试覆盖浮点、整数、uint8、空 buffer，与原逐个广播结果逐元素完全一致。
原 batch 4 配置和源码备份，以及完整日志、GPU 采样、吞吐汇总在 `work_dirs/g5_qat_throughput/`。

```bash
conda activate bevdet_qat
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
bash tools/dist_train.sh configs/bevdet/ep2_bevdet+flashocc/bevdet-b0-det-occ-g5-official-qat.py 4
```

正式输出：`work_dirs/ep2_g5_official_qat/`；后台控制台：`launcher.log`。本次切换发生在旧训练首个 epoch 内，尚无 QAT checkpoint，因此重新从 G5 初始化。原训练日志保留。

最终 batch12+合并同步：62.0245 samples/s，0.773888 s/step；GPU 稳定段平均利用率 72.125%，nvidia-smi 峰值使用 31908 MiB。正式训练已于 09:46 启动，epoch 1 iter 40 稳定段 0.760 s/step，loss=0.7973、grad_norm=5.1206；无验证。
