# Ascend、MLU、MetaX DiT 单步耗时对比

- 整理日期：2026-07-29
- 覆盖范围：11 个单卡配置、1 个双卡配置、11 个八卡配置，共 23 个配置
- 结果状态：三个平台均为 23/23 个离线推理配置最终通过
- 核心指标：一次有效离线推理中，所有主 DiT（primary `infer_main`）调用的平均耗时；数值越低越好

## 统计口径

1. 每个平台先从离线推理最终报告中取 `selected_success`，不使用失败、中断、旧版本误判或未被最终报告选中的运行。
2. 使用当前统一解析器重新读取所选 `run.log`：
   - 单卡：逐个主 DiT 调用耗时求算术平均；通常取 `infer_main`，MetaX Self-Forcing 因异步主机计时不可用而取同步的 device-event；
   - 多卡：每个逻辑 step 先取各 rank 的最大耗时，再对全部 step 求算术平均；
   - 首个 step 计入平均值；
   - 仅接受 `complete=true`、`synchronized=true`、`authoritative=true` 的逐步记录。
3. 23 × 3 份所选日志均通过上述完整性检查。
4. Wan2.1 Self-Forcing 实际执行 7 个 temporal segment，每段 4 次主 DiT，共 28 次调用，因此按这 28 次调用求平均；不计 7 次 `infer_main_in_rerun` 和 chunk 周边开销。
5. Ascend 的旧最终报告生成于逐步解析逻辑完善之前，其中 `dit_seconds_per_step` 是 `Run DiT` 总时长除以配置步数。为保证三平台口径一致，本报告不直接使用该旧字段，而是与 MLU、MetaX 一样从逐步日志回溯计算。
6. CPU placement/offload 取自每个 `selected_success` 的实际运行配置，而不是当前配置文件：
   - `DiT:model`、`DiT:block` 分别表示主 DiT 按整模型、逐 block CPU offload；
   - `TE:*` 表示对应文本编码器 CPU offload；`VAE` 表示 VAE CPU offload；
   - 组件未显式设置时，按实际实现回退到全局 `cpu_offload`；全局关闭时忽略不生效的 `offload_granularity`；
   - `empty_cache`、tiling、异步 VAE 和 rank0 组件归属不是 CPU placement，不用于拆行。
7. 同一 workload 的有效 CPU placement/offload 签名不同时拆成不同物理行；只有同一签名行内、且至少两个平台有数值时才评选最低值。

## 全量结果

单卡和多卡结果仍放在同一张表中。表内粗体仅表示同一 CPU placement/offload 签名下的最低平均 DiT 单步耗时；`—` 表示该平台没有运行此精确签名，因此不可作同配置比较。签名只列启用项，未列出的相关组件均驻留设备端；`TE:QwenVL` 指 Qwen2.5-VL 文本编码器。

| 分组 | 模型 | 任务 | 卡数 | 并行 | 输出规格 | 主 DiT 调用数 | CPU placement / offload | Ascend (s) | MLU (s) | MetaX (s) | 同签名最低 |
|---|---|---:|---:|---|---|---:|---|---:|---:|---:|---|
| 单卡 | Z-Image-Turbo | T2I | 1 | 单卡 | 1664×928 | 9 | 无 | 1.037471 | **0.720033** | 1.105694 | MLU |
| 单卡 | Wan2.1 Self-Forcing 1.3B | T2V | 1 | 单卡 | 832×480×81 | 4×7 段 | 无 | 0.359632 | **0.274299** | 0.497018 | MLU |
| 单卡 | Wan2.1 1.3B | T2V | 1 | 单卡 | 832×480×81 | 50 | 无 | **3.657893** | 4.959325 | 6.875425 | Ascend |
| 单卡 | LongCat-Image | T2I | 1 | 单卡 | 1344×768 | 50 | 无 | **0.400073** | 0.596893 | 0.656357 | Ascend |
| 单卡 | Qwen-Image-2512 | T2I | 1 | 单卡 | 1664×928 | 50 | TE:QwenVL | 1.236078 | — | — | — |
| 单卡 | Qwen-Image-2512 | T2I | 1 | 单卡 | 1664×928 | 50 | 无 | — | 1.982083 | — | — |
| 单卡 | Qwen-Image-2512 | T2I | 1 | 单卡 | 1664×928 | 50 | DiT:model；TE:QwenVL | — | — | 3.124710 | — |
| 单卡 | FLUX.2-dev | T2I | 1 | 单卡 | 1344×768 | 50 | DiT:block；TE:Mistral3；VAE | 45.905810 | — | **1.999385** | MetaX |
| 单卡 | FLUX.2-dev | T2I | 1 | 单卡 | 1344×768 | 50 | DiT:model；TE:Mistral3 | — | 1.828039 | — | — |
| 单卡 | Wan2.2 MoE A14B | T2V | 1 | 单卡 | 832×480×81 | 40 | DiT:model | 21.480221 | — | — | — |
| 单卡 | Wan2.2 MoE A14B | T2V | 1 | 单卡 | 832×480×81 | 40 | 无 | — | 25.707417 | — | — |
| 单卡 | Wan2.2 MoE A14B | T2V | 1 | 单卡 | 832×480×81 | 40 | DiT:model；TE:T5 | — | — | 40.920633 | — |
| 单卡 | Wan2.2 MoE A14B | T2V | 1 | 单卡 | 1280×720×81 | 40 | DiT:model | 89.886009 | — | — | — |
| 单卡 | Wan2.2 MoE A14B | T2V | 1 | 单卡 | 1280×720×81 | 40 | 无 | — | 101.977013 | — | — |
| 单卡 | Wan2.2 MoE A14B | T2V | 1 | 单卡 | 1280×720×81 | 40 | DiT:model；TE:T5 | — | — | 141.801269 | — |
| 单卡 | HunyuanVideo-1.5 | T2V | 1 | 单卡 | 848×480×121 | 50 | TE:QwenVL | **18.082941** | — | 40.144035 | Ascend |
| 单卡 | HunyuanVideo-1.5 | T2V | 1 | 单卡 | 848×480×121 | 50 | 无 | — | 25.094830 | — | — |
| 单卡 | HunyuanVideo-1.5 | T2V | 1 | 单卡 | 1264×720×121 | 50 | TE:QwenVL | **82.441129** | — | 110.426954 | Ascend |
| 单卡 | HunyuanVideo-1.5 | T2V | 1 | 单卡 | 1264×720×121 | 50 | 无 | — | 109.359756 | — | — |
| 单卡 | LTX-2.3 22B dev | S2V | 1 | 单卡 | 768×512×241 | 30 | DiT:model；TE:Gemma | **14.354104** | 17.352747 | 19.590502 | Ascend |
| 多卡 | Z-Image-Turbo | T2I | 2 | SP2 | 1664×928 | 9 | 无 | 0.688291 | **0.494025** | 0.672701 | MLU |
| 多卡 | Wan2.1 1.3B | T2V | 8 | CFG2×SP4 | 832×480×81 | 50 | 无 | **0.514387** | 0.692286 | 0.975208 | Ascend |
| 多卡 | LongCat-Image | T2I | 8 | CFG2×SP4 | 1344×768 | 50 | 无 | **0.135231** | 0.156441 | 0.172231 | Ascend |
| 多卡 | Qwen-Image-2512 | T2I | 8 | CFG2×SP4 | 1664×928 | 50 | TE:QwenVL | 0.328629 | — | — | — |
| 多卡 | Qwen-Image-2512 | T2I | 8 | CFG2×SP4 | 1664×928 | 50 | 无 | — | 0.365240 | — | — |
| 多卡 | Qwen-Image-2512 | T2I | 8 | CFG2×SP4 | 1664×928 | 50 | DiT:model；TE:QwenVL | — | — | 0.715319 | — |
| 多卡 | FLUX.2-dev | T2I | 8 | TP8 | 1344×768 | 50 | 无 | **0.285134** | 0.419370 | — | Ascend |
| 多卡 | FLUX.2-dev | T2I | 8 | TP8 | 1344×768 | 50 | TE:Mistral3 | — | — | 0.460026 | — |
| 多卡 | Wan2.2 MoE A14B | T2V | 8 | CFG2×SP4 | 832×480×81 | 40 | DiT:model | 3.769141 | — | — | — |
| 多卡 | Wan2.2 MoE A14B | T2V | 8 | CFG2×SP4 | 832×480×81 | 40 | 无 | — | 3.368436 | — | — |
| 多卡 | Wan2.2 MoE A14B | T2V | 8 | CFG2×SP4 | 832×480×81 | 40 | DiT:model；TE:T5 | — | — | 4.483104 | — |
| 多卡 | Wan2.2 MoE A14B | T2V | 8 | TP8 | 832×480×81 | 40 | 无 | 4.498276 | **4.296255** | 7.549338 | MLU |
| 多卡 | Wan2.2 MoE A14B | T2V | 8 | CFG2×SP4 | 1280×720×81 | 40 | DiT:model | 11.945277 | — | — | — |
| 多卡 | Wan2.2 MoE A14B | T2V | 8 | CFG2×SP4 | 1280×720×81 | 40 | 无 | — | 13.034071 | — | — |
| 多卡 | Wan2.2 MoE A14B | T2V | 8 | CFG2×SP4 | 1280×720×81 | 40 | DiT:block；TE:T5 | — | — | 13.993054 | — |
| 多卡 | Wan2.2 MoE A14B | T2V | 8 | TP8 | 1280×720×81 | 40 | 无 | 15.263588 | **15.167144** | 21.033103 | MLU |
| 多卡 | HunyuanVideo-1.5 | T2V | 8 | CFG2×SP4 | 848×480×121 | 50 | TE:QwenVL | **2.390244** | — | 3.836540 | Ascend |
| 多卡 | HunyuanVideo-1.5 | T2V | 8 | CFG2×SP4 | 848×480×121 | 50 | 无 | — | 3.305429 | — | — |
| 多卡 | HunyuanVideo-1.5 | T2V | 8 | CFG2×SP4 | 1264×720×121 | 50 | TE:QwenVL | **10.864043** | — | 13.520038 | Ascend |
| 多卡 | HunyuanVideo-1.5 | T2V | 8 | CFG2×SP4 | 1264×720×121 | 50 | 无 | — | 13.954177 | — | — |
| 多卡 | LTX-2.3 22B dev | S2V | 8 | SP8 | 768×512×241 | 30 | DiT:model；TE:Gemma | **4.196967** | 4.536204 | 5.096624 | Ascend |

## 结果摘要

- 23 个 workload 中有 12 个存在跨平台 CPU placement/offload 差异；拆分后主表为 41 个签名行。
- 11 个 workload 的三平台签名完全相同；另有 6 个 workload 存在两平台同签名子组，共形成 17 个至少两平台可比的签名组。
- 只统计这些同签名组：Ascend 有 11 组最低，MLU 有 5 组最低，MetaX 有 1 组最低；单平台签名不计胜负。
- Qwen-Image 单/多卡及四个 Wan2.2 单卡/CFG2×SP4 workload 均为三平台三种签名，没有严格的跨平台同签名对照。
- FLUX.2 单卡中，Ascend 与 MetaX 同为 `DiT:block；TE:Mistral3；VAE`，MetaX 的 1.999385 秒低于 Ascend 的 45.905810 秒；MLU 的 1.828039 秒属于 `DiT:model；TE:Mistral3`，不与前两者评胜负。
- Wan2.2 720p TP8 的 MLU 与 Ascend 非常接近，分别为 15.167144 秒和 15.263588 秒，相差约 0.64%。
- Ascend FLUX.2 单卡的 45.905810 秒是明显离群值，而同平台 TP8 为 0.285134 秒。该差异与实际 offload、并行配置和运行栈相关，不能解释为单纯的卡数扩展效率。

## 使用限制

- 本报告比较的是各平台已经通过严格产物校验的实际配置，不是控制所有软件变量后的纯硬件 micro-benchmark。
- 三个平台的硬件容量、驱动和框架、LightX2V 工作树、算子实现及 offload 策略并不完全相同。
- DiT 单步耗时不包含模型加载、文本编码、VAE、音频处理、文件编码和落盘，因此不能直接替代端到端耗时或 Server 延迟。
- 即使差异只发生在 TE/VAE offload，本报告仍为配置可追溯性而拆行；这些阶段本身不进入 DiT 单步计时。
- 拆行只控制 CPU placement/offload，不代表完整配置相同。MetaX Self-Forcing 的异步 VAE 及 MetaX Qwen 的 VAE 前缓存清理等非 offload 差异不据此拆行。
- 首步计入均值，因此首步编译或缓存建立的影响也保留在结果中。
- “最低值”仅描述本批记录中的观测值，不代表统计显著性；每个配置没有进行多轮独立重复测试。

## 数据追溯

- [Ascend 离线最终报告](../../logs/ascend_npu/infer/suites/final_report.json)
- [MLU 离线最终报告](../../logs/mlu/infer/suites/final_report.json)
- [MetaX 离线最终报告](../../logs/metax/infer/suites/final_report.json)
- [统一逐步解析实现](../../scripts/run_record.py)

每个最终报告的 `cases[].selected_success.paths` 可继续追溯到所选 `run.log`、`run.json` 和生成产物。Ascend 报告保留了采集机器上的旧绝对前缀 `/data/wushuo1/Lightx2v-Platform-Run-Examples/`；在当前工作区追溯时，应将其映射为仓库根 `/data/Lightx2v-Platform-Run-Examples/`。
