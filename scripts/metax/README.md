# MetaX C500 离线推理

本目录提供与 Ascend 64 GiB 基线同规格的 MetaX C500 推理入口：

- 11 个单卡用例：`infer/single/`
- 1 个双卡 SP2 用例：`infer/dist_2/`
- 11 个八卡用例：`infer/dist_8/`
- 完整套件、脱离终端运行和断点恢复工具

固定使用以下本机资源：

| 资源 | 路径 |
| --- | --- |
| Examples 项目 | `/data/Lightx2v-Platform-Run-Examples` |
| LightX2V 项目 | `/data/LightX2V-metax` |
| 模型权重 | `/data/models` |
| MetaX 配置 | `/data/Lightx2v-Platform-Run-Examples/configs/metax` |

## 性能配置

配置使用当前全重 BF16，不启用蒸馏、权重量化或特征缓存。

- Attention 使用本机实测最快且保持 BF16 精确结果的 `flash_attn2`。
- RoPE、LayerNorm、RMSNorm 和 modulation 使用兼容的 PyTorch 实现。
- 正式运行固定使用 `PROFILING_DEBUG_LEVEL=2`，保存同步后的 Level1
  `step_pre`、`infer_main`、`step_post` 单步明细以及 Level2 全流程计时。
  `run.json` 会按 step 保留各 rank 原始值，并以每步最慢 rank 作为多卡
  用例的作业级 DiT 耗时，不能跨 rank 求和。
- `CFG2×SP4` 的四卡 SP 组分别落在 C500 的 `0–3`、`4–7` MXLink 岛内。
- MetaX 初始化会在创建 MCCL 进程组前按 `LOCAL_RANK` 绑定 C500。单轴 SP2/SP8 直接复用默认 WORLD 进程组；`CFG2×SP4` 不在多个 MCCL communicator 间切换，而是把逻辑 SP/CFG 通信映射为 WORLD 上一次提交的批量 P2P。
- 生产 BF16 A2A 尺寸 `(4,8190,3,128)` 经 10 次预热、60 次稳态实测，批量 WORLD-P2P 的 8-rank 最大延迟中位数为 `0.548 ms`（p95 `0.687 ms`），WORLD8 全量 all-gather 中位数为 `2.954 ms`；前者快约 `5.39×`。变长逻辑组 gather 也已覆盖 Qwen 正/负提示词长度不同的情况。
- LTX-2.3 S2V 最终直接复用输入音频做 mux，并跳过随后必然被覆盖的 audio-VAE decode。SP8 的纯音频条件入口还会只在 global rank 0 加载/执行 Gemma 与 VAE、编码输入音频和解码保存结果；其余 rank 通过 WORLD 接收音频 latent 与四个文本 context。这样避免 C500 同时构建 8 份组件时出现 `MX_QUEUE_COMPUTE_MQL (type:21)` 队列争用，并让每张非主卡少驻留约 1.69 GiB VAE 权重。
- Z-Image-Turbo 使用同一 MXLink 岛内的 GPU `0,1` 做 SP2。
- Qwen-Image CFG2×SP4 在 50 步结束后先同步 8 个 rank、回收 DiT 引用和 allocator cache，再执行并行 VAE decode；该顺序用于给 C500 的 Conv3d workspace 留出连续显存。
- Wan2.2 720p CFG2×SP4 在首次 denoise 前及随后每 4 步清理一次 allocator cache。C500 的 expandable-segments 选项因显存页大小不受支持，实测长序列即使仍有十余 GiB 总空闲，也可能因找不到 500 MiB 连续块而失败；按步清理只作用于这个 720p 配置。
- Wan2.2 720p 未启用 QKV fusion：该序列下三个约 184.6 MiB 投影会被打包成一次约 553.7 MiB 连续分配，正好放大 C500 的碎片风险，而预期算子收益不足 1%。HunyuanVideo 720p 同样保留分离投影，避免在不足 100 MiB 的实测显存余量上增加打包 workspace。

64 GiB 下的 offload 策略：

| 模型/模式 | 策略 |
| --- | --- |
| Wan2.1、Self-Forcing、LongCat、Z-Image | 无 offload |
| Wan2.2 单卡、CFG2×SP4 | DiT model offload；T5 流式 offload |
| Wan2.2 TP8 | 无 offload |
| HunyuanVideo-1.5 | 仅 Qwen2.5-VL offload |
| Qwen-Image-2512 | DiT model offload；Qwen2.5-VL 编码后 offload |
| LTX-2.3 单卡 | DiT model 和 Gemma offload，VAE 常驻 |
| LTX-2.3 SP8 | DiT model 和 rank-0 Gemma offload；VAE 仅 rank 0 常驻 |
| FLUX.2-dev 单卡 | DiT block offload |
| FLUX.2-dev TP8 | Mistral3 文本编码后 offload；TP DiT 和 VAE 常驻 |

## 实测验证

正式套件 `metax_20260725T072300Z` 已在 MetaX C500 `8 × 64 GiB` 节点完成 `23/23` 个用例，结束时间为 `2026-07-26T12:59:18Z`。表中耗时取各成功 `run.json` 的 `timing.duration_seconds`，是单样本端到端 wall time，包含模型加载、推理、产物保存及运行包装器收尾；它不是纯算子时间，也不是多轮统计中位数。因此这些数字用于确认当前权重和配置能够完整运行，并提供本机速度参考，不应直接解释为稳定吞吐基准。

该套件最初在 LightX2V `main@d658c11edb77124387f275c1c080fcf9612ba8e1` 上启动，随后在 `ltx_tmp@9fe703cdbed4dbb18d4dc2a276fd7e3552af7998` 及 MetaX 工作树补丁上恢复。表中 `main`、`ltx_tmp` 表示产物对应 `run.json` 记录的 LightX2V 基线；Examples 基线为 `main@7a5ba4c944bbef2485fbc6bc78a875f4f3cfd5c2`。最终审计会另外保存两个工作树的 tracked patch、untracked source archive、逐文件清单和 SHA-256，使实际执行的 dirty working tree 可以复原。

| # | 分组 | 用例 | 并行方式 | LightX2V 基线 | wall time (s) | 产物状态 |
| ---: | --- | --- | --- | --- | ---: | --- |
| 01 | `single` | `z_image_turbo_t2i_1664x928` | 单卡 | `main` | 152.335 | 直接成功 |
| 02 | `dist_2` | `z_image_turbo_t2i_1664x928_sp2` | SP2 | `main` | 241.794 | 直接成功 |
| 03 | `single` | `wan21_1_3b_self_forcing_t2v_480p_81f` | 单卡 | `ltx_tmp` | 143.604 | 直接成功 |
| 04 | `single` | `wan21_1_3b_t2v_480p_81f` | 单卡 | `main` | 449.423 | 直接成功 |
| 05 | `dist_8` | `wan21_1_3b_t2v_480p_81f_cfg2_sp4` | CFG2×SP4 | `ltx_tmp` | 203.282 | 直接成功 |
| 06 | `single` | `longcat_image_t2i_1344x768` | 单卡 | `main` | 173.416 | 直接成功 |
| 07 | `dist_8` | `longcat_image_t2i_1344x768_cfg2_sp4` | CFG2×SP4 | `ltx_tmp` | 232.427 | 直接成功 |
| 08 | `single` | `qwen_image_2512_t2i_1664x928` | 单卡 | `ltx_tmp` | 426.045 | 直接成功 |
| 09 | `dist_8` | `qwen_image_2512_t2i_1664x928_cfg2_sp4` | CFG2×SP4 | `ltx_tmp` | 415.028 | 直接成功 |
| 10 | `single` | `flux2_dev_t2i_1344x768` | 单卡 | `main` | 531.695 | 直接成功 |
| 11 | `dist_8` | `flux2_dev_t2i_1344x768_tp8` | TP8 | `ltx_tmp` | 685.373 | 直接成功 |
| 12 | `single` | `wan22_moe_a14b_t2v_480p_81f` | 单卡 | `main` | 2144.119 | 校验恢复¹ |
| 13 | `dist_8` | `wan22_moe_a14b_t2v_480p_81f_cfg2_sp4` | CFG2×SP4 | `ltx_tmp` | 912.463 | 直接成功 |
| 14 | `dist_8` | `wan22_moe_a14b_t2v_480p_81f_tp8` | TP8 | `ltx_tmp` | 771.737 | 直接成功 |
| 15 | `single` | `wan22_moe_a14b_t2v_720p_81f` | 单卡 | `main` | 6110.638 | 校验恢复¹ |
| 16 | `dist_8` | `wan22_moe_a14b_t2v_720p_81f_cfg2_sp4` | CFG2×SP4 | `ltx_tmp` | 1299.695 | 直接成功（a6） |
| 17 | `dist_8` | `wan22_moe_a14b_t2v_720p_81f_tp8` | TP8 | `ltx_tmp` | 1304.545 | 直接成功 |
| 18 | `single` | `hunyuan_video_15_t2v_480p_121f` | 单卡 | `main` | 2406.604 | 校验恢复¹ |
| 19 | `dist_8` | `hunyuan_video_15_t2v_480p_121f_cfg2_sp4` | CFG2×SP4 | `ltx_tmp` | 577.338 | 直接成功 |
| 20 | `single` | `hunyuan_video_15_t2v_720p_121f` | 单卡 | `main` | 5860.720 | 校验恢复¹ |
| 21 | `dist_8` | `hunyuan_video_15_t2v_720p_121f_cfg2_sp4` | CFG2×SP4 | `ltx_tmp` | 1106.551 | 直接成功 |
| 22 | `single` | `ltx2_3_22b_dev_s2v_768x512_241f` | 单卡 | `ltx_tmp` | 964.962 | 直接成功 |
| 23 | `dist_8` | `ltx2_3_22b_dev_s2v_768x512_241f_sp8` | SP8 | `ltx_tmp` | 645.989 | 直接成功（a4） |

¹ 用例 12、15、18、20 的模型推理和产物写入已经完成，但当时系统 `ffmpeg` 因缺少 `libfreetype.so.6` 使包装器返回 74。恢复流程没有重新生成或替换文件，而是仅对 allowlist 中原 SHA-256 完全一致的产物使用修正后的 PyAV 校验器做全量解码；通过后在 `suite.json` 中写入 `validation_only_recovery`、原始错误、原产物 SHA-256 和旧 `run.json` 归档路径。因此这些条目是“校验恢复”，不是在 `ltx_tmp` 上重新推理成功。

每项直接成功都要求子进程返回 0，结果文件存在且非空，并满足目标规格。PNG 由 Pillow 验证容器和尺寸，较新的记录还检查 RGB extrema；MP4 使用 PyAV 校验容器、编码、宽高、帧数、帧率，并解码全部预期视频帧。LTX S2V 还要求 AAC 音轨存在并可完整解码。用例 16 的 a6 产物为 H.264 `1280×720`、`81` 帧，大小 `1,835,999` bytes，SHA-256 为 `af01df7a3020f87e3d46fc5036cedd935d146788afe9fcf2f985e96691842ac5`；用例 23 的 a4 产物为 H.264 `768×512`、`241` 帧、`24 fps`，带 AAC 音轨，PyAV 全量解码通过。

LTX SP8 的 rank-0 Gemma/VAE 路径另做过一次不计入正式性能表的 1-step canary：`metax_ltx_sp8_smoke_20260726T115506Z` 在 `ltx_tmp@9fe703c` 上以 8 rank 完成，端到端耗时 `487.707 s`，没有 `type:21` retry，supervisor 返回 0。输出为 H.264 `768×512`、`241` 帧、`24 fps`，带 AAC 音轨；PyAV 全量解码通过，文件大小 `3,989,391` bytes，SHA-256 为 `eae4f7a2d6e1ce7e0708dea4bb468354cd5e399d2034a6ac1b60e46c185453ce`。诊断日志与结果分别位于：

```text
logs/metax/infer/diagnostics/dist_8/ltx2_3_22b_dev_s2v_768x512_241f_sp8_1step_smoke/metax_ltx_sp8_smoke_20260726T115506Z/
results/metax/infer/diagnostics/dist_8/ltx2_3_22b_dev_s2v_768x512_241f_sp8_1step_smoke/metax_ltx_sp8_smoke_20260726T115506Z/
```

为排除正式套件中早期 `main` 结果对当前工作树兼容性结论的影响，又在
`ltx_tmp@9fe703c` 和最终 MetaX 源码补丁上执行了补充套件
`metax_current_verify_20260726T130100Z`，结果为 `2/2` 成功，结束时间
`2026-07-26T13:12:48Z`：

| 用例 | run_id | wall time (s) | 严格校验结果 |
| --- | --- | ---: | --- |
| `z_image_turbo_t2i_1664x928_sp2` | `metax_current_verify_20260726T130100Z_01_a1` | 271.651 | PNG `1664×928`，Pillow load/verify 与 RGB extrema 通过 |
| `qwen_image_2512_t2i_1664x928_cfg2_sp4` | `metax_current_verify_20260726T130100Z_02_a1` | 408.787 | PNG `1664×928`，Pillow load/verify 与 RGB extrema 通过 |

Qwen 补充产物的 SHA-256 仍为
`146ec61e56bdde06eb5c2200dd37d14929b93a97eda6c1e2b604fdc0037e0364`，
与正式套件中的当前代码产物完全一致。补充套件两个 attempt 均无
`type:21` retry、无 MCCL 共享内存残留，子进程和包装器返回码均为 0。

最终审计命令会输出本次实际生成的 `final_*` 目录；交付说明应记录该精确
路径。只有 `audit_summary.json` 报告 23 项严格复验全部通过、当前
config/script 无哈希漂移、源码归档不含 MLU 内容时，才视为审计成功。

完整套件结束后的最终审计目录保存在：

```text
logs/metax/infer/suites/metax_20260725T072300Z/audits/final_<UTC>_<suffix>/
├── audit_summary.json
├── latest_success_file_manifest.json
├── source_repositories.json
├── untracked_metax_sources.tar
└── SHA256SUMS
```

## 运行单个用例

```bash
cd /data/Lightx2v-Platform-Run-Examples

# 单卡
bash scripts/metax/infer/single/run_wan21_1_3b_t2v_480p_81f.sh

# 八卡 CFG2×SP4
bash scripts/metax/infer/dist_8/run_wan21_1_3b_t2v_480p_81f_cfg2_sp4.sh

# 双卡 SP2
bash scripts/metax/infer/dist_2/run_z_image_turbo_t2i_1664x928_sp2.sh
```

入口默认使用 GPU 0、GPU 0–1 或 GPU 0–7。需要映射到其他物理卡时，可在启动前设置数量匹配的 `CUDA_VISIBLE_DEVICES`。

## 日志和结果

每次运行生成独立 `run_id`，单卡、双卡和八卡分别保存：

```text
logs/metax/infer/single|dist_2|dist_8/<case_id>/<run_id>/
├── run.log
└── run.json

results/metax/infer/single|dist_2|dist_8/<case_id>/<run_id>/
└── output.png|mp4
```

成功状态要求 LightX2V 返回码为 0，且产物存在、非空并通过 PNG/MP4 格式与目标规格校验。失败或中断时，已有日志、结构化记录和产物同样保留。

`run.json` 的 `metrics.dit_step_profile.steps` 保存每一步和每个 rank 的
`infer_main` 时间；`dit_seconds_per_step` 和首步只会在
profile 同步、rank 数完整且步数完整时写入。Self-Forcing 的异步 VAE
路径使用不插入逐步 barrier 的 device event，并在异步流水线结束后统一
读取；原有 `(non-sync)` CPU enqueue 记录仍保留用于审计，但解析器会优先
采用 device event，避免把入队时间误报为 GPU 单步时间。

多卡入口会在启动 MCCL 前动态读取宿主机 `/dev/shm` 的实际容量、可用空间及已有 `mccl-*` 文件，而不假定固定 tmpfs 大小；当前节点挂载约为 `1008 GiB`。预检要求至少保留 48 MiB 可用空间，空间不足时会在创建 communicator 前失败并报告文件数量及大小，避免 mmap 后首次写入触发 SIGBUS。套件只会隔离文件名 PID 属于本次已退出进程组的新残留，其他作业或基线文件不会被移动；隔离目录保存在对应 attempt 的日志目录中。

## 完整套件

只读查看或预演 23 个用例：

```bash
python scripts/metax/run_infer_suite.py --list
python scripts/metax/run_infer_suite.py --dry-run
```

以独立 session 顺序运行全部用例；关闭终端或 VS Code 不会停止任务：

```bash
bash scripts/metax/run_all_detached.sh
```

启动命令会输出 `suite_id`、后台 PID、controller 日志和状态文件。套件默认在单项失败后继续，其状态保存在：

```text
logs/metax/infer/suites/<suite_id>/
├── controller.log
├── controller.lock
├── controller.pid
└── suite.json
```

套件结束时会自动聚合所有 MetaX suite，生成与 Ascend 相同结构的最终
报告：

```text
logs/metax/infer/suites/final_report.md
logs/metax/infer/suites/final_report.json
```

也可以在不运行推理的情况下手工重新生成：

```bash
python scripts/aggregate_infer_reports.py --platform metax
```

恢复失败、运行中或未完成的用例：

```bash
bash scripts/metax/resume_detached.sh <suite_id>
```

恢复时会用 suite 锁和全局 GPU 锁防止多个 MetaX suite 并发占卡，并在跳过成功项前重新核对结果文件大小和 SHA-256。每个分布式用例前默认保留 30 秒运行时释放窗口。已出现至少 12 次的 C500 `type:21` 计算队列重试若持续 300 秒且没有真实业务进度，会被提前判定为队列死锁；其他用例若连续 1800 秒没有 `run.log`、`run.json` 或结果文件进度，也会被终止。清理会先处理入口进程组，再按 PID 与 `/proc` starttime 双重身份、由深到浅处理 torchrun 创建的独立 session 后代，避免漏掉 worker 或误伤 PID 复用后的无关进程，诊断证据会写回 attempt 记录。

`--scope single`、`--scope multi` 或重复传入 `--only <case_id>` 可用于新套件的子集，不与 `--resume` 混用。所有入口强制使用本地权重并设置 Hugging Face/Transformers 离线模式，因此断开 VS Code、终端或网络不影响已经启动的套件。
