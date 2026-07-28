# MetaX C500 LightX2V 平台测试报告

- 报告日期：2026-07-26
- 测试平台：MetaX C500，8 卡，单卡 64 GiB
- 测试项目：LightX2V 离线推理单卡/多卡脚本、稳定性与结果有效性
- 模型目录：`/data/models`
- Examples 项目：`/data/Lightx2v-Platform-Run-Examples`
- LightX2V 项目：`/data/LightX2V-metax`
- 总体结论：**通过（PASS）**

## 1. 结论摘要

- MetaX 脚本矩阵共 23 项：11 个单卡、1 个双卡 SP2、11 个八卡用例，正式 suite 最终 **23/23 成功**。
- 23 项中 19 项为模型进程、包装器和产物校验直接成功；4 项为模型推理及产物写入成功后，针对原始 SHA-256 不变产物执行的校验恢复。
- 在最终 `ltx_tmp` 代码和 MetaX 工作树补丁上另行执行 2 项代表性回归，结果 **2/2 成功**：
  - Z-Image SP2 覆盖普通多卡序列并行路径。
  - Qwen-Image CFG2×SP4 覆盖 WORLD-P2P 逻辑通信、变长文本和显存余量较紧路径。
- Wan2.2 720p CFG2×SP4 在加入首次 denoise 前及每 4 步 allocator cache 清理后完成 40/40 步、VAE 解码和 MP4 落盘。
- LTX-2.3 SP8 在采用 rank-0 Gemma/VAE/音频 I/O 后完成 30/30 步，避免了 8 份重组件同时初始化导致的 `MX_QUEUE_COMPUTE_MQL (type:21)` 队列争用。
- 最终审计重新严格校验全部 23 个产物，结果 **23/23 通过**；46 个当前配置/入口脚本的大小和 SHA-256 与成功 `run.json` 全部一致。
- 自动化测试结果：Examples 31 项通过、LightX2V 31 项通过、8 进程 CPU/Gloo 集合通信 1 项通过。
- 任务结束后无残留 LightX2V/torchrun/suite 进程，8 卡均恢复到约 858 MiB 基线显存，`/dev/shm` 无 MCCL 文件残留。

## 2. 测试目标与范围

本轮测试验证以下目标：

1. 参考 Ascend 910B2C 的 64 GiB 规格，为 C500 提供同等覆盖的单卡和多卡入口。
2. 所有脚本固定使用 `/data/LightX2V-metax` 和 `/data/models` 下的本地权重。
3. 日志和结果按 `single`、`dist_2`、`dist_8` 分目录保存，每次运行使用独立 `run_id`。
4. 验证 C500 上 BF16、CPU offload、SP、TP、CFG2×SP4、LTX S2V 和多卡通信的兼容性。
5. 对显存碎片、MCCL 共享内存、后台持久运行、失败恢复和残留进程清理进行稳定性验证。
6. 对 PNG/MP4 产物执行格式、尺寸、帧数、帧率、像素和音频完整性校验。

本轮只测试单机最多 8 卡，不包含跨节点推理。

## 3. 测试环境

### 3.1 硬件与系统

| 项目 | 实测值 |
| --- | --- |
| 主机名 | `dev-metax-0716-0` |
| GPU | 8 × MetaX C500 |
| 单卡显存 | 64 GiB（65,536 MiB） |
| MX-SMI | 2.3.1 |
| Kernel Mode Driver | 3.6.11 |
| MACA | 3.7.1.5 |
| BIOS | 1.31.1.0 |
| 操作系统 | Linux 5.15.0-138-generic，x86_64 |
| `/dev/shm` 最终验证容量 | 约 1008 GiB |

### 3.2 软件栈

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.11 |
| PyTorch | `2.8.0+metax3.7.1.4` |
| flash-attn | `2.6.3+metax3.7.1.4torch2.8` |
| SageAttention | `2.0.1+metax3.7.1.4torch2.8` |
| xFormers | `0.0.22+metax3.7.1.4torch2.8` |
| Transformers | 4.57.6 |
| Diffusers | 0.39.0 |
| Safetensors | 0.8.0 |
| Pillow | 11.2.1 |
| NumPy | 1.26.4 |
| Loguru | 0.7.3 |

### 3.3 代码基线与可复原性

| 仓库 | 分支/提交 | 状态 |
| --- | --- | --- |
| Examples | `main@7a5ba4c944bbef2485fbc6bc78a875f4f3cfd5c2` | dirty，包含 MetaX 脚本及运行框架补丁 |
| LightX2V 最终工作树 | `ltx_tmp@9fe703cdbed4dbb18d4dc2a276fd7e3552af7998` | dirty，包含 MetaX 兼容及性能补丁 |
| LightX2V 正式 suite 早期基线 | `main@d658c11edb77124387f275c1c080fcf9612ba8e1` | 部分早期成功产物来源 |

正式 suite 跨越了早期 `main` 与最终 `ltx_tmp` 工作树，因此第 7 节逐项标明产物对应基线。最终审计保存了两个仓库的 tracked binary patch、未跟踪 MetaX 源码 tar、逐文件清单、恢复计划和 SHA-256，可复原实际运行的 dirty 工作树。

## 4. 被测实现

### 4.1 脚本矩阵

| 分组 | 卡数 | 脚本数量 | 保存目录 |
| --- | ---: | ---: | --- |
| `single` | 1 | 11 | `logs/metax/infer/single/`；`results/metax/infer/single/` |
| `dist_2` | 2 | 1 | `logs/metax/infer/dist_2/`；`results/metax/infer/dist_2/` |
| `dist_8` | 8 | 11 | `logs/metax/infer/dist_8/`；`results/metax/infer/dist_8/` |
| 合计 | — | 23 | 按 case 与 run_id 继续分层 |

### 4.2 关键适配

- 在创建 MCCL 进程组前按 `LOCAL_RANK` 调用 `torch.cuda.set_device()`。
- 单轴 SP2/SP8 复用默认 WORLD 进程组；CFG2×SP4 将逻辑 SP/CFG 通信映射为 WORLD 上的批量 P2P，避免多个 MCCL communicator 切换。
- 生产 BF16 A2A 形状 `(4,8190,3,128)` 的实测中，WORLD-P2P 8-rank 最大延迟中位数为 0.548 ms，WORLD8 全量 all-gather 为 2.954 ms，前者快约 5.39 倍。
- LTX SP8 只在 global rank 0 加载和执行 Gemma、VAE、输入音频编码及最终解码保存，其余 rank 接收广播结果。
- Qwen CFG2×SP4 在 50 步结束后先同步、释放 DiT 引用和 allocator cache，再执行 VAE decode。
- Wan2.2 720p CFG2×SP4 在首次 denoise 前及每 4 步强制释放 allocator cache，降低连续大块分配失败风险。
- 多卡入口动态检查 `/dev/shm` 容量、可用空间及已有 MCCL 文件；suite 支持独立 session、锁、断点恢复、卡死 watchdog 和递归后代清理。

### 4.3 精度与 offload

所有配置使用当前全重 BF16，不使用权重量化、蒸馏或特征缓存。主要 offload 策略如下：

| 模型/模式 | C500 64 GiB 策略 |
| --- | --- |
| Wan2.1、Self-Forcing、LongCat、Z-Image | 无 offload |
| Wan2.2 单卡、CFG2×SP4 | DiT model offload；T5 流式 offload |
| Wan2.2 TP8 | 无 offload |
| HunyuanVideo-1.5 | 仅 Qwen2.5-VL offload |
| Qwen-Image-2512 | DiT model offload；Qwen2.5-VL 编码后 offload |
| LTX-2.3 单卡 | DiT model 和 Gemma offload，VAE 常驻 |
| LTX-2.3 SP8 | DiT model 和 rank-0 Gemma offload，VAE 仅 rank 0 常驻 |
| FLUX.2-dev 单卡 | DiT block offload |
| FLUX.2-dev TP8 | Mistral3 编码后 offload；TP DiT 和 VAE 常驻 |

## 5. 验收标准

### 5.1 直接成功

用例需要同时满足：

- LightX2V 子进程返回码为 0。
- 包装器返回码为 0。
- 结果文件存在且非空。
- `run.json` 为 `status=succeeded` 且 `artifact.valid=true`。
- PNG 或 MP4 满足预期尺寸和任务规格。
- 多卡进程组正常销毁，没有残留 worker、MCCL 文件或持续 `type:21` retry。

### 5.2 产物严格校验

- PNG：Pillow load/verify、宽高、RGB channel extrema，拒绝全黑或全白结果。
- MP4：容器和 H.264 codec、宽高、帧数、帧率、时长，并完整解码全部视频帧。
- LTX S2V：除视频要求外，还要求 AAC 音轨存在，音频帧和采样数均可完整解码。
- 校验恢复：只允许使用 allowlist 中原 SHA-256 不变的产物，并保留原失败记录、恢复语义和旧 `run.json` 归档。

### 5.3 耗时口径

表中时间来自成功 `run.json` 的 `timing.duration_seconds`，为单样本冷启动端到端 wall time，包含模型加载、首次初始化、文本编码、DiT、VAE/音频处理、产物落盘和包装器收尾。

每个 case 不是多轮统计中位数，也不是纯算子耗时；因此这些数字用于验收和本机速度参考，不用于宣称稳定吞吐。

## 6. Suite 概况

| 项目 | 结果 |
| --- | --- |
| 正式 suite | `metax_20260725T072300Z` |
| 正式结束时间 | `2026-07-26T12:59:18Z` |
| 正式结果 | 23/23 success |
| 直接成功 | 19 |
| 校验恢复 | 4（case 12、15、18、20） |
| 当前代码补充 suite | `metax_current_verify_20260726T130100Z` |
| 补充结束时间 | `2026-07-26T13:12:48Z` |
| 补充结果 | 2/2 success |
| 最终审计 | 23/23 strict revalidation passed |

## 7. 正式 23 项测试结果

| # | 分组 | 用例 | 并行方式 | LightX2V 基线 | 端到端耗时（s） | 结果 |
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
| 12 | `single` | `wan22_moe_a14b_t2v_480p_81f` | 单卡 | `main` | 2144.119 | 校验恢复 |
| 13 | `dist_8` | `wan22_moe_a14b_t2v_480p_81f_cfg2_sp4` | CFG2×SP4 | `ltx_tmp` | 912.463 | 直接成功 |
| 14 | `dist_8` | `wan22_moe_a14b_t2v_480p_81f_tp8` | TP8 | `ltx_tmp` | 771.737 | 直接成功 |
| 15 | `single` | `wan22_moe_a14b_t2v_720p_81f` | 单卡 | `main` | 6110.638 | 校验恢复 |
| 16 | `dist_8` | `wan22_moe_a14b_t2v_720p_81f_cfg2_sp4` | CFG2×SP4 | `ltx_tmp` | 1299.695 | 直接成功（a6） |
| 17 | `dist_8` | `wan22_moe_a14b_t2v_720p_81f_tp8` | TP8 | `ltx_tmp` | 1304.545 | 直接成功 |
| 18 | `single` | `hunyuan_video_15_t2v_480p_121f` | 单卡 | `main` | 2406.604 | 校验恢复 |
| 19 | `dist_8` | `hunyuan_video_15_t2v_480p_121f_cfg2_sp4` | CFG2×SP4 | `ltx_tmp` | 577.338 | 直接成功 |
| 20 | `single` | `hunyuan_video_15_t2v_720p_121f` | 单卡 | `main` | 5860.720 | 校验恢复 |
| 21 | `dist_8` | `hunyuan_video_15_t2v_720p_121f_cfg2_sp4` | CFG2×SP4 | `ltx_tmp` | 1106.551 | 直接成功 |
| 22 | `single` | `ltx2_3_22b_dev_s2v_768x512_241f` | 单卡 | `ltx_tmp` | 964.962 | 直接成功 |
| 23 | `dist_8` | `ltx2_3_22b_dev_s2v_768x512_241f_sp8` | SP8 | `ltx_tmp` | 645.989 | 直接成功（a4） |

### 7.1 校验恢复说明

Case 12、15、18、20 的模型推理和文件写入已经完成，但当时系统 `/usr/bin/ffmpeg` 缺少 `libfreetype.so.6`，使包装器在推理完成后返回 74。

恢复过程没有重新推理、生成或替换文件，而是：

1. 核对原文件路径、大小和 SHA-256。
2. 只对 allowlist 中原 SHA-256 完全一致的文件执行 PyAV 全量解码。
3. 验证通过后在 suite 中记录 `validation_only_recovery`、原错误、旧记录归档和不可变产物信息。
4. 最终审计再次独立验证恢复语义及产物完整性。

因此这 4 项可用于“当前文件有效、正式 suite 完整”的验收结论，但不能写成在 `ltx_tmp` 上重新推理成功。

## 8. 当前代码补充回归

为确认最终 `ltx_tmp@9fe703c` 工作树能够兼容早期正式结果所覆盖的路径，执行了以下补充 suite：

| 用例 | run_id | 端到端耗时（s） | 产物 | 结果 |
| --- | --- | ---: | --- | --- |
| `z_image_turbo_t2i_1664x928_sp2` | `metax_current_verify_20260726T130100Z_01_a1` | 271.651 | PNG 1664×928，1,905,970 bytes，SHA-256 `dbe62028200587d79711aeb27eedd55ddc3fca1ca213b963a1962a17707f81b1` | 成功 |
| `qwen_image_2512_t2i_1664x928_cfg2_sp4` | `metax_current_verify_20260726T130100Z_02_a1` | 408.787 | PNG 1664×928，1,808,154 bytes，SHA-256 `146ec61e56bdde06eb5c2200dd37d14929b93a97eda6c1e2b604fdc0037e0364` | 成功 |

两个产物均通过 Pillow load/verify、尺寸和 RGB extrema 校验；子进程和包装器返回码均为 0，没有 `type:21` retry 或 MCCL 共享内存残留。

两个补充产物分别与正式 suite 中对应产物的文件大小和 SHA-256 完全相同，说明跨分支恢复后的两次独立运行输出一致。

## 9. 重点稳定性验证

### 9.1 Wan2.2 720p CFG2×SP4

- Case：`wan22_moe_a14b_t2v_720p_81f_cfg2_sp4`
- 最终 run：`metax_20260725T072300Z_16_a6`
- 配置：BF16、CFG2×SP4、DiT model offload、T5 流式 offload、VAE 常驻
- 修正：首次 denoise 前清一次 cache，之后每 4 步清理一次
- 结果：40/40 步完成，VAE 解码和文件保存成功
- 端到端耗时：1299.695036 秒
- 产物：H.264 MP4，1280×720，81 帧，16 fps
- 文件大小：1,835,999 bytes
- SHA-256：`af01df7a3020f87e3d46fc5036cedd935d146788afe9fcf2f985e96691842ac5`
- 校验：PyAV 全帧解码及 RGB extrema 通过

此前失败表现为仍有总空闲显存时无法找到约 500 MiB 连续块，或首次大 Conv3d 无有效 workspace。最终配置在保持性能优先的 model/T5 offload 方案下解决了 allocator 碎片窗口，没有启用更慢的逐 block offload。

### 9.2 LTX-2.3 SP8 S2V

- Case：`ltx2_3_22b_dev_s2v_768x512_241f_sp8`
- 最终 run：`metax_20260725T072300Z_23_a4`
- 配置：BF16、SP8、DiT/Gemma offload、rank-0 Gemma/VAE/音频 I/O
- 结果：30/30 步完成，视频解码、音频 mux 和保存成功
- 端到端耗时：645.988555 秒
- 产物：H.264 MP4，768×512，241 帧，24 fps，含 AAC 音轨
- 文件大小：915,312 bytes
- SHA-256：`4737b01834ff55f765faa72171c3e0c997efe1204fc3e2c54d4b2a58f08ea415`
- 校验：241 个视频帧、433 个音频帧和 443,392 个音频 samples 完整解码通过

另执行 1-step canary `metax_ltx_sp8_smoke_20260726T115506Z`，8 rank supervisor 返回 0，没有 `type:21` retry。其输出同样为 768×512、241 帧、24 fps、带 AAC 音轨的有效 MP4。

## 10. 功能和自动化测试

| 测试项 | 命令/范围 | 结果 |
| --- | --- | --- |
| Examples 运行框架单测 | `python -m pytest -q scripts/tests` | 31 passed |
| LightX2V MetaX/LTX 单测 | rank-0 helper、cache interval、distributed warmup | 31 passed |
| WORLD-P2P 集合通信 | 8 进程 CPU/Gloo；all-gather、all-to-all、变长逻辑组 | 1 passed |
| Python lint | Examples 和 LightX2V 目标文件 `ruff check` | 通过 |
| LightX2V formatter | 19 个目标文件 `ruff format --check` | 通过 |
| Shell 语法 | 所有 `scripts/metax/**/*.sh` 执行 `bash -n` | 通过 |
| JSON 语法 | 所有 `configs/metax/**/*.json` 执行 `json.tool` | 通过 |
| Suite 清单 | `run_infer_suite.py --list` | 23 项，未修改状态 |
| Suite 预演 | `run_infer_suite.py --dry-run` | 23 项，未启动进程 |
| Git 空白检查 | 两个仓库 `git diff --check` | 通过 |

补充观察：

- Examples 的 pytest 因仓库根目录缓存写权限产生 1 条 `PytestCacheWarning`，不影响 31 项测试结果。
- Examples 的 `ruff format --check` 仅报告共享文件 `scripts/run_record.py` 可按 200 字符行宽做大范围样式压缩。为避免对共享 MLU 运行时引入无功能收益的格式化 churn，未执行该样式改写；Ruff lint、单测和实际产物验证均通过。

## 11. 最终审计结果

审计目录：

```text
/data/Lightx2v-Platform-Run-Examples/logs/metax/infer/suites/
  metax_20260725T072300Z/audits/final_20260726T131914Z_2cea63c4/
```

| 审计项 | 结果 |
| --- | --- |
| suite 状态 | success |
| suite case 数 | 23 |
| 严格产物复验 | 23/23 passed |
| 结果文件与 `run.json` 大小/SHA 匹配 | 全部匹配 |
| 当前 config/script 漂移 | 0 |
| 校验恢复语义 | case 12、15、18、20 全部通过 |
| 历史 suite/run 文件改写 | 否 |
| 源码稳定性复核 | 通过 |
| tracked patch | 2 份 |
| untracked MetaX 源文件 | 65 个 |
| MLU 内容进入归档 | 否 |
| `__pycache__` 进入归档 | 否 |
| `SHA256SUMS` | 全部 OK |

关键审计文件：

- `audit_summary.json`：审计结论。
- `case_01_strict_revalidation.json` 至 `case_23_strict_revalidation.json`：逐项严格校验。
- `latest_success_file_manifest.json`：成功配置、脚本、记录和结果清单。
- `source_repositories.json`：两个仓库的分支、提交和 dirty 文件。
- `examples.tracked.binary.patch`、`lightx2v.tracked.binary.patch`：tracked 工作树补丁。
- `untracked_metax_sources.tar`：未跟踪 MetaX 源码。
- `source_restore_plan.json`：源码恢复步骤。
- `SHA256SUMS`：审计目录完整性校验。

## 12. 问题与修正记录

| 问题 | 影响 | 修正与验证 |
| --- | --- | --- |
| MetaX 多卡初始化前未明确按 local rank 绑卡 | communicator 或首个算子可能落到错误设备 | `metax_cuda.py` 按 `LOCAL_RANK` 调用 `set_device`；多卡实跑通过 |
| CFG2×SP4 多逻辑组触发 MCCL communicator 不稳定 | 集合通信失败或队列卡死 | 改为 WORLD 批量 P2P；micro-benchmark、8 进程 Gloo 和正式多卡用例通过 |
| `/dev/shm` 过小或残留 MCCL mmap 文件 | 首次写入可能 SIGBUS | 动态预检容量/余量/文件，隔离仅属于已退出本 suite 的残留；最终无残留 |
| LTX SP8 八份 Gemma/VAE 同时初始化 | `type:21` 队列争用 | rank-0-only 组件与 I/O；1-step canary 和正式 30 步均通过 |
| Qwen VAE decode 前显存连续空间不足 | Conv3d workspace 风险 | 全 rank 同步后释放 DiT/cache，再 decode；正式及补充回归均通过 |
| Wan 720p CFG 长序列 allocator 碎片 | 500 MiB 连续分配失败或 Conv3d 算法失败 | 首次 denoise 前及每 4 步清 cache；a6 完成 40 步并严格验证 |
| 系统 ffmpeg 缺 `libfreetype.so.6` | 已生成视频的包装器校验失败 | 使用 PyAV 全量解码 fallback；4 项按不可变 SHA 做校验恢复 |
| VS Code/终端关闭 | 前台作业可能被终止 | `nohup + setsid` 独立 controller、suite 状态和 resume；后台恢复流程实测通过 |
| worker/session 残留 | 后续用例占卡或 MCCL 污染 | PID/starttime 身份校验、递归后代清理、锁与 cooldown；最终资源回到基线 |

## 13. 已知限制与风险

1. **耗时不是统计 benchmark。** 每个 case 主要记录一次冷启动端到端耗时，可能受文件缓存、首次编译和主机 I/O 影响。
2. **正式 suite 代码来源混合。** 表中已明确区分早期 `main` 和最终 `ltx_tmp`；只额外补跑了 Z-Image SP2 与 Qwen CFG 两项当前代码回归，没有把早期 9 项描述成 `ltx_tmp` 全量重跑。
3. **4 项为校验恢复。** Case 12、15、18、20 的原推理产物有效，但不是当前分支重新推理结果。
4. **系统 ffmpeg 仍可能不可直接使用。** 当前验证器会自动使用 PyAV fallback；若外部流程强依赖 `/usr/bin/ffmpeg`，仍需补齐 `libfreetype.so.6`。
5. **C500 不支持当前 allocator 的 expandable segments。** 运行时会忽略该选项，720p 长序列依赖显式 cache 清理控制碎片。
6. **未覆盖跨节点。** 当前只验证单机 1/2/8 卡。
7. **没有高频显存采样报告。** `run.json` 保存启动时 MX-SMI 环境快照，但本报告不把采样值解释为精确峰值显存。
8. **工作树尚未提交。** 复现时应使用最终审计中的 patch、tar 和 restore plan，不能只依赖 Git HEAD。

## 14. 日志、结果与复现入口

### 14.1 原始状态

- 正式 suite：
  `/data/Lightx2v-Platform-Run-Examples/logs/metax/infer/suites/metax_20260725T072300Z/suite.json`
- 补充 suite：
  `/data/Lightx2v-Platform-Run-Examples/logs/metax/infer/suites/metax_current_verify_20260726T130100Z/suite.json`
- 最终审计：
  `/data/Lightx2v-Platform-Run-Examples/logs/metax/infer/suites/metax_20260725T072300Z/audits/final_20260726T131914Z_2cea63c4/`
- 使用说明：
  `/data/Lightx2v-Platform-Run-Examples/scripts/metax/README.md`

### 14.2 日志和结果布局

```text
logs/metax/infer/single|dist_2|dist_8/<case_id>/<run_id>/
├── run.log
└── run.json

results/metax/infer/single|dist_2|dist_8/<case_id>/<run_id>/
└── output.png|mp4
```

### 14.3 运行命令

```bash
cd /data/Lightx2v-Platform-Run-Examples

# 查看全部 23 项
python scripts/metax/run_infer_suite.py --list

# 只读预演
python scripts/metax/run_infer_suite.py --dry-run

# 脱离终端执行完整 suite
bash scripts/metax/run_all_detached.sh

# 恢复已有 suite
bash scripts/metax/resume_detached.sh <suite_id>
```

## 15. 最终判定

本轮 MetaX C500 平台适配满足单卡、多卡、日志/结果隔离、后台持久运行、失败恢复和严格产物校验要求。

正式矩阵 23/23 成功，当前代码补充回归 2/2 成功，最终产物审计 23/23 通过，未发现 config/script 哈希漂移、MLU 内容混入、残留 GPU 进程或 MCCL 共享内存文件。

**测试结论：通过（PASS）。**
