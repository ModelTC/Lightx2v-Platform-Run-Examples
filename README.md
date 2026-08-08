# LightX2V Platform Run Examples

本项目用于记录 [LightX2V](https://github.com/ModelTC/LightX2V) 在国产化计算平台上的运行示例，主要包括：

- 单卡和分布式推理脚本；
- 模型推理配置；
- 推理服务启动脚本；
- T2I、T2V 服务测速脚本及测试数据；
- 不同平台、模型和配置下的测速结果留档。

当前仓库收录 **昇腾 NPU（Ascend）**、**寒武纪 MLU590** 与 **MetaX C500** 的运行示例。

## 目录结构

```text
.
├── configs/                 # 各平台的模型推理配置
│   ├── ascend_npu/
│   │   ├── single/          # 单卡配置
│   │   ├── dist_2/          # 两卡配置
│   │   └── dist_8/          # 八卡配置
│   ├── mlu/
│   │   ├── single/
│   │   ├── dist_2/
│   │   └── dist_8/
│   └── metax/
│       ├── single/
│       ├── dist_2/
│       └── dist_8/
├── scripts/
│   ├── lib/
│   │   └── infer_runtime.sh     # 推理归档、通用预检、结果校验和进程管理
│   ├── preflight_infer.py       # 入口、配置、设备及公共输入预检
│   ├── run_record.py        # 生成并完成结构化 run.json
│   ├── logging.sh           # 旧脚本兼容日志入口
│   ├── mlu/
│   │   ├── infer/           # MLU 单卡、双卡和八卡离线推理
│   │   ├── run_infer_suite.py
│   │   ├── run_all_detached.sh
│   │   ├── resume_detached.sh
│   │   └── auto_resume_detached.sh
│   ├── metax/
│   │   ├── infer/           # C500 单卡、双卡和八卡离线推理
│   │   ├── run_infer_suite.py
│   │   ├── run_all_detached.sh
│   │   └── resume_detached.sh
│   └── ascend/
│       ├── infer/           # 离线推理脚本
│       │   ├── single/
│       │   ├── dist_2/
│       │   └── dist_8/
│       └── server/          # 推理服务启动脚本
│           ├── single/
│           └── dist/
├── tests/                   # 服务测速脚本
├── data/                    # 本机测速用 JSONL、图片和音频
├── logs/                    # 运行日志（不提交到 Git）
└── results/                 # 推理及测速结果（不提交到 Git）
```

目前仓库包含 Wan2.1、Wan2.1 Self-Forcing、Wan2.2 MoE、HunyuanVideo 1.5、LTX-2.3、FLUX.2-dev、Qwen-Image、LongCat-Image 和 Z-Image-Turbo 等模型示例，覆盖 T2I（文生图）、T2V（文生视频）和 S2V 等任务。

当前单样本单卡正式测速范围是下文列出的 11 个全重 BF16 用例。Wan2.1 1.3B 仅保留官方推荐的 480p；Wan2.2 MoE 和 HunyuanVideo 1.5 分别保留 480p、720p；图像模型只保留 16:9 原生分辨率档。

## MetaX C500 离线推理

MetaX 配置覆盖同一组 11 个单卡、1 个双卡和 11 个八卡用例，固定使用 `/data/LightX2V-metax` 与 `/data/models`。当前 64 GiB C500 的算子、offload、MXLink 拓扑、运行命令、分组日志/结果目录及断点恢复说明见 [`scripts/metax/README.md`](scripts/metax/README.md)。

## MLU590 离线推理

MLU 入口按当前测试机的固定资源编写：

| 资源 | 固定路径 |
| --- | --- |
| Examples 项目 | `/data/Lightx2v-Platform-Run-Examples` |
| LightX2V 主项目 | `/data/LightX2V-mlu` |
| 模型权重根目录 | `/data/models` |
| MLU 推理配置 | `/data/Lightx2v-Platform-Run-Examples/configs/mlu` |
| LTX-2.3 S2V 默认音频 | `/data/LightX2V-mlu/assets/inputs/audio/seko_input.mp3` |

配置覆盖与 Ascend 相同的 11 个单卡、1 个双卡和 11 个八卡用例，模型规格、分辨率、帧数、步数、seed 及并行策略与下文两个表一致。MLU 使用本地全重 BF16，不启用量化、蒸馏或缓存；Wan2.1 与 Wan2.2 使用 `mlu_sage_attn`，LTX-2.3 DiT 使用 `torch_sdpa` 且 Gemma 使用 `mlu_flash_attn`，其余模型使用 `mlu_flash_attn`。在当前 MLU 算子版本和 Wan2.2 的真实 480p/720p attention shape 下，Sage 微测比 Flash 分别快约 11.5% 和 11.9%，因此正式配置统一选择 Sage。FLUX.2-dev 的 Mistral3 单独注册 `mlu_flash_attn`，避免原生 MLU SDPA 产生 NaN；在 512-token、32Q/8KV 的实际 GQA 形状下，该实现比稳定的 eager 路径快约 17.8%。

入口在加载 LightX2V `base.sh` 后把 `PROFILING_DEBUG_LEVEL` 重置为 `0`，避免逐算子性能统计影响正式推理速度；完整 stdout/stderr、配置和运行元数据仍会归档。

MLU590 每卡为 80 GiB，offload 策略针对当前 `/data/models` 权重优化：

| 用例 | MLU590 策略 |
| --- | --- |
| Wan2.2 单卡、CFG2×SP4、TP8 | 全部关闭 CPU offload；80 GiB 可省去 Ascend 64 GiB 配置中的 model 搬运 |
| Qwen-Image、HunyuanVideo 单卡及 CFG2×SP4 | Qwen2.5-VL 与主模型均不 offload |
| LTX-2.3 单卡及 SP8 | 约 51.3 GiB Gemma/文本投影编码后回 CPU；约 35.4 GiB DiT 采用 model offload，只在 30 步去噪首尾各搬运一次；约 1.7 GiB VAE/audio 常驻。SP8 仅切序列、不切权重，全常驻至少约 88.4 GiB，不能装入 80 GiB |
| FLUX.2-dev 单卡 | Mistral3 编码后回 CPU，DiT 采用 model offload 并在 50 步去噪期间整模常驻，VAE 常驻；约 45 GiB 文本编码器与约 60 GiB DiT 不会同时占用 MLU |
| FLUX.2-dev TP8 | DiT、Mistral3、VAE 均不 offload |
| Wan2.1、Self-Forcing、LongCat、Z-Image | 无 offload |

运行单个用例：

```bash
cd /data/Lightx2v-Platform-Run-Examples
bash scripts/mlu/infer/single/run_wan21_1_3b_t2v_480p_81f.sh
bash scripts/mlu/infer/dist_8/run_wan21_1_3b_t2v_480p_81f_cfg2_sp4.sh
```

单独运行入口时，单卡默认使用 MLU 0，双卡默认使用 0、1，八卡默认使用 0–7；可以在启动前显式设置 `MLU_VISIBLE_DEVICES` 映射到其他物理卡。完整 mixed suite 为避免不同卡数组继承同一列表，会固定按 `0`、`0,1`、`0,1,2,3,4,5,6,7` 分配设备。

每个用例的日志和结果按卡数分开保存，失败和中断记录也会保留：

```text
logs/mlu/infer/single|dist_2|dist_8/<case_id>/<run_id>/
├── run.log
└── run.json

results/mlu/infer/single|dist_2|dist_8/<case_id>/<run_id>/
└── output.png|mp4
```

顺序执行完整 23 用例并脱离当前终端或 VSCode 会话：

```bash
cd /data/Lightx2v-Platform-Run-Examples
bash scripts/mlu/run_all_detached.sh
```

启动器会立即输出 suite ID、后台 PID、controller 日志和状态文件路径。套件日志位于 `logs/mlu/infer/suites/<suite_id>/controller.log`，可恢复状态位于同目录的 `suite.json`。套件默认在单个用例失败后继续，确保其余模型仍被执行；同样以脱离会话的方式恢复失败或未完成的用例：

```bash
bash scripts/mlu/resume_detached.sh <suite_id>
```

如果正在运行的套件修正了失败配置，可预先挂接一次自动恢复。它会等待当前 controller 退出，再以新代码仅重跑失败或产物无效的用例：

```bash
bash scripts/mlu/auto_resume_detached.sh <suite_id> [controller_pid]
```

等待日志保存为同一 suite 目录下的 `auto_resume.log`，恢复阶段的输出继续追加到 `controller.log`。自动恢复进程同样使用独立 session，不依赖 VS Code 或当前终端。

也可用 `--scope single`、`--scope multi` 或重复传入 `--only <case_id>` 运行子集。MLU 环境强制使用本地权重并设置 Hugging Face/Transformers 离线模式，因此断开编辑器或网络不影响已经启动的套件。

## Ascend 使用前准备

本仓库的昇腾推理入口按当前测试机编写，路径和设备号直接写在各入口顶部，不通过路径环境变量覆盖：

| 资源 | 本机固定路径 |
| --- | --- |
| Examples 项目 | `/data/wushuo1/Lightx2v-Platform-Run-Examples` |
| LightX2V 主项目 | `/data/wushuo1/LightX2V` |
| 模型权重根目录 | `/data/wushuo1/models` |
| 推理配置 | `/data/wushuo1/Lightx2v-Platform-Run-Examples/configs/ascend_npu` |
| 服务测速数据 | `/data/wushuo1/Lightx2v-Platform-Run-Examples/data` |
| LTX-2.3 S2V 默认音频 | `/data/wushuo1/LightX2V/assets/inputs/audio/seko_input.mp3` |

运行前确认：

1. `/data/wushuo1/LightX2V` 可以在当前昇腾环境正常推理。
2. 对应用例的权重已经位于 `/data/wushuo1/models/<模型目录>`。
3. 配置文件和本地测试数据存在。
4. 需要 `ffprobe` 或 PyAV 之一来完整校验视频编码、分辨率和帧数；优先使用 `ffprobe`，未安装时使用 PyAV 全量解码，两者均不可用会将产物校验记为失败。
5. 单卡入口固定使用 NPU 0，Z-Image-Turbo 两卡入口固定使用 NPU 0、1，八卡入口固定使用 NPU 0–7。

每个入口都采用与 LightX2V 主项目平台脚本一致的可读结构：先声明小写路径和用例参数，显式加载 `scripts/base/base.sh`，并在 `lightx2v_infer` 函数中完整写出 `python` 或 `torchrun` 命令。模型类别、任务、prompt、seed、配置和输出规格均以入口脚本为准。

`scripts/lib/infer_runtime.sh` 不生成或隐藏模型命令。它只负责为入口创建本次运行目录、保存 stdout/stderr、执行公共预检、管理子进程和信号、生成 `run.json`，并校验最终图片或视频。

## 单样本单卡推理

所有配置均使用当前全重 BF16，不启用量化、蒸馏或特征缓存。配置文件名使用 `<case_id>.json`，入口脚本使用 `run_<case_id>.sh`，其中 `case_id` 明确包含模型版本、任务、分辨率及视频帧数。

| 用例 | 目标规格 | 步数 | 首选 offload | 入口脚本 |
| --- | --- | ---: | --- | --- |
| Wan2.1 T2V 1.3B | 832×480，81 帧 | 50 | 无 | `run_wan21_1_3b_t2v_480p_81f.sh` |
| Wan2.2 MoE T2V A14B | 832×480，81 帧 | 40 | DiT model | `run_wan22_moe_a14b_t2v_480p_81f.sh` |
| Wan2.2 MoE T2V A14B | 1280×720，81 帧 | 40 | DiT model | `run_wan22_moe_a14b_t2v_720p_81f.sh` |
| Wan2.1 Self-Forcing 1.3B | 832×480，81 帧 | 4 | 无 | `run_wan21_1_3b_self_forcing_t2v_480p_81f.sh` |
| HunyuanVideo-1.5 T2V | 848×480，121 帧 | 50 | 仅 Qwen2.5-VL 编码器 | `run_hunyuan_video_15_t2v_480p_121f.sh` |
| HunyuanVideo-1.5 T2V | 1264×720，121 帧 | 50 | 仅 Qwen2.5-VL 编码器 | `run_hunyuan_video_15_t2v_720p_121f.sh` |
| LTX-2.3 S2V 22B dev | 768×512，241 帧 | 30 | DiT model + Gemma | `run_ltx2_3_22b_dev_s2v_768x512_241f.sh` |
| Qwen-Image-2512 T2I | 1664×928 | 50 | 仅 Qwen2.5-VL 编码器 | `run_qwen_image_2512_t2i_1664x928.sh` |
| LongCat-Image T2I | 1344×768 | 50 | 无 | `run_longcat_image_t2i_1344x768.sh` |
| Z-Image-Turbo T2I | 1664×928 | 9（实际 8 NFE） | 无 | `run_z_image_turbo_t2i_1664x928.sh` |
| FLUX.2-dev T2I | 1344×768 | 50 | block | `run_flux2_dev_t2i_1344x768.sh` |

在项目根目录执行对应脚本。例如，使用单张昇腾 NPU 运行 Wan2.1 T2V：

```bash
cd /data/wushuo1/Lightx2v-Platform-Run-Examples
bash scripts/ascend/infer/single/run_wan21_1_3b_t2v_480p_81f.sh
```

每次执行会生成不会互相覆盖的 `run_id`，日志和结果固定归档在本项目下：

```text
logs/ascend_npu/infer/<case_id>/<run_id>/
├── run.log       # 从预检开始的完整 stdout/stderr、命令和运行结果摘要
└── run.json      # 配置、版本、参数、状态、耗时指标及产物校验信息

results/ascend_npu/infer/<case_id>/<run_id>/
└── output.png|mp4
```

只有 LightX2V 子进程退出码为 0 且结果文件存在、非空并通过格式可读性校验时，`run.json` 才会标记为 `succeeded`。失败或中断的 `run.log`、`run.json` 和已产生的文件同样保留。`run.json` 通过结果路径、文件大小和 SHA-256 与产物对应。

`run.json` 会保留配置快照和 SHA-256、入口及主项目参考脚本、LightX2V Git commit、Python/关键依赖版本、NPU 信息、完整 prompt/seed，以及从 LightX2V `[Profile]` 日志提取的模型加载、文本编码、DiT、VAE、Pipeline、Total Cost。存在对应原始指标时，还会计算 DiT 单步耗时、视频生成帧吞吐或图片吞吐；缺失值保持 `null`，不会猜测。

## 单样本多卡推理

多卡入口只保留能让同一个样本有效使用全部进程的配置。八卡配置统一位于 `configs/ascend_npu/dist_8/`，两卡 Z-Image-Turbo 配置位于 `configs/ascend_npu/dist_2/`；不保留四卡目录。

| 模型 | 卡数与并行策略 | 目标规格 | 首选 offload | 配置与入口 |
| --- | --- | --- | --- | --- |
| Wan2.1 T2V 1.3B | 8，CFG2×SP4 | 832×480，81 帧 | 无 | `wan21_1_3b_t2v_480p_81f_cfg2_sp4.json` / `run_wan21_1_3b_t2v_480p_81f_cfg2_sp4.sh` |
| Wan2.2 MoE T2V A14B | 8，CFG2×SP4 | 832×480，81 帧 | DiT model | `wan22_moe_a14b_t2v_480p_81f_cfg2_sp4.json` / `run_wan22_moe_a14b_t2v_480p_81f_cfg2_sp4.sh` |
| Wan2.2 MoE T2V A14B | 8，TP8 | 832×480，81 帧 | 无 | `wan22_moe_a14b_t2v_480p_81f_tp8.json` / `run_wan22_moe_a14b_t2v_480p_81f_tp8.sh` |
| Wan2.2 MoE T2V A14B | 8，CFG2×SP4 | 1280×720，81 帧 | DiT model | `wan22_moe_a14b_t2v_720p_81f_cfg2_sp4.json` / `run_wan22_moe_a14b_t2v_720p_81f_cfg2_sp4.sh` |
| Wan2.2 MoE T2V A14B | 8，TP8 | 1280×720，81 帧 | 无 | `wan22_moe_a14b_t2v_720p_81f_tp8.json` / `run_wan22_moe_a14b_t2v_720p_81f_tp8.sh` |
| HunyuanVideo-1.5 T2V | 8，CFG2×SP4 | 848×480，121 帧 | Qwen2.5-VL | `hunyuan_video_15_t2v_480p_121f_cfg2_sp4.json` / `run_hunyuan_video_15_t2v_480p_121f_cfg2_sp4.sh` |
| HunyuanVideo-1.5 T2V | 8，CFG2×SP4 | 1264×720，121 帧 | Qwen2.5-VL | `hunyuan_video_15_t2v_720p_121f_cfg2_sp4.json` / `run_hunyuan_video_15_t2v_720p_121f_cfg2_sp4.sh` |
| LTX-2.3 S2V 22B dev | 8，SP8 | 768×512，241 帧 | DiT model + Gemma | `ltx2_3_22b_dev_s2v_768x512_241f_sp8.json` / `run_ltx2_3_22b_dev_s2v_768x512_241f_sp8.sh` |
| FLUX.2-dev T2I | 8，TP8 | 1344×768 | 无 | `flux2_dev_t2i_1344x768_tp8.json` / `run_flux2_dev_t2i_1344x768_tp8.sh` |
| Qwen-Image-2512 T2I | 8，CFG2×SP4 | 1664×928 | Qwen2.5-VL | `qwen_image_2512_t2i_1664x928_cfg2_sp4.json` / `run_qwen_image_2512_t2i_1664x928_cfg2_sp4.sh` |
| LongCat-Image T2I | 8，CFG2×SP4 | 1344×768 | 无 | `longcat_image_t2i_1344x768_cfg2_sp4.json` / `run_longcat_image_t2i_1344x768_cfg2_sp4.sh` |
| Z-Image-Turbo T2I | 2，SP2 | 1664×928 | 无 | `z_image_turbo_t2i_1664x928_sp2.json` / `run_z_image_turbo_t2i_1664x928_sp2.sh` |

Wan2.1 Self-Forcing 关闭了 CFG，并且 12 个 attention heads 不能被 8 整除，因此没有创建无效的八卡入口。Z-Image-Turbo 有 30 个 attention heads，同样不支持 SP8，按两卡 SP2 提供。

Wan2.2 的 TP8 与 CFG2×SP4 是两组独立对照：TP8 会切分权重，使用无 offload 速度配置；CFG2×SP4 不切分权重，因此在 64 GiB 显存上保留 model 级 CPU offload。

例如运行八卡 Wan2.1 480p：

```bash
cd /data/wushuo1/Lightx2v-Platform-Run-Examples
bash scripts/ascend/infer/dist_8/run_wan21_1_3b_t2v_480p_81f_cfg2_sp4.sh
```

八卡入口固定使用 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，Z-Image-Turbo 两卡入口固定使用 `0,1`。所有多卡入口均在脚本中显式写出 `torchrun` 命令，并沿用单卡相同的 `run.log`、`run.json` 和结果文件归档格式；`run.json` 额外记录 `benchmark.parallel_strategy`。多卡日志中的规范化 Profile 指标取各 rank 的最大耗时，表示决定整次推理速度的最慢 rank。

## 启动服务

使用单卡配置启动 FLUX.2-dev T2I 服务：

```bash
cd /data/wushuo1/Lightx2v-Platform-Run-Examples
bash scripts/ascend/server/single/start_server_flux2_dev_t2i_1344x768.sh
```

多卡服务入口与 `scripts/ascend/infer/dist_2`、`dist_8` 的测试配置一一对应。例如启动八卡 Wan2.1 T2V 服务：

```bash
cd /data/wushuo1/Lightx2v-Platform-Run-Examples
bash scripts/ascend/server/dist/start_server_wan21_1_3b_t2v_480p_81f_cfg2_sp4.sh
```

服务默认监听 `8000` 端口，Prometheus 指标端口为 `8001`。可通过 `PORT`、`METRIC_PORT` 和 `MASTER_PORT` 覆盖。

## 服务测速

测速数据为 JSONL 格式。T2I/T2V 每行包含 `prompt` 和 `seed`；S2V 额外包含 `audio_path`：

```json
{"prompt": "A cat running on the grass.", "seed": 42}
{"prompt": "A woman speaking.", "seed": 42, "audio_path": "audios/01.mp3"}
```

推荐先用 dry-run 校验当前 12 个分布式服务、配置、数据和命令：

```bash
cd /data/wushuo1/Lightx2v-Platform-Run-Examples
python scripts/run_service_suite.py --dry-run
```

执行完整服务测速：

```bash
python scripts/run_service_suite.py
```

默认对每个服务执行一次不计入统计的 warm-up，再以并发 1 测量 10 个样本。T2I 使用同步 PNG 接口；T2V 和 LTX2.3 S2V 使用异步任务接口。只执行指定 case：

```bash
python scripts/run_service_suite.py \
  --only wan21_1_3b_t2v_480p_81f_cfg2_sp4
```

日志写入 `logs/ascend_npu/server/<suite_id>/`，结果写入 `results/ascend_npu/server/<suite_id>/`，两边使用相同的 `suite_id`。每个 case 保留：

- 服务、warm-up 和正式客户端日志；
- 测试前后的 Prometheus 指标及 NPU 状态；
- p50、p90、平均/最小/最大端到端延迟、吞吐量和成功率；
- 每个请求的 task ID、耗时、错误、结果路径、文件大小和 SHA256；
- 生成的 PNG/MP4、Git 状态、配置和数据文件校验值。

任何正式样本失败都会使该 case 失败。统计中不包含 p95 和 p99。

也可在已经启动服务时单独运行通用 benchmark：

```bash
python bench_t2i_service.py --help
python bench_t2v_service.py --help
```

suite 只会终止自己创建的进程组。若测试前发现无法确认归属的 NPU 进程或端口占用，会停止测试而不会直接清理外部进程。

## 结果记录建议

为了便于横向对比，每次测速建议至少记录以下信息：

- 芯片型号、设备数量及驱动/CANN 版本；
- LightX2V 版本或 Git commit；
- 模型、任务类型和配置文件；
- 输入分辨率、生成长度、推理步数及精度；
- 并发数、请求数、延迟、吞吐量和峰值显存。

新增平台时，建议沿用 `configs/<platform>/{single,dist_<卡数>}` 和 `scripts/<platform>/infer/{single,dist_<卡数>}` 的目录约定。
