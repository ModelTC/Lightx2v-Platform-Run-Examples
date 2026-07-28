# Cambricon MLU590 Server 测试

> 当前状态：脚本已写好，但尚未启动过服务，也尚未进行 MLU smoke 或正式测试。

这套入口复用 `scripts/run_service_suite.py`，对应 Ascend 的 12 个分布式
Server case。所有 detached launcher 都使用 `nohup` + `setsid`，因此关闭
VSCode 或终端不会终止控制器。PID 文件由控制器取得 suite lock 后原子写入，
launcher 不会提前覆盖它。launcher 不会按 PID 主动发信号；控制器只清理带有
本次精确 `RUN_ID` 且进程身份未变化的自有服务进程。

## 正式测试：1 次预热 + 10 次测量

从仓库根目录启动完整 12 项测试：

```bash
cd /data/Lightx2v-Platform-Run-Examples
bash scripts/mlu/server/run_all_detached.sh
```

launcher 实际使用的固定正式口径是：

```bash
python scripts/run_service_suite.py \
  --platform mlu \
  --suite-id ID \
  --warmup-count 1 \
  --sample-count 10 \
  --concurrency 1
```

正式 launcher 同时强制 `LIGHTX2V_SERVICE_SUITE_KIND=formal` 和
`INFER_PROFILE_LEVEL=0`，不会继承调用 shell 中残留的 diagnostic/profile
环境。

可以用 `SUITE_ID` 指定可重复引用的 ID：

```bash
SUITE_ID=mlu_server_formal_001 \
  bash scripts/mlu/server/run_all_detached.sh
```

查看状态和日志尾部：

```bash
bash scripts/mlu/server/status.sh mlu_server_formal_001
```

控制器已经退出且 suite 未全部通过时，可断点恢复。恢复逻辑复核已有成功
case 的 warmup 和测量产物，只重新执行缺失、失败或校验失效的 case。恢复还会
严格比对 manifest 中的仓库 revision/diff、脚本、配置、数据和
工具 SHA；任何输入发生变化都会拒绝混合版本恢复，此时必须创建新的 suite：

```bash
bash scripts/mlu/server/resume_detached.sh mlu_server_formal_001
```

恢复命令的核心调用是：

```bash
python scripts/run_service_suite.py \
  --platform mlu \
  --resume mlu_server_formal_001
```

恢复 launcher 会清除调用者的 suite kind/profile 环境，再从原 summary 恢复
这两个值。

## Smoke：仅诊断，不进入正式报告

在正式测试之前可以启动固定的四项 smoke：

- Z-Image Turbo，2 卡 SP2
- Wan2.1 1.3B，8 卡 CFG2+SP4
- FLUX.2-dev，8 卡 TP8
- LTX-2.3，8 卡 SP8

```bash
cd /data/Lightx2v-Platform-Run-Examples
bash scripts/mlu/server/run_smoke_detached.sh
```

Smoke 固定为 Level 0、每项 1 次预热、1 次测量、并发 1。日志目录会写入
`DIAGNOSTIC_ONLY` 标记；smoke suite 只能作为诊断资料，不能作为正式
benchmark source，也不应以正式 suite 参数交给报告聚合器。

## Profiling 级别

MLU Server 正式测试固定使用 Level 0，关闭 profiling 同步和日志开销：

```bash
INFER_PROFILE_LEVEL=0 \
  SUITE_ID=mlu_server_formal_level0_001 \
  bash scripts/mlu/server/run_all_detached.sh
```

正式 launcher 不接受 Level 1/2 覆盖。Level 0 不生成逐步 DiT 日志，Server
报告以客户端端到端延迟、吞吐、产物和显存采样为准；推理套件原有的单步 DiT
测试口径不受这里影响。

LTX-2.3 的 241 帧配置使用 block 级 DiT offload，并保留 Gemma/VAE CPU
offload、启用 VAE tiling，同时关闭 double-precision RoPE。Smoke 实测整模型
offload 在卡 0 的 DiT 搬入阶段达到约 80.6–80.7GB；切到 block offload 后
30 步 DiT 可以完成，但非 tiling VAE 解码仍需额外 1.42GB 并 OOM。因此这组
offload/tiling 配置是保持同权重、分辨率和 241 帧条件下所需的显存回退。
MLU LTX runner 允许该配置单独指定 VAE tile；这里使用 512px 空间 tile 和
64 帧时间 tile（24 帧重叠）。此外在最后一个 DiT step 后把两个 block
预取缓冲移回 CPU，VAE 解码结束后的下一请求首步再恢复，避免无用的 DiT
缓冲与 VAE 峰值叠加。

## 保存目录

每个 suite 使用独立目录：

```text
logs/mlu/server/<suite_id>/
├── controller.log
├── controller.pid
├── suite.log
└── <case_id>/...

results/mlu/server/<suite_id>/
├── manifest.json
├── summary.json
├── summary.md
└── <case_id>/...
```

Smoke 日志目录还会包含：

```text
logs/mlu/server/<smoke_suite_id>/DIAGNOSTIC_ONLY
```

最终聚合报告应写入单独目录，例如：

```text
results/mlu/server/final_service_benchmark_<date>/final_report.md
results/mlu/server/final_service_benchmark_<date>/final_report.json
```

正式 suite 完成后，使用下面的命令生成报告（聚合器只接受 Level 0 下完整的
12 个正式 case 和 120 个测量结果）：

```bash
cd /data/Lightx2v-Platform-Run-Examples
python scripts/aggregate_service_reports.py \
  --platform mlu \
  --source-suite mlu_server_formal_001 \
  --output-dir results/mlu/server/final_service_benchmark_001
```

报告会校验产物大小和 SHA，并用 Pillow/ffprobe 检查 PNG/MP4 的分辨率、
帧数及 LTX 音频。条件不完整时报告仍会保存，但命令返回非零且不会把正式
报告标记为通过。

启动正式测试前，应先确认 8 张 MLU 均可独占、HTTP/metrics/master 端口未被
占用，并先运行四项 smoke。控制器会以 fail-closed 方式解析完整的 0–7 卡
CNMON 内存与进程表；缺卡、未知进程行或无明确空进程标记都会拒绝运行。默认
空闲/释放绝对上限均为每卡 256 MiB。
