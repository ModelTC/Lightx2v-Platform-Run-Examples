# MetaX C500 Server 测试脚本

> 当前状态：脚本已补齐并完成无 GPU 静态验证；尚未启动 Server、smoke
> 或正式测试。

## 固定测试口径

- 严格对齐 Ascend 的 12 个多卡 Server case。
- 正式测试每项 1 次预热（不计入统计）+ 10 个正式样本，并发 1。
- 正式测试固定 `PROFILING_DEBUG_LEVEL=0`，避免 profiling 干扰客户端
  端到端 P50/P90。
- Smoke 固定为 1 次预热 + 1 个正式请求，只用于诊断，永不进入正式报告。
- 正式报告保存 E2E P50/P90/Avg/Min/Max、吞吐、启动、预热、峰值显存、
  120 个正式产物及其 SHA256/媒体校验；不展示单步 DiT。

## 与 MLU 任务隔离

MetaX 所有入口、控制器、runtime 和报告器均位于本目录。两个较大的
`_service_*_core.py` 是审核时冻结的私有核心，避免 MLU 对公共 Server
文件的并行修改改变 MetaX 行为。MetaX 脚本不会修改或写入：

- `scripts/mlu/`
- `configs/mlu/`
- `logs/mlu/`
- `results/mlu/`
- 公共 `scripts/run_service_suite.py`
- 公共 `scripts/aggregate_service_reports.py`
- 公共 `scripts/lib/server_runtime.sh`

客户端请求与产物落盘只读复用稳定的 `bench_t2i_service.py`、
`bench_t2v_service.py` 和 `scripts/service_benchmark_common.py`，并在控制器
启动及每次客户端启动前核对固定 SHA256；如果这些共享客户端文件被另一任务
改动，MetaX 控制器会 fail closed，要求重新审核，而不会静默采用新行为。

MetaX Server 与 MetaX 离线套件共同持有
`logs/metax/infer/suites/metax_gpu.lock`，因此两类任务不能同时占用 C500。
进程清理只针对当前 `RUN_ID` 及 PID/starttime 验证过的后代，不执行全局
`pkill`。每个分布式 case 启动前执行 `/dev/shm` 预检，退出后只隔离本
case 产生、且归属于已退出进程树的新 `mccl-*` 普通文件。

## 将来执行（本轮没有执行）

单项 smoke（仅 `z_image_turbo_t2i_1664x928_sp2`）：

```bash
cd /data/Lightx2v-Platform-Run-Examples
bash scripts/metax/server/run_smoke_detached.sh
```

完整 12 项正式测试：

```bash
cd /data/Lightx2v-Platform-Run-Examples
bash scripts/metax/server/run_all_detached.sh
```

指定 Suite ID：

```bash
SUITE_ID=metax_server_formal_001 \
  bash scripts/metax/server/run_all_detached.sh
```

查看状态：

```bash
bash scripts/metax/server/status.sh metax_server_formal_001
```

断点恢复：

```bash
bash scripts/metax/server/resume_detached.sh metax_server_formal_001
```

聚合正式报告时必须显式给出正式 suite；诊断 suite 只能用
`--diagnostic-suite` 作为追溯信息，不能提供正式指标：

```bash
bash scripts/metax/server/generate_report.sh \
  --source-suite metax_server_formal_001 \
  --diagnostic-suite metax_server_smoke_001
```

默认输出：

```text
logs/metax/server/<suite_id>/
results/metax/server/<suite_id>/
results/metax/server/final_service_benchmark_<date>/final_report.md
results/metax/server/final_service_benchmark_<date>/final_report.json
```

## 12 个正式 case

1. `z_image_turbo_t2i_1664x928_sp2`
2. `flux2_dev_t2i_1344x768_tp8`
3. `longcat_image_t2i_1344x768_cfg2_sp4`
4. `qwen_image_2512_t2i_1664x928_cfg2_sp4`
5. `wan21_1_3b_t2v_480p_81f_cfg2_sp4`
6. `wan22_moe_a14b_t2v_480p_81f_cfg2_sp4`
7. `wan22_moe_a14b_t2v_480p_81f_tp8`
8. `wan22_moe_a14b_t2v_720p_81f_cfg2_sp4`
9. `wan22_moe_a14b_t2v_720p_81f_tp8`
10. `hunyuan_video_15_t2v_480p_121f_cfg2_sp4`
11. `hunyuan_video_15_t2v_720p_121f_cfg2_sp4`
12. `ltx2_3_22b_dev_s2v_768x512_241f_sp8`
