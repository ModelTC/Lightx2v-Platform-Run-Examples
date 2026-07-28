# Cambricon MLU590 离线推理

MLU 套件与 Ascend、MetaX 使用同一份结构化记录和最终报告契约。

- 正式入口默认使用 `PROFILING_DEBUG_LEVEL=2`。
- `run.log` 保存 Level1 单步和 Level2 全流程 profile。
- `run.json` 的 `metrics.dit_step_profile.steps` 保存逐步、逐 rank
  `infer_main` 数据；多卡作业级时间取每步最慢 rank，不跨 rank 求和。
- 只有同步、rank 完整且步数完整的数据才进入 DiT/step 和首步汇总。
- 离线推理每个脚本只测试一个样本，报告端到端耗时和逐步 DiT 耗时；
  P50/P90 仅用于独立的 10 样本 Server 测试报告。

完整套件：

```bash
bash scripts/mlu/run_all_detached.sh
```

套件结束时会自动生成：

```text
logs/mlu/infer/suites/final_report.md
logs/mlu/infer/suites/final_report.json
```

不运行推理、只重新聚合已有记录：

```bash
python scripts/aggregate_infer_reports.py --platform mlu
```

如需进行不计入正式报告的探索性无 profile 运行，可以显式设置
`INFER_PROFILE_LEVEL=0`。
