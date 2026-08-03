# Ascend、MLU、MetaX Server 性能对比

- 整理日期：2026-07-29
- 测试范围：12 个多卡 Server 配置
- 测试负载：1 次预热（不计入统计）+ 10 个正式请求，并发 1
- 核心指标：客户端端到端延迟；P50 越低越好

## 结果选择规则

1. 只纳入 `formal` 正式批次及正式 retry。
2. smoke 和 diagnostic 通常只有 1 个样本，不参与性能选择。
3. 同一平台、同一配置只有在正式请求 10/10 成功时才具备性能比较资格。
4. 若存在多个合格批次，选择 P50 最低的一轮；失败配置保留为“失败”，不使用部分完成请求计算性能。
5. 采用修订配置后的 retry 时，必须明确标记配置变化，不能将其写成原 formal 配置原样重跑。
6. CPU placement/offload 以最终采用批次的实际模型配置为准；失败的 LTX 使用最后一个正式失败轮次。签名覆盖主 DiT 的 model/block offload、文本编码器、VAE、Gemma staged 和 offload buffer 释放。
7. 同一 workload 的有效 CPU placement/offload 签名不同时拆行；只在同一签名行内、且至少两个平台有合格数值时评选最低 P50。

按此规则整理后的覆盖情况：

- Ascend：12/12 通过。
- MLU：11/12 通过；LTX-2.3 SP8 失败。
- MetaX：原 formal 为 10/12；纳入修订配置的正式 retry 后为 11/12，LTX-2.3 SP8 仍失败。

本批数据中，每个最终通过项都只有一轮满足上述资格的成功结果，因此“最低 P50”规则没有在多个成功正式轮次之间发生二次取舍。

## 按 CPU placement/offload 分组的 P50 对比

表内粗体为同一 CPU placement/offload 签名下的最低合格 P50。`—` 表示该平台没有运行此精确签名，不能作同配置比较；“失败”表示该平台确实以此签名发起了正式测试，但没有合格结果。签名只列启用项，未列出的相关组件均驻留设备端；`TE:QwenVL` 指 Qwen2.5-VL 文本编码器。

| 配置 | 任务 | 卡数/并行 | CPU placement / offload | Ascend P50 (s) | MLU P50 (s) | MetaX P50 (s) | 同签名最低 |
|---|---:|---|---|---:|---:|---:|---|
| `z_image_turbo_t2i_1664x928_sp2` | T2I | 2/SP2 | 无 | 6.0180 | **4.5325** | 6.5198 | MLU |
| `flux2_dev_t2i_1344x768_tp8` | T2I | 8/TP8 | 无 | **13.5237** | 14.5637 | — | Ascend |
| `flux2_dev_t2i_1344x768_tp8` | T2I | 8/TP8 | TE:Mistral3 | — | — | 56.5486 | — |
| `longcat_image_t2i_1344x768_cfg2_sp4` | T2I | 8/CFG2×SP4 | 无 | **6.0200** | 6.5733 | 9.2702 | Ascend |
| `qwen_image_2512_t2i_1664x928_cfg2_sp4` | T2I | 8/CFG2×SP4 | TE:QwenVL | 20.3004 | — | — | — |
| `qwen_image_2512_t2i_1664x928_cfg2_sp4` | T2I | 8/CFG2×SP4 | 无 | — | 16.3107 | — | — |
| `qwen_image_2512_t2i_1664x928_cfg2_sp4` | T2I | 8/CFG2×SP4 | DiT:model；TE:QwenVL | — | — | 66.0582 | — |
| `wan21_1_3b_t2v_480p_81f_cfg2_sp4` | T2V | 8/CFG2×SP4 | 无 | **26.0947** | 35.2265 | 45.1631 | Ascend |
| `wan22_moe_a14b_t2v_480p_81f_cfg2_sp4` | T2V | 8/CFG2×SP4 | DiT:model | 161.5873 | — | — | — |
| `wan22_moe_a14b_t2v_480p_81f_cfg2_sp4` | T2V | 8/CFG2×SP4 | 无 | — | 135.8295 | — | — |
| `wan22_moe_a14b_t2v_480p_81f_cfg2_sp4` | T2V | 8/CFG2×SP4 | DiT:block；TE:T5 | — | — | 169.8818† | — |
| `wan22_moe_a14b_t2v_480p_81f_tp8` | T2V | 8/TP8 | 无 | 182.6718 | **177.1256** | 289.1855 | MLU |
| `wan22_moe_a14b_t2v_720p_81f_cfg2_sp4` | T2V | 8/CFG2×SP4 | DiT:model | 484.1679 | — | — | — |
| `wan22_moe_a14b_t2v_720p_81f_cfg2_sp4` | T2V | 8/CFG2×SP4 | 无 | — | 523.5647 | — | — |
| `wan22_moe_a14b_t2v_720p_81f_cfg2_sp4` | T2V | 8/CFG2×SP4 | DiT:block；TE:T5 | — | — | 557.9611 | — |
| `wan22_moe_a14b_t2v_720p_81f_tp8` | T2V | 8/TP8 | 无 | 618.6909 | **617.9540** | 845.3132 | MLU |
| `hunyuan_video_15_t2v_480p_121f_cfg2_sp4` | T2V | 8/CFG2×SP4 | TE:QwenVL | **141.2936** | — | 223.3367 | Ascend |
| `hunyuan_video_15_t2v_480p_121f_cfg2_sp4` | T2V | 8/CFG2×SP4 | 无 | — | 186.2176 | — | — |
| `hunyuan_video_15_t2v_720p_121f_cfg2_sp4` | T2V | 8/CFG2×SP4 | TE:QwenVL | **582.6520** | — | 740.5749 | Ascend |
| `hunyuan_video_15_t2v_720p_121f_cfg2_sp4` | T2V | 8/CFG2×SP4 | 无 | — | 745.5035 | — | — |
| `ltx2_3_22b_dev_s2v_768x512_241f_sp8` | S2V | 8/SP8 | DiT:model；TE:Gemma(staged)；VAE | 143.7747§ | — | — | — |
| `ltx2_3_22b_dev_s2v_768x512_241f_sp8` | S2V | 8/SP8 | DiT:block；TE:Gemma；VAE；释放buffer | — | 失败 | — | — |
| `ltx2_3_22b_dev_s2v_768x512_241f_sp8` | S2V | 8/SP8 | DiT:block；TE:Gemma；VAE | — | — | 失败 | — |

† MetaX Wan2.2 480p CFG2×SP4 使用修订配置的正式 `retry_block` 结果，详见“配置变化”。

§ Ascend LTX-2.3 使用修订配置和修订实现的 `ltx_final2` 结果，详见“配置变化”。

## 最佳正式结果明细

“吞吐”统一写为样本/分钟，以兼容 T2I、T2V 和 S2V。峰值设备内存是各卡采样最大值，不同平台采样工具的精度可能不同。

| 平台 | 配置 | CPU placement / offload | 状态 | 正式请求 | P50 (s) | P90 (s) | Avg (s) | 吞吐（样本/分） | 峰值设备内存 (MB) | 来源批次 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Ascend | `z_image_turbo_t2i_1664x928_sp2` | 无 | 通过 | 10/10 | 6.0180 | 6.0308 | 6.0255 | 9.943488 | 31904 | `service_suite_20260726T082203Z_p562104` |
| Ascend | `flux2_dev_t2i_1344x768_tp8` | 无 | 通过 | 10/10 | 13.5237 | 13.6049 | 13.5817 | 4.415329 | 65306 | `service_suite_20260726T0830Z_retry11` |
| Ascend | `longcat_image_t2i_1344x768_cfg2_sp4` | 无 | 通过 | 10/10 | 6.0200 | 6.2089 | 6.1378 | 9.764363 | 38330 | `service_suite_20260726T0830Z_retry11` |
| Ascend | `qwen_image_2512_t2i_1664x928_cfg2_sp4` | TE:QwenVL | 通过 | 10/10 | 20.3004 | 21.4030 | 20.6664 | 2.901816 | 62847 | `service_suite_20260726T0830Z_retry11` |
| Ascend | `wan21_1_3b_t2v_480p_81f_cfg2_sp4` | 无 | 通过 | 10/10 | 26.0947 | 26.1480 | 26.1171 | 2.296909 | 21981 | `service_suite_20260726T0830Z_retry11` |
| Ascend | `wan22_moe_a14b_t2v_480p_81f_cfg2_sp4` | DiT:model | 通过 | 10/10 | 161.5873 | 162.6012 | 161.5382 | 0.371399 | 49429 | `service_suite_20260726T0830Z_retry11` |
| Ascend | `wan22_moe_a14b_t2v_480p_81f_tp8` | 无 | 通过 | 10/10 | 182.6718 | 183.1742 | 182.6726 | 0.328447 | 31333 | `service_suite_20260726T0830Z_retry11` |
| Ascend | `wan22_moe_a14b_t2v_720p_81f_cfg2_sp4` | DiT:model | 通过 | 10/10 | 484.1679 | 485.8885 | 483.9249 | 0.123980 | 52287 | `service_suite_20260726T0830Z_retry11` |
| Ascend | `wan22_moe_a14b_t2v_720p_81f_tp8` | 无 | 通过 | 10/10 | 618.6909 | 618.8332 | 618.5107 | 0.097005 | 40881 | `service_suite_20260726T0830Z_retry11` |
| Ascend | `hunyuan_video_15_t2v_480p_121f_cfg2_sp4` | TE:QwenVL | 通过 | 10/10 | 141.2936 | 141.6302 | 141.0233 | 0.425443 | 56263 | `service_suite_20260726T0830Z_retry11` |
| Ascend | `hunyuan_video_15_t2v_720p_121f_cfg2_sp4` | TE:QwenVL | 通过 | 10/10 | 582.6520 | 583.2634 | 582.3536 | 0.103029 | 54192 | `service_suite_20260726T0830Z_retry11` |
| Ascend | `ltx2_3_22b_dev_s2v_768x512_241f_sp8` | DiT:model；TE:Gemma(staged)；VAE | 通过§ | 10/10 | 143.7747 | 145.4376 | 143.8445 | 0.417074 | 43776 | `service_suite_20260726T_ltx_final2` |
| MLU | `z_image_turbo_t2i_1664x928_sp2` | 无 | 通过 | 10/10 | 4.5325 | 4.5380 | 4.3840 | 13.644147 | 26166 | `mlu_server_formal_level0_001` |
| MLU | `flux2_dev_t2i_1344x768_tp8` | 无 | 通过 | 10/10 | 14.5637 | 14.5876 | 14.5706 | 4.114571 | 58518 | `mlu_server_formal_level0_001` |
| MLU | `longcat_image_t2i_1344x768_cfg2_sp4` | 无 | 通过 | 10/10 | 6.5733 | 6.5945 | 6.5790 | 9.105061 | 31510 | `mlu_server_formal_level0_001` |
| MLU | `qwen_image_2512_t2i_1664x928_cfg2_sp4` | 无 | 通过 | 10/10 | 16.3107 | 16.5665 | 16.3121 | 3.674786 | 56822 | `mlu_server_formal_level0_001` |
| MLU | `wan21_1_3b_t2v_480p_81f_cfg2_sp4` | 无 | 通过 | 10/10 | 35.2265 | 35.2410 | 35.0797 | 1.709910 | 15702 | `mlu_server_formal_level0_001` |
| MLU | `wan22_moe_a14b_t2v_480p_81f_cfg2_sp4` | 无 | 通过 | 10/10 | 135.8295 | 135.8750 | 135.6680 | 0.442223 | 69430 | `mlu_server_formal_level0_001` |
| MLU | `wan22_moe_a14b_t2v_480p_81f_tp8` | 无 | 通过 | 10/10 | 177.1256 | 177.2376 | 177.1902 | 0.338602 | 24246 | `mlu_server_formal_level0_001` |
| MLU | `wan22_moe_a14b_t2v_720p_81f_cfg2_sp4` | 无 | 通过 | 10/10 | 523.5647 | 524.1007 | 523.4676 | 0.114618 | 71798 | `mlu_server_formal_level0_001` |
| MLU | `wan22_moe_a14b_t2v_720p_81f_tp8` | 无 | 通过 | 10/10 | 617.9540 | 618.2253 | 617.8096 | 0.097115 | 31478 | `mlu_server_formal_level0_001` |
| MLU | `hunyuan_video_15_t2v_480p_121f_cfg2_sp4` | 无 | 通过 | 10/10 | 186.2176 | 186.5447 | 186.2032 | 0.322211 | 55670 | `mlu_server_formal_level0_001` |
| MLU | `hunyuan_video_15_t2v_720p_121f_cfg2_sp4` | 无 | 通过 | 10/10 | 745.5035 | 746.2429 | 745.4436 | 0.080487 | 57462 | `mlu_server_formal_level0_001` |
| MLU | `ltx2_3_22b_dev_s2v_768x512_241f_sp8` | DiT:block；TE:Gemma；VAE；释放buffer | **失败** | 0/0 | — | — | — | — | — | `mlu_server_formal_level0_001` |
| MetaX | `z_image_turbo_t2i_1664x928_sp2` | 无 | 通过 | 10/10 | 6.5198 | 6.6214 | 6.6196 | 9.051388 | 37596 | `metax_server_20260727T092540Z` |
| MetaX | `flux2_dev_t2i_1344x768_tp8` | TE:Mistral3 | 通过 | 10/10 | 56.5486 | 60.5483 | 57.6126 | 1.041307 | 64365 | `metax_server_20260727T092540Z` |
| MetaX | `longcat_image_t2i_1344x768_cfg2_sp4` | 无 | 通过 | 10/10 | 9.2702 | 9.6722 | 9.4203 | 6.364102 | 48360 | `metax_server_20260727T092540Z` |
| MetaX | `qwen_image_2512_t2i_1664x928_cfg2_sp4` | DiT:model；TE:QwenVL | 通过 | 10/10 | 66.0582 | 70.6893 | 67.0013 | 0.895368 | 65519 | `metax_server_20260727T092540Z` |
| MetaX | `wan21_1_3b_t2v_480p_81f_cfg2_sp4` | 无 | 通过 | 10/10 | 45.1631 | 45.1760 | 45.0660 | 1.331179 | 32616 | `metax_server_20260727T092540Z` |
| MetaX | `wan22_moe_a14b_t2v_480p_81f_cfg2_sp4` | DiT:block；TE:T5 | 通过† | 10/10 | 169.8818 | 170.6480 | 169.8411 | 0.353259 | 17410 | `metax_server_retry_block_20260727T210300Z` |
| MetaX | `wan22_moe_a14b_t2v_480p_81f_tp8` | 无 | 通过 | 10/10 | 289.1855 | 289.2075 | 289.1269 | 0.207517 | 34216 | `metax_server_20260727T092540Z` |
| MetaX | `wan22_moe_a14b_t2v_720p_81f_cfg2_sp4` | DiT:block；TE:T5 | 通过 | 10/10 | 557.9611 | 559.6668 | 558.5460 | 0.107420 | 21506 | `metax_server_20260727T092540Z` |
| MetaX | `wan22_moe_a14b_t2v_720p_81f_tp8` | 无 | 通过 | 10/10 | 845.3132 | 845.8766 | 845.5047 | 0.070963 | 42536 | `metax_server_20260727T092540Z` |
| MetaX | `hunyuan_video_15_t2v_480p_121f_cfg2_sp4` | TE:QwenVL | 通过 | 10/10 | 223.3367 | 228.6338 | 224.8647 | 0.266821 | 65449 | `metax_server_20260727T092540Z` |
| MetaX | `hunyuan_video_15_t2v_720p_121f_cfg2_sp4` | TE:QwenVL | 通过 | 10/10 | 740.5749 | 743.8168 | 741.6220 | 0.080903 | 65536‡ | `metax_server_20260727T092540Z` |
| MetaX | `ltx2_3_22b_dev_s2v_768x512_241f_sp8` | DiT:block；TE:Gemma；VAE | **失败** | 0/0 | — | — | — | — | — | `metax_server_retry_block_20260727T210300Z`（最后正式失败） |

‡ MetaX HunyuanVideo 720p 的 case 记录含 `device sampler errors: 1`。10/10 请求延迟有效，但峰值内存采样应谨慎解读。

## 主要结论

- 12 个 workload 中有 7 个存在跨平台 CPU placement/offload 差异；拆分后 P50 主表为 23 个签名行。
- 5 个 workload 的三平台签名完全相同；FLUX.2 TP8 和两个 HunyuanVideo 配置另有两平台同签名子组，共形成 8 个至少两平台可比的签名组。
- 只统计这些同签名组：Ascend 有 5 组最低，MLU 有 3 组最低，MetaX 没有最低组；单平台签名和失败签名不计胜负。
- Ascend 是唯一完成全部 12 个正式 Server 配置的平台。
- Qwen-Image、两个 Wan2.2 CFG2×SP4 和 LTX-2.3 的三平台签名各不相同，没有严格的跨平台同签名 P50 对照。
- 在同为无 offload 的 FLUX.2 TP8 子组中，Ascend 的 13.5237 秒低于 MLU 的 14.5637 秒；MetaX 使用 `TE:Mistral3`，单独列为 56.5486 秒。
- 两个 HunyuanVideo 配置中，Ascend 与 MetaX 同为 `TE:QwenVL`，Ascend 均较低；MLU 的无 offload 结果单独列出。
- Wan2.2 720p TP8 的 MLU 与 Ascend 接近：617.9540 秒和 618.6909 秒，相差约 0.12%。

## 配置变化、失败与修复

### Ascend

- 最终 12 项由三个正式来源批次合并：Z-Image 来自首批，10 项来自 `retry11`，LTX-2.3 来自 `ltx_final2`。
- 首批其余 11 项受端口占用影响；精确追踪和清理 suite 自有进程后恢复。
- LTX-2.3 是修订配置和修订实现后的结果：模型配置 SHA-256 从 `17d2c9acdc16f417bd9e78cb3c690b3c4a8e17f7de2714720b644881887bbf7d` 变为 `9e3f7a99afa1e18b00a32aad213e113160f546c6502e7a6a754a811804fc754c`；新增 Gemma text-only、Gemma 分阶段 CPU offload，并将 VAE CPU offload 和 tiled VAE 从关闭改为开启。
- LTX-2.3 的 LightX2V revision 从 `9fe703cdbed4dbb18d4dc2a276fd7e3552af7998` 变为 `c70ddad2c0780cc5b495b6a20fd8d87e36ec97d2`，且带有工作区补丁。实现变化包括移除不用的 vision tower/multimodal projector、绕过不需要的 vocabulary logits、分阶段执行 Gemma、让 tiled VAE 的整个 generator 处于 `torch.inference_mode()`，以及为 NPU allocator 启用 expandable segments。
- 最终 LTX-2.3 仍保持全量权重、241 帧和 30 个扩散步并完成 10/10，但不能解释为初始配置原样重跑。

### MLU

- 正式批次中 11 项均为 10/10。
- LTX-2.3 SP8 的正式配置为 `DiT:block；TE:Gemma；VAE；释放buffer`，在 warm-up 的 VAE/Conv3d 路径发生设备内存不足，未进入正式请求，因此不提供延迟或吞吐数据。
- 后续 LTX diagnostic 均未形成合格的 10 请求正式结果，未纳入主表。
- `logs/mlu/server/suites/final_report.md` 生成于 formal 批次运行之前，是过期的中间状态；本报告使用更新的 `results/mlu/server/mlu_server_formal_level0_001/summary.json`。

### MetaX

- 原 formal 批次为 10/12；Wan2.2 480p CFG2×SP4 和 LTX-2.3 SP8 未通过。
- Wan2.2 480p CFG2×SP4 的三轮模型配置 SHA-256 分别为：
  - 原 formal：`37ea577be05b1d0c5601307cd1410a1b7fc4eddfc59b6f1b600b418c9677fe0a`
  - 首次正式 retry：`c04006052850f17f4a412d143694b80df163aff0081bc049e513208b1ccf6b7c`
  - 通过的 `retry_block`：`7d555a866b74690882bd1fddeb827c32a2dbe13119265e9fe200c6530aeb96dd`
- 原 formal 使用 model offload；首次 retry 新增 `empty_cache_step_interval=1`；通过的 `retry_block` 进一步改为 block offload 并保留该清缓存设置。
- benchmark、服务入口、离线入口和输入数据的哈希在这三轮保持一致，但模型配置发生变化。因此主表将最后一轮作为“修订配置后的最佳正式结果”，没有把它描述为原配置重跑。
- LTX-2.3 在主批次和正式 retry 中均未通过。表格使用最后一个正式 `retry_block` 失败轮次的 `DiT:block；TE:Gemma；VAE` 签名；其模型配置 SHA-256 为 `168a5619c8d6f1e92f7a804078521c32b8c73e76fdf0672c502fa9fc6d6f9d58`。
- 后续 diagnostic 又加入过 CPU compute 等设置，但仍出现显存不足、连接或通信异常以及任务超时；这些非正式配置未用于主表。

## 使用限制

- 三个平台的硬件容量、驱动和框架、LightX2V 工作树、算子实现及 offload 策略并不完全相同；MLU 为 8×80 GiB，Ascend 和 MetaX 为 8×64 GiB。表格反映的是完整软硬件栈及当前有效配置，不是只改变芯片型号的严格 A/B 实验。
- P50/P90 仅由 10 个成功正式请求计算；没有 P95 或 P99 数据。
- 吞吐由各 benchmark 的整体测量窗口计算，不应简单以 `60 / Avg` 反推替换；并发 1 的结果也不代表服务容量上限。
- 峰值设备内存来自外部周期采样，是观测下界；平台间采样工具和误差不同。
- “通过”的产物检查强度并非跨平台完全一致：Ascend 最终聚合包含严格输出校验；MLU、MetaX 的原始 Server 汇总可确认非空输出及 SHA，但没有统一的严格解码校验字段。三者均不等同于人工画质评分。

## 数据追溯

- [Ascend 最终 Server 报告](../../results/ascend_npu/server/final_service_benchmark_20260726/final_report.json)
- [MLU formal 汇总](../../results/mlu/server/mlu_server_formal_level0_001/summary.json)
- [MetaX 原 formal 汇总](../../results/metax/server/metax_server_20260727T092540Z/summary.json)
- [MetaX 首次正式 retry](../../results/metax/server/metax_server_retry_20260727T200501Z/summary.json)
- [MetaX 通过的 retry_block](../../results/metax/server/metax_server_retry_block_20260727T210300Z/summary.json)

各汇总中的 `result_dir`、`benchmark_result.request_results` 和 `run_config` 可继续追溯到请求明细、生成产物、配置哈希和日志目录。Ascend 最终报告保留了采集机器上的旧绝对前缀 `/data/wushuo1/Lightx2v-Platform-Run-Examples/`；在当前工作区追溯时，应将其映射为仓库根 `/data/Lightx2v-Platform-Run-Examples/`。
