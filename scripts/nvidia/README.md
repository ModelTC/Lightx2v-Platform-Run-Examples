# NVIDIA CUDA 离线推理

本目录提供 NVIDIA CUDA 推理入口。除新增的 Hunyuan3D 单卡端到端用例外，
其余用例与 `scripts/metax/infer/` 对应：

- 12 个单卡用例：`infer/single/`
- 1 个双卡 SP2 用例：`infer/dist_2/`
- 11 个八卡 CFG2×SP4、SP8 或 TP8 用例：`infer/dist_8/`

固定使用以下本机资源：

| 资源 | 路径 |
| --- | --- |
| Examples 项目 | `/data/nvme1/wushuo/Lightx2v-Platform-Run-Examples` |
| LightX2V 项目 | `/data/nvme1/wushuo/LightX2V` |
| 模型权重 | `/data/nvme1/models` |
| NVIDIA 配置 | `/data/nvme1/wushuo/Lightx2v-Platform-Run-Examples/configs/nvidia` |

除 Hunyuan3D shape 配置使用 `torch_sdpa` 外，其余配置均使用 LightX2V
注册的 FA2 名称 `flash_attn2`，并显式设置 `"use_compile": false`。
其它推理步数、分辨率、offload 和并行策略与 MetaX 对应用例保持一致。

入口默认使用 GPU 0、GPU 0–1 或 GPU 0–7。可以通过
`CUDA_VISIBLE_DEVICES` 映射到其它数量匹配的物理卡。

```bash
cd /data/nvme1/wushuo/Lightx2v-Platform-Run-Examples

# 单卡
bash scripts/nvidia/infer/single/run_wan21_1_3b_t2v_480p_81f.sh

# 双卡 SP2
bash scripts/nvidia/infer/dist_2/run_z_image_turbo_t2i_1664x928_sp2.sh

# 八卡 CFG2×SP4
bash scripts/nvidia/infer/dist_8/run_wan21_1_3b_t2v_480p_81f_cfg2_sp4.sh
```

除会保存 mesh 和贴图结果的 Hunyuan3D 用例外，其余入口沿用 MetaX 基准的
no-save 模式：完整执行文本编码、DiT 和 VAE，但不写出最终 PNG/MP4。
每次运行的日志和结构化记录保存在：

```text
logs/nvidia/infer/single|dist_2|dist_8/<case_id>/<run_id>/
```
