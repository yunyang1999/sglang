# DeepEP v2 / EPv2 SM120 集成

本文记录 SGLang 中 `epv2` 后端在 **SM120 / 纯 PCIe / 无 NVL** 机器上的集成、部署和验证结果。当前验证机器为 **5k10**：`10.6.142.10`，工作目录 `/root/menyu`，容器 `sglang_epv2_5k10_menyu`，SGLang 工作树 `/root/menyu/sglang_epv2`。

本文三部分：**一、部署配置  二、运行方式与测试数据  三、代码修改与优化总结**。

---

## 一、部署配置

### 1.1 目标分支

SM120 专用整理分支：

```bash
git clone -b epv2-integration-sm120 https://github.com/MengYu10151/sglang.git
cd sglang
pip install -e python
```

本分支基于 `epv2-integration`，保留 `epv2` 作为独立 MoE A2A backend，不复用 legacy `deepep` 的 dispatcher、mode 语义或 dispatch/combine 数据结构。

### 1.2 硬件与容器

- GPU：SM120，8 卡，纯 PCIe / 无 NVL。
- 当前验证节点：5k10，`10.6.142.10`。
- Docker：`sglang_epv2_5k10_menyu`。
- 容器建议参数：`--privileged --network host --ipc host --shm-size 64g --gpus all`。
- 模型目录：`/root/menyu/models/DeepSeek-V4-Flash`。

### 1.3 NCCL 2.30.7

当前 5k10 使用官方 NCCL 2.30.7 clean build：

```bash
/root/menyu/nccl                         # git branch: v2.30.7-official-clean
/usr/local/nccl-v2307-official           # installed NCCL_HOME
```

运行期环境：

```bash
export NCCL_HOME=/usr/local/nccl-v2307-official
export CPATH=/usr/local/nccl-v2307-official/include:${CPATH:-}
export LIBRARY_PATH=/usr/local/nccl-v2307-official/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/nccl-v2307-official/lib:${LD_LIBRARY_PATH:-}
```

本 SM120 EPv2 direct 验证不依赖 NCCL-EP patch；这里使用的是官方 NCCL 2.30.7。

### 1.4 DeepEP v2

当前验证使用 DeepEP v2 源码：

```bash
/root/menyu/DeepEP_epv2
commit d4f41e4
```

安装方式：

```bash
cd /root/menyu/DeepEP_epv2
TORCH_CUDA_ARCH_LIST=12.0 python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall --no-deps
```

验证：

```bash
python3 - <<'PY'
from deep_ep import ElasticBuffer
import deep_ep
print(deep_ep.__file__)
PY
```

### 1.5 DeepGEMM PR #324

SM120 / MXFP4 路径依赖 DeepGEMM PR #324，当前独立安装并优先于 SGLang vendored/import path：

```bash
/root/menyu/DeepGEMM_pr324
commit aced12c2
```

验证：

```bash
python3 - <<'PY'
import deep_gemm
print(deep_gemm.__file__)
PY
```

当前 5k10 输出：

```text
/root/menyu/DeepGEMM_pr324/deep_gemm/__init__.py
```

### 1.6 运行期环境变量

当前 5k10 env 文件：

```bash
/root/menyu/epv2_env.sh
```

关键配置：

```bash
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_CUDA_VMM=0
export NVSHMEM_QP_DEPTH=4096
export NVSHMEM_IBGDA_NIC_HANDLER=cpu

export NCCL_CUMEM_ENABLE=1
export NCCL_WIN_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_LSA_TEAM_SIZE=1
export NCCL_NET_MERGE_LEVEL=LOC
export NCCL_NVLS_ENABLE=0

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_DSV4_FP4_EXPERTS=1
export SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export SGLANG_EPV2_NUM_SMS=8
```

`NCCL_LSA_TEAM_SIZE=1` 和 `NCCL_NET_MERGE_LEVEL=LOC` 是 5k10 这类纯 PCIe 机器上跑 EPv2 direct 的关键规避配置。

---

## 二、运行方式与测试数据

### 2.1 支持范围

当前 SM120 分支只验证 **DeepSeek-V4-Flash + DeepGEMM + EPv2 direct + FP8 dispatcher output**。

| 场景 | 状态 | 说明 |
| --- | --- | --- |
| `--moe-a2a-backend epv2 --epv2-mode direct --epv2-dispatcher-output-dtype fp8 --moe-runner-backend deep_gemm` | 支持 | SM120 主线 |
| `epv2 hybrid` | 未作为 5k10 主线验证 | 5k10 当前只做 direct 调优适配 |
| Triton / BF16 runner | 未纳入本轮 SM120 性能矩阵 | 后续单独补 |
| CUDA graph | 本轮性能矩阵关闭 | 结果不是 graph-on 上限 |

### 2.2 EPv2 direct 启动命令

本轮性能矩阵使用如下 server 参数：

```bash
source /root/menyu/epv2_env.sh
cd /root/menyu/sglang_epv2

export PYTHONPATH=/root/menyu/sglang_epv2/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_OPT_FIX_MEGA_MOE_MEMORY=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=true
export SGLANG_OPT_USE_JIT_EP_ACTIVATION=1
export SGLANG_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export SGLANG_EPV2_NUM_SMS=8

python3 -m sglang.launch_server \
  --model-path /root/menyu/models/DeepSeek-V4-Flash \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 32124 \
  --context-length 16384 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.80 \
  --attention-backend dsv4 \
  --sampling-backend flashinfer \
  --mamba-backend triton \
  --moe-runner-backend deep_gemm \
  --tp-size 8 \
  --dp-size 8 \
  --ep-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend epv2 \
  --epv2-mode direct \
  --epv2-dispatcher-output-dtype fp8 \
  --disable-overlap-schedule \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --disable-radix-cache \
  --disable-chunked-prefix-cache \
  --disable-shared-experts-fusion \
  --disable-flashinfer-autotune \
  --chunked-prefill-size 2048 \
  --page-size 256 \
  --max-running-requests 256 \
  --sm-group-num 8 \
  --random-seed 42 \
  --skip-server-warmup
```

注意：本轮 SM120 性能矩阵为了和 TP / DP+TP / NCCL_EP 历史矩阵对齐，显式关闭 CUDA graph。不要把这组结果理解成 EPv2 direct 的 graph-on 性能上限。

### 2.3 TP / DP+TP baseline 环境

TP / DP+TP baseline 不能直接套 EPv2 的 fused clamp env。DSV4 DeepGEMM TP path 在某些 varlen shape 下会触发：

```text
AssertionError: swiglu_limit (DeepSeek V4) requires SGLANG_OPT_USE_JIT_EP_ACTIVATION=True
```

原因是 `SGLANG_OPT_SWIGLU_CLAMP_FUSION=true` 时会把 `swiglu_limit` 传入 `_varlen_deep_gemm_silu_mul_quant`；而 `N % 4 != 0 or G % 4 != 0` 会关闭 JIT activation，最终触发 assert。

已验证 TP / DP+TP baseline 应使用：

```bash
export SGLANG_OPT_FIX_MEGA_MOE_MEMORY=false
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_OPT_USE_JIT_EP_ACTIVATION=1
```

启动参数与 EPv2 相同，只是不加 `--moe-a2a-backend epv2`：

```bash
# TP8
--tp-size 8

# DP+TP
--tp-size 8 --dp-size 8 --enable-dp-attention
```

验证日志：

```text
/root/menyu/logs/tp_start_debug_20260630_032157/summary.tsv
```

| FIX_MEGA | SWIGLU_CLAMP | 结果 |
| --- | --- | --- |
| true | true | ERROR |
| false | true | ERROR |
| false | false | READY |

最小 serving smoke：

```text
/root/menyu/logs/tp_bench_debug_20260630_032756/bench_tp8_1024_1.log
/root/menyu/logs/dp8_tp_bench_debug_20260630_033018/bench_dp8_tp_1024_1.log
```

### 2.4 Benchmark 命令

性能矩阵使用 `sglang.benchmark.serving`：

```bash
python3 -m sglang.benchmark.serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port ${PORT} \
  --model /root/menyu/models/DeepSeek-V4-Flash \
  --tokenizer /root/menyu/models/DeepSeek-V4-Flash \
  --dataset-name random-ids \
  --random-input-len ${ISL} \
  --random-output-len ${OSL} \
  --random-range-ratio 1.0 \
  --num-prompts 3 \
  --max-concurrency 1 \
  --tokenize-prompt \
  --disable-tqdm \
  --output-file ${DETAIL_JSONL}
```

矩阵：

```bash
CONFIGS="tp8 dp8_tp dp8_ep_epv2_direct"
SHAPES="8192:1024 1024:1024 1024:8192 1024:1 8192:1 1:1024"
NUM_PROMPTS=3
CONCURRENCY=1
```

结果路径：

```text
# TP / DP+TP baseline
/root/menyu/logs/epv2_corrected_perf_20260630_033641

# EPv2 direct 完整 PASS 矩阵
/root/menyu/logs/epv2_vs_ncclep_matrix_20260630_005158

# 合并报告
/root/menyu/logs/epv2_corrected_perf_20260630_033641/corrected_tp_dp_tp_plus_epv2_report.md
/root/menyu/logs/epv2_corrected_perf_20260630_033641/ttft_tpot_detailed_report.md
```

说明：`epv2_corrected_perf_20260630_033641` 中有一条 EPv2 `BENCH_FAIL` 是手动中断重复 EPv2 重跑造成的，不代表 EPv2 failure。合并报告使用的是 `epv2_vs_ncclep_matrix_20260630_005158` 中完整 PASS 的 EPv2 direct 数据。

### 2.5 Correctness

DSV4 Flash FP8 本地 tokenizer 缺 `chat_template`，raw `/generate` plain prompt 可能出现模板碎片，不作为 strict correctness。Strict correctness 使用 `/v1/chat/completions`：

1. 事实问答：中国和日本首都，期望包含北京/东京。
2. 算术：`17*23+19`，期望 `410`。
3. 翻译：`The quick brown fox jumps over the lazy dog`，期望合理中文翻译。

已验证日志：

```text
/root/menyu/logs/epv2_direct_deepgemm_psum_clean_correctness_20260629_111625.log
/root/menyu/logs/epv2_direct_deepgemm_psum_clean_outputs_*.txt
```

结论：EPv2 direct + DeepGEMM psum layout correctness PASS。

### 2.6 性能总吞吐

| ISL | OSL | TP8 total tok/s | DP+TP total tok/s | EPv2 direct total tok/s | EPv2 vs TP8 | EPv2 vs DP+TP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 1024 | 82.24 | 79.59 | 90.53 | +10.1% | +13.7% |
| 1024 | 1024 | 22.68 | 21.82 | 25.45 | +12.2% | +16.6% |
| 1024 | 8192 | 13.30 | 12.93 | 15.41 | +15.9% | +19.2% |
| 1024 | 1 | 343.18 | 346.23 | 342.00 | -0.3% | -1.2% |
| 8192 | 1 | 329.00 | 351.65 | 354.31 | +7.7% | +0.8% |
| 1 | 1024 | 11.74 | 11.55 | 13.86 | +18.1% | +20.0% |

### 2.7 TTFT / TPOT mean + P95

| ISL | OSL | Config | TTFT mean / P95 ms | TPOT mean / P95 ms | Total tok/s |
| ---: | ---: | --- | ---: | ---: | ---: |
| 8192 | 1024 | TP8 | 24943.42 / 24997.14 | 85.15 / 85.24 | 82.24 |
| 8192 | 1024 | DP+TP | 26232.17 / 26392.62 | 87.54 / 88.08 | 79.59 |
| 8192 | 1024 | EPv2 direct | 26738.34 / 27212.68 | 73.37 / 74.93 | 90.53 |
| 1024 | 1024 | TP8 | 2986.41 / 2987.14 | 85.35 / 85.56 | 22.68 |
| 1024 | 1024 | DP+TP | 4976.53 / 5817.30 | 86.87 / 87.33 | 21.82 |
| 1024 | 1024 | EPv2 direct | 5635.19 / 6503.12 | 73.16 / 73.25 | 25.45 |
| 1024 | 8192 | TP8 | 2985.89 / 2986.18 | 84.24 / 84.37 | 13.30 |
| 1024 | 8192 | DP+TP | 2830.52 / 2855.13 | 86.69 / 87.02 | 12.93 |
| 1024 | 8192 | EPv2 direct | 2805.05 / 2838.23 | 72.69 / 73.30 | 15.41 |
| 1024 | 1 | TP8 | 2984.95 / 2986.10 | 0.00 / 0.00 | 343.18 |
| 1024 | 1 | DP+TP | 2957.33 / 3104.66 | 0.00 / 0.00 | 346.23 |
| 1024 | 1 | EPv2 direct | 2994.16 / 3154.77 | 0.00 / 0.00 | 342.00 |
| 8192 | 1 | TP8 | 24901.19 / 24911.24 | 0.00 / 0.00 | 329.00 |
| 8192 | 1 | DP+TP | 23295.50 / 23472.86 | 0.00 / 0.00 | 351.65 |
| 8192 | 1 | EPv2 direct | 23120.48 / 23389.92 | 0.00 / 0.00 | 354.31 |
| 1 | 1024 | TP8 | 89.80 / 90.36 | 85.23 / 85.49 | 11.74 |
| 1 | 1024 | DP+TP | 92.64 / 93.83 | 86.62 / 87.11 | 11.55 |
| 1 | 1024 | EPv2 direct | 90.04 / 91.33 | 72.21 / 72.42 | 13.86 |

### 2.8 结果解读

- Decode-heavy `1/1024`：EPv2 direct 的 TPOT 为 `72.21 / 72.42 ms`，明显低于 TP8 `85.23 / 85.49 ms` 和 DP+TP `86.62 / 87.11 ms`。
- Long-output `1024/8192`：EPv2 direct 的 TTFT 和 TPOT 均优于 TP8 / DP+TP。
- Prefill-only `1024/1`：三者接近，EPv2 direct 没明显优势。
- Long prefill `8192/1`：EPv2 direct 与 DP+TP 接近，略高于 TP8。
- 本轮数据 CUDA graph 关闭；graph-on 需要另测，不能直接用这里的数据代表生产 decode 上限。

---

## 三、代码修改与优化总结

### 3.1 SM120 DeepGEMM enable

涉及文件：

```text
python/sglang/srt/layers/deep_gemm_wrapper/configurer.py
python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py
```

修改点：

- 取消原先对 `sm_version == 120` 的 DeepGEMM 禁用。
- `DEEPGEMM_BLACKWELL` 扩展到 `sm100 or sm120`。
- `DEEPGEMM_SCALE_UE8M0` 在 SM120 下开启，匹配 DeepGEMM PR #324 的 packed UE8M0 scale path。
- `get_mn_major_tma_aligned_tensor` 增加新旧 DeepGEMM API 兼容 fallback。
- `grouped_gemm_nt_f8f8bf16_contig` 增加 `use_psum_layout` / `expected_m_for_psum_layout` 参数，适配 DeepGEMM PR #324 的 psum contiguous path。

### 3.2 EPv2 direct 到 DeepGEMM psum layout

涉及文件：

```text
python/sglang/srt/layers/moe/token_dispatcher/epv2.py
python/sglang/srt/layers/moe/moe_runner/deep_gemm.py
python/sglang/srt/layers/moe/utils.py
```

修改点：

- EPv2 direct expanded output 不再先 repack 成 masked slab，而是直接走 DeepGEMM psum contiguous layout。
- `DeepGemmRunnerInput` 增加 `use_psum_layout`。
- `pre_permute_epv2_to_deep_gemm` 在 expanded/direct 路径中将 `psum_num_recv_tokens_per_expert` 作为 grouped layout 信息传入 DeepGEMM。
- 对 non-expanded path，`num_recv_tokens_per_expert` 可保持 GPU tensor，减少 Python list / host sync。
- `expected_m` 继续作为 DeepGEMM 调度 hint，而不是模型语义 token limit。

### 3.3 Top-k duplicate 与 padding 处理

涉及文件：

```text
python/sglang/srt/layers/moe/token_dispatcher/epv2.py
python/sglang/srt/layers/moe/ep_moe/kernels.py
python/sglang/srt/layers/moe/moe_runner/deep_gemm.py
```

修改点：

- `_deduplicate_topk_for_epv2`：同一 token 的重复 expert id 会把权重累加到第一条 lane，后续 duplicate lane 标记为 `-1/0`，避免 EPv2 dispatch epilogue 对重复 local expert id 的限制。
- `m_indices` padding 初始化为 `-1`，避免 DeepGEMM 读取未初始化 padding expert id。
- scatter / expand m_indices init kernel 中 padding row 写 `-1`。
- `ep_gather` 增加 `APPLY_WEIGHTS` 开关，expanded combine 语义下可避免重复乘权重。

### 3.4 FP4 / MXFP4 quant config 传递

涉及文件：

```text
python/sglang/srt/layers/quantization/fp8.py
python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
```

修改点：

- dispatcher quant config 中增加 `is_fp4_experts`。
- Quark MXFP4 MoE 显式传 `torch.float4_e2m1fn_x2` 与 `is_fp4_experts=True`。
- EPv2/DeepGEMM path 可以按 FP8/FP4 expert weight 语义选择 recipe 和 scale layout。

### 3.5 Debug 开关

默认不启用，仅用于定位 correctness / layout 问题：

```bash
export SGLANG_EPV2_DEBUG_TENSOR=1
export SGLANG_EPV2_DEBUG_TOPK=1
```

功能：

- 打印 dispatch input / recv hidden / recv scale / combine input-output 的 shape、dtype、NaN/Inf、absmax。
- 打印 top-k duplicate、local duplicate、weight sum 分布和样例。

### 3.6 当前限制

- 当前 SM120 验证只覆盖 DeepGEMM FP8 direct path；hybrid / Triton / BF16 需要另测。
- 当前性能矩阵关闭 CUDA graph；graph-on 的 decode 上限需另开专门矩阵。
- TP / DP+TP baseline 必须使用 `FIX_MEGA=false + SWIGLU_CLAMP=false`，否则会触发 DSV4 DeepGEMM fused clamp varlen assert。
- EPv2 `direct` / `hybrid` 仍是 server 生命周期固定模式，不是 DeepEP v1 `auto` 式 prefill/decode 动态切换。
- 5k10 纯 PCIe 场景依赖 `NCCL_LSA_TEAM_SIZE=1` 和 `NCCL_NET_MERGE_LEVEL=LOC`。

### 3.7 后续建议

1. 补充 CUDA graph enabled 的 EPv2 direct decode 矩阵，并与本轮 graph-off 矩阵分开记录。
2. 单独验证 hybrid/prefill path，不和 5k10 direct 主线混在同一结论里。
3. 给 TP / DP+TP baseline 脚本增加 env guard，避免再次把 EPv2 fused clamp env 套到 TP baseline。
4. 将 DeepGEMM PR #324 dependency 写入安装脚本或 requirements 文档，避免 fallback 到 vendored/旧 deep_gemm。
