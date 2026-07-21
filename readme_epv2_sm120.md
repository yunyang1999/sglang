# DeepEP v2 / EPv2 SM120 集成与验证

本文记录 SGLang `deepep_v2` 后端在 **SM120、8 卡、纯 PCIe、无 NVL**
节点上的集成、部署、正确性和性能结果。本文只覆盖：

- 模型：DeepSeek-V4-Flash。
- MoE runner：DeepGEMM。
- Dispatcher output：FP8。
- EPv2 通信模式：`direct`。
- 当前验证节点：5k11（`10.6.142.11`）。
- 当前容器：`5k11_epv2_sm120_sync`。
- 当前工作树：`/root/menyu/sglang_epv2_sm120_sync`。

当前 SM120 分支基于：

```text
origin/epv2-deepep-v2-backend@a791257700
```

旧 SM120 分支使用过 psum-contiguous decode 实验路径。当前版本已经与最新
`epv2-deepep-v2-backend` 对齐：prefill/extend 使用 contiguous adapter，decode
使用 expanded dispatch + masked DeepGEMM adapter。旧性能表不能与当前结果直接混用。

---

## 1. 支持范围

| 配置 | 状态 | 说明 |
| --- | --- | --- |
| `deepep_v2 + direct + fp8 + deep_gemm` | 已验证 | 本分支主线 |
| `deepep_v2 + hybrid` | 未验证 | SM120 纯 PCIe 节点当前只调优 direct |
| Triton / BF16 runner | 未验证 | 本轮只验证 DeepGEMM |
| CUDA graph | 未纳入性能矩阵 | 本文数据均显式关闭 graph |
| MegaMoE | 不涉及 | EPv2 不调用 MegaMoE kernel，也不使用 swizzled gate/up layout |

EPv2 是独立 MoE A2A backend，不复用 legacy DeepEP normal/low-latency dispatcher、
handle 或 dispatch/combine 数据结构。

---

## 2. 依赖与运行环境

### 2.1 容器建议

```bash
docker run --privileged --network host --ipc host \
  --shm-size 64g --gpus all ...
```

模型目录：

```text
/root/menyu/models/DeepSeek-V4-Flash
```

### 2.2 NCCL

本轮 server 显式加载自编译 NCCL：

```text
source: /root/menyu/nccl
HEAD:   28b9118fee74761c2e892f73a00713863891a4e7
lib:    /root/menyu/nccl/build/lib
```

启动时必须确保：

```bash
export LD_LIBRARY_PATH=/root/menyu/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
```

当前源码工作树含本地 NCCL/EP 调试改动，正式发布前需要把 NCCL 二进制来源整理成
可复现的 clean commit 或安装脚本。EPv2 SM120 direct 验证不依赖 NCCL-EP backend，
但依赖 NCCL symmetric-memory/RDMA 运行能力。

### 2.3 DeepEP v2

```text
source: /root/menyu/DeepEP_b306af0
commit: b306af06afd412c88e51e71802951606e40b7358
import: /root/menyu/DeepEP_b306af0/deep_ep/__init__.py
```

### 2.4 DeepGEMM SM120

使用 DeepGEMM PR #324 的 SM120/MXFP4 实现：

```text
source: /root/menyu/dg-sm120
commit: aced12c2c8882a945c568ace9d4a7e5778aae410
import: /usr/local/lib/python3.12/dist-packages/deep_gemm/__init__.py
torch:  2.11.0+cu130
```

独立安装的 `deep_gemm` 优先于 SGLang vendored 版本。

### 2.5 纯 PCIe/RDMA 环境

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
```

`NCCL_LSA_TEAM_SIZE=1` 和 `NCCL_NET_MERGE_LEVEL=LOC` 是当前纯 PCIe 节点运行
EPv2 direct 的必要配置。

---

## 3. 当前数据流

EPv2 的通信模式在 server 初始化时固定为 `direct`，但 MoE runner layout 按推理阶段选择：

| 推理阶段 | EPv2 dispatch layout | DeepGEMM 输入 | CPU sync |
| --- | --- | --- | --- |
| Prefill / extend | non-expanded contiguous | contiguous grouped GEMM | exact count readback |
| Decode / non-extend | native expanded | `expand_to_masked_slab` 后进入 masked grouped GEMM | `do_cpu_sync=False` |

Decode 路径为：

```text
BF16 hidden states
  -> FP8 pre-quant + scales
  -> EPv2 direct expanded dispatch
  -> expand_to_masked_slab([E_local, max_m, hidden])
  -> DeepGEMM masked gate/up GEMM
  -> clamp + SiLU + FP8 quant
  -> DeepGEMM masked down GEMM
  -> masked_slab_to_expand
  -> EPv2 native combine
```

`masked_m` 保存每个本地 expert 的真实 token 数；`masked_max_m` 使用固定
`capacity * ep_group_size`，避免 ragged DP 下其他 rank 的较大 batch 溢出本 rank buffer。

### 3.1 与 MegaMoE 的边界

EPv2 不需要 MegaMoE：

```bash
export SGLANG_OPT_FIX_MEGA_MOE_MEMORY=false
```

该变量设为 `false` 是为了覆盖容器可能继承的旧环境。EPv2 adapter 不读取
MegaMoE backend、不写 `disable_swizzle` 标记，也不调用 MegaMoE kernel。

Clamp fusion 是独立的 DeepGEMM activation 优化：

```bash
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=true
export SGLANG_OPT_USE_JIT_EP_ACTIVATION=1
```

---

## 4. EPv2 启动命令

```bash
cd /root/menyu/sglang_epv2_sm120_sync

export PYTHONPATH=/root/menyu/sglang_epv2_sm120_sync/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/root/menyu/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_CUMEM_ENABLE=1
export NCCL_WIN_ENABLE=1
export NCCL_LSA_TEAM_SIZE=1
export NCCL_NET_MERGE_LEVEL=LOC
export NCCL_NVLS_ENABLE=0
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_CUDA_VMM=0
export NVSHMEM_QP_DEPTH=4096
export NVSHMEM_IBGDA_NIC_HANDLER=cpu
export CUDA_DEVICE_MAX_CONNECTIONS=1

export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_DSV4_FP4_EXPERTS=1
export SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export SGLANG_DEEPEP_V2_NUM_SMS=8
export SGLANG_OPT_FIX_MEGA_MOE_MEMORY=false
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=true
export SGLANG_OPT_USE_JIT_EP_ACTIVATION=1

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
  --moe-a2a-backend deepep_v2 \
  --deepep-v2-mode direct \
  --deepep-v2-dispatcher-output-dtype fp8 \
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

---

## 5. 正确性验证

### 5.1 单元测试

```bash
python3 -m pytest -q \
  test/registered/unit/server_args/test_server_args.py::TestDeepEPv2Args \
  test/registered/unit/layers/moe/test_deepep_v2_masked_slab.py
```

结果：

```text
20 passed
```

### 5.2 E2E 输出

使用 `/v1/chat/completions`、`temperature=0`、`max_tokens=256`：

| 问题 | 当前 EPv2 输出 | 结果 |
| --- | --- | --- |
| 中国和日本首都 | `中国和日本的首都分别是北京市和东京都。` | PASS |
| `17*23+19` | `17 × 23 = 391，391 + 19 = 410，所以答案是 410。` | PASS |
| 英译中 | `敏捷的棕色狐狸跳过了懒狗。` | PASS |

验证配置：`direct + FP8 + DeepGEMM + FIX_MEGA=false + CLAMP_FUSION=true`。
当前输出无乱码。

---

## 6. 性能测试方法

### 6.1 Benchmark 命令

```bash
python3 -m sglang.benchmark.serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 32124 \
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
  --output-details \
  --output-file ${DETAIL_JSON}
```

完整矩阵：

```text
configs: TP8, DP8+TP, DP8+EPv2-direct
shapes:  8192/1024, 1024/1024, 1024/8192, 1024/1, 8192/1, 1/1024
prompts: 3
CC:      1
graph:   OFF
```

### 6.2 为什么分成两套性能结果

DSV4 Flash 的 TP8 当前不能使用 fused clamp。JIT activation guard 包含：

```python
if N % 4 != 0 or G % 4 != 0 or D // 8 < E:
    use_jit_ep_activation = False
```

DSV4 Flash 有 256 个 expert、MoE intermediate size 2048：

| 配置 | 本地 experts `E` | 本地 intermediate `D` | `D/8 >= E` |
| --- | ---: | ---: | --- |
| TP8 | 256 | 256 | 不满足 |
| EP8 | 32 | 2048 | 满足 |

因此不存在“TP8 和 EP8 同时开启 fused clamp”的当前可运行矩阵。本文分别记录：

1. **公平 ablation**：三边全部关闭 clamp fusion。
2. **当前可部署配置**：EPv2 开启 clamp fusion，TP8 使用非融合 fallback。

两套结果回答的问题不同，不能混成一张“EP 一定比 TP 快”的表。

---

## 7. 公平矩阵：三边 clamp fusion 全部关闭

共同环境：

```bash
export SGLANG_OPT_FIX_MEGA_MOE_MEMORY=false
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_OPT_USE_JIT_EP_ACTIVATION=1
```

18/18 benchmark case PASS，三组 server correctness gate PASS。

| ISL/OSL | 配置 | Total tok/s | TTFT mean/P95 ms | TPOT mean/P95 ms |
| --- | --- | ---: | ---: | ---: |
| 8192/1024 | TP8 | 88.11 | 22197.86 / 22200.47 | 80.54 / 81.03 |
| 8192/1024 | DP8+TP | 82.13 | 26720.66 / 27028.66 | 83.56 / 84.18 |
| 8192/1024 | DP8+EPv2 | 60.76 | 25614.39 / 25718.17 | 123.22 / 123.28 |
| 1024/1024 | TP8 | 23.93 | 2701.27 / 2701.54 | 81.03 / 81.10 |
| 1024/1024 | DP8+TP | 23.27 | 2833.56 / 2879.17 | 83.27 / 83.61 |
| 1024/1024 | DP8+EPv2 | 15.79 | 4582.75 / 4683.41 | 122.30 / 122.37 |
| 1024/8192 | TP8 | 13.82 | 2702.03 / 2702.40 | 81.11 / 81.21 |
| 1024/8192 | DP8+TP | 13.59 | 2835.66 / 2890.73 | 82.46 / 82.86 |
| 1024/8192 | DP8+EPv2 | 9.15 | 2724.10 / 2735.84 | 122.62 / 122.69 |
| 1024/1 | TP8 | 379.19 | 2701.25 / 2701.81 | N/A |
| 1024/1 | DP8+TP | 365.14 | 1870.27 / 2806.40 | N/A |
| 1024/1 | DP8+EPv2 | 344.68 | 2970.34 / 2996.48 | N/A |
| 8192/1 | TP8 | 368.72 | 22218.03 / 22224.83 | N/A |
| 8192/1 | DP8+TP | 356.52 | 15296.37 / 23040.39 | N/A |
| 8192/1 | DP8+EPv2 | 367.50 | 22290.38 / 22368.63 | N/A |
| 1/1024 | TP8 | 12.29 | 84.42 / 84.76 | 81.42 / 81.49 |
| 1/1024 | DP8+TP | 12.03 | 90.81 / 92.07 | 83.21 / 84.04 |
| 1/1024 | DP8+EPv2 | 8.19 | 87.00 / 89.11 | 122.23 / 122.37 |

日志：

```text
/root/menyu/logs/epv2_sm120_fair_clamp_off_20260720_235800
```

结论：关闭 fused clamp 后，EPv2 output-heavy case 的 TPOT 上升到约 122 ms。
这是 DeepGEMM activation 前后处理退化，不是 EPv2 direct 通信正确性问题。

---

## 8. 当前可部署配置：EPv2 clamp fusion 开启

cleanup 后配置：

```bash
export SGLANG_OPT_FIX_MEGA_MOE_MEMORY=false
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=true
export SGLANG_OPT_USE_JIT_EP_ACTIVATION=1
```

代表性复测：

| ISL/OSL | 配置 | Total tok/s | TTFT mean/P95 ms | TPOT mean/P95 ms |
| --- | --- | ---: | ---: | ---: |
| 1024/1 | DP8+EPv2 | 338.56 | 3024.05 / 3088.80 | N/A |
| 1/1024 | DP8+EPv2 | 12.78 | 359.63 / 360.23 | 78.04 / 78.35 |

日志：

```text
/root/menyu/logs/epv2_no_mega_cleanup_20260721
```

与 cleanup 前、同一 5k11 上的 EPv2 optimized 结果对比：

| ISL/OSL | cleanup 前 | cleanup 后 | 变化 |
| --- | ---: | ---: | ---: |
| 1024/1 total tok/s | 344.80 | 338.56 | -1.8% |
| 1/1024 total tok/s | 12.98 | 12.78 | -1.5% |
| 1/1024 TPOT mean | 77.12 ms | 78.04 ms | +1.2% |

该差异处于短样本运行波动范围，说明移除 EPv2/MegaMoE 耦合没有造成显著性能回退。

使用公平矩阵中的当前 TP8 baseline 作参考：

| ISL/OSL | TP8 | EPv2 optimized | EPv2 相对 TP8 |
| --- | ---: | ---: | ---: |
| 1024/1 total tok/s | 379.19 | 338.56 | -10.7% |
| 1/1024 total tok/s | 12.29 | 12.78 | +4.0% |
| 1/1024 TPOT mean | 81.42 ms | 78.04 ms | -4.2% |

当前只能得出：

- Decode representative case：EPv2 optimized 略优于 TP8。
- Prefill representative case：EPv2 慢于 TP8。
- 旧 README 中“EPv2 全矩阵优于 TP”的结论不再保留。
- 完整 optimized EPv2 矩阵仍需按当前 commit 重新测试。

---

## 9. SM120 分支代码修改

### 9.1 DeepGEMM SM120 enable

涉及文件：

```text
python/sglang/srt/layers/deep_gemm_wrapper/configurer.py
python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py
```

- 允许 SM120 启用外部 DeepGEMM PR #324。
- `DEEPGEMM_BLACKWELL` 扩展到 SM100/SM120。
- SM120 使用 packed UE8M0 scale。
- 兼容新旧 `get_mn_major_tma_aligned_tensor` import API。

### 9.2 UE8M0 scale packing

涉及文件：

```text
python/sglang/srt/layers/moe/moe_runner/deep_gemm.py
```

- 显式按 4 个 uint8 exponent 打包到一个 int32。
- 避免对非 contiguous uint8 tensor 直接 `.view(torch.int32)`。

### 9.3 FP4/MXFP4 capability

涉及文件：

```text
python/sglang/srt/layers/quantization/fp8.py
python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
```

- dispatcher quant config 传递 `is_fp4_experts`。
- Quark MXFP4 显式传递 FP4 semantic dtype。

### 9.4 Clamp fusion 与 MegaMoE 解耦

涉及文件：

```text
python/sglang/srt/layers/moe/moe_runner/deep_gemm.py
```

- `SGLANG_OPT_SWIGLU_CLAMP_FUSION` 独立控制 fused activation。
- 不在 `DeepGemmRunnerCore` 新增 `get_moe_a2a_backend()` 依赖。
- 不在 EPv2 adapter 写 MegaMoE/swizzle 特判。
- upstream 原有 MegaMoE 开关和 kernel 不做修改。

---

## 10. 当前限制与后续工作

1. 重新跑当前 commit 的完整 EPv2 clamp-on 矩阵，不能继续引用旧 psum-contiguous 数据。
2. 单独补 CUDA graph enabled 的 decode 矩阵；本文结果不是 graph-on 上限。
3. 若需要 TP8/EP8 完全相同的 fused clamp 条件，需要扩展 JIT activation 对
   `D/8 < E` shape 的支持，或实现支持 `swiglu_limit` 的 fused fallback。
4. 对通信库本身做公平比较时，应使用 EP library UT 或固定相同 MoE compute path；
   e2e best-config 对比会混入 activation、layout 和 scheduler 差异。
5. 当前只验证 direct + DeepGEMM + FP8。Hybrid、Triton/BF16 和其他模型需要单独验证。
6. 将 NCCL、DeepEP v2 和 DeepGEMM PR #324 固化到可复现安装脚本。
