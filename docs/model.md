# metal-indextts TTS 整体结构分析与优化路线

> 基线环境：Apple M3 Ultra（60 核 GPU，fp32 峰值约 28 TFLOPS，统一内存带宽约 800 GB/s）
> 基线数据：`今天天气不错,去划船吗?`（2 段，2.51s 音频）推理 17.5s，RTF ≈ 7.0x
> 模型 bundle：7.7 GB fp32（gpt 3.5G / s2mel 1.2G / semantic_codec 0.18G / bigvgan 0.45G）

## 一、流水线总览

```
文本 ──► Tokenizer/分段 (CPU) ──► 每段依次执行：
          │
          ├─ [1] Frontend      文本→ids；voice conditioning + emovec
          │       (subsampling conv → conformer ×N → perceiver)        ~0.55s/段
          │
          ├─ [2] GPT decode    自回归生成 mel codes（24 层，KV cache）   ~3.5s/段
          │       prefix = conds(34) + text(~13) → 逐 token 解码 ~60 codes
          │
          ├─ [3] Condition     codes → semantic codec 量化 → s2mel
          │       length regulator → 拼 prompt → condition[T,512]       ~2.1s/段
          │
          └─ [4] Acoustic      CFM 25 步 euler × DiT estimator(CFG batch=2)
                  → mel → BigVGAN vocoder → wav                          ~1.9s/段
          
段间拼接（200ms 静音）──► 输出 wav
```

### 各模块结构与计算量（T = prompt 629 + 生成 ~100 ≈ 730 tokens）

| 模块 | 结构 | 主要张量 | 每段计算量(约) |
|---|---|---|---|
| Frontend | conv2d subsampling + conformer 栈 + perceiver cross-attn | spk_cond_emb [364,1024] | ~50 GFLOP |
| GPT | 24 层 decoder（width 1280, heads 20, MLP 5120），mel_head [8194,1280] | KV cache [24,2,T_kv,1280] | 每 token ~0.5 GFLOP（GEMV，带宽瓶颈） |
| Condition | semantic codec（codebook 8192×8）+ length regulator（conv k=3）+ 投影 | condition [T,512] | ~30 GFLOP |
| CFM/DiT | 13 层 transformer（512 宽，RoPE attn 8×64）+ 8 层 wavenet（k=5, 512→1024）+ CFG 双分支 | x [2T,80→512] | 25 步 × ~20 GFLOP = 500 GFLOP |
| BigVGAN | conv_pre + 6 级上采样 ×3 resblocks（snake 激活） | mel[T,80] → wav[T×256] | ~80 GFLOP |

### 已落地优化（截至 2026-06-11）

- CFM estimator 单 command buffer pass 化（11392 → 492 个 CB）
- 大 GEMM 全部走 MPSMatrixMultiplication；wavenet k=5 conv 拆 5 tap 累加 GEMM
- attention kernel：RoPE 预旋转 + simd 归约 + V 阶段 1024 线程并行
- 10 个归约 kernel 的 threadgroup 竞态修复（曾导致 >512 token 不确定输出）
- CFM 步内权重 CPU 拷贝消除（residentExists 跳过，省 ~330MB/步 memcpy）

## 二、当前性能画像与瓶颈定位

实测分解（2 段合计，推理 17.5s）：

| 阶段 | 耗时 | GPU 实际计算 | 主要损耗 |
|---|---|---|---|
| Frontend | 1.1s | ~0.6s | 每段新建 MetalContext（编译 Metal 库 + 权重上传） |
| GPT | 7.0s | ~3.5s | **每 token 重传整个 KV cache（~170MB/token）**；GEMV 带宽 fp32 |
| Condition | 4.2s | ~1.3s | 逐 op 提交+等待；每段重建 context/权重 |
| Acoustic | 5.2s | ~4.8s | CFM 仍 25 步；BigVGAN ~400 个独立 CB |

三个**结构性事实**（优化空间的根源）：

1. **每段×每阶段新建 `MetalContext`**（tts_synthesis.inl 中 `{ mit2::MetalContext gpt_metal; ... }` 块）：
   每次构造都 `newLibraryWithSource` 重新编译全部 kernel（~0.2-0.5s）并重新上传该阶段全部
   resident 权重（GPT 约 1.9GB）。N 段 × 4 阶段 = 4N 次重复。
2. **GPT KV cache 存在 CPU**：`gpt_cached_attention_f32_pass(cache.k, cache.v, ...)` 每个
   token 把 24 层 × 2 × kv_tokens × 1280 floats 上传一遍（kv≈700 时 ~172MB/token，
   60 token ≈ 10GB 纯上传），并且每层 QKV 再读回 CPU 追加 cache。
3. **fp32 权重**：GPT 解码是典型带宽瓶颈（每 token 读 ~1.9GB 权重），fp32 浪费一半带宽。

---

## 三、优化点可行性与收益分析

### 1. CPU → GPU 计算迁移

**现状盘点**（仍在 CPU 的计算）：

| 项目 | 量级 | 迁移可行性 | 收益 |
|---|---|---|---|
| Tokenizer/分段 | <1ms | 无必要 | — |
| GPT 每 token 采样/argmax（8194 logits 读回） | 每 token 一次同步 | argmax 可做成 GPU kernel，token id 留 GPU | 小（消除 ~60 次同步往返，约 0.1-0.3s/段） |
| GPT KV cache 追加（passRead QKV → CPU append） | 每 token 24 次读回 | **应直接在 GPU 上 append（见第 3 点）** | 大（与拷贝消除合并计算） |
| CFM euler 更新后 x 读回再上传 | 233KB×2×25/段 | x 全程驻留 GPU（pass 跨步复用 slot） | 中（消除 50 次同步） |
| weight-norm/tap 转置预计算 | 已缓存 | 可挪到 bundle 构建期 | 小（首步延迟改善） |

**结论**：纯"计算搬到 GPU"剩余空间不大（计算本体已基本在 GPU），真正收益在
**消除迁移伴随的同步点**——GPT 每 token 的 logits 读回 + KV 读回是当前最贵的同步链。
建议与优化点 3 合并实施。

### 2. 固定模型的结构整合 / 计算合并

模型权重固定（不考虑升级），可做静态折叠与预计算：

| 方案 | 说明 | 可行性 | 预估收益 |
|---|---|---|---|
| **t_embedder 查表** | CFM 的 25 个 timestep 值固定（t=0, 1/25, …），t1/t2 嵌入（512 维 ×2）可对整张表预计算一次，跨段/跨请求复用 | 高，纯查表 | 每段省 50 个独立 CB（~0.1s/段） |
| **weight-norm 折叠进 bundle** | s2mel/bigvgan 的 weight_g/weight_v 在构建 bundle 时折叠成 dense 权重；conv tap 转置同理 | 高，离线一次 | 首步延迟 ~0.2s；运行时代码简化 |
| **QKV+RoPE 融合** | DiT 的 QKV GEMM 输出后单独跑 rotate kernel；可在 GEMM 后处理或将旋转并入下游 | 中 | 小（rotate 已 <1ms） |
| **LayerNorm/AdaRMSNorm + GEMM 融合** | 每层 norm→linear 是固定序列，可写 fused kernel | 中（手写 kernel） | 小-中（减少 ~30% 小 dispatch，约 0.2-0.4s/段 acoustic） |
| **CFG 双分支共享 prompt 计算** | null 分支的 cond/style 全零、prompt_x 全零——第 0 层的输入投影中 null 分支大部分输入恒定，可预计算第 0 层 null 分支的常量部分 | 中（只省第一层一半） | 小 |
| **CFM 常量跨步驻留** | cond_bat/px_bat/style/mask 25 步不变，目前每步 beginPass 重新上传（~6MB×25） | 高 | 中（~0.15GB 上传/段 + alloc 开销，约 0.2-0.3s/段） |
| **GPT prefill 与 decode 共用 pass** | prefill 是一次大 GEMM 批（已较优）；decode 24 层已 pass 化 | 已具备 | — |

**结论**：单项都不大，但 t_embedder 查表 + CFM 常量驻留 + norm 融合合计可再省
acoustic ~20-30%。优先级低于第 3、4 点。

### 3. 减少 CPU↔GPU 数据拷贝（**收益最大**）

| 方案 | 现状 | 改法 | 预估收益 |
|---|---|---|---|
| **GPT KV cache GPU 驻留** | 每 token 上传 ~172MB + 24 次 QKV 读回 | cache 用持久 MTLBuffer（24 层 × 2 × max_tokens × 1280 × 4B ≈ 350MB 预分配），attention kernel 直接读，QKV 由 kernel 原位 append | **GPT 3.5s → 预计 1.0-1.5s/段（2-3 倍）**，全管线最大单项 |
| **跨段/跨阶段共享 MetalContext** | 每段每阶段重建（库编译 + 权重重传） | 单进程一个 context（或每阶段一个、跨段复用）；权重 resident 一次 | 省 ~1.5-2.5s/段 的隐性开销（计入各 stage 秒数）；API 服务形态下必做 |
| **阶段间数据 GPU 直通** | conds/codes/condition 经 CPU vector（含调试用文件导出） | 阶段间用 PassSlot/MTLBuffer 句柄传递；文件导出加开关 | 中（每段 ~10MB×多次 + 文件 IO，约 0.2-0.4s/段） |
| **CFM x 驻留 GPU** | 每步 euler 后读回→下步重传 | 一个 pass 跑满 25 步（或 x 用持久 buffer） | 中（50 次同步 + 上传） |
| **logits 不读回** | 每 token 读 32KB + CPU argmax | GPU argmax kernel，仅读回 1 个 uint（采样模式需扩展） | 小-中（同步链缩短） |
| **非 pass op 的 arena 进出拷贝** | condition 阶段每 op memcpy 进出 | 把 condition 主链路 pass 化 | 中（condition 2.1 → ~1.2s/段） |

**结论**：这是当前最划算的方向。仅 KV cache 驻留 + 共享 context 两项,
预计把整体 RTF 从 7.0x 压到 **3.5-4.5x**。

### 4. 计算精度量化

| 方案 | 可行性 | 收益 | 风险 |
|---|---|---|---|
| **fp16 权重存储 + fp32 累加（GEMM/GEMV）** | 高：MPS 原生支持 MPSDataTypeFloat16；自有 kernel 用 `half` 读、`float` 累加 | GPT 解码带宽减半 → **GPT 再快 ~1.8x**；MPS GEMM 在 Apple GPU 上 fp16 吞吐≈2×fp32；bundle 体积 7.7G→3.9G，权重上传时间减半 | 低：fp32 累加下 GEMM 误差 ~1e-3 相对量级；CFM 是迭代 ODE，轨迹会偏移（输出是"另一个有效采样"，与现 MPS 改动同性质）；建议听感 A/B |
| **KV cache fp16** | 高 | cache 内存/带宽减半，配合驻留方案 | 低（注意 softmax 前转 fp32） |
| **int8 权重（per-channel）** | 中：MPS 无现成 int8 GEMM，需手写 simdgroup kernel + 反量化 | GPT 带宽再减半（理论再 ~1.5-1.8x） | 中：GPT 自回归对量化误差敏感（codes 可能漂移）；工程量大 |
| **bf16** | Apple GPU 无原生 bf16 算力优势 | 不如 fp16 | — |
| **激活 fp16** | 中（逐 kernel 改） | 带宽进一步降低 | 中：norm/softmax 需保 fp32 |

**结论**：**fp16 权重是性价比最高的量化档位**——一次 bundle 转换 + GEMM dtype 切换，
GPT/acoustic 同时受益，预计整体再提速 1.5-1.8x。int8 暂不建议（工程量大、
自回归质量风险，等 fp16 落地后按需评估）。

### 5. 其他可能方案（待讨论，暂不执行）

| 方案 | 思路 | 预估收益 | 备注 |
|---|---|---|---|
| **减少 CFM 步数** | 25 → 12~16 步（euler 换 midpoint/heun 可在更少步数保质量） | acoustic 按比例（~1.5-2x） | 质量旋钮，需听感验证；也可做蒸馏 few-step（训练工作，超出"不升级模型"约束） |
| **段间流水线并行** | 段 N 的 acoustic 与段 N+1 的 GPT 并行（各自 command queue） | 多段文本墙钟 ~1.5-2x | 依赖共享 context 改造；单段无收益 |
| **GPT 投机解码** | 小 draft 模型或 n-gram 草稿 + 批量验证 | GPT 1.5-2.5x | 需 draft 来源；批量验证路径已具备（prefill 即批量） |
| **CFG 提前退出/间隔** | 后期步骤 CFG 影响小，可隔步算 null 分支 | acoustic ~1.3x | 有论文支持（CFG-cache）；需听感验证 |
| **BigVGAN pass 化 + 上采样融合** | ~400 个独立 CB → 单 pass；snake 激活与 conv 融合 | ~0.3-0.5s/段 | 工程直给，收益中等 |
| **prompt 区 mel 不送 BigVGAN** | 已只送生成区 ✓ | — | 已是现状 |
| **流式输出** | BigVGAN 分块流式 + 段级流式播放，首包延迟 ≈ 首段推理时间 | 体感巨大（非吞吐） | API 服务化时一并设计 |
| **Metal residency set / heap** | 权重统一 heap 管理，省 per-CB 绑定开销 | 小 | 共享 context 后再评估 |

---

## 四、建议实施顺序（按收益/成本比）

| 优先级 | 项目 | 预计整体 RTF |
|---|---|---|
| P0 | 3-a GPT KV cache GPU 驻留（含 GPU append、argmax kernel） | 7.0x → ~5.0x |
| P0 | 3-b 共享 MetalContext（跨段跨阶段，权重只传一次） | → ~4.0x |
| P1 | 4 fp16 权重（bundle 转换 + MPS/kernels dtype） | → ~2.5x |
| P1 | 3-c condition 阶段 pass 化 + 阶段间 GPU 直通 | → ~2.2x |
| P2 | 2 t_embedder 查表 + CFM 常量跨步驻留 + BigVGAN pass 化 | → ~2.0x |
| P3 | 5 CFM 步数/CFG 间隔（质量旋钮）、段间流水线、投机解码 | → 1.0-1.5x 可期 |

> 注：RTF 为推理口径（不含模型加载）。P0+P1 全部落地后，2.5s 量级短句有望接近
> 实时边界；叠加 P3 的质量旋钮可越过 1.0x。


---

## 五、Apple Silicon 高配（48+ 核 GPU / 10+ 核 CPU）专项方案

> 基线（2026-06-11 优化后）：短句 RTF 3.09x、长句 1.57x（CFM 16 步）。
> 剩余瓶颈本质：**不是算力不够，是喂不饱 60 个 GPU 核** —— GPT 是串行微 dispatch
> （每 token 124 个，核占用 <10%），CFM 是 ~4800 个串行 barrier 的小 dispatch（~30%）。

### 关键认知（决定方案取舍）

1. **MPS 的 encoder 边界 = 隐式全序列化点**：MPS GEMM 必须独占 encoder，而 hazard
   tracking 在 encoder 边界按"整个 buffer"粒度串行化。因此只要 GEMM 走 MPS，
   "并发 dispatch encoder" 的收益就被夹在 GEMM 之间，空间有限。真正解锁并发要么
   换自写 simdgroup_matrix GEMM（全部进单 encoder），要么 untracked buffer + 手动
   fence（高风险，曾在此项目踩过竞态坑）。
2. **t1/t2 条件量是"每步常量"**：CFM 的 timestep 嵌入查表后，所有 adaLN 投影
   （13 层 × attn/ffn）、final norm 投影、final layer 调制、wavenet cond 卷积
   都只依赖 (step, layer)，可整表预计算 —— 零数值变化地砍掉每步 ~30 个小
   dispatch 及其 barrier 气泡。
3. **统一内存 + AMX**：GPT 解码本质是每 token 读 ~0.95GB fp16 权重的 GEMV，
   纯带宽活。M3 Ultra 的 AMX 走同一条 820GB/s 总线且零 kernel 启动延迟，
   llama.cpp 已证明该形态下 CPU 解码可打平/超过 GPU。
4. **ANE 是闲置的第三算力**（16 核，~35 TOPS fp16，与 GPU 真并行）：BigVGAN
   纯卷积、权重固定，是 CoreML 离线转换的理想对象；token 数可变需按桶 padding。
5. **多队列段间重叠**：上一轮失败的原因是第二个 MetalContext 重复编译/传权重；
   正确形态是**单 context 双 MTLCommandQueue**（resident 共享），用 MTLEvent 管
   跨队列依赖。仅对小 dispatch（占不满核）的负载有叠加收益。

### 实施顺序（每步独立可验证，完成后做音质确认）

| 步骤 | 内容 | 数值影响 | 预估收益 |
|---|---|---|---|
| 1 | **t 条件常量整表预计算**（adaLN wb / final 调制 / wavenet cond 并入 t 表） | 无（bit 级一致） | CFM 每步 -30 dispatch，~5-10% acoustic |
| 2 | **CFM fp16 激活**（MPS GEMM A/C 转 fp16）+ w1/w3 合并单 GEMM | 有（等价采样漂移，需听感） | acoustic 1.3-1.8x |
| 3 | **GPT fp16 KV cache + ICB**（每 token 图录制一次重复提交） | KV fp16 微小；ICB 无 | GPT 1.5-2x |
| 4 | **单 context 双队列段间重叠**（修正版流水线） | 无 | 长文本墙钟 1.3-1.5x |
| 5 | **CPU/AMX 整解码循环**（Accelerate 重写 GPT decode） | 微小（CPU fp 顺序） | GPT 再 2-3x（上限最高，工程量最大） |
| 6 | **ANE BigVGAN**（CoreML 离线转换 + IOSurface 交换） | fp16 ANE 精度 | 释放 GPU ~0.4s/段 + 三硬件流水 |
| 7 | int8 权重（per-channel，kernel 内反量化） | 中（argmax 可能翻转） | GPT 带宽再减半，最后做质量评估 |

> 预期：步骤 1-3 落地后长句 RTF ~1.0x（实时线）；4-6 奔 0.5-0.7x。
