# 保持 TTS 精度的推理加速计划

日期：2026-09-07

状态：P0–P2 用户听感验证通过，P0 的跨后端数值门槛未通过记录保留。P2、P3 均未证明稳定提速，实验路径默认关闭；P3 已完成逐位一致和性能验证，尚未收到本轮听感判定。用户随后要求分析更激进方案，见第 15 节；该节为候选分析，尚未实施。

最新变更：用户明确要求 M3 Ultra 默认开启 P0，已调整设备选择策略，记录见第 16 节。第 15 节所述未默认启用是此次变更前的状态。

执行规则（用户要求）：分步执行，每一步用同一句话、同一音色、相同参数生成优化前后两段音频；必须等用户听感判定后再进入下一步。听感结论与数值精度结论分别记录。

## 1. 目标与约束

在当前模型和原生 Metal 运行时基础上，降低 TTS 生成耗时，优先消除重复计算和运行时开销。

- 保持模型权重、现有计算精度、CFM 步数、CFG、prompt 长度、文本分段和采样参数不变。
- 不以减少生成内容、改变音频长度或增加静音来改善 RTF。
- 不引入新的量化，不将激活或采样概率计算降精度。
- 性能和精度分别验收：跑得更快不代表音质已经达标。
- 区分两类候选：可要求逐位一致的编排优化，以及可能改变浮点累加顺序、必须额外验证音质的算子优化。

本计划不承诺未经测量的端到端加速倍数。

## 2. 当前证据与基线

### 已有优化

当前代码已经包含权重常驻、GPU KV cache、greedy GPT ICB、CFM/BigVGAN 单次提交路径、音色条件 LRU 缓存及常驻 MetalContext。这些能力不再列为待新增优化。

段间流水线已有 `MIT2_SEGMENT_PIPELINE=1` 实验入口，但源码记录它在单 GPU 上曾出现负收益，不能直接作为默认提速方案。

### 本次阶段实测

环境：Apple M3 Ultra，使用仓库已有 `build/mtts` 和 `bin` 模型。此次没有重新构建二进制，因此实施前应从当前源码构建 Release 版本并重新建立基线。

BigVGAN 使用同一组合成 mel，430 帧，输出 110080 个采样点，约 4.99 秒音频。每个进程先预热一次，再测量多次调用。

| 路径 | 首轮每次耗时，2 次均值 | 复测每次耗时，3 次均值 | 复测声码器 RTF |
| --- | ---: | ---: | ---: |
| 默认/显式关闭备用后端 | 1.75567 秒 | 1.75608 秒 | 0.351758 |
| `MIT2_BIGVGAN_IM2COL=1` | 0.616599 秒 | 0.616122 秒 | 0.123415 |

该形状下声码器阶段约加速 2.85 倍，耗时减少约 65%，每次节省约 1.14 秒。以上不是端到端结果，也不构成波形精度或听感验证。

CFM 合成输入基准：850 帧、16 步，预热后 2 次均值为 0.635582 秒，每步约 39.72 毫秒。此用例与 BigVGAN 用例独立，不能相加作为一次真实 TTS 的阶段耗时。

复现命令：

```bash
./build/mtts --bench-cfm bin 850 16 2
MIT2_BIGVGAN_IM2COL=0 ./build/mtts --bench-bigvgan bin 430 3
MIT2_BIGVGAN_IM2COL=1 ./build/mtts --bench-bigvgan bin 430 3
```

GPU 基准顺序运行，避免多个实验竞争 GPU。

## 3. 执行顺序

| 顺序 | 工作项 | 预期价值 | 精度要求 |
| --- | --- | --- | --- |
| P0 | 验证 BigVGAN 备用卷积后端并调整选择策略 | 已有明确阶段速度证据，潜在收益最大 | 保持 dtype；累加顺序变化需波形和音质验证 |
| P1 | 将 CFM 条件投影移出步进循环 | 小改动、确定消除重复计算 | 以逐位一致为目标 |
| P2 | sampling GPT 复用 ICB，保留 CPU 采样 | 减少采样路径逐 token 的命令编码开销 | 固定 seed 的 codes 和 WAV 一致 |
| P3 | GPT LayerNorm/GEMV 重复计算微基准 | 可能减少大量重复归一化 | 保持归约算法，先验证数值再评估速度 |

若业务只使用 greedy，P2 不产生直接收益，可跳过。若必须严格保持 WAV 逐位一致，则优先实施 P1/P2；P0 只有通过逐位一致检查才能纳入该严格范围。

## 4. P0：BigVGAN 卷积后端

### 问题与位置

- `runtime/metal_context.mm`：`bigvgan_im2col_enabled_for_device()`。
- `runtime/metal_context.mm`：`conv1d_dilated_same_f32_pass()`。
- `runtime/impl/dit_cfm_bigvgan.cpp`：`run_bigvgan_vocoder_metal_single_pass()` 和 `run_bench_bigvgan()`。

初始默认策略只在 M1 Max 启用备用后端，M3 Ultra 默认关闭；第 16 节已按用户要求增加 M3 Ultra 默认启用。虽然开关名为 im2col，当前残差卷积备用实现实际按卷积核位置拆成多个 MPS GEMM 并累加。

### 实施步骤

1. 从当前源码构建 Release 二进制，复现两条路径的性能差异。
2. 为同一真实 mel 输入导出默认路径和备用路径的 float32 waveform，保证比较发生在 PCM 量化之前。
3. 覆盖短、中、长音频及多个音色，测量最大绝对误差、RMSE、频谱差异、有限值和削波情况。
4. 使用已有 waveform golden 和基线音频验证，并检查发音、音色、齿音、高频伪影和稳定性。
5. 精度通过后，评估 M3 Ultra 默认启用；若不同层或长度表现不同，按通道数、卷积核、dilation 和序列长度选择后端。
6. 保留环境变量覆盖和原有后端，便于复现与回退。

### 验收

- GPT codes、CFM mel、音频采样点数不变。
- 不改变权重与激活 dtype。
- 若 waveform 非逐位一致，必须通过现有精度门槛和音质对照，不能仅凭相关系数宣布无损。
- 同时报告声码器阶段和真实请求端到端收益；检查不同长度的性能回退和内存增长。

## 5. P1：CFM 不变条件投影外提

### 问题与位置

`runtime/impl/dit_cfm_bigvgan.cpp` 的 `run_cfm_euler_metal_single_pass()` 在每个 diffusion step 内重复调用 `cond_projection(cond)`。同一次合成中，其输入、权重和输出形状不变，且 `cond_proj_slot` 已是持久 workspace 槽位。

### 实施步骤

1. 将条件投影编码移到 step 循环之前，只执行一次。
2. 保持原有算子、输入、矩阵形状、dtype 和输出槽位。
3. 检查 scratch 重置和后续操作不会覆盖投影输出，并保留所需依赖屏障。
4. 在 12、16、25 步及不同输入长度上比较 CFM 输出和耗时。

### 验收

- 16 步时该投影从 16 次减至 1 次；不能将此描述为 CFM 整体加速 16 倍。
- 每步状态、最终 mel 和 WAV 以逐位一致为目标。
- 不跨请求错误复用不同 condition；本项仅在单次合成内部复用。
- 报告真实减少的阶段耗时，不以算子次数代替性能测量。

## 6. P2：sampling GPT 复用 ICB

### 问题与位置

`runtime/impl/gpt.cpp` 的 `run_gpt_kv_greedy_metal()` 仅在非 sampling 模式进入现有 ICB 路径。sampling 每 token 仍重新编码 GPU 计算、同步读取 logits，再执行 CPU 采样。

### 实施步骤

1. 录制可复用的单 token Transformer 和 logits 计算图。
2. 每步更新 token、位置和 KV 状态，复用 ICB，继续将 logits 交给现有 CPU 采样函数。
3. 保留 temperature、top-k、top-p、repetition penalty、排序平局处理和 SplitMix64 随机数消耗顺序。
4. 分别统计 CPU 编码、GPU 执行、等待、logits 读取和 CPU 采样耗时，确认收益来源。

### 验收

- 固定输入和多个 seed，对比每步 logits、处理后 logits、codes、停止位置和最终 WAV。
- 覆盖连续请求、不同长度及 KV 容量变化，检查 ICB 资源绑定是否需要更新。
- 保留 CPU 采样意味着逐 token 同步仍在；不能声称已获得 greedy 的 8-token 批次收益。

本阶段不将采样搬到 GPU。当前采样使用 double 概率累计，改写为 GPU 浮点运算可能改变固定 seed 的结果，不属于已证明等价的优化。

## 7. P3：GPT LayerNorm/GEMV 微基准

### 问题与位置

`metal/kernels.metal` 的 `mit2_gpt_fused_gemv_f16w_f32` 在每个 threadgroup 内重新计算同一输入的 LayerNorm。当前每组处理 8 个输出通道：

- QKV 的 3840 个输出，对应约 480 次相同归一化。
- FFN 扩展的 5120 个输出，对应约 640 次相同归一化。
- 上述工作在每个 token 的 24 层中重复。

### 实施步骤

1. 增加实验路径：同一输入归一化一次，保存结果，供后续 GEMV 使用。
2. 保持现有均值、方差归约顺序和计算精度，避免直接换用数值行为不同的归一化 kernel。
3. 在 ICB 中测量额外 dispatch 和中间读写是否抵消计算节省。
4. 对比完整 token 解码，而非只比较孤立 LayerNorm。

### 验收

- logits/codes 精度门槛先通过，再评估速度。
- 覆盖不同 KV 长度，报告 token 延迟及端到端收益。
- 无稳定收益则保留当前融合实现；不因减少理论计算量而默认合入。

## 8. 统一验证与报告

每项独立修改、独立比较，最后再测试组合效果。

- 记录源码 revision、构建参数、二进制、模型和音色标识、硬件、系统版本及相关环境变量。
- 固定文本、seed、分段、prompt、CFM/CFG 和采样参数。
- 区分进程冷启动、首请求与预热后的稳态请求。
- 使用短句、中等文本和多段长文本，覆盖多个音色；涉及解码的修改同时覆盖 greedy 和 sampling。
- 分开记录 frontend、GPT、condition、CFM、BigVGAN 和请求墙钟时间，以及生成 codes 数、mel 帧数、音频时长和 RTF。
- 测量峰值内存、提交次数及重复运行稳定性；正式结论使用多轮结果和中位数，必要时增加样本量观察尾延迟。
- 算子优化比较 float32 中间输出，避免 PCM 量化掩盖误差。
- 优先复用已有 golden、固定 seed 和 benchmark 工具；补充验证必须实际覆盖候选生产路径。

注意：当前 `--bench-bigvgan-breakdown` 使用 `multi_submit_readback` 路径，不能直接代表默认 single-pass 路径的层耗时。若需定位生产路径热点，应为 single-pass 增加对应测量。

每项报告包含：修改内容、精度结果、阶段耗时、端到端耗时、内存影响、是否接受，以及回退方式。

## 9. 本轮不实施的方向

- 减少 CFM 步数、截断参考 prompt、改变文本分段或增加量化：可能改变音质、韵律或计算精度。
- 直接开启段间流水线：已有负收益记录，需独立重测资源争用，不能假设并发就更快。
- 删除 GPT latent forward：虽然存在生成后重算，但生成位置编码有特殊处理，未证明解码隐藏状态可直接替代。
- 跨 CFM 步缓存完整 prompt 的 Transformer 隐藏状态：完整注意力上下文随生成状态变化，不能仅因原始 prompt 不变就认为其深层结果不变。
- 把已有缓存和常驻优化重新列为新增工作。

## 10. 进度清单

- [x] 检查当前推理路径与已落地优化。
- [x] 使用现有二进制测量 CFM 阶段基准。
- [x] 使用现有二进制复测 BigVGAN 两种后端的阶段速度。
- [x] 从当前源码构建 Release 并建立可追溯基线。
- [ ] P0：完成 BigVGAN 真实输入精度验证和后端选择评估。
- [x] P1：实现 CFM 条件投影外提并验证本句逐位一致；用户听感验证通过。
- [x] P2：实现保留 CPU 采样的 ICB 路径并验证确定性；用户听感通过，但未通过性能验收，默认关闭。
- [x] P3：完成 LayerNorm/GEMV 拆分实验、逐步采样对比及完整解码性能验证；无性能收益，默认关闭，等待用户试听。
- [ ] 完成组合优化的端到端精度、性能与内存回归。

## 11. P0 首轮执行记录：用户听感通过，数值门槛未通过

日期：2026-09-07。源码基线：`eda855b2e9d7cfaa4269380f42880d41b060c4d2`，使用本次重新构建的 Release 二进制。

本轮复用已有备用卷积后端，仅增加默认关闭的 `MIT2_TTS_DUMP_AUDIO_TENSORS=1` 诊断开关，保存声码器输入 mel 和 PCM 转换前的 float32 waveform。未修改自动后端选择，未执行 P1。

- 文本：今天的天气不错，我们去划船吧。微风吹过湖面，远处传来孩子们的笑声。
- 音色：本地“琴”，`voices/bundles/voice_6546b24e17b0d1.pt`。
- CFM：16 步；CFG：0.7；完整默认 prompt；原有分段逻辑。
- GPT：sampling，seed 20240605，temperature 1.0，top-k 30，top-p 0.8，repetition penalty 10。
- A：`MIT2_BIGVGAN_IM2COL=0`；B：`MIT2_BIGVGAN_IM2COL=1`。
- 两条路径各在独立常驻进程中运行一次同文预热，再运行一次正式请求。耗时包含诊断文件写入，仅为本句首轮结果，不是多轮统计结论。

| 项目 | A：原后端 | B：备用后端 |
| --- | ---: | ---: |
| 音频时长 | 6.48998 秒 | 6.48998 秒 |
| 热请求墙钟 | 5.33143 秒 | 3.84117 秒 |
| 热请求 RTF | 0.821486 | 0.591862 |
| 首请求墙钟 | 12.6887 秒 | 11.0485 秒 |
| 削波采样数 | 0 | 0 |

热请求耗时减少约 27.95%。text IDs、GPT codes、condition、noise 和生成 mel 均逐位一致；每条路径自身的冷、热两次 waveform 也逐位一致。

跨后端 float32 waveform 不逐位一致：

- 最大绝对误差：0.01463585，超过已有 waveform golden 门槛 0.001。
- RMSE：0.00012940；波形差异 SNR：60.11 dB。
- 143104 个采样点中，73 个绝对差异超过 0.001。
- 最大差异位于约 5.385 秒。

**判定：速度改善得到本句验证，但严格数值验收未通过，不能标为无损或精度通过。维持默认路径，等待用户试听；根据反馈继续修正 P0 或跳过，未经用户判定不进入 P1。**

后续用户反馈：**“听觉验证通过”**。据此进入 P1，对比双方均显式使用已经试听认可的 `MIT2_BIGVGAN_IM2COL=1`，以隔离 P1 的增量影响。本反馈是本句听感结论，不修改历史数值结果，也不等同于多音色、多长度的全面精度验收。自动后端选择仍未更改。

本轮产物（位于 Git 忽略的 artifacts 目录）：

- [A：优化前音频](../artifacts/tts-optimization/p0/before.wav)
- [B：候选优化后音频](../artifacts/tts-optimization/p0/after.wav)
- [数值和耗时结果](../artifacts/tts-optimization/p0/metrics.json)
- [配置及二进制 SHA-256](../artifacts/tts-optimization/p0/metadata.json)
- [复现脚本](../artifacts/tts-optimization/p0/run_comparison.py)

同目录保存测试二进制 `mtts`、诊断改动 `source.patch`、各阶段日志和 float32 张量。复现命令：`python3 artifacts/tts-optimization/p0/run_comparison.py`，会覆盖本轮同名产物。

## 12. P1 执行记录：本句逐位一致，用户听感通过

改动：`run_cfm_euler_metal_single_pass()` 在 Euler 循环前计算一次条件投影，输出保存在原有 `cond_proj_slot` 中。循环内继续使用相同的输入合并、Transformer、Wavenet 和 Euler 更新。未更改算子、形状、dtype、步数或 scratch 分配布局，保留 Metal helper 自带的依赖屏障。

本轮 A/B：A 使用 P0 保存的二进制；B 使用重新构建的 Release 二进制。两者都显式启用已经试听认可的 BigVGAN 备用后端，其余文本、音色和参数与 P0 完全相同。每条路径运行一次同文预热和一次正式合成。

| 项目 | A：P1 优化前 | B：P1 优化后 |
| --- | ---: | ---: |
| 音频时长 | 6.48998 秒 | 6.48998 秒 |
| 热请求墙钟 | 3.81777 秒 | 3.81000 秒 |
| 热请求 RTF | 0.588256 | 0.587060 |
| 浮点波形最大差异 | — | 0 |
| WAV 文件逐位一致 | — | 是 |

text IDs、GPT codes、condition、noise、生成 mel 和 float32 waveform 均逐位一致；两条路径各自冷、热输出也一致。A 的波形与用户通过听感的 P0 B 波形相同。

两份 WAV 的 SHA-256 均为 `2bab7ba8353eb2ec2b6549e714d17899b4072ab072dc79b3f840c6b1a0dbd7eb`。

独立 CFM 阶段基准关闭诊断导出，每个进程先预热一次：

| 总帧数 | 步数 | 每组测量次数 | A 每次耗时 | B 每次耗时 |
| --- | ---: | ---: | ---: | ---: |
| 128 | 12 | 3 | 100.020 毫秒 | 99.270 毫秒 |
| 850 | 16 | 8，A→B | 636.828 毫秒 | 635.022 毫秒 |
| 850 | 16 | 8，B→A | 636.182 毫秒 | 634.471 毫秒 |
| 850 | 25 | 3 | 1001.620 毫秒 | 989.937 毫秒 |

**结论：本句 16 步的输出逐位一致；阶段基准仅显示小幅收益。850 帧、16 步两轮约节省 1.7–1.8 毫秒，约 0.27%–0.28%。本句端到端差约 7.8 毫秒，接近测量波动范围，不能宣称显著提速。** 12/25 步表格是性能结果，不代表这些配置已完成逐位一致验证。

Release 构建和 `git diff --check` 通过。用户随后反馈 **“通过，继续”**，据此进入 P2。

本轮产物：

- [A：P1 优化前音频](../artifacts/tts-optimization/p1/before.wav)
- [B：P1 优化后音频](../artifacts/tts-optimization/p1/after.wav)
- [精度与请求耗时](../artifacts/tts-optimization/p1/metrics.json)
- [CFM 多轮基准](../artifacts/tts-optimization/p1/cfm_benchmark.json)
- [配置及二进制哈希](../artifacts/tts-optimization/p1/metadata.json)
- [音频复现脚本](../artifacts/tts-optimization/p1/run_comparison.py)
- [CFM 基准复现脚本](../artifacts/tts-optimization/p1/bench_cfm.py)

同目录保留 `mtts_after` 和源码差异 `source.patch`；前置基线二进制位于 P0 产物目录。复现脚本会覆盖同名产物。

## 13. P2 执行记录：输出一致，但未证明提速，默认关闭

基线 revision：`9ff210f`（包含 P1 和诊断张量导出）。本轮只实现采样模式的 ICB，不执行 P3。

### 实现

- greedy 和 sampling 共用 Transformer 图录制流程，以解码模式区分图缓存。
- sampling 每次执行一个 token，不录制 greedy 的 argmax、token 记录或状态推进操作。
- sampling 的 mel head 继续使用原 `linear_f32_pass` 对应的 GEMV kernel、数据类型和线程配置，避免直接替换成 greedy 的融合 GEMV。
- CPU 继续负责原有 top-k/top-p、temperature、repetition penalty、double 概率累计及 SplitMix64 随机数。
- 保留原位置编码规则，每次重放更新 token、KV token 数和位置。
- KV 分配布局变化时继续使录制图失效；切换 greedy/sampling 时重录。
- 录制结束时去除重复的只读资源声明，避免每个 token 反复声明相同资源。
- 由于性能没有稳定改善，**最终源码默认关闭 sampling ICB**；`MIT2_GPT_SAMPLED_ICB=1` 显式启用，`MIT2_GPT_ICB=0` 仍可关闭 ICB 总路径。

### 精度验证

新增 `--test-gpt-sampled-icb-parity BUNDLE CONDS TEXT_IDS`，在同一进程内比较旧 pass 与 sampling ICB：

- 3 个 seed，4 组配置，分别生成最多 16/32/48/64 个 token。
- 检查每步原始 logits 和采样处理后的 logits 逐位一致，原始 logits 全部有限。
- 检查 codes、停止位置和重复生成结果。
- 覆盖完整/较短文本前缀、两种位置编码模式、top-k/top-p/repetition penalty 的启用与禁用。
- 覆盖 greedy → sampling → greedy 切换及 KV 容量从 2048 增长到 3072。

4 组测试全部通过。最终默认关闭版本使用以下命令显式测试候选路径：

```bash
MIT2_GPT_SAMPLED_ICB=1 ./build/mtts --test-gpt-sampled-icb-parity \
  bin \
  artifacts/tts-optimization/p1/after.wav.conds.f32 \
  artifacts/tts-optimization/p1/after.wav.text_ids.u32
```

整句音频沿用 P0/P1 的文本、“琴”音色、16 步 CFM 和全部采样参数。A 使用 P1 二进制，B 使用 P2 候选；两者均启用已试听认可的 BigVGAN 备用后端。

text IDs、GPT codes、condition、noise、mel、float32 waveform 及 WAV 文件全部逐位一致。输出 325 个有效 codes、559 帧 mel、143104 个采样点（6.48998 秒）。WAV SHA-256 仍为 `2bab7ba8353eb2ec2b6549e714d17899b4072ab072dc79b3f840c6b1a0dbd7eb`，与已接受的 P1 相同。

### 性能验证

第一次单次对比显示小幅改善，但复测结果接近波动范围，因此另行关闭诊断导出，在每个独立常驻进程内先预热一次，再连续测量 4 次；本轮顺序为 B→A。

| 项目 | A：原采样 pass | B：sampling ICB |
| --- | ---: | ---: |
| 热请求 1 | 3.75782 秒 | 3.83898 秒 |
| 热请求 2 | 3.75166 秒 | 3.80402 秒 |
| 热请求 3 | 3.75423 秒 | 3.80553 秒 |
| 热请求 4 | 3.77416 秒 | 3.81187 秒 |
| 请求墙钟中位数 | 3.756025 秒 | 3.808700 秒 |
| GPT 阶段中位数 | 1.948955 秒 | 1.984220 秒 |
| 每请求 GPT command buffers | 329 | 329 |

B 的请求墙钟中位数约慢 1.40%，GPT 阶段约慢 1.81%。上述多次请求的 WAV 均与基线逐位一致。

**判定：P2 精度通过，但当前 M3 Ultra 上未证明稳定提速，不能作为有效加速默认启用。CPU 采样仍要求逐 token 同步，提交数没有减少。保留可选实验实现，默认走原采样 pass；不放宽精度、不改变采样方法来追求速度。** 尚未独立拆出 CPU 编码/采样耗时，因此不把上述总耗时差异全部归因于某个 CPU 环节。

用户随后反馈 **“通过，继续”**，P2 听感通过，据此执行 P3。P2 未通过性能验收的结论不变，正常采样仍使用原 pass。

### 产物

- [A：P2 优化前音频](../artifacts/tts-optimization/p2/before.wav)
- [B：P2 候选音频](../artifacts/tts-optimization/p2/after.wav)
- [逐步采样一致性测试](../artifacts/tts-optimization/p2/final_icb_parity.json)
- [整句精度与单次耗时](../artifacts/tts-optimization/p2/metrics.json)
- [4 次热请求性能结果](../artifacts/tts-optimization/p2/warm_benchmark.json)
- [配置与二进制标识](../artifacts/tts-optimization/p2/metadata.json)
- [最终构建的显式启用和默认回退检查](../artifacts/tts-optimization/p2/final_build_checks.json)
- [A/B 复现脚本](../artifacts/tts-optimization/p2/run_comparison.py)
- [热请求基准脚本](../artifacts/tts-optimization/p2/bench_warm.py)

目录保留性能测试时的候选二进制 `mtts_after`，以及最终默认关闭版本 `mtts_final`；源码差异保存为 `source.patch`。复现脚本会覆盖同名产物。

## 14. P3 执行记录：输出一致，两种路径均未提速，默认关闭

基线 revision：`cd2f4a1`（包含 P2 的可选 ICB 路径）。本轮只评估 LayerNorm/GEMV 拆分，不改变采样、CFM、prompt 或分段。

### 实现

- 新增 `mit2_gpt_layernorm_256_f32`，严格沿用原融合 kernel 的 256 线程归约、求和顺序、方差计算、float32 仿射变换。
- QKV 和 FFN 扩展投影的输入各归一化一次，将结果写入 float32 workspace，再交给原有 GEMV、GELU 和 residual 运算。
- 同时支持 pass 和 ICB；不使用现有 1024 线程通用 LayerNorm 代替原计算顺序。
- 切换拆分模式时使 ICB 失效，重录相应图；仅在启用拆分时增加所需 workspace 和 48 个 ICB command 容量。
- 每 token 的 24 层共增加 48 次独立归一化 dispatch。重复计算减少，但提交图中的 dispatch 和中间读写增加。
- 默认关闭，通过 `MIT2_GPT_SPLIT_LAYERNORM=1` 显式启用；设为 `0` 保留原融合实现。

### 精度验证

新增测试入口：

```bash
MIT2_GPT_SAMPLED_ICB=1 ./build/mtts --test-gpt-split-layernorm-parity \
  bin \
  artifacts/tts-optimization/p2/before.wav.conds.f32 \
  artifacts/tts-optimization/p2/before.wav.text_ids.u32
```

该测试在同一 context 中切换拆分模式，复用 P2 的测试夹具：3 个 seed、4 组采样配置、16/32/48/64 个 token、完整/较短文本前缀、两种位置编码模式、KV 增长及 greedy/sampling 切换。分别比较 pass 和 ICB；每步原始 logits、处理后 logits、codes 和停止位置全部一致，4 组测试通过。

同句音频继续使用“琴”音色、16 步 CFM 及原采样参数：

- 两条主对比路径均启用已通过听感的 BigVGAN 备用后端。
- 两者均关闭 P2 sampling ICB，只改变 P3 拆分开关。
- text IDs、codes、condition、noise、mel、float32 waveform 以及 WAV 文件逐位一致。
- 两条路径各自的冷、热音频一致，输出仍为 6.48998 秒。
- WAV SHA-256 仍为 `2bab7ba8353eb2ec2b6549e714d17899b4072ab072dc79b3f840c6b1a0dbd7eb`，与已接受的 P0 B、P1、P2 相同。

### 性能验证

使用完整 TTS 请求内的 GPT 阶段计时及请求墙钟时间，避免仅凭孤立归一化算子的计算量判断收益。各进程先预热一次，再测量 4 次；关闭诊断张量导出，顺序为 B→A。

| 执行路径 | A：原融合实现，墙钟中位数 | B：拆分实现，墙钟中位数 | A GPT 中位数 | B GPT 中位数 |
| --- | ---: | ---: | ---: | ---: |
| 正常 sampling pass | 3.813110 秒 | 3.854985 秒 | 1.985305 秒 | 2.027855 秒 |
| 实验 sampling ICB | 3.752255 秒 | 3.800490 秒 | 1.937100 秒 | 1.992110 秒 |

正常 pass 墙钟约慢 1.10%，ICB 墙钟约慢 1.29%。两组补测的所有 WAV 均与主对比基线逐位一致。不能跨这两组运行直接判定 ICB 优于 pass；它们用于各自组内比较拆分开关。

**判定：P3 已验证本句精度等价，但未证明性能收益。新增 dispatch 和中间存取使减少重复计算没有转化为完整解码提速。保留默认关闭的实验入口，继续使用原融合实现；不通过降低归约精度或放宽采样一致性追求速度。** 尚未独立分离每个 kernel 的时间，因此不把整个耗时差归因于某一个算子。

Release 构建及 `git diff --check` 通过。P3 等待用户试听；在该判定之前，不进行后续组合配置变更。

### 产物

- [A：P3 优化前音频](../artifacts/tts-optimization/p3/before.wav)
- [B：P3 拆分候选音频](../artifacts/tts-optimization/p3/after.wav)
- [逐步 logits 和采样对比](../artifacts/tts-optimization/p3/ln_parity.json)
- [整句精度及资源指标](../artifacts/tts-optimization/p3/metrics.json)
- [正常 sampling pass 热请求基准](../artifacts/tts-optimization/p3/warm_benchmark.json)
- [ICB 路径热请求基准](../artifacts/tts-optimization/p3/icb/warm_benchmark.json)
- [配置及二进制哈希](../artifacts/tts-optimization/p3/metadata.json)
- [音频复现脚本](../artifacts/tts-optimization/p3/run_comparison.py)
- [pass 基准脚本](../artifacts/tts-optimization/p3/bench_warm.py)
- [ICB 基准脚本](../artifacts/tts-optimization/p3/icb/bench_warm.py)

同目录保存测试二进制 `mtts_after` 和源码差异 `source.patch`；复现脚本会覆盖同名产物。

## 15. 更激进方案分析：先明确收益上限，再选择实验

日期：2026-09-07。用户反馈当前提速不明显，要求分析更激进的方法。本轮只核查源码、历史记录、研究资料并运行现有合成输入基准；没有修改推理代码、模型或实际生成参数。

### 15.1 当前速度没有明显提升的原因

1. **本节分析时，P0 的有效配置尚未成为 M3 Ultra 的源码默认值。** 当时 `runtime/metal_context.mm` 的 `bigvgan_im2col_enabled_for_device()` 只对 M1 Max 自动开启；M3 Ultra 需要显式设置 `MIT2_BIGVGAN_IM2COL=1`。P0 热请求从 5.33143 秒降至 3.84117 秒，耗时减少约 28%，是显式切换后端的结果。这里只确认源码策略，未检查用户正在使用的服务进程环境。随后用户要求默认开启，实施记录见第 16 节。
2. P1 消除条件投影重复计算，但本句端到端差异接近噪声。P2/P3 均未取得稳定收益，默认关闭正确；它们不能累计为有效加速。
3. **继续只减少 CPU 编码、缓存或拆分小算子，缺乏明显收益证据。** P3 正常路径 GPT 墙钟约 1.99 秒，其中已报告的 GPU elapsed 约 1.75 秒。两者差约 0.23 秒，不能全算作 CPU 采样，也不能把 GPU elapsed 当作纯算术时间；但已有证据不支持“仅把采样搬到 GPU 就让整句快一倍”。

后续实验应先核实服务实际后端，固定一个已接受的基线。需要分别报告“原默认配置 → P0”和“P0 → 新候选”的增量，避免重复计算收益。P0 本句听感已通过，但波形不是逐位一致，也未完成跨音色推广验证。

### 15.2 本机计算预算及新增基准

已开启 P0、关闭 P2/P3 的历史真实热请求：生成 6.48998 秒音频，墙钟中位数 3.813110 秒，GPT 1.985305 秒。GPT 占约 52%，因此要显著降低整句耗时，应同时处理 GPT 和声学生成。

本轮在当前 `build/mtts` 上顺序运行以下现有基准；每个进程预热一次，再测 3 次均值。没有并发 GPU 基准。

| 合成输入基准 | 每次耗时 | 用途 |
| --- | ---: | --- |
| CFM，1188 帧、16 步 | 0.938527 秒 | 与真实请求总序列长度相同 |
| CFM，1188 帧、8 步 | 0.470416 秒 | 单独估计减步的计算收益 |
| CFM，859 帧、8 步 | 0.323708 秒 | 估计减步并缩短参考序列后的计算量 |
| BigVGAN，559 帧，备用后端 | 0.769938 秒 | 与真实请求输出 mel 长度相同 |

这些不是新音频的端到端结果。尤其 `run_bench_cfm()` 的 prompt 长度固定为总帧数的 6/7，真实请求则为 629/1188；本轮只对齐总序列计算形状，没有完成真实 prompt 裁剪实验。不同进程的阶段均值也不能直接相加声称精确的真实请求分解。

复现命令：

```bash
./build/mtts --bench-cfm bin 1188 16 3
./build/mtts --bench-cfm bin 1188 8 3
./build/mtts --bench-cfm bin 859 8 3
MIT2_BIGVGAN_IM2COL=1 ./build/mtts --bench-bigvgan bin 559 3
```

二进制 SHA-256：`07a501f123864e337fbde7f6947faad18f33f9b118141106153e27ae72b76df7`。记录见 `artifacts/tts-optimization/aggressive-analysis/measurements.json`。

以历史 3.813110 秒为参照，仅替换对应阶段成本作粗估：8 步约节省 0.468 秒，约 12%；8 步加较短序列约节省 0.615 秒，约 16%。实际整句收益及听感必须另测。

### 15.3 精度要求需要分别讨论

- **固定 seed 下逐位一致：** 保留原模型与数值路径，主要考虑内核布局、权重组织和执行优化；本轮没有证据保证还能取得大幅提速。
- **保持原模型的目标采样分布：** 可研究带严格接受/拒绝校正的投机解码；不等于相同 seed 产生相同 codes 或 WAV。
- **允许数值和采样结果变化，以内容、音色、自然度无可感知退化为目标：** 才适合减步、量化、裁剪参考及蒸馏。它们是待验证候选，不能预先承诺“不影响精度”。

原计划第 1 节的限制继续适用于已实施的 P0–P3。下列候选明确列出需要放宽的条件，分析不等于已经批准这些候选的质量结论。

### 15.4 可选路线及在当前代码上的落点

| 路线 | 具体改变 | 收益依据及边界 | 音质与实现代价 |
| --- | --- | --- | --- |
| A：CFM 减步与非均匀时间表 | 先 16→12，再单独评估 8 步；校准时间节点以保留重要区间 | 8 步的阶段耗时约减半，整句粗估约省 12% | 不需重训，但会改变 mel；现有产品 CLI 仅允许 12–25，8 步须新增实验入口 |
| B：缩短声学参考序列 | 本句 prompt 629→约 300 帧，生成区仍保留 559 帧；保留完整参考提取的音色特征 | 总序列 1188→859，减少每个 CFM 步的工作；与 A 合用粗估整句省约 16% | 改变声学条件，可能影响音色和韵律；须同步裁剪对齐的 prompt mel/condition 并保留生成区 noise |
| C：GPT 矩阵内核优化及选择性量化 | 对实际 QKV/FFN 形状重做小规模 GEMV 基准；评估 W8，再选择性 W4 | GPT 约占 52%，是最大可攻阶段；若 GPT 实测加速 2 倍，整句粗估约省 26% | FP16 内核重排也可能改变归约顺序；量化会改变 logits，需校准与完整生成验证 |
| D：GPT 投机解码 | 小 draft 一次提出 4/8 个声学 token，原 24 层 GPT 批量验证 | 将多个串行 GEMV 转为较高复用的多行计算；收益取决于 draft 成本、接受率和验证延迟 | 严格校正可保持目标分布；当前没有兼容 draft，需训练/蒸馏及新增批量 KV 验证路径，工程量大 |
| E：少步 CFM 蒸馏 | 用现有模型作为 teacher，训练 4/2 步 student；进一步研究把 CFG 引导蒸馏进 student | 若 CFM 实测快 4 倍，按当前预算整句粗估省约 18% | 需要训练数据与训练流程，属于模型工作；直接减少到 2 步或关闭 CFG 不等价于蒸馏 |
| F：轻量声码器 | 训练/适配 Vocos、iSTFT 类结构或较小 BigVGAN student，替换当前六级上采样路径 | 当前备用 BigVGAN 同形状约 0.77 秒；若声码器快 4 倍，整句粗估省约 15% | 需要匹配当前 mel 定义、训练分布及原生 Metal 算子，可能改变高频、气声和齿音 |

表中的阶段加速倍数是计算整句收益的假设，不是本项目已实现的结果；A/B 的阶段时间是本轮实测，但对应真实语音质量尚未测量。

**A/B：近期最容易形成可试听结果。** 当前 CFM 已把条件分支与无条件分支合为 batch=2，16 次 Euler 步仍逐次运行整个 DiT。修改时间表还需要同步更新 timestep embedding、调制表及缓存键。不能直接照搬其他模型的节点；[EPSS 研究](https://arxiv.org/abs/2505.19931) 在 F5-TTS/E2 TTS 上探索了无需训练的非均匀减步策略，支持这一实验方向，但论文的 RTX 3090 性能和质量结论不代表 IndexTTS/M3 Ultra 已验证。

CFM 配置为非因果注意力：参考输入虽然固定，中间 hidden states 仍会受每步变化的生成区影响。因此“缓存完整 prompt 的全部中间 K/V，后续不再计算”不是严格等价优化。B 需要真正减少输入上下文并接受质量验证。

**C：建议作为追求明显总耗时下降的主攻方向。** 当前 GPT 是 24 层、宽度 1280、FFN 宽度 5120，权重已经为 FP16，激活和主要累加为 FP32。现有融合 kernel 已有 `half4` 读取和 SIMD 归约，不能把已有向量化包装成新优化。先测真实形状下的权重布局、每组输出行数、寄存器占用、内存吞吐与完整 decode；P3 的负结果只否定那种拆分方式，不能证明所有 GPT 内核优化均无价值。

如允许数值变化，再做仅量化权重的 W8/W4 GEMV：从 QKV/FFN 开始，先保持 LayerNorm、激活、采样算法和输出 head 原路径。需要在运算中读取压缩权重并反量化；启动时解压回 FP16 不能持续减少 decode 读权重的流量。先完成算子微基准和量化误差校准，再接入 GPT，避免先移植整个框架。Apple 的 [MLX 官方介绍](https://developer.apple.com/videos/play/wwdc2025/315/) 提供量化矩阵乘与按层量化的实现参考。当前数据尚不能区分带宽、GPU 调度和计算占比，不能承诺 4 bit 带来 4 倍速度，也不能将模型体积下降等同于生成速度提升。

**D：如果更重视保持原模型分布，投机解码比直接替换小 GPT 更合适。** draft 必须兼容本项目的 8194 维声学 token 词表及文本/音色条件，普通文本 LLM 不能直接担当。原 GPT 需要新增带历史 KV 的多 token 因果验证、接受后的 KV 截断/回滚，以及与当前位置编码一致的路径；已有 full-forward/prefill 可以复用部分算子，但不是现成验证器。

接受/拒绝校正必须基于应用 temperature、top-k、top-p、repetition penalty 后的分布，处理拒绝后的残差采样及 EOS。算法层面的目标分布保持见 [Speculative Decoding 原论文](https://proceedings.mlr.press/v202/leviathan23a.html)。批量矩阵计算的浮点差异仍须检查，固定 seed 的随机数消耗也会改变。最小可行验证应先测真实语音上的接受长度与下式，而非直接建设完整服务：

`(draft 提案耗时 + teacher 批量验证及校正耗时) / 平均实际推进 token 数 < 原逐 token 解码耗时`

同一块 GPU 上的 draft 计算要计入成本。没有高接受率且足够快的 draft 时，该方案可能负收益。

**E/F：更大的潜力需要模型训练配合。** [FlashSpeech](https://arxiv.org/abs/2404.14700) 展示了通过一致性模型实现极少步语音生成的研究方向，但不是本仓库可直接启用的 IndexTTS 蒸馏权重。应从当前 teacher 的真实生成分布训练 student，并单独验证内容、音色、韵律和长句稳定性。

[Vocos](https://arxiv.org/abs/2306.00814) 通过预测 Fourier 系数并用逆变换重建音频，减少时域逐级上采样。迁移时必须匹配本项目声学输出的 22050 Hz、80 mel、hop 256、FFT 1024，以及 mel 滤波和数值归一化；不可误用 `dataset` 部分的 24 kHz/100 mel 配置。相同 mel 通道数也不等于预训练声码器可直接替换，应使用本模型实际输出 mel 做适配与听感验证。

### 15.5 建议的新实验顺序及停止条件

1. **先校准真实产品基线。** 检查实际服务是否使用 P0 后端及常驻热路径；分别记录首请求、热请求、GPT 和声学阶段。模型加载、参考音色处理与热推理不能混为一个指标。
2. **先做 A 的 16→12 单变量试听。** 通过后再独立评估 8 步及非均匀时间表；B 作为另一独立实验，不能首次同时减步、裁剪参考和量化。
3. **把主要内核投入转到 C。** 在当前 FP16 路径与 W8/W4 候选间，用真实 GPT 形状先检验可达到的收益。建议以完整 GPT 至少约 20% 的稳定提升作为继续扩大实现的筛选目标；整句收益必须另测。若没有可观的完整 decode 收益，停止该候选，不继续堆叠小改动。
4. **要保持目标采样分布时，选择 D 的可行性实验。** 先解决 draft 来源和接受率，再决定是否投入完整实现；不能把当前不存在的 draft 当作已具备的条件。
5. **目标接近整句快 2 倍时，再评估组合或训练。** 按当前预算，假设 GPT 快 2 倍、CFM 快 4 倍，整句约为 `3.813 - 1.985/2 - 0.939×3/4 ≈ 2.12 秒`，约快 1.8 倍；仍需声码器等额外收益才接近 1.9 秒。不同方案作用于同一阶段时不能重复叠加收益。

继续遵守用户的逐步试听规则：每一步交付相同文本、同音色、可比参数的前后两段音频，收到听感判定后才推进下一个实施步骤。对减步/声码器先复用相同 codes 和噪声以隔离差异；GPT 变化造成 codes 或时长变化时，应同时报告绝对耗时、token 数、音频长度和 RTF，不能用少说内容或提前 EOS 充当加速。模型分布发生变化后，用波形逐位相等作唯一门槛已不适用；需补充多 seed、多音色、数字/中英混读及长句的内容和听感检查。

流式输出可以缩短“第一次听见声音”的等待，但不保证整句完成更快。当前 CFM 是非因果模型，真正的声学分块需要边界/上下文设计甚至训练，不能简单拆循环宣称精度不变。同 GPU 增加线程、段间并发主要影响吞吐，已有负收益记录；重写到 MLX/Core ML、启用 ANE 或直接换模型也应先通过小规模实测证明价值，暂不列为默认下一步。

## 16. 按用户要求为 M3 Ultra 默认开启 P0

日期：2026-09-07。用户明确要求“默认为 M3 Ultra 开启 P0”。

### 变更

`runtime/metal_context.mm` 的 `bigvgan_im2col_enabled_for_device()` 现在对 M1 Max 和 M3 Ultra 自动选择 P0 的 MPS tap-GEMM 后端。显式环境变量仍优先：`MIT2_BIGVGAN_IM2COL=0` 关闭，`1` 开启。已完成 Release 构建，更新 `build/mtts`；已经运行的旧服务进程需要重启后才会使用新二进制和默认策略。

### 验证

使用与前几轮相同的句子和“琴”音色，CFM 16 步、CFG 0.7、seed 20240605 及原采样参数。前后两组均清除全部 `MIT2_*` 环境变量，以验证设备默认策略；关闭诊断音频张量导出。两组顺序运行，各先生成一次，再测量 3 次热请求。

| 配置 | 首请求墙钟 | 3 次热请求墙钟 | 热请求中位数 |
| --- | ---: | --- | ---: |
| 修改前二进制，无环境覆盖 | 12.5806 秒 | 5.32867 / 5.32624 / 5.32649 秒 | 5.32649 秒 |
| 修改后二进制，无环境覆盖 | 10.9286 秒 | 3.82894 / 3.81002 / 3.81871 秒 | 3.81871 秒 |

本句热请求耗时减少 **28.31%**。两组输出均为 143104 个采样点、22050 Hz、6.48998 秒，各组冷/热生成的 WAV 一致。

- 新默认 WAV 与此前已接受的 P0 输出逐位一致，SHA-256 为 `2bab7ba8353eb2ec2b6549e714d17899b4072ab072dc79b3f840c6b1a0dbd7eb`。
- 新二进制显式设置 `MIT2_BIGVGAN_IM2COL=0` 后，WAV 与修改前默认输出逐位一致，确认覆盖开关有效。
- 前后 codes、condition、noise、text IDs 一致；前后 WAV 不相同，仍属于已记录的 P0 跨后端数值差异，未将其宣称为跨后端逐位等价。
- Release 构建、实际默认路径生成、关闭覆盖生成和 `git diff --check` 通过。本轮仅落实已授权的默认后端选择，未开始减步、量化或其他候选实验。

音频与复现记录保存在 `artifacts/tts-optimization/p0-default/`：`before.wav`、`after.wav`、`override_off.wav`、`results.json`、`metadata.json`、`verify_default.py`、前后二进制及 `source.patch`。保留同句两段音频供用户复听，不以新的听感结论覆盖此前记录。
