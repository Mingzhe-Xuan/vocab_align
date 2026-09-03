# Planner→Thinker Training-free Soft-Token Transport（STT）实验计划

## 1. 目标

在保留 C2C 的协作任务、OpenHermes 训练语料、下游任务和评测协议的前提下，实现训练免费的 Planner→Thinker 协作：sender A 是 planner，receiver B 是 thinker；同一题目分别进入两侧的原生提示词，两侧均显式开启 CoT。A 先生成 planner think，再把 `sender prompt + sender think` 的全部上下文 hidden states 通过固定词表传输矩阵对齐成 B embedding，并前置拼接到 B 自己显式编码的题目提示词；B 随后按 thinker 提示词继续思考并回答。主模型对仍为 tokenizer 和架构差异明显的 Qwen3→Mistral-Nemo。

原 C2C：

$$
(K^A,V^A) \xrightarrow{\text{trained projector}} (K^B,V^B).
$$

STT 不训练 projector。令 $P_A(q)$、$P_B(q)$ 为分别包含同一题目 $q$ 的 planner/thinker 原生 chat prompt，$r_A$ 为 A 显式 CoT 生成的 planner think，$h_t^A$ 为完整 sender context $P_A(q)\Vert r_A$ 在位置 $t$ 的最终 hidden state。通过 A 的 LM head 与固定、稀疏、列随机 $T_{A\to B}$ 对齐：

$$
p_t^A=\operatorname{softmax}(\operatorname{LMHead}_A(h_t^A)/\tau),\qquad
e_t^B=W_{\mathrm{in}}^B T_{A\to B}p_t^A.
$$

receiver 的 prefill 输入严格为

$$
[e_1^B,\ldots,e_n^B]\ \Vert\ W_{\mathrm{in}}^B\operatorname{Tok}_B(P_B(q)),
$$

即概念顺序 `sender prompt + sender think + receiver prompt`。前两段是 aligned continuous embeddings，后一段是 B 自己 tokenizer 的显式文本 embeddings；B 再显式 CoT 生成最终答案。

问题：**training-corpus-fitted、但没有任何梯度训练的 soft-token transport，能否优于 receiver-only/T2T，并接近或补充可训练的 KV-C2C？**

## 2. 定义与边界

- A 为 sender/planner，B 为 receiver/thinker；所有对照固定相同的 A→B 方向。
- “training-free”指 A、B、$T$、adapter 均不经梯度更新；用训练语料做 byte-span 计数是无标签离线统计，必须报告为 *training-free, corpus-fitted transport*，而非无数据或 zero-shot mapping。
- STT 不再是 KV-C2C 的等价实现；它保留的是 C2C 的协作任务和模型设置，改变的是通信表示空间。
- 主协议传输 A 的完整上下文而非 prompt-only：A 先生成一次有限预算的 planner think，然后对 `sender prompt + sender think` 全位置做 exact STT；A 在 B 开始生成后不再同步。
- 题目必须同时出现在 $P_A(q)$ 和 $P_B(q)$ 中。禁止只让 A 看题，也禁止用 A 的 rendered prompt 替代 B 的原生 prompt。
- sender 与 receiver 的 system/user instruction 均须明确要求逐步思考；tokenizer 支持原生 thinking 开关时必须设置 `enable_thinking=true`，evaluator 的 `use_cot` 也固定为 `true`。
- 不使用 benchmark test 或 dev 数据拟合 $T$；不训练 LoRA、projector、蒸馏头或任何可学习桥接模块。

## 3. 固定设置与数据隔离

### 3.1 首个模型对

| 角色 | 模型 | 用途 |
|---|---|---|
| Source A | `Qwen/Qwen3-8B` | prompt 前向，提供 logits |
| Receiver B | `mistralai/Mistral-Nemo-Instruct-2407` | 接收 soft embedding，生成最终答案 |
| C2C 对照 | 同一 A/B | 使用为该模型对新训练的 projector checkpoint；不得复用旧 Qwen 小模型对 checkpoint |

该模型对的 tokenizer、special token、chat template、词表大小和模型架构均明显不同，作为主实验直接检验跨 tokenizer transport。开始构建 $T$ 前必须保存全词表审计：两侧词表大小、exact-byte 覆盖率、语料加权覆盖率、tokenization 长度比、special/control token 对照表，以及无法安全映射的 token 清单。不得假设共享 token ID 具有相同含义。

模型和 tokenizer revision 必须锁定到明确 commit。若尚无该 A/B 的 C2C checkpoint，C2C 结果标记为 `pending-new-projector-training`，不能用旧模型对的 C2C 分数代替；STT、R/S/T2T 可先独立完成预实验。

### 3.2 数据划分

沿用 `OpenHermesChatDataset` 50 万样本和 seed 42，先保存确定性的样本-id manifest：

| 分区 | 比例 | C2C 用途 | STT 用途 |
|---|---:|---|---|
| `transport_train` | 99% | projector SFT 训练 | 构建 $T$ 的无标签统计 |
| `transport_dev` | 1% | 原训练期验证 | 仅审计 $T$、选择预先限定的近似参数 |
| benchmark test | 不参与构建 | 下游评测 | 下游评测 |

构建 $T$ 时对同一条 message 的**原始内容**分别 tokenization，不对各自完整 chat template 的字符串做 span 统计。BOS/EOS/PAD、角色模板、控制符等必须通过显式规则表处理，避免模板差异污染对齐。

### 3.3 固定评测条件

所有方法固定相同的 benchmark 样本、canonical messages、few-shot、双侧 CoT、贪心解码以及独立的 sender/receiver `max_new_tokens`。A、B 分别使用已冻结的原生 chat template；逐题保存 canonical problem、sender/receiver rendered prompt、sender think 文本与 token IDs。固定每个模型的精度、device map、GPU、warm-up 和 offload 规则。每题保存 sender prompt/think/context 长度、aligned prefix 长度、receiver prompt/output 长度、各段 latency、峰值显存及逐题输出。

由于两个模型无法在当前 2 GB 本地 GPU 上同时驻留，本地主机只用于 tokenizer 对齐、$T$ 构建和短序列 CPU 数值 oracle。端到端质量实验使用能同时容纳 A/B 的 GPU 或明确的多 GPU device map；使用 CPU/disk offload 的运行只报告功能正确性，不进入正式 latency 主表。

T2T 与 Planner→Thinker STT 必须使用相同的 sender think 预算并报告实际通信 token 数；STT 额外执行完整 sender context forward 以取得所有 hidden-state alignment 输入，该成本必须单独计入，不得沿用旧 prompt-only latency。

## 4. 构建固定词表矩阵 $T_{A\to B}$

$T\in\mathbb{R}_+^{V_B\times V_A}$ 的列定义为

$$
T_{ji}=P(B\text{ token}=j\mid A\text{ token}=i),\qquad \sum_jT_{ji}=1.
$$

按如下优先级构建稀疏列：

1. **special-token 规则**：同功能 BOS/EOS/PAD/换行/角色标记 one-hot 映射；不可安全映射的控制 token 使用指定 fallback，禁止与普通 token 混合。
2. **exact-byte 规则**：规范化后的原始 bytes 相同的普通 token 优先 one-hot。
3. **byte-span 统计**：在 `transport_train` 的同一原始文本上取得两 tokenizer 的 byte offsets；每次 A token 与所有重叠 B token 按重叠 byte 长度累计 $M_{ji}$。实现必须流式处理，不构造稠密 $V_A\times V_B$ 矩阵。
4. **低频 fallback**：低计数或零覆盖列用共同外部 embedding 空间的 B-token ANN top-$k$ 候选形成先验 $P^{(0)}$。禁止直接比较两 LLM 未校准的 embedding 行。
5. **合并并归一化**：

$$
\widetilde T_{ji}=M_{ji}+\lambda_iP_{ji}^{(0)},\qquad
T_{ji}=\frac{\widetilde T_{ji}}{\sum_r\widetilde T_{ri}}.
$$

每列保留 top-$k$ 后重新归一化，保存为 CSC/等价稀疏格式。记录 tokenizer 文件哈希、规范化规则、语料版本、$k$、低频阈值、$\lambda_i$、ANN 模型、随机种子和代码版本。上述逐列归一化结果保留为不约束目标边际的 local baseline；正式的 $T$ 在同一稀疏候选图上使用下述 Sinkhorn 算法拟合。

### 4.1 边际分布与熵正则化 OT 目标

记 source A 的词表边际为 $a\in\Delta^{V_A}$，receiver B 的词表边际为 $b\in\Delta^{V_B}$。两者均从 `transport_train` 的 token 频率估计，并使用相同的 special-token 规则、低频平滑和归一化协议，满足

$$
a_i>0,\qquad b_j>0,\qquad \sum_i a_i=\sum_j b_j=1.
$$

若某些 token 保留零质量，必须先从 OT 的有效支撑中移除，不能在后续计算 $\operatorname{Diag}(a)^{-1}$ 时除以零。

令代价矩阵和 coupling 的方向固定为

$$
C,\Pi\in\mathbb R^{V_B\times V_A},
$$

即行对应 receiver B token，列对应 source A token。由第 4 节的匹配证据构造代价，例如先定义

$$
S_{ji}=M_{ji}+\lambda_iP_{ji}^{(0)},
$$

再取

$$
C_{ji}=-\log\frac{S_{ji}+\delta}{\sum_rS_{ri}+\delta V_B}.
$$

其中 $\delta>0$ 只用于数值平滑。special-token 和 exact-byte 候选可按预注册规则赋予更低代价；候选图外的边设为 $C_{ji}=+\infty$，等价于 $K_{ji}=0$。具体代价组成、规则权重和候选来源必须写入构建配置，不能使用 benchmark test 调参。

求解熵正则化最优传输：

$$
\Pi^*=\arg\min_{\Pi\ge0}
\left\langle C,\Pi\right\rangle+
\epsilon\sum_{j,i}\Pi_{ji}(\log\Pi_{ji}-1),
$$

约束为

$$
\Pi\mathbf1_{V_A}=b,\qquad
\Pi^T\mathbf1_{V_B}=a.
$$

$\epsilon>0$ 控制传输计划的平滑度：较小的 $\epsilon$ 更接近未正则化 OT，但更难数值稳定；较大的 $\epsilon$ 会产生更平滑的 coupling。$\epsilon$ 只能在 `transport_dev` 上从预先限定的候选集合中选择。

### 4.2 Sinkhorn 拟合算法

根据代价矩阵构造 Gibbs 核：

$$
K_{ji}=\exp\left(-\frac{C_{ji}}{\epsilon}\right),
\qquad K\in\mathbb R_+^{V_B\times V_A}.
$$

初始化 $v^{(0)}=\mathbf1_{V_A}$，然后迭代：

$$
u^{(t+1)}=b\oslash(Kv^{(t)}),
$$

$$
v^{(t+1)}=a\oslash(K^Tu^{(t+1)}),
$$

其中 $u\in\mathbb R^{V_B}$、$v\in\mathbb R^{V_A}$，$\oslash$ 表示按元素相除。收敛后输出

$$
\Pi^*=\operatorname{Diag}(u)K\operatorname{Diag}(v).
$$

伪代码：

```text
Input:
    cost C [V_B, V_A]
    source marginal a [V_A]
    target marginal b [V_B]
    epsilon, tolerance, max_iter

K = exp(-C / epsilon)
v = ones(V_A)

for iteration in 1 .. max_iter:
    u = b / (K @ v)
    v = a / (K.T @ u)

    row_residual = ||u * (K @ v) - b||_1
    col_residual = ||v * (K.T @ u) - a||_1
    if max(row_residual, col_residual) <= tolerance:
        break

Pi = diag(u) @ K @ diag(v)
T  = Pi @ diag(1 / a)
return T, Pi, convergence_report
```

若 source 与 target 的词表和经验边际完全相同，即 $V_A=V_B$ 且 $a=b$，算法退化为对称形式：

$$
u=a\oslash(Kv),\qquad v=a\oslash(K^Tu).
$$

一般的跨 tokenizer 场景必须分别使用 $a$ 和 $b$，不能在两次更新中误用同一个边际。由于本文固定 $K,\Pi$ 的形状为 $V_B\times V_A$，row scaling 使用 $b$，column scaling 使用 $a$；若代码采用转置方向，更新也必须相应转置，并通过边际测试确认。

### 4.3 从 coupling 得到条件传输矩阵

最终用于 STT 的列随机矩阵为

$$
T_{A\to B}=\Pi^*\operatorname{Diag}(a)^{-1},
\qquad
T_{ji}=\frac{\Pi^*_{ji}}{a_i}.
$$

因此

$$
\sum_jT_{ji}=1,\qquad Ta=b.
$$

这一步不能省略：$\Pi$ 是带 source 频率质量的联合 coupling，而 STT 需要的是给定 A token 后 B token 的条件分布。

### 4.4 数值稳定、稀疏实现与停止条件

小词表或裁剪候选先实现普通 Sinkhorn，作为可读的数值 oracle；正式大词表实现使用 log-domain Sinkhorn 或等价的稳定缩放，避免 $\exp(-C/\epsilon)$ 下溢。

实现必须满足：

- 每个 $a_i>0$ 的 source token 至少有一条候选边；
- 每个 $b_j>0$ 的 receiver token 至少有一条候选边；
- 稀疏候选图存在满足两侧边际的可行 coupling；
- 分母可设置机器精度相关下界，但不能静默掩盖无可行边的 token；
- 记录 `max_iter`、实际迭代数、`tolerance`、最终行/列边际残差和是否收敛；
- 未收敛或出现 NaN/Inf 时构建失败，不能保存为有效 artifact。

停止条件为

$$
\max\left(
\left\|\Pi\mathbf1-b\right\|_1,
\left\|\Pi^T\mathbf1-a\right\|_1
\right)\le\texttt{tolerance}.
$$

精度采用分层要求：

- toy vocab、dense oracle、有限差分和小图算法回归继续使用 `1e-9` 或测试中原有的更严格阈值；这些测试用于发现方向、转置和数值实现错误。
- 真实 Qwen3→Mistral-Nemo full-vocabulary 构建使用 `tolerance = 2e-3`，即上述两侧 L1 residual 的最大值不超过 `0.002`。该阈值是工程/实验 artifact 的预注册近似精度，不替代小图数值 oracle。
- convergence report 和 audit 必须同时保存阈值及实际 row/column residual。出现 NaN/Inf、不可行支撑或 residual 超过对应层级阈值时仍失败，不能保存有效 artifact。

该真实图阈值于 2026-09-02 根据用户确认调整；Job 234 的 row/column residual 为 `1.6915304665e-3`/`6.2669379185e-14`，在新阈值内，但它仍使用旧 `1e-9` 配置且没有产出 artifact，因此不能把旧 checkpoint 标记为有效。按新配置重跑的 preview Job 236 已完成原子保存和独立稀疏审计。正式 OpenHermes transport_train Job 240 进一步绑定 495,000 samples/997,233 canonical messages，row/column residual 为 `1.9655245213e-3`/`1.0560509249e-13`，`max_column_sum_error=1.2299050667e-12`，checkpoint 为 `complete/fresh`，峰值 RSS 为 `8,036,128 KiB`、0 swap；该正式 artifact 满足本节工程验收阈值。

优先在 top-$k$ 候选支撑图上直接运行稀疏 Sinkhorn，而不是先求稠密 $\Pi$ 再裁剪。若在 Sinkhorn 后再次 top-$k$ 并逐列归一化，会破坏目标边际 $b$；该结果必须标记为 `sparsified-approximate` 并重新报告边际误差。稠密全词表 Sinkhorn 仅用于小规模数值 oracle，不作为第一版大词表实现。

### 4.5 Artifact 与审计

保存 $T$ 为 CSC 或等价稀疏格式，并同时保存：

- source/target 边际 $a,b$；
- $\epsilon$、`tolerance`、`max_iter` 和 convergence report；
- 代价构建配置与候选支撑图；
- tokenizer 文件哈希、规范化规则和语料版本；
- $k$、低频阈值、$\lambda_i$、ANN 模型和随机种子；
- 代码版本。

保存前至少验证：

$$
\min_{j,i}T_{ji}\ge-\delta_{\mathrm{num}},
$$

$$
\max_i\left|\sum_jT_{ji}-1\right|\le\delta_{\mathrm{col}},
$$

$$
\left\|Ta-b\right\|_1\le\delta_{\mathrm{marginal}}.
$$

对真实 full-vocabulary artifact，`delta_marginal` 取 `2e-3`，并与构建时 `tolerance` 一同写入 metadata；toy/dense oracle 的对应审计继续采用 `1e-9` 或测试原有更严阈值。`delta_num` 和列随机性 `delta_col` 仍按存储 dtype 的数值精度设置，不因边际近似阈值调整而放宽非负性或逐列归一化检查。

正式 artifact 审计必须直接在 CSC/等价稀疏结构上计算列和、两侧边际、`Ta-b`、每列熵、transport cost 和正则目标，空间复杂度为 O(nnz + $V_A$ + $V_B$)。不得为 full-vocabulary audit 构造 $V_B\times V_A$ dense transport、coupling、mask 或 `where` 中间数组；dense 转换只允许用于有明确尺寸上限的 toy oracle。

## 5. STT 推理协议

```text
problem q
  → planner prompt P_A(q), enable_thinking=true
  → A no-grad generate → sender think r_A
  → A no-grad full-context hidden states H_A(P_A(q) + r_A)
  → LMHead_A(H_A) → softmax(/tau) → fixed sparse T[A→B]
  → aligned sender embeddings E_A
problem q
  → thinker prompt P_B(q), enable_thinking=true → B native embeddings
  → concat [E_A ; Emb_B(P_B(q))] → receiver B inputs_embeds prefill
  → B 的标准 KV cache 与标准 generate
  → B 显式 CoT 思考并生成答案
```

receiver 在 prefill 中同时接收 aligned sender context 和自身 tokenizer 显式编码的 thinker prompt。题目在 planner 与 thinker prompt 中各出现一次；拼接发生在 embedding 维，不能把 A token IDs 交给 B，也不能把 A 文本偷偷改由 B 重编码为 T2T。

每个 $h_t^A$ 都对应一个 aligned embedding，必须覆盖 sender prompt 与 sender think 的全部有效位置。因此新主协议不再使用旧版“receiver 起始 embedding + shifted source logits”，固定 `causal_shift=false`；receiver 的原生 BOS/chat boundary 已包含在 $P_B(q)$ 中。mask 与 position IDs 在两段拼接后从零连续重建，padding 不能进入有效长度。

精确 oracle：

$$
e_t^B=W_{\mathrm{in}}^BT\operatorname{softmax}(z_t^A/\tau).
$$

实施顺序：先全量 softmax 的短样本精确版；再做 source top-$m$ 截断并报告保留概率质量；随后验证稀疏列累积与预计算/分块 $C=W_{\mathrm{in}}^BT$ 的等价性；最后加入 `algo_detail.md` 的 ORF 核近似。通信温度 $\tau$ 与生成温度不同；若用 dev 选择 $\tau$，须明确报告为验证超参数选择。

## 6. 对照组与消融

| ID | 方法 | 训练的参数 | A→B 通信 |
|---|---|---:|---|
| R | Receiver-only | 0 | 无 |
| S | Source-only | 0 | 无 |
| T2T | 原 two-stage text | 0 | A 生成背景文本，B 重编码 |
| C2C | 原 C2C | projector | A KV → B KV |
| TH | Transport-hard | 0 | `argmax(Tp^A)` 对应的 B embedding |
| TS | Transport-soft（主方法） | 0 | $W_{in}^BTp^A$ |
| TK | Transport-soft + ORF | 0 | TS 的核近似实现 |

T2T 使用现有 `TwoStageInference`，固定 A 生成背景、B 生成答案。C2C 必须在同一 `transport_train` 分区上为 Qwen3-8B→Mistral-Nemo-Instruct-2407 新训练 projector；旧 recipe 超参数仅作为起点，需先核实两侧层数、KV head、head dimension、cache layout 和 projector 接口，并将最终配置完整报告。

必要消融：

| 维度 | 对照 | 目的 |
|---|---|---|
| $T$ 来源 | special/exact；+span；+ANN | 识别对齐证据贡献 |
| $T$ 拟合 | local 列归一化 vs Sinkhorn；多个 $\epsilon$ | 检验目标边际约束及熵正则强度 |
| 输出形式 | TH vs TS | 检验 soft expectation embedding 的价值 |
| 温度 | 多个 $\tau$ | 选择应保留的 source 不确定性 |
| 近似 | exact、top-$m$、ORF | 建立误差—效率曲线 |
| 位置 | shift/no-shift | 验证因果时间定义 |
| 模型对 | Qwen3→Mistral-Nemo；反向或第二个异 tokenizer 对 | 验证方向性与词表对齐的适用范围 |

主结果参数须预注册或仅用 `transport_dev` 确定；禁止在 benchmark test 上反复选择 $k,m,\tau$。

## 7. 指标与评测顺序

先用原 `unified_evaluator.py` 的 MMLU-Redux 跑通全部组别，再扩展 GSM8K、MATH-500、LongBench 及可选 MMMLU/MMLU-Pro/GPQA/CEval/ARC。

端到端报告：任务分数、按 subject/category 分数、每题正确性、首 token/总延迟、吞吐、source/transport/receiver 的分段耗时、峰值显存、长度统计。对同一批样本的正确性差异使用 paired bootstrap 置信区间或 McNemar 检验。

对 $T$ 及实现单独报告：

- $\min T_{ji}$ 与最大列和误差；
- Sinkhorn 是否收敛、迭代数、行/列边际残差与 $\lVert Ta-b\rVert_1$；
- OT transport cost、entropy 和正则化目标值；
- special token 审计通过率；
- exact/span/ANN 的覆盖列数和质量；
- 每列候选数、熵、最大权重、截断质量分位数；
- `transport_dev` 上的 span 覆盖和漂移；
- 随机 $p^A$ 的质量守恒；
- 精确、top-$m$、ORF 输出 embedding 的 cosine/相对误差；
- source top-$m$ 丢弃的概率质量。

## 8. 工程阶段与验收

### 阶段 0：冻结基线

记录 C2C commit、模型/tokenizer revision、依赖、硬件；先完成 Qwen3→Mistral-Nemo 的 tokenizer 全量审计和 R/S/T2T 小规模 smoke test；再验证现有 C2C projector 接口能否接收该模型对，并为新 projector 生成独立 recipe/checkpoint。保存 split manifest、canonical messages、rendered prompt、生成配置和逐题输出。

**验收：** 原 C2C 路径不变；R/S/T2T 可复现；新模型对的 tokenizer 审计完整；C2C 若进入主表，必须有同一 A/B、同一数据分区的新 checkpoint 和成功 smoke test。

### 阶段 1：构建并审计 $T$

导出 token bytes/offset/special metadata；实现流式 span 计数、规则映射、ANN fallback、边际估计、代价矩阵和稀疏候选图；先以小规模普通 Sinkhorn 建立数值 oracle，再实现大词表 log-domain/稀疏 Sinkhorn、条件矩阵转换、稀疏保存和审计报告；在 dev 验证不变量。

**验收：** Sinkhorn 按精度分层收敛：真实 full-vocabulary 构建的两侧最大 L1 residual 与 $\lVert Ta-b\rVert_1$ 不超过 `2e-3`，toy/dense oracle 保持 `1e-9` 或原测试阈值；$T$ 的非负性和列和审计通过；special token 无危险误配；artifact 可独立加载与复算。报告必须保存实际 residual 和采用的阈值。

### 阶段 2：Planner→Thinker STT 精确原型

新增独立 `TrainingFreeTransportModel`；sender 显式 CoT generate → 完整 sender context hidden/logits → fixed transport → 与 receiver 原生题目 prompt embeddings 拼接 → receiver 显式 CoT 生成。短序列上用显式矩阵和手工拼接作数值 oracle。

**验收：** 无梯度/optimizer state；题目在两侧 prompt 中均存在且两侧 CoT 打开；sender prompt/think 全位置均被对齐；receiver 原生 prompt 位于 aligned prefix 之后；中间 shape/mask/position/cache 与独立生成预算均被测试。

**历史证据（已被新协议取代）：** Job 245 只证明旧 prompt-only exact STT 可执行，不能作为 Planner→Thinker 协议验收；新协议必须重新跑真实 smoke 与 benchmark。

### 阶段 3：统一评测

在 evaluator 的 `training_free_transport` 分支构造 planner/thinker 两套含同题目的 prompt，显式开启两侧 CoT，记录 sender think、完整拼接长度和新增分段 latency；继续复用原题目格式化、答案解析、分卡和结果 schema。

**验收：** 各方法题目与 prompt 相同，可逐样本比较，失败样本有日志而非静默跳过。

### 阶段 4：近似与消融

> 2026-09-03 优先级修订：近似与消融实现及实验暂缓，不作为当前一轮交付的验收条件。当前先完成精确 STT 在 MMLU-Redux、GSM8K、MATH-500、LongBench 上的固定小样本测试与结果报告；近似误差—效率曲线和第 6 节消融仅在后续明确恢复时执行，且仍不得使用 test set 调参。

加入 TH、top-$m$、ORF；形成精确误差—速度曲线；跑第 6 节的必要消融。

**验收：** 主结论不依赖未报告的 test-set 调参。

### 阶段 5：Planner→Thinker benchmark 验收

> 2026-09-03 目标修订：反向/第二模型对实验不再是当前验收前置；已实现的方向安全配置保留供以后使用。当前先用新的双题目、双 CoT、完整 sender context + receiver native prompt 协议，在 MMLU-Redux、GSM8K、MATH-500、LongBench 上重新运行 exact 小样本并报告质量、失败、各段 latency、长度和显存。近似与消融继续暂缓。

## 9. 代码边界、风险与交付物

新增实现应与原 C2C 并存：

```text
C2C/rosetta/transport/
  vocab_transport.py      # 构建、加载、验证 T
  soft_transport.py       # logits → T → embeddings
  wrapper.py              # TrainingFreeTransportModel
C2C/script/transport/
  build_vocab_transport.py
  audit_vocab_transport.py
C2C/recipe/transport_recipe/
  qwen3_8b_to_mistral_nemo_instruct_2407.yaml
```

不删除或重写 `projector.py`、原 `wrapper.py` 的 C2C 路径、`SFT_train.py`、原 checkpoint/recipe。STT 只复用 evaluator、tokenizer、数据切分和基础生成能力。

主要风险及缓解：B 未在 virtual soft prompt 上训练可能导致性能低（这是 training-free 核心限制，需如实报告）；跨 tokenizer 覆盖不足或 special token 不可安全对应（全词表审计 + span 统计 + 显式 fallback/失败清单）；模板污染 span 统计（canonical 内容 + 规则表）；两个大模型同时驻留成本高（资源门槛预检，多 GPU/明确 device map，本地仅做 CPU 对齐 oracle）；新模型对没有现成 C2C checkpoint（独立训练并清楚标记 pending，不混用旧结果）；全词表成本高（精确 oracle → top-m → ORF）；T2T 预算不受控（固定并记录通信长度）；将语料统计误称 zero-shot（统一采用 corpus-fitted 表述）。

最终交付物：split manifest；版本化 T artifact、构建配置和审计报告；STT 精确/近似实现与测试；R/S/T2T/C2C/TH/TS/TK 可复现配置；逐题结果、延迟和显存记录；主表、消融表、置信区间和失败案例分析。
