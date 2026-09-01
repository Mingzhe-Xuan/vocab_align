# A 到 B 的词表对齐计划

## 1. 目标与边界

构造固定、稀疏、列随机的词表迁移矩阵

$$
T_{A\to B}\in\mathbb R_+^{V_B\times V_A},
\qquad
T_{ji}=P(B\text{ token}=j\mid A\text{ token}=i).
$$

列索引是 A token，行索引是 B token。对任意 A 侧概率列向量
$p^A\in\Delta^{V_A-1}$，定义

$$
p^B=T_{A\to B}p^A.
$$

必须满足

$$
T_{ji}\ge0,
\qquad
\sum_{j=1}^{V_B}T_{ji}=1\quad(\forall i).
$$

本计划只构造静态的 soft-token 映射；它不试图把一个 A token 精确展开为 B 的变长 token 序列。因此，矩阵的含义是 B 词表上的条件概率分布，而非 tokenizer 的无损转换器。

得到 $T$ 后，后续映射为

$$
C=W_{\mathrm{in}}^B T\in\mathbb R^{d_B\times V_A},
\qquad
h_0^B=Cp^A.
$$

## 2. 先决条件与固定产物

在构造 $T$ 前固定并记录：

- A、B 的模型版本、tokenizer 文件、词表大小 $V_A,V_B$；
- tokenizer 的 decode / byte-level 行为、Unicode 规范化方式及 special-token 清单；
- 代表性语料及其版本、语言分布和切分方式；
- B 的输入 embedding 矩阵方向：$W_{\mathrm{in}}^B\in\mathbb R^{d_B\times V_B}$；
- 对齐策略、超参数、随机种子和构建代码版本。

对每个 token 生成一条规范记录：`id`、展示文本、原始 bytes、是否含词首/空白标记、special-token 类别、以及可用时的语料频率。不得仅依据 tokenizer 的展示字符串判断 token 相同。

## 3. 分层构造策略

优先采用“同一原始语料的 span/byte 对齐统计”，嵌入近邻仅作为覆盖不足 token 的 fallback；全局 OT 是可选研究项，而不是第一版依赖。

### 3.1 特殊 token 与精确表面匹配

先建立显式规则表：

1. 两侧语义和控制功能相同的 BOS、EOS、PAD、换行、角色标记等，映射为 one-hot；
2. 不可安全对应的特殊 token 映射到约定的 B unknown / control token，或标记为待人工处理；
3. 对相同规范 bytes 的普通 token，优先 one-hot 映射；
4. 禁止将控制 token、纯 byte fallback、空白边界 token 与一般语义 token 混合匹配。

这些规则先写入稀疏计数矩阵 $M\in\mathbb R_+^{V_B\times V_A}$，并保留规则来源作为审计元数据。

### 3.2 代表性语料上的 span 对齐

对每段相同的原始文本，分别以 A、B tokenizer 编码，并取得每个 token 对应的原始 byte span。对 A 的一次 token 出现 $i$ 与所有重叠的 B token $j$，累加非负质量：

$$
M_{ji}\leftarrow M_{ji}+w\bigl(\operatorname{span}_A(i),\operatorname{span}_B(j)\bigr).
$$

默认权重取重叠 byte 长度；若一侧 token 完全覆盖另一侧，可额外给予较高权重。实现必须保留空白、标点和 Unicode 边界，避免只用 decode 后字符串再查找位置。

该步骤学习的是“在目标语料中，A token 出现位置对应哪些 B token”的经验条件分布，而非仅凭孤立 token 字符串猜测语义。

### 3.3 低频或未覆盖 token 的嵌入 fallback

对计数不足的 A token，使用一个共同的、最好是多语的外部嵌入模型产生可比较特征。不得直接比较 A、B 两个语言模型各自的 embedding 行，除非已使用可靠锚点把它们对齐。

为降低子词孤立编码的不稳定性，对 token 文本使用多个 carrier context 编码并平均；同时把 bytes、词首/空白标记等表面特征用于过滤候选。为全部 B token 建立 ANN 索引；每个 A token 仅检索 top-$k$ 候选 $J_i$，不物化 $V_A\times V_B$ 相似度矩阵。

对候选写入平滑先验，例如

$$
P^{(0)}_{ji}\propto
\exp\!\left(\frac{\operatorname{cos}(a_i,b_j)}{\gamma}\right)
\mathbf 1[j\in J_i].
$$

这里 $\gamma$ 为相似度温度。对于同语言 tokenizer，优先将该先验用于未覆盖列，而不是覆盖统计的替代品。

### 3.4 合并与逐列归一化

令 $n_i=\sum_jM_{ji}$ 为第 $i$ 列的 span 统计质量。用频次自适应的平滑系数 $\lambda_i$ 合并数据与先验：

$$
\widetilde T_{ji}=M_{ji}+\lambda_iP^{(0)}_{ji},
\qquad
T_{ji}=\frac{\widetilde T_{ji}}{\sum_r\widetilde T_{ri}}.
$$

高频 token 应主要由 span 统计决定；低频和零频 token 增大 $\lambda_i$。每列只保留候选集合中的 top-$k$ 项，截断后重新归一化，并记录截断质量

$$
\rho_i=1-\sum_{j\in J_i}T_{ji}.
$$

最终使用 CSC（按 A token 列访问）或等价稀疏列格式保存 $T$。

## 4. 实施阶段

### 阶段 A：接口与精确基线

1. 核实 A 输出头前的 Norm、$W_{\mathrm{out}}^A$、bias 与矩阵方向；
2. 核实 B token embedding 后还会施加的 position embedding、scale、Norm 等流程；
3. 实现小规模精确路径 $p^A\to Tp^A\to W_{\mathrm{in}}^Bp^B$；
4. 验证它与 $C\,p^A$ 严格一致。

### 阶段 B：token 元数据与规则映射

1. 导出并审计两侧 token 记录；
2. 建立 special-token 和 exact-byte 规则；
3. 对规则结果人工抽样，确认 whitespace、控制 token、byte fallback 未误配。

### 阶段 C：语料对齐统计

1. 选取与实际任务相近的训练/构建语料，另留独立验证语料；
2. 实现 byte-span tokenizer wrapper；
3. 流式累计稀疏 $M$，不保存所有 token occurrence；
4. 输出每列覆盖次数、候选数和熵。

### 阶段 D：fallback 与稀疏化

1. 为低频列构建共同嵌入空间与 B 侧 ANN 索引；
2. 生成 top-$k$ 先验候选并同统计合并；
3. 逐列归一化、稀疏化，记录 $\rho_i$ 与 fallback 占比；
4. 生成版本化的 $T$、构建配置和审计报告。

### 阶段 E：接入核化映射

1. 以稀疏列计算 $c_i=W_{\mathrm{in}}^BT_{:,i}$，避免物化 $C$；
2. 用 $c_i$ 分块累计核近似所需的 $S,z$；
3. 将 $T$ 的质量评估和核近似误差评估分开报告。

## 5. 必须通过的验证

### 数学与数值不变量

对最终 $T$ 检查：

$$
\min_{j,i}T_{ji}\ge-\delta_{\mathrm{num}},
\qquad
\max_i\left|\sum_jT_{ji}-1\right|\le\delta_{\mathrm{col}}.
$$

对随机 $p^A$ 检查：

$$
\left|\mathbf1^\top Tp^A-1\right|\le\delta_{\mathrm{mass}},
\qquad
\min_j(Tp^A)_j\ge-\delta_{\mathrm{num}}.
$$

并验证稀疏实现的 $W_{\mathrm{in}}^B(Tp^A)$ 与按列预计算的 $Cp^A$ 一致。

### 对齐质量

- 按 token 频率、语言、词类、token 长度和 special-token 类别分桶抽样审查 top 候选；
- 报告 exact-rule、span-statistics、embedding-fallback 各自覆盖的列数和概率质量；
- 报告每列候选数、熵、最大权重和截断质量 $\rho_i$ 的分位数与最大值；
- 在独立语料上重新统计 span 对齐，检测训练/验证分布漂移；
- 对高频 token 人工检查错误案例，特别关注词首空白、标点、数字、Unicode 和控制符。

### 端到端质量

固定 $T$ 后，分别报告精确 soft-token 映射 $F$ 与核近似 $\hat F$ 的差异，例如 cosine similarity、相对误差，以及注入 B 后的下游行为。不得将 $T$ 的语义误差归因于随机特征近似。

## 6. 第一版决策

第一版采用：**显式 special-token 规则 + exact-byte 匹配 + 代表性语料上的 byte-span 对齐 + 低频 token 的共同嵌入 ANN fallback + top-$k$ 稀疏列归一化**。

不在第一版采用完整熵正则 OT，原因是其稠密代价矩阵与耦合在大词表上成本过高，而且缺少可靠共同特征时并不能自动解决 tokenizer 的一对多切分问题。若第一版验证显示局部候选质量不足，再将 OT 限制在已有 top-$k$ 候选图上，作为后续实验。
