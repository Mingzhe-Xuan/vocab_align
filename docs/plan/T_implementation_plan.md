# Training-free Soft-Token Transport 分阶段实施计划

本文是 [`T_plan.md`](./T_plan.md) 的工程落地补充，规定阶段 0—5 的文件结构、模块职责和测试范围。研究问题、数学定义和实验口径以 `T_plan.md` 为准；本文不修改其中的算法目标。

## 1. 实施原则

1. STT 与原 C2C 路径并存，不修改 `rosetta/model/projector.py` 和原 `rosetta/model/wrapper.py` 的公开行为。
2. `rosetta/transport/` 只放可复用、可测试的库代码；`script/transport/` 只负责参数解析、编排和落盘。
3. 配置进入 `recipe/transport_recipe/`，生成物进入 `local/transport/` 或运行时指定的输出目录。大型矩阵、模型和逐题结果不提交到源码目录。
4. 单元测试不得下载远程模型或数据集。tokenizer、模型和稀疏矩阵均使用 tiny fixture；真实模型检查作为 smoke/integration test 单独标记。
5. 每个 artifact 都必须包含 `schema_version`、输入指纹、构建配置、随机种子和代码版本；加载时先校验兼容性，再参与推理或评测。
6. 研究性能不是工程验收条件。工程验收要求结果可复现、失败显式、统计口径一致；STT 是否优于基线由实验结果回答。

## 2. 目标文件结构

以下为全部阶段完成后的目标结构。标注“现有”的文件应增量修改，不整体重写。

```text
C2C/
  rosetta/
    transport/
      __init__.py
      config.py                    # 配置 dataclass、枚举和跨字段校验
      token_metadata.py            # token bytes、offset、special/control metadata
      vocab_transport.py           # T 的公共构建/加载/验证入口；保留现有小规模原型
      candidate_graph.py           # special/exact/span/ANN 候选边与来源标签
      marginals.py                 # a、b 的流式频率估计和平滑
      sinkhorn.py                  # dense oracle 与 sparse/log-domain Sinkhorn
      artifact.py                  # 稀疏 T/Pi、manifest 和版本化序列化
      audit.py                     # 不变量、覆盖率、危险 special 映射和漂移报告
      soft_transport.py            # logits -> distribution -> T -> B embedding
      wrapper.py                   # TrainingFreeTransportModel
      approximations.py            # hard、top-m 和分块/预计算近似
      orf.py                       # ORF 核近似
      metrics.py                   # transport、近似误差和分段 latency 指标
  script/
    dataset/
      build_transport_manifest.py  # 固定 transport_train/dev 样本 ID
    transport/
      compare_tokenizers.py        # 现有；扩展为全词表审计入口
      build_small_vocab_transport.py # 现有；阶段 1 dense/local oracle
      freeze_baseline.py
      build_vocab_transport.py
      audit_vocab_transport.py
      smoke_stt.py
      run_transport_ablation.py
      summarize_transport.py
    evaluation/
      unified_evaluator.py         # 现有；增加 STT adapter/分段计时
  recipe/
    transport_recipe/
      qwen3_8b_to_mistral_nemo_instruct_2407.yaml
      mistral_nemo_to_qwen3_8b.yaml
      schema.yaml                  # 字段说明、默认值和允许的枚举
    eval_recipe/
      stt_mmlu_redux.yaml
      stt_ablation.yaml
  test/
    transport/
      conftest.py                  # tiny tokenizer/model/sparse graph fixtures
      test_config.py
      test_token_metadata.py
      test_vocab_transport.py
      test_candidate_graph.py
      test_marginals.py
      test_sinkhorn.py
      test_artifact.py
      test_audit.py
      test_soft_transport.py
      test_wrapper.py
      test_approximations.py
      test_orf.py
      test_metrics.py
    evaluation/
      test_stt_evaluator.py
      test_transport_statistics.py
    integration/
      test_real_tokenizers.py
      test_stt_generation.py
      test_evaluator_smoke.py
  local/
    transport/
      manifests/                   # split manifest、canonical messages
      audits/                      # tokenizer/T 审计 JSON 和 Markdown
      artifacts/                   # 稀疏 T、Pi、a、b 和 metadata
      results/                     # 逐题输出、性能和统计汇总
```

若实现初期模块规模很小，可先将 `token_metadata.py`、`candidate_graph.py`、`marginals.py` 保留在 `vocab_transport.py` 内；一旦单文件同时承担两类以上职责，再按上述边界拆分。现有 `test/test_vocab_transport.py` 应迁移到 `test/transport/` 并保留其 exact/span 回归用例。

## 3. 阶段 0：冻结基线

### 3.1 文件与功能

| 文件 | 功能 |
|---|---|
| `rosetta/transport/config.py` | 定义模型 revision、精度、device map、生成参数、数据 split 和输出 schema；拒绝未锁定 revision 或互相冲突的配置。 |
| `script/dataset/build_transport_manifest.py` | 以 seed 42 从 OpenHermes 样本 ID 生成确定性的 `transport_train`/`transport_dev` manifest；检测重复和交叉污染。 |
| `script/transport/compare_tokenizers.py` | 在现有比较脚本上增加全词表 token bytes、special/control 分类、exact-byte 覆盖、语料加权覆盖和长度比。 |
| `script/transport/freeze_baseline.py` | 汇总代码版本、依赖、硬件、模型/tokenizer commit、canonical messages、rendered prompts 和 generation config。 |
| `recipe/transport_recipe/qwen3_8b_to_mistral_nemo_instruct_2407.yaml` | 固定主模型对与构建参数；C2C checkpoint 缺失时显式写 `pending-new-projector-training`。 |

阶段 0 不训练 projector。它只验证原 C2C 接口是否与新模型对结构兼容，并生成独立训练 recipe；实际训练属于外部实验任务。

### 3.2 单元测试

- `test_config.py`
  - 合法配置可稳定序列化/反序列化。
  - 缺少 revision、seed 或输出路径时失败。
  - test split 被配置成 transport 构建输入时失败。
  - `pending-new-projector-training` 不可被当作可加载 checkpoint。
- `test_token_metadata.py`
  - UTF-8、多字节字符和 byte-level BPE token 能恢复正确 bytes。
  - special token 不进入普通 exact-byte 匹配集合。
  - 两个相同 token ID、不同 token bytes 不会被误判为匹配。
- manifest 测试
  - 相同 seed 和输入 ID 得到字节级相同 manifest。
  - train/dev 无交集、无重复，比例符合约定。
  - 输入顺序变化不改变基于稳定 sample ID 的划分。
- baseline 快照测试
  - canonical messages 与两个模型的 rendered prompt 分开保存。
  - 硬件或 checkpoint 缺失会标记 unavailable，而不是伪造结果。

### 3.3 非单元验收

- 使用真实 Qwen3/Mistral-Nemo tokenizer 跑全词表审计。
- R、S、T2T 在固定 2—5 条样本上完成 smoke test，并产生相同结果 schema。
- 原 C2C 测试保持通过；新模型对没有 checkpoint 时主表状态必须为 pending。

## 4. 阶段 1：构建并审计词表传输矩阵

### 4.1 文件与功能

| 文件 | 功能 |
|---|---|
| `token_metadata.py` | 统一导出 token bytes、字符/byte offset、频率和 special/control 类别。 |
| `candidate_graph.py` | 按 special → exact-byte → byte-span → ANN fallback 优先级构建稀疏边；每条边记录来源和原始证据。 |
| `marginals.py` | 流式统计 `a`、`b`，应用同一 special 规则、低频平滑和有效支撑过滤。 |
| `sinkhorn.py` | 提供小矩阵 dense Sinkhorn oracle，以及大词表 sparse/log-domain 实现；返回 `Pi` 和收敛报告。 |
| `vocab_transport.py` | 作为公共 facade 编排候选图、代价、Sinkhorn 和 `T = Pi Diag(a)^-1`；保留 local 列归一化 baseline。 |
| `artifact.py` | 保存 CSC/等价稀疏结构、`a`、`b`、候选图、配置、指纹和收敛信息；支持安全加载与 schema migration。 |
| `audit.py` | 直接在 CSC 上以 O(nnz + vocab) 检查非负性、列和、两侧边际、`Ta=b`、覆盖率、熵、目标值及危险 special 映射；dense helper 仅限 tiny oracle。 |
| `build_vocab_transport.py` | 流式读取 manifest，支持 resume/checkpoint，构建正式 artifact。 |
| `audit_vocab_transport.py` | 只读加载 artifact，重算关键不变量并输出 JSON/Markdown 报告。 |

ANN 只在共同外部 embedding 空间中生成候选边，禁止直接比较两侧 LLM 未校准的 embedding 行。候选图必须保证每个正质量 token 至少有一条边；无法满足时构建失败。

### 4.2 单元测试

- 候选与 span
  - special 映射优先于 exact/span/ANN。
  - exact-byte 重复候选按明确策略处理，结果确定。
  - ASCII、中文、emoji、组合字符的 byte overlap 计数正确。
  - message 内容统计不包含 chat template/BOS/EOS 污染。
  - 低频或零覆盖列进入 ANN/fallback；无安全 fallback 时显式失败。
- 边际与代价
  - `a_i > 0`、`b_j > 0` 且两者分别归一化为 1。
  - 被移出有效支撑的零质量 token 不参与除法。
  - `C`/`Pi` 始终使用 `[V_B, V_A]`，转置错误由非方阵用例捕获。
  - 图外边的 kernel 质量严格为零。
- Sinkhorn
  - 2×3、3×2 非方阵的 dense oracle 满足两侧边际。
  - sparse/log-domain 输出与 dense oracle 在小图上误差低于 `1e-9` 或用例原有更严阈值；真实图的近似验收阈值不得反向降低该单元测试标准。
  - 极小 epsilon、极端频率下不产生 NaN/Inf。
  - 不可行支撑图、未收敛和超过 `max_iter` 时构建失败。
  - 收敛报告包含迭代数、row/column residual 和 converged 状态。
- 条件矩阵与 artifact
  - `T = Pi Diag(a)^-1` 后逐列和为 1，且 `Ta` 与 `b` 一致。
  - save/load round trip 保持稀疏索引、dtype、数值和 metadata。
  - tokenizer 指纹或 schema 不匹配时拒绝加载。
  - 损坏、缺列或非有限数值的 artifact 不可通过审计。
  - 大 shape/低 nnz artifact 的正式 audit 不调用 dense 转换；稀疏统计与小矩阵手算结果一致。

### 4.3 非单元验收

- 用 toy vocab 同时运行 dense 与 sparse 构建，保存可复算的 oracle 报告。
- 用真实 tokenizer 和小语料构建预览 artifact，确认所有正质量行/列有可行支撑。真实 Qwen3→Mistral-Nemo full-vocabulary coupling 的两侧最大 L1 residual 验收阈值为 `2e-3`。
- 正式语料运行必须记录 checkpoint/resume 状态、实际 row/column residual 和使用的 tolerance；不将半成品标为有效 artifact。旧 Job 234 虽达到新精度，但因在 `1e-9` 配置下失败且未生成 artifact，不能作为有效产物。按新配置重跑的 preview Job 236 已完成原子保存和独立稀疏审计；进一步的正式 OpenHermes transport_train Job 240 绑定 495,000 samples/997,233 canonical messages，row/column residual 为 `1.9655245213e-3`/`1.0560509249e-13`，checkpoint 为 `complete/fresh`，峰值 RSS 为 `8,036,128 KiB`，满足本阶段正式 artifact 验收。

## 5. 阶段 2：STT 精确推理原型

### 5.1 文件与功能

| 文件 | 功能 |
|---|---|
| `soft_transport.py` | 实现 temperature softmax、精确 `Tp`、`W_in^B Tp`、source top-m 可选路径和质量统计。 |
| `wrapper.py` | 实现 `TrainingFreeTransportModel`：A no-grad prefill、因果 shift、B virtual prompt prefill、标准 KV cache 生成。 |
| `metrics.py` | 记录 source、transport、receiver 分段耗时、输入/virtual/output 长度和峰值显存。 |
| `script/transport/smoke_stt.py` | 加载两侧模型与 T，执行短样本并保存中间 shape、配置和输出。 |

`wrapper.py` 不继承或覆盖原 C2C wrapper 的投影逻辑。共同能力通过小型纯函数复用，不能以条件分支改变旧 C2C 行为。

### 5.2 单元测试

- `test_soft_transport.py`
  - 稀疏 `Tp` 与显式 dense 矩阵结果一致。
  - `W_in^B Tp` 与先算组合矩阵 `C = W_in^B T` 的结果一致。
  - 概率和、dtype、device 与 batch/sequence 维度保持正确。
  - `tau <= 0`、词表维不匹配和非法 T 会明确报错。
  - source top-m 报告被丢弃概率质量，`m=V_A` 退化为精确路径。
- `test_wrapper.py`
  - source forward 全程 `no_grad`，模型参数无 `.grad`，不存在 optimizer state。
  - “起始 embedding + shifted logits”的序列长度、位置和因果关系正确。
  - shift/no-shift 是显式配置，默认值与 recipe 一致。
  - attention mask、position ids、past key values 长度随 prefill/generation 正确更新。
  - B 生成期使用自身 tokenizer/embedding，而不是继续调用 A。
  - transport 关闭时走独立 receiver-only 输入路径，并与直接调用 receiver 的结果一致。
  - batch size 1/2、单 token prompt、padding 和提前 EOS 均可处理。
- `test_metrics.py`
  - 分段耗时之和与总耗时误差在计时精度内。
  - CPU 环境下显存字段为 `null/unavailable`，不能填入伪零值。

### 5.3 非单元验收

- tiny 随机模型完成端到端 generate，不依赖网络。
- 真实模型短序列 smoke test 能生成并保存逐步 diagnostics。
- CPU/offload 运行只标记功能正确性，不进入正式 latency 表。

真实 Qwen3-8B→Mistral-Nemo Job 245 已完成本阶段功能验收：在 RTX 5090/Blackwell 上使用隔离 `blackwell-cu128` profile（torch 2.7.1+cu128），同一 prompt 的 Receiver-only 与精确 STT 均确定性生成 2 tokens；STT virtual prompt 为 `[1, 7, 5120]`，retained/active support mass 在 float32 舍入内为 1，top-m 丢弃质量为 0。报告 schema v2 原子保存且无 partial，记录模型/tokenizer/artifact/runtime/compiled arch、分段 metrics 和两路输出；Job 245 为 0:16.67、Exit 0、MaxRSS 16,560,968 KiB。该耗时只作功能诊断，不进入正式 latency 表。

## 6. 阶段 3：统一评测

### 6.1 文件与功能

| 文件 | 功能 |
|---|---|
| `script/evaluation/unified_evaluator.py` | 增加 `training_free_transport` 初始化与推理分支，复用现有题目格式化、答案解析、分卡和保存逻辑。 |
| `rosetta/transport/metrics.py` | 定义与 evaluator 兼容的分段 latency、长度和显存 schema。 |
| `recipe/eval_recipe/stt_mmlu_redux.yaml` | 固定样本、few-shot/CoT、解码、`max_new_tokens`、硬件和输出路径。 |
| `script/transport/summarize_transport.py` | 聚合逐题正确性、类别分数、失败原因、吞吐和置信区间输入。 |

建议在 evaluator 内引入窄接口（如 `generate_one(sample) -> EvaluationRecord`），让 HF、T2T、C2C 和 STT 共享外层循环，避免为 STT 复制整段评测代码。

### 6.2 单元测试

- 配置 `model_name: training_free_transport` 能创建正确 adapter，其他 model name 行为不变。
- R/S/T2T/C2C/STT 对同一 fixture 产生相同 `sample_id`、canonical message 和评测 prompt 元数据。
- STT 的 source/transport/receiver latency 和通信长度写入统一结果 schema。
- 单样本异常写入 bad-sample 日志并继续后续样本；异常样本不得静默计为错误或被丢弃。
- 多 rank 合并保持样本唯一、顺序确定，不重复计分。
- answer parser 对 STT 不启用特殊宽松规则。
- 恢复运行跳过已有成功样本，但会重试明确失败或不完整记录。

### 6.3 非单元验收

- 固定 MMLU-Redux 小子集运行 R/S/T2T/TS，验证逐样本可配对。
- 完整评测前执行资源门槛检查；失败时输出缺失模型、显存或 artifact，而不是启动半完整实验。
- 正式结果同时保存 summary 和逐题记录，后者是统计检验的唯一输入。

## 7. 阶段 4：近似与消融

### 7.1 文件与功能

| 文件 | 功能 |
|---|---|
| `approximations.py` | TH、source top-m、稀疏列累积、分块和预计算 `W_in^B T`。 |
| `orf.py` | 按 `algo_detail.md` 实现 ORF 特征采样、固定 seed、误差/内存统计。 |
| `run_transport_ablation.py` | 展开预注册的 T 来源、local/Sinkhorn、epsilon、tau、shift 和近似组合。 |
| `recipe/eval_recipe/stt_ablation.yaml` | 明确 dev 可选超参数集合和 test 冻结值。 |
| `summarize_transport.py` | 输出误差—速度曲线、覆盖/熵分位数和配对统计。 |

### 7.2 单元测试

- TH 等于 `argmax(Tp)` 对应的 B embedding，tie-break 确定。
- top-m 在 `m=V_A` 时等于精确结果；随着 m 增大，丢弃质量不增加。
- sparse accumulation、分块和预计算路径与 dense oracle 在容忍度内一致。
- 固定 seed 的 ORF 特征和输出可复现；特征数增加时 shape、内存估算正确。
- cosine/相对误差处理零范数，不产生静默 NaN。
- 消融展开器只生成预注册组合，test 配置不能覆盖由 dev 冻结的参数。
- 结果聚合按同一 sample ID 做配对，缺失样本会报告而非自动取不一致交集。

### 7.3 非单元验收

- 在固定 dev 子集上生成 exact、top-m、ORF 的误差—延迟曲线。
- 主测试集只运行冻结配置；任何补跑或参数变更进入审计日志。
- TH/TS/TK 均输出统一 schema，可直接进入同一统计脚本。

## 8. 阶段 5：泛化与结果分析

### 8.1 文件与功能

| 文件 | 功能 |
|---|---|
| `recipe/transport_recipe/mistral_nemo_to_qwen3_8b.yaml` | 反向模型对；独立 T、边际、special 规则和 artifact，禁止简单转置正向 T。 |
| 第二模型对 recipe | 使用相同 schema 声明新的异 tokenizer A→B 方向。 |
| `script/transport/summarize_transport.py` | paired bootstrap、McNemar、类别切片、latency 分解和失败案例索引。 |
| `test/evaluation/test_transport_statistics.py` | 统计方法与结果聚合的确定性测试。 |

数学、长上下文等新增 benchmark 应通过 evaluator 的既有 dataset adapter 接入；STT 不单独实现一套 prompt 或答案解析。

### 8.2 单元测试

- 反向配置交换 A/B 后，所有矩阵 shape、边际和 tokenizer 指纹随方向正确变化。
- 正向 artifact 不能被反向 wrapper 加载。
- paired bootstrap 在固定 seed 下可复现，置信区间边界有序。
- McNemar 计数与手工构造的四格表一致，并正确处理零分母/全相同预测。
- category/subject 聚合与逐题总数守恒。
- latency 分解明确区分 source prefill、transport、receiver prefill 和 decode。
- 长上下文截断必须发生在 canonical 规则指定的位置，并记录截断前后长度。

### 8.3 非单元验收

- 至少一个反向或第二异 tokenizer 模型对完成最小主表。
- 数学或长上下文扩展沿用冻结 prompt/解码协议。
- 报告包含显著性、失败案例、资源条件和负结果，不以缺失实验替代零提升结论。

## 9. 测试分层与执行约定

在 `pyproject.toml` 注册以下 marker，避免默认单元测试误触发模型下载或 GPU 作业：

```toml
[tool.pytest.ini_options]
markers = [
  "integration: local files or multiple components are required",
  "model: requires locally available real model/tokenizer artifacts",
  "gpu: requires CUDA and declared memory threshold",
  "slow: runtime is unsuitable for the default unit suite",
]
```

建议执行层级：

```text
默认提交检查：pytest test/transport test/evaluation -m "not integration and not model and not gpu and not slow"
本地集成检查：pytest test/integration -m "integration and not gpu"
GPU smoke：     pytest test/integration -m "gpu" --model-cache <path>
正式实验：      通过 recipe 调用脚本，不作为 pytest 单元测试
```

公共 fixture 必须覆盖：非方阵词表、重复 exact bytes、不可映射 special token、中文/emoji offsets、极端边际、不可行稀疏图、padding batch 和最短生成序列。

## 10. 阶段依赖和完成定义

```text
阶段 0（冻结输入与协议）
  -> 阶段 1（可审计 T artifact）
       -> 阶段 2（精确 STT 推理）
            -> 阶段 3（统一评测）
                 -> 阶段 4（近似与消融）
                 -> 阶段 5（泛化与统计）
```

阶段 4 和阶段 5 可在阶段 3 的小子集闭环稳定后并行准备，但正式 test 结果必须使用阶段 3 冻结的协议。

每个阶段只有同时满足以下条件才算完成：

1. 目标模块具有类型清晰的公共接口，脚本不承载核心算法。
2. 对应 CPU 单元测试通过，旧 C2C 回归测试未退化。
3. 该阶段要求的 smoke/integration test 有机器可读结果。
4. 配置、artifact 和结果均带版本与输入指纹，可从 manifest 复算。
5. 已知限制和未完成的外部实验明确标记为 pending，不用替代数据填充。

## 11. 建议实施顺序

1. 先整理现有 `vocab_transport.py` 和测试 fixture，补齐 artifact round trip 与非方阵 dense Sinkhorn oracle。
2. 完成阶段 0 的 manifest/config，使后续所有构建共享冻结输入。
3. 完成阶段 1 的 sparse/log-domain Sinkhorn 和审计 CLI，产出第一个正式 T artifact。
4. 以 tiny 模型完成阶段 2，再使用真实模型做最短 smoke test。
5. 接入 evaluator 并只跑固定小子集；schema 稳定后再扩展完整评测。
6. 最后实现 top-m/ORF、消融和第二模型对，避免近似优化掩盖精确路径错误。
