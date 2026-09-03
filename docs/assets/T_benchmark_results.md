# Exact STT benchmark smoke 结果

## 范围与口径

本报告记录 Qwen3-8B → Mistral-Nemo-Instruct-2407 的 training-free
soft-token transport（STT）固定小样本测试。所有作业使用正式
OpenHermes-500k transport artifact、完整词表与完整稀疏 $T$、greedy
解码、source CPU/receiver GPU；没有启用 hard、top-m、precomputed、ORF
或消融参数。

这些结果用于验证 benchmark 接入、长序列 exact 执行、逐题记录和初步
生成质量，不是完整 benchmark 主表。`execution success` 表示模型推理和
记录成功，不等于答案正确。LongBench Qasper 在统一记录中保留官方参考
字段，但当前由外部 task scorer 评分，因此 summary 不伪造 boolean
accuracy。

## 固定 smoke 结果

| Benchmark | Slurm job | 样本 | Execution | 得分 | 平均总耗时/题 | 平均 CUDA peak |
|---|---:|---:|---:|---:|---:|---:|
| MMLU-Redux / abstract_algebra | 247 | 5 | 5/5 success | 0/5 | 60.23s | 26.19GiB |
| GSM8K / main | 248 | 3 | 3/3 success | 0/3 | 55.43s | 25.82GiB |
| MATH-500 / all | 249 | 3 | 3/3 success | 0/3 | 62.97s | 26.24GiB |
| LongBench / Qasper | 253 | 1 | 1/1 success | external scorer 未运行 | 909.34s | 23.69GiB |

前三项的 parser prediction 均为 null。MMLU-Redux 五条输出均为 `>`；
GSM8K 输出为重复 `the`、重复短片段和 `>`；MATH-500 输出为重复 `and`
和两个 `>`。因此三个 0 分结果是实际生成质量，而不是 execution failure
或失败样本被错误计入分母。当前小样本证据说明 exact STT 链路可以运行，
但尚未产生可用的任务质量。Qasper 的有效输出同样为 `>`，参考答案为
`Ground truth is not established in the paper`；记录正确标记为
`external_required`，未运行官方 task scorer，也未把它伪装成 accuracy。

## LongBench 诊断与修复

首次 Qasper Job 250 仅作为诊断保留，不计入有效结果：LongBench formatter
已经输出 source tokenizer 的 chat prompt，旧 adapter 又渲染一次；同时
exact sparse transport 对约 2048-token 输入一次物化 query-by-edge
contributions，尝试额外申请 21.10GiB 并 OOM。有效实现作出两项保持语义的
修复：

1. LongBench sample 显式标记 source prompt 已渲染，adapter 直接编码一次。
2. exact transport 沿 source sequence 使用 32-token query chunks；每块仍执行
   完整词表 softmax 和完整稀疏 $T$，再按原顺序拼接，因此这是等价内存优化，
   不是近似。

本地 65-token 测试实际产生 `[32, 32, 1]` 三块；完整回归 196/196 通过。
Job 253 使用上述修复和同一 2048-token 输入上限成功完成：实际 source/
virtual prompt 均为 2060 tokens，source/transport/receiver-prefill/decode
分别为 864.25/44.67/0.34/0.07s，总耗时 909.34s；CUDA peak 为
25,437,179,392 bytes（23.69GiB）。逐题 diagnostics 为
`source_prompt_rendered=true`、`approximation_mode=exact`、无 top-m，active/
retained support mass 约 1.00000024。

Job 253 records/summary SHA-256 分别为
`a8918457e5d142e2ff647da2fcdf162efd6508173c06c31e6087d086651f46e8` 和
`009ca3f440db1a758287ffcd72530ceac54347502441995b446b0835ad084b12`；
无 bad-samples 或 `.partial`。有效 code version 为 `e02f4ad...`。

## 可复现性

- 正式 artifact SHA-256：
  `1495d5224aa83d979134fb7a225989af724753d9c9a0e7d755012d9b0b0aba97`；
  shape `131069 × 151669`，2,733,518 nnz。
- GSM8K revision：`740312add88f781978c0658806c59bc2815b9866`。
- MATH-500 revision：`6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`；
  本地固定文件 SHA-256：
  `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`。
- LongBench revision：`36dca87b51c492bf896705ab6953d3a6141dd012`；
  Qasper 文件 SHA-256：
  `7c6bf3a2a402b557d001808ba345a23921a211c39bf2d36d925d1d70e21b3f03`。
- Runtime：Python 3.10.12、torch 2.7.1+cu128、transformers 4.52.4、
  accelerate 1.9.0、RTX 5090 `sm_120`。

近似误差—效率曲线与消融按当前需求延期，不属于本报告的验收范围。
