# 正式 T artifact：位置、格式与使用方法

## 1. 当前文件

正向已验收的 transport artifact 对应：

```text
sender/source:   Qwen/Qwen3-8B
receiver/target: mistralai/Mistral-Nemo-Instruct-2407
direction:       Qwen3-8B -> Mistral-Nemo-Instruct-2407
```

它有两个内容相同的副本：

- Guqq：`~/vocab_align/C2C/local/transport/artifacts/qwen3_8b_to_mistral_nemo_openhermes_500k.npz`
- 本地仓库根目录：[`qwen3_8b_to_mistral_nemo_openhermes_500k.npz`](../../qwen3_8b_to_mistral_nemo_openhermes_500k.npz)

完整性信息：

| 项目 | 值 |
|---|---|
| 文件大小 | `39,951,267` bytes |
| SHA-256 | `1495d5224aa83d979134fb7a225989af724753d9c9a0e7d755012d9b0b0aba97` |
| shape | `[131069, 151669]` |
| 布局 | `[target vocabulary, source vocabulary]` |
| 非零元素 | `2,733,518` |
| 权重 dtype | `float64` |
| source tokenizer fingerprint | `c39a0519aee714a6c8a3eca850849443a5cc6eb7e960211daa11cf5a896061de` |
| target tokenizer fingerprint | `12be2a776e12655422f69aabc496e70843d20b1b118c0089314e14d542beca92` |
| artifact input fingerprint | `ca9aefcff66a50600e508f5e9bd1d3f17dec3ea4bc344cff0c3101839dfa459a` |

本地文件已由根目录 `.gitignore` 精确排除。它是模型生成物且大于
10 MiB，不得执行 `git add -f` 或通过普通 Git 提交。

独立求解并已验收的反向 artifact 对应：

```text
sender/source:   mistralai/Mistral-Nemo-Instruct-2407
receiver/target: Qwen/Qwen3-8B
direction:       Mistral-Nemo-Instruct-2407 -> Qwen3-8B
construction:    fresh marginals + reverse ANN + Sinkhorn
```

- Guqq：`~/vocab_align/C2C/local/transport/artifacts/mistral_nemo_to_qwen3_8b_openhermes_500k.npz`
- 本地仓库根目录：[`mistral_nemo_to_qwen3_8b_openhermes_500k.npz`](../../mistral_nemo_to_qwen3_8b_openhermes_500k.npz)

| 项目 | 值 |
|---|---|
| Slurm job | `327` |
| 文件大小 | `39,412,012` bytes |
| SHA-256 | `78a01689490224824db6c54460c992091572694cd5c603327e5d84fe3efff84d` |
| shape | `[151643, 131072]` |
| 布局 | `[Qwen target, Mistral source]` |
| 非零元素 | `2,693,524` |
| source active support | `131,072` Mistral token IDs（完整词表） |
| target active support | `151,643` Qwen ordinary token IDs |
| source tokenizer fingerprint | `ea2f93fc906df0a1bdc3e1c4501666dcc2d56413f17ce99acc019af0acfe4fae` |
| target tokenizer fingerprint | `a15bfb56a5f7330ca968a5b72cb34377671ed372d8d52155b3fe77165dbd895d` |
| artifact input fingerprint | `21947b8e10b37f13c766c773c85e391375b3ba791022b613d7ff811e3fd700a3` |
| reverse ANN SHA-256 | `33fb337bea7c28d3d271f24ed02c336ae2f36942a645672392166c59dae6fc7d` |
| 最大列和误差 | `7.1783e-12` |
| transported marginal L1 | `1.9999668e-3` |
| audit | `valid: true`，无危险 special mapping |

该工件从 OpenHermes-2.5 的 495,000 个 `transport_train` 样本重新统计
Mistral source 与 Qwen target marginals，使用独立 Mistral→Qwen ANN 候选图并
从 fresh checkpoint 运行 Sinkhorn；metadata 不含 `derivation`，也没有读取
正向 T。Job 324 的 Bayes 派生版本只在 Guqq 以
`mistral_nemo_to_qwen3_8b_bayes_reverse_openhermes_500k.npz` 保留为对照，
不是正式反向文件。正式反向文件由根目录 `.gitignore` 精确排除，不进入 Git。

## 2. 文件内部结构

该文件是禁止 pickle 的 NumPy `.npz`，其中 T 使用 CSC 稀疏布局：

```text
indptr, indices, data
shape
source_marginal, target_marginal
source_token_ids, target_token_ids
candidate_rows, candidate_columns
candidate_evidence, candidate_sources
metadata
```

`shape[0]` 是 receiver/target 词表轴，`shape[1]` 是 sender/source
词表轴。单纯转置或 Bayes 条件化正向 CSC 只能得到同一联合耦合的反向表示，
不能代替交换模型后重新估计 marginals、构建候选图和求解 OT。正式反向矩阵
正是后一种独立问题的解；Bayes 版本仅作诊断对照。两个方向的 artifact 都不能
复用于其他 tokenizer revision，也不得 dense 化，因为十亿量级 dense 元素会
产生不可接受的内存占用。

## 3. 在 Planner -> Thinker STT 中的作用

新协议先让 planner 生成 think，再对完整的 `sender prompt + sender think`
重新执行 sender forward。对每个有效位置，STT 的概念计算为：

```text
sender hidden state
  -> sender LM head logits
  -> softmax(logits / tau)
  -> sparse T transport
  -> receiver embedding expectation
```

得到的 aligned sender embeddings 按以下顺序送入 receiver：

```text
[aligned sender prompt]
[aligned sender think]
[receiver native prompt]
```

第三段由 receiver tokenizer 原生编码，并显式包含同一道题。sender 与
receiver 的 chat template 都开启 thinking。当前 exact 主协议固定
`causal_shift: false`，因此 sender 完整 context 的每个有效位置都有一枚
aligned embedding。

## 4. 推荐使用方式：evaluation recipe + Slurm

不要手工读取稀疏数组并自行拼接。推荐让 evaluation adapter 完成双提示词、
planner generation、fingerprint 校验、STT、receiver prefill 和记录落盘。

四个现有入口是：

- [`stt_mmlu_redux.yaml`](../../C2C/recipe/eval_recipe/stt_mmlu_redux.yaml)
- [`stt_gsm8k.yaml`](../../C2C/recipe/eval_recipe/stt_gsm8k.yaml)
- [`stt_math500.yaml`](../../C2C/recipe/eval_recipe/stt_math500.yaml)
- [`stt_longbench_qasper.yaml`](../../C2C/recipe/eval_recipe/stt_longbench_qasper.yaml)

它们的 `model.artifact` 应指向：

```yaml
artifact: local/transport/artifacts/qwen3_8b_to_mistral_nemo_openhermes_500k.npz
```

真实模型加载和推理只能在 Guqq 通过 Slurm 执行。例如：

```bash
cd ~/vocab_align/C2C
sbatch script/transport/slurm/evaluate_stt_benchmark.sbatch \
  recipe/eval_recipe/stt_gsm8k.yaml
```

其余 benchmark 只需替换最后一个 recipe 路径。不要在登录节点直接运行
`python -m script.evaluation.unified_evaluator`。

## 5. 程序化加载

低层 loader 位于
[`artifact.py`](../../C2C/rosetta/transport/artifact.py)，最小只读检查为：

```python
from pathlib import Path

from rosetta.transport.artifact import load_transport_artifact

artifact = load_transport_artifact(
    Path("../qwen3_8b_to_mistral_nemo_openhermes_500k.npz"),
    source_fingerprint=(
        "c39a0519aee714a6c8a3eca850849443a5cc6eb7e960211daa11cf5a896061de"
    ),
    target_fingerprint=(
        "12be2a776e12655422f69aabc496e70843d20b1b118c0089314e14d542beca92"
    ),
)

assert artifact.shape == (131069, 151669)
assert artifact.data.size == 2733518
```

按本轮“暂时忽略 fingerprint”的范围，只做反向 artifact 数值检查时可省略
loader 的两个 expected fingerprint 参数：

```python
reverse_artifact = load_transport_artifact(
    Path("../mistral_nemo_to_qwen3_8b_openhermes_500k.npz")
)
assert reverse_artifact.shape == (151643, 131072)
assert reverse_artifact.data.size == 2693524
assert "derivation" not in reverse_artifact.metadata
assert reverse_artifact.metadata["build_config"]["ann"]["sha256"] == (
    "33fb337bea7c28d3d271f24ed02c336ae2f36942a645672392166c59dae6fc7d"
)
```

这只表示跳过 live tokenizer fingerprint 对比；schema、列归一化、marginal、
有限值和稀疏坐标验证仍会执行。它不授权在真实模型推理中关闭 wrapper 的
fingerprint 门禁。

该示例应从 `C2C/` 目录运行。它只加载和验证 artifact，不加载模型。
实际推理应继续使用
[`transport_adapter.py`](../../C2C/script/evaluation/transport_adapter.py)
创建 wrapper；低层
[`TrainingFreeTransportModel`](../../C2C/rosetta/transport/wrapper.py)
接口要求调用者已经正确生成 sender context 并编码 receiver prompt，直接调用
更容易遗漏协议步骤。

## 6. 当前 fingerprint 兼容性警告

正向 artifact 内记录的 source/target fingerprints 分别是 `c39a...` 和
`12be...`，与当前正向 recipe 的 `1a385...` 和 `8542...` 不同。独立反向
artifact 使用当前锁定 tokenizer 重新计算得到 `ea2f...` 和 `a15b...`，不是
交换旧指纹。严格模型运行 loader 仍应拒绝与对应 recipe 不一致的组合；不得
删除或绕过 fingerprint 检查来强行推理。

在继续真实 benchmark 前，需要用锁定 revision 的当前 tokenizer fingerprint
实现复核哪一组值正确，并使 recipe、运行时 tokenizer 和 artifact metadata
三者一致。如果当前 fingerprint 算法与 artifact 构建时不同，应重建 artifact
或执行有审计记录的 schema/provenance 迁移，而不是直接改 metadata。

## 7. 常用校验命令

本地 PowerShell：

```powershell
Get-Item .\qwen3_8b_to_mistral_nemo_openhermes_500k.npz | Select-Object Length
Get-FileHash -Algorithm SHA256 .\qwen3_8b_to_mistral_nemo_openhermes_500k.npz
Get-Item .\mistral_nemo_to_qwen3_8b_openhermes_500k.npz | Select-Object Length
Get-FileHash -Algorithm SHA256 .\mistral_nemo_to_qwen3_8b_openhermes_500k.npz
```

Guqq 登录节点只读校验：

```bash
cd ~/vocab_align
stat -c '%s %n' \
  C2C/local/transport/artifacts/qwen3_8b_to_mistral_nemo_openhermes_500k.npz
sha256sum \
  C2C/local/transport/artifacts/qwen3_8b_to_mistral_nemo_openhermes_500k.npz
sha256sum \
  C2C/local/transport/artifacts/mistral_nemo_to_qwen3_8b_openhermes_500k.npz
```

需要重新构建独立反向 artifact 时，先提交
[`build_reverse_ann_candidates.sbatch`](../../C2C/script/transport/slurm/build_reverse_ann_candidates.sbatch)，
再提交
[`build_reverse_formal_transport.sbatch`](../../C2C/script/transport/slurm/build_reverse_formal_transport.sbatch)。
两者都只能通过 Slurm 运行，不能在登录节点直接执行批量构建。

这些命令只检查文件，不产生明显计算负载。每次连接 Guqq 仍需遵循
`AGENTS.md` 的连接记录和首项 `git pull` 要求。
