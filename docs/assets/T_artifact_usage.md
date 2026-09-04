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

由同一联合耦合派生的反向 artifact 对应：

```text
sender/source:   mistralai/Mistral-Nemo-Instruct-2407
receiver/target: Qwen/Qwen3-8B
direction:       Mistral-Nemo-Instruct-2407 -> Qwen3-8B
derivation:      bayes-joint-reversal-v1
```

- Guqq：`~/vocab_align/C2C/local/transport/artifacts/mistral_nemo_to_qwen3_8b_openhermes_500k.npz`
- 本地仓库根目录：[`mistral_nemo_to_qwen3_8b_openhermes_500k.npz`](../../mistral_nemo_to_qwen3_8b_openhermes_500k.npz)

| 项目 | 值 |
|---|---|
| Slurm job | `324` |
| 文件大小 | `40,694,539` bytes |
| SHA-256 | `77905324ee9e063aef33c0e01a73c26bf4ac7907c8a48f972c463f5af3eb486f` |
| shape | `[151669, 131069]` |
| 布局 | `[Qwen target, Mistral source]` |
| 非零元素 | `2,733,518` |
| source active support | `131,069` Mistral token IDs |
| target active support | `151,669` Qwen token IDs |
| source tokenizer fingerprint | `12be2a776e12655422f69aabc496e70843d20b1b118c0089314e14d542beca92`（未做 live 验证） |
| target tokenizer fingerprint | `c39a0519aee714a6c8a3eca850849443a5cc6eb7e960211daa11cf5a896061de`（未做 live 验证） |
| artifact input fingerprint | `9d42ecb84aa005828aa2e5535b7feda4c60991d996f424cbb7e651b3efa4bba8` |
| 最大列和误差 | `4.9072e-14` |
| transported marginal L1 | `1.0520e-13` |
| audit | `valid: true`，无危险 special mapping |

反向文件也由根目录 `.gitignore` 精确排除，不进入 Git。

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
词表轴。单纯把 CSC 矩阵转置不能得到反向条件分布，因为新列一般不归一化。
本仓库的
[`reversal.py`](../../C2C/rosetta/transport/reversal.py)
先恢复联合质量
`J[target, source] = T[target, source] * source_marginal[source]`，再用
正向实际 transported target marginal 做 Bayes 条件化；因此逐边联合质量保持
不变。正向的 recorded target marginal 与 realized marginal 相差
`0.0019655245`，反向必须采用后者，不能直接交换 metadata marginal。

这个派生反向矩阵表示同一个正向 OT 联合耦合的反向条件分布，不等同于交换
模型后重新构图、估计边缘并求解一次新的 OT。两种研究问题应使用不同文件名
和 provenance。两个正式 artifact 都不能复用于其他 tokenizer revision，也
不得 dense 化，因为
`131069 x 151669` 的 dense 矩阵会产生不可接受的内存占用。

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
assert reverse_artifact.shape == (151669, 131069)
assert reverse_artifact.metadata["derivation"]["method"] == (
    "bayes-joint-reversal-v1"
)
assert reverse_artifact.metadata["derivation"]["joint_mass_preserved"] is True
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

artifact 内记录的 source/target fingerprints 分别是 `c39a...` 和
`12be...`。当前 transport recipe 中冻结的值是 `1a385...` 和 `8542...`，
二者并不相同。反向 artifact 按方向交换了原 artifact 中的两个 fingerprint，
并明确记录 `fingerprint_validation: not-performed`。严格模型运行 loader 应当
拒绝与当前 recipe 不一致的组合；不得删除或绕过 fingerprint 检查来强行推理。

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

需要从正向 artifact 重新派生反向 artifact 时，只能提交 CPU Slurm 入口
[`reverse_formal_transport.sbatch`](../../C2C/script/transport/slurm/reverse_formal_transport.sbatch)，
不能在登录节点直接执行转换 CLI。

这些命令只检查文件，不产生明显计算负载。每次连接 Guqq 仍需遵循
`AGENTS.md` 的连接记录和首项 `git pull` 要求。
