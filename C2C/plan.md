# Training-free Alignment 接入 C2C 的實施計畫

## 1. 目標

在不破壞原有 C2C 訓練、checkpoint 和評估流程的前提下，將倉庫外的
`alignment.py` 接入 C2C，並提供統一的 `--method` 選項，使同一套 evaluator
可以比較以下五種方法：

| method | 定義 | 是否訓練 | 相容性要求 |
|---|---|---:|---|
| `mlp` | 原始 C2C KV-cache projector/fuser | 是 | 不新增詞表或 hidden dimension 限制 |
| `identical` | source final hidden 直接送入 receiver embedding 空間，並可做 norm scaling | 否 | 詞表完全一致；source hidden dimension 與 receiver input embedding dimension 相同 |
| `linear` | 由 source output weight 到 receiver input embedding 求 ridge least-squares map | 否 | 詞表及 token ID 完全一致；兩邊 dimension 可不同 |
| `kernel` | 用 ORF positive features 近似 softmax expectation | 否 | 詞表及 token ID 完全一致；兩邊 dimension 可不同 |
| `exact` | 顯式計算 source logits 的 softmax，再對 receiver input embeddings 求期望 | 否 | 詞表及 token ID 完全一致；兩邊 dimension 可不同 |

`mlp` 是原有 C2C 路徑，不是新增的 hidden-to-embedding MLP。它保留原本的
checkpoint 載入、KV-cache projector 和 tokenizer alignment 行為。

本計畫中的「詞表完全一致」指 token string 到 token ID 的完整映射一致，而不只是
vocabulary size 相同。「模型維度完全對應」在 `identical` 中具體指：

```text
source final hidden size == receiver input embedding size
```

不要求兩模型 layer 數、attention head 數或 KV head 數相同，因為 training-free
方法不傳遞逐層 KV cache。

## 2. 向後相容原則

1. 不修改原有 `C2CProjector`、`RosettaModel` checkpoint 格式和
   `projector_config.json`。
2. 不改變 `script/train/SFT_train.py` 的既有訓練語義。原 C2C fuser 仍由原腳本訓練。
3. `unified_evaluator.py` 未傳 `--method` 時：
   - 若 recipe 未指定 method，默認為 `mlp`；
   - 因而現有命令和 recipe 繼續走原始 C2C 路徑。
4. CLI 的 `--method` 優先於 recipe，實際採用的 method 必須寫入結果 metadata。
5. `mlp` 回歸測試必須證明接入前後使用相同 checkpoint、prompt、seed 和 generation
   config 時，輸出及評分不變。

## 3. 建議的文件結構

```text
C2C/
├── rosetta/
│   └── model/
│       ├── alignment.py              # 移入並完善現有 alignment primitives
│       ├── alignment_compat.py       # tokenizer/模型相容性檢查
│       └── alignment_wrapper.py      # TrainingFreeAlignmentModel
├── rosetta/
│   └── utils/
│       └── evaluate.py               # 新增 training-free loader
├── script/
│   └── evaluation/
│       └── unified_evaluator.py      # --method 與模型分派
├── recipe/
│   └── eval_recipe/
│       ├── alignment_identical.yaml
│       ├── alignment_linear.yaml
│       ├── alignment_kernel.yaml
│       └── alignment_exact.yaml
└── test/
    └── alignment/
        ├── test_compat.py
        ├── test_methods.py
        ├── test_shift.py
        ├── test_wrapper.py
        └── test_mlp_regression.py
```

使用 `alignment.py` 而不是擴展現有 `aligner.py`。後者負責不同 tokenizer 的離散
token/位置對齊，與本功能的 latent-space alignment 應保持獨立。

## 4. Alignment API

### 4.1 統一方法名稱

將現有型別擴展為：

```python
AlignMethod = Literal["identical", "linear", "kernel", "exact"]
```

`mlp` 不加入 `AlignmentState`，因為它由原 `RosettaModel` 和 C2C projector 執行。
模型 factory/evaluator 負責在 `mlp` 和 training-free wrapper 之間分派。

### 4.2 State builder

提供統一入口：

```python
def build_alignment_state(
    method: AlignMethod,
    source_model,
    receiver_model,
    *,
    ridge: float = 1e-4,
    feature_count: int = 1024,
    temperature: float = 1.0,
    seed: int = 42,
    chunk_size: int = 2048,
    normalize_output: bool = True,
) -> AlignmentState:
    ...
```

需要補上 `build_identical_state()`。`exact` 不必預計算大矩陣，但 state 需保存
temperature、normalization 選項，以及執行所需的 source output head 和 receiver
input embedding 引用或明確由 wrapper 傳入。

所有解析構建使用 FP32；推理輸出最後轉回 receiver dtype。state 每個 evaluator
worker 只建立一次，不得逐題重建。

### 4.3 Exact oracle

對 source final hidden `h`，嚴格計算：

```python
logits = source_model.get_output_embeddings()(h)
probs = torch.softmax(logits.float() / temperature, dim=-1)
aligned = probs @ receiver_model.get_input_embeddings().weight.float()
```

要求：

- 不得使用 top-k、top-p、稀疏截斷或 ORF；
- softmax 和期望計算使用 FP32；
- source output head bias 必須包含在 logits 中；
- temperature 的定義統一為對完整 logits 做
  `softmax(logits / temperature)`；
- `kernel` 的 bias/temperature 處理必須與 exact 的定義一致；
- 若 full logits 引起 OOM，先降低 evaluation batch/sequence chunk，而不是把
  近似結果標成 exact；
- 可後續實現數學等價的分塊 log-sum-exp，但必須先以顯式 full-softmax 版本作
  golden oracle，並通過數值等價測試。

### 4.4 Normalization

現有 `identical`、`linear` 會做 embedding-norm scaling，而 `kernel` 不會。為避免
比較時混入未記錄的差異，將其改成統一配置：

```yaml
normalize_output: true
```

主實驗固定一個值，另做 `true/false` 消融。`exact` 默認不額外 normalize，因為
softmax expectation 本身已有明確含義；若開啟 normalization，結果必須標記為
`exact+norm`，不能和純 exact oracle 混稱。

## 5. 嚴格相容性檢查

新增：

```python
def validate_alignment_compatibility(
    method: str,
    source_model,
    receiver_model,
    source_tokenizer,
    receiver_tokenizer,
) -> CompatibilityReport:
    ...
```

### 5.1 詞表完全一致

對 `identical`、`linear`、`kernel`、`exact`，至少檢查：

1. `get_vocab()` 完整 `token -> id` dictionary 相同；
2. `get_added_vocab()` 相同；
3. vocabulary length 相同；
4. BOS/EOS/PAD/UNK 及 additional special token 的 token 和 ID 相同；
5. fast tokenizer backend 的 model、normalizer、pre-tokenizer、post-processor
   和 decoder 配置相同；
6. chat template 相同，或由 evaluator 明確指定唯一 canonical template；
7. 固定 probe strings 的 encode 結果相同；
8. 每個實際 prompt 在運行時產生的 source/receiver `input_ids` 逐元素相同。

檢查失敗必須在第一個 model forward 前 raise，錯誤信息包含第一個不一致的 token、
ID、special-token 字段或 prompt 位置。不能只 warning 後繼續。

### 5.2 Dimension 檢查

`identical` 額外檢查：

```python
source_model.get_output_embeddings().weight.shape[1]
    == receiver_model.get_input_embeddings().weight.shape[1]
```

並在實際 forward 後再次檢查：

```python
source_hidden.shape[-1]
    == receiver_model.get_input_embeddings().embedding_dim
```

`linear`、`kernel`、`exact` 允許兩邊 dimension 不同。

`mlp` 不執行以上新增限制，只沿用原 C2C loader/projector 自身的合法性檢查。

### 5.3 可審計報告

每次實驗保存：

```json
{
  "method": "kernel",
  "source_model": "...",
  "receiver_model": "...",
  "source_tokenizer_fingerprint": "...",
  "receiver_tokenizer_fingerprint": "...",
  "vocab_equal": true,
  "special_tokens_equal": true,
  "backend_equal": true,
  "source_hidden_dim": 896,
  "receiver_embedding_dim": 1024
}
```

fingerprint 只用於記錄和快速比較，不能替代結構及實際 encode 檢查。

## 6. TrainingFreeAlignmentModel

新增一個獨立 wrapper，不把 training-free 邏輯塞入 `RosettaModel`：

```python
class TrainingFreeAlignmentModel(nn.Module):
    def __init__(
        self,
        source_model,
        receiver_model,
        source_tokenizer,
        receiver_tokenizer,
        alignment_state,
        *,
        causal_shift: bool = True,
    ):
        ...
```

初始化時：

- 兩個模型設為 `eval()`；
- 所有參數 `requires_grad_(False)`；
- 禁止建立 optimizer；
- 執行一次完整 compatibility validation；
- 建立並緩存 alignment state。

### 6.1 Prefill 流程

主路徑：

```text
canonical prompt
→ source/receiver tokenizer runtime equality check
→ source forward(output_hidden_states=True, use_cache=False)
→ 取得送入 source LM head 的 final hidden states
→ causal shift
→ apply_alignment
→ receiver inputs_embeds prefill
→ receiver 原生 KV cache
→ receiver decode
```

causal shift 定義為：

```python
first_embed = receiver_input_embedding(input_ids[:, :1])
aligned = apply_alignment(source_hidden[:, :-1], state)
receiver_inputs_embeds = torch.cat([first_embed, aligned], dim=1)
```

即 source position `t` 的 hidden 預測 token `t+1`，因此放到 receiver position
`t+1`。`causal_shift=false` 只作消融，不能默認開啟。

必須確認取到的是 source LM head 實際消費的 final normalized hidden state，而不是
任意中間層。若某模型的 `hidden_states[-1]` 與 LM head 輸入不一致，使用 forward
hook 捕獲 LM head input，並為該模型加測試。

### 6.2 Generation

優先實現顯式兩階段生成，避免依賴不同 Transformers model 對
`generate(inputs_embeds=...)` 的不一致處理：

1. receiver 以 `inputs_embeds` 完成一次 prefill，取得 receiver `past_key_values`
   和 next-token logits；
2. 依 generation config 選出第一個 token；
3. 後續 token 使用 receiver 原生 `input_ids + past_key_values` autoregressive
   decode；
4. 返回僅包含新生成 token 的統一 output。

第一版至少支持主實驗需要的 greedy decoding：

```yaml
do_sample: false
max_new_tokens: 64
```

採樣模式可在 greedy 路徑驗收後補上 temperature/top-p/top-k，且其 temperature
必須與 alignment communication temperature 分開命名。

wrapper 應返回或記錄：

- generated token IDs；
- source forward latency；
- alignment latency；
- receiver prefill latency；
- receiver decode latency；
- peak GPU memory。

## 7. Loader 與 CLI 接入

### 7.1 統一 loader

在 `rosetta/utils/evaluate.py` 新增：

```python
def load_training_free_alignment_model(
    model_config,
    eval_config,
    device,
    generation_config=None,
):
    ...
```

沿用現有配置名稱：

- `base_model`：receiver；
- `teacher_model`：source。

training-free 方法不要求 `checkpoints_dir`。`mlp` 仍要求原 C2C checkpoint。

### 7.2 `--method`

在 `script/evaluation/unified_evaluator.py` 增加：

```text
--method {mlp,identical,linear,kernel,exact}
```

解析優先級：

```text
CLI --method
→ model.alignment_config.method
→ mlp
```

模型分派：

```python
if method == "mlp":
    model, tokenizer = load_rosetta_model(...)
else:
    model, tokenizer = load_training_free_alignment_model(...)
```

`--method` 只控制評估/推理方法。`SFT_train.py` 保持原狀，因為只有 `mlp` 路徑需要
訓練。若未來也在訓練入口加入同名參數，非 `mlp` 必須明確拒絕啟動訓練，而不是
靜默建立 optimizer。

### 7.3 Recipe

統一格式示例：

```yaml
model:
  model_name: Rosetta
  rosetta_config:
    base_model: Qwen/receiver
    teacher_model: Qwen/source
    checkpoints_dir: path/to/c2c/final  # 僅 mlp 使用

  alignment_config:
    method: kernel
    causal_shift: true
    normalize_output: false
    temperature: 1.0
    feature_count: 1024
    ridge: 1.0e-4
    seed: 42
    chunk_size: 2048

  generation_config:
    do_sample: false
    max_new_tokens: 64
```

CLI 可覆蓋：

```bash
python script/evaluation/unified_evaluator.py \
  --config recipe/eval_recipe/alignment_kernel.yaml \
  --method exact
```

輸出路徑或結果文件名必須包含 method，避免不同方法互相覆蓋。

## 8. 測試計畫

### 8.1 Alignment 單元測試

1. `identical` shape、dtype、norm scaling；
2. `identical` dimension mismatch fail-fast；
3. `linear` 與小矩陣顯式 ridge 解一致；
4. `linear` 記錄相對 reconstruction residual；
5. `exact` 與手工 full-softmax expectation 一致；
6. `exact` 包含 output-head bias；
7. `kernel` 在小詞表上逼近 exact；
8. 增加 feature count 時記錄 kernel error 曲線；
9. temperature 對 exact/kernel 使用相同定義；
10. denominator 非有限或非正時 fail-fast。

主要誤差指標：

```text
cosine similarity
relative L2 error
norm ratio
max absolute error
```

### 8.2 Compatibility 測試

覆蓋：

- vocabulary size 相同但 token ID permutation 不同；
- vocab 相同但 added token 不同；
- special token ID 不同；
- backend normalizer/pre-tokenizer 不同；
- chat template 不同；
- probe encode 不同；
- 實際 prompt runtime IDs 不同；
- identical dimension mismatch；
- mlp 不受新增的 strict compatibility gate 阻斷。

### 8.3 Causal shift 測試

以人工 logits/embedding 建立小例子，驗證：

```text
receiver[0] = 真實第一 token embedding
receiver[t+1] = align(source_hidden[t])
```

同時驗證 attention mask、position、prefill cache length 和 prompt length 一致。

### 8.4 Wrapper 整合測試

- `max_new_tokens=1` 的第一 token 與手工 receiver prefill logits 一致；
- 多 token greedy decode 與手工 autoregressive loop 一致；
- 所有參數均無梯度；
- repeated call 不污染 KV cache；
- batch size 1 和 padding case；
- EOS 提前停止；
- 結果只包含新生成 token，evaluator 不做錯誤 prompt slicing。

### 8.5 MLP 回歸測試

用現有 checkpoint 和固定短樣本比較接入前後：

- projector 數量及 state dict 不變；
- `projector_dict` 不變；
- prompt token IDs 不變；
- generated IDs 不變；
- benchmark sample count、跳過數和 accuracy 不變。

## 9. 分階段實施與驗收

### 階段 0：凍結原始基線

記錄當前 C2C commit、依賴、模型 revision、checkpoint、固定 prompt、generation
config 和輸出。跑 receiver-only、source-only、T2T、原 C2C(`mlp`) 小樣本。

驗收：

- 保存逐題輸出；
- 原始 C2C 可重現；
- 後續有可比較的 regression artifact。

### 階段 1：移入 primitives 並補 exact

將根目錄 `alignment.py` 移入 `rosetta/model/alignment.py`，補
`build_identical_state()`、統一 factory、normalization 配置和 `exact`。

驗收：

- alignment 單元測試通過；
- exact 與手工 full-softmax 一致；
- kernel 以 exact 為 oracle 報告誤差。

### 階段 2：Compatibility gate

實現 tokenizer/model 檢查和 fingerprint/report。

驗收：

- 四個 training-free 方法在不合法模型對上於 forward 前失敗；
- 錯誤能指出首個差異；
- `mlp` 原路徑不受影響。

### 階段 3：Training-free wrapper

實現 source forward、final hidden 捕獲、causal shift、receiver prefill 和 greedy
decode。

驗收：

- 單 prompt 能由四個 training-free 方法生成；
- exact 第一 token 與手工計算一致；
- cache、mask、position 和輸出長度測試通過；
- trainable parameter 數為 0。

### 階段 4：Evaluator 與 CLI

加入 `--method`、loader 分派、recipe、結果 metadata 和分段 latency。

驗收：

- 同一 recipe 可由 CLI 切換五種方法；
- 不傳 `--method` 仍運行原始 `mlp`；
- 不同 method 不覆蓋結果；
- 每題使用相同 prompt 和 generation config。

### 階段 5：對比實驗

先在固定 MMLU-Redux 子集 smoke test，再跑全量。主表：

| 方法 | Accuracy | E2E latency | Alignment latency | Peak memory | Trainable params |
|---|---:|---:|---:|---:|---:|
| Receiver-only | | | 0 | | 0 |
| Source-only | | | 0 | | 0 |
| T2T | | | N/A | | 0 |
| MLP/C2C | | | N/A | | >0 |
| Identical | | | | | 0 |
| Linear | | | | | 0 |
| Kernel | | | | | 0 |
| Exact oracle | | | | | 0 |

固定：

- source/receiver/tokenizer revision；
- benchmark 樣本及順序；
- canonical prompt；
- dtype、device、attention implementation；
- greedy decoding 和 max tokens；
- seed。

至少做：

- causal shift / no shift；
- normalization on/off；
- kernel feature count；
- communication temperature；
- linear ridge；
- kernel 相對 exact 的精度—速度曲線。

逐題結果用 paired bootstrap confidence interval 或 McNemar test 比較，不能只報總
accuracy。

## 10. 完成定義

以下條件全部滿足才視為完成：

1. `alignment.py` 成為可安裝 `rosetta` package 的一部分；
2. evaluator 支持
   `--method {mlp,identical,linear,kernel,exact}`；
3. 不帶 `--method` 的原始 C2C 實驗不被打斷；
4. compatibility 規則按 method 正確執行並 fail-fast；
5. exact 嚴格使用完整 softmax，不包含近似或截斷；
6. kernel 有 exact oracle 數值對照；
7. training-free wrapper 完成 causal shift、prefill 和 decode；
8. 五種 method 能在同一固定子集上輸出可比較的逐題結果；
9. 所有單元、整合與 MLP regression 測試通過；
10. 結果中保存 method、模型/tokenizer fingerprint、配置、latency 和 peak memory。
