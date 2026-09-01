# 测试记录

## 2026-09-01：transport 核心与 artifact 实现单元

计划范围：

- 保留现有 exact-byte 与 byte-span 回归行为。
- 使用 2×3 和 3×2 代价矩阵验证 dense Sinkhorn 的方向、两侧边际和收敛报告。
- 验证不可行支撑、非法边际和未收敛均显式失败。
- 验证 `T = Pi Diag(a)^-1` 的列和及 `Ta=b`。
- 验证 artifact 保存/加载保持稀疏索引、dtype、数值和 metadata；schema 或指纹不匹配时拒绝加载。
- 运行 transport 单元测试及现有 C2C 测试，预期全部通过且不访问网络。

实际结果：

- `python -m pytest -o addopts= -q`：8 passed（0.27s）。使用 `-o addopts=` 是因为本地既有环境未安装项目可选的 `pytest-cov`，首次启动在收集前即因未知 `--cov` 参数退出。
- `python -m compileall -q rosetta/transport`：通过。
- 对测试目录执行 `compileall` 时因沙箱拒绝创建其 `__pycache__` 而失败；测试文件已由 pytest 成功导入执行，因此不作为源码编译失败。
- `git diff --check`：通过。

## 2026-09-01：sparse/log-domain Sinkhorn 实现单元

计划范围：

- 候选证据转为 `[V_B,V_A]` 稀疏边代价，图外 kernel 质量严格为零。
- 2×3、3×2 稀疏结果与 dense oracle 在容忍度内一致并满足两侧边际。
- 极小 epsilon 与极端正边际不产生 NaN/Inf。
- 重复边、缺 row/column 支撑、不可行图或 max_iter 未收敛均显式失败。
- 稀疏 convergence report 与 dense 口径一致，包含迭代数和 row/column residual。

实际结果：

- `python -m pytest -o addopts= test/transport/test_sparse_sinkhorn.py -q`：5 passed（0.67s）。
- `python -m pytest -o addopts= -q`：38 passed（29.67s）。
- `python -m compileall -q rosetta/transport`：通过。
- `git diff --check`：通过。

## 2026-09-01：facade、artifact graph 与 audit 实现单元

计划范围：

- facade 串联边际、候选图、sparse Sinkhorn、`T=Pi Diag(a)^-1`，并将 active support 压缩为 artifact 坐标。
- artifact 保存 source/target 原始 token ID 映射及完整候选边来源/证据；旧 schema-1 无候选数组 artifact 可迁移加载。
- 保存/加载后可重算非负性、列和、`Ta=b`、两侧 coupling 残差、候选覆盖、熵和来源统计。
- tokenizer 指纹/方向不匹配、危险 special 映射、损坏候选数组或不收敛构建显式失败。
- toy vocab 同时运行 dense 与 sparse oracle，保存/加载并生成 JSON/Markdown audit，二者数值误差在容忍度内。

实际结果：

- `python -m pytest -o addopts= test/transport/test_artifact.py test/transport/test_audit.py test/transport/test_vocab_transport_facade.py -q`：8 passed（3.10s）。
- `python -m pytest -o addopts= test/transport/test_build_vocab_transport_cli.py test/transport/test_vocab_transport_facade.py -q`：3 passed（4.59s）。
- `python -m pytest -o addopts= -q`：44 passed（27.50s）。
- `python -m compileall -q rosetta/transport script/transport`：通过。
- `git diff --check`：通过。
- 正式 toy 构建：生成 `local/transport/artifacts/toy_oracle.npz`、checkpoint、JSON/Markdown audit（均被 `.gitignore` 排除）。audit 为 valid，shape 4×2、nnz 4、candidate edges 4、dense oracle max error 0、row/column/transported marginal L1 均为 0。
- `--resume`：通过，只读加载并重新审计已验证 artifact，checkpoint 记录 `loaded-valid-artifact`。

## 2026-09-01：exact soft transport 与 metrics 实现单元

计划范围：

- `smoothing>0` 为所有未排除 vocab token 提供正质量，使零覆盖 source 列进入候选 fallback；`smoothing=0` 继续过滤零质量。
- 稀疏 `Tp` 与显式 dense 矩阵一致，`W_in^B(Tp)` 与组合矩阵路径一致。
- batch/sequence 维、概率和、dtype/device 保持正确；artifact 原 token ID 映射正确 gather/scatter。
- `tau<=0`、source vocab 不匹配、非完整 active support 或非法 top-m 显式失败。
- top-m 报告丢弃概率质量，`m=V_A` 与精确路径一致，m 增大时丢弃质量不增加。
- metrics 分段耗时与总耗时在容忍度内；CPU peak memory 为 `None` 而非伪零。

实际结果：

- `python -m pytest -o addopts= test/transport/test_marginals.py test/transport/test_soft_transport.py test/transport/test_metrics.py test/transport/test_optional_torch_import.py -q`：9 passed（3.90s）。
- `python -m pytest -o addopts= -q`：51 passed（24.62s）；仅有本机 pandas 对既有 numexpr/bottleneck 版本的两条 warning。
- `python -m compileall -q rosetta/transport`：通过。
- `git diff --check`：通过；仅报告工作树 LF/CRLF 转换 warning。

第二次服务器运行：revision 字段已正确回填，但 completion audit 发现 JSON 缺少所有 artifact 通用的 schema/input fingerprint/build config/seed/code version，故仍不标记阶段 0 审计完成。已补 provenance 字段与输入敏感性测试，待最终重跑。

provenance 修复本地结果：

- `python -m pytest -o addopts= test/transport/test_tokenizer_audit.py -q`：3 passed（23.51s）。
- `python -m pytest -o addopts= -q`：27 passed（26.64s）。
- `python -m compileall -q script/transport`：通过。
- `git diff --check`：通过。

服务器同步异常：推送 `902ca9c` 后，服务器 HTTPS `git pull` 连续一次 GnuTLS 中断、两次超时，尚未同步 provenance 修复。已按三次失败规则记录经验，下一步改用 GitHub SSH transport；最终审计仍待执行。

SSH transport 结果：`git pull git@github.com:Mingzhe-Xuan/vocab_align.git main` 因服务器无 GitHub public-key 权限失败。已退出会话；不以 scp 覆盖源码，真实审计保持 pending。

最终服务器集成结果：

- Guqq 已通过 HTTPS `git pull` 从 `4ecd48b` fast-forward 到 `36e6224`，未触发 `bash net.sh`。
- 使用 `/home/xmz/vocab_align/C2C/.venv`（Python venv）及两个锁定 revision 完成真实 tokenizer 审计；未安装/加载 PyTorch 或模型权重。
- 产物 provenance 验收通过：`schema_version=1`，input fingerprint、build config、seed、code version、source/target revision 与 tokenizer fingerprint 均存在且 revision 等于 recipe 锁定 SHA。
- 指标：共享唯一 byte strings 67,858；source/target exact-byte 词表覆盖率分别为 0.4474498038/0.5177273039；样本 occurrence 覆盖率 0.8641975309；target/source 平均长度比 1.2177489177。
- 本地接收的忽略目录产物 `C2C/local/transport/audits/qwen3_8b_to_mistral_nemo_instruct_2407.json` SHA-256 为 `31E69CCC0EEBE322FD1D2A278683DADD0493E821C9846E9C64482CC4CAE5BAC5`。

## 2026-09-01：TrainingFreeTransportModel wrapper 实现单元

计划范围：

- source forward 必须运行于 `no_grad`，wrapper 不创建 optimizer state，调用后 source 参数无梯度。
- shift 模式按每条 mask 的首个有效位置注入 receiver 起始 token embedding，其余有效位置使用前一 source 时刻 logits；no-shift 使用同位置 logits，二者长度均与 source prompt 一致。
- 验证 batch size 1/2、单 token prompt、左右 padding；拒绝内部不连续 mask，并检查 position IDs、attention mask 与 cache 长度。
- transport 生成首步使用最后有效 virtual prompt 位置的 receiver logits；后续 decode 仅使用 receiver 原生 token ID/embedding 与 KV cache，不再调用 source。
- EOS 批次可分别提前结束，已结束序列后续补 PAD；生成 token 属于 receiver 词表。
- transport 关闭时走独立 receiver-only 路径，调用结果与 receiver 自身 `generate` 完全一致。
- temperature、top-m、起始 token、输入 shape/vocab 不匹配和缺失 cache 等非法协议显式失败。
- 参考 `docs/assets/alignment.py` 的 row-vector、完整 logits（含 bias）、浮点校验与分块原则；测试确保跨词表 artifact 映射仍生效。

实际结果：

- 首轮定向测试 13 passed/1 failed；失败因左 padding oracle 错把另一行 token 当作前一有效 token。按原因果时序修正测试数据，不修改实现或降低断言。
- 最终 `python -m pytest -o addopts= test/transport/test_config.py test/transport/test_wrapper.py test/transport/test_soft_transport.py test/transport/test_optional_torch_import.py -q`：25 passed（5.88s）。
- `python -m pytest -o addopts= -q`：65 passed（35.86s）；仅有本机 pandas 对既有 numexpr/bottleneck 版本的两条 warning。
- `python -m compileall -q rosetta/transport`：通过。
- Black 对 4 个本单元 Python 文件检查通过，均无需变更。
- `git diff --check`：通过；仅报告工作树 LF/CRLF 转换 warning。

## 2026-09-01：STT smoke diagnostics 实现单元

计划范围：

- wrapper 的结构化生成结果包含 receiver token IDs、virtual prompt shape、transport 质量统计和 source/transport/receiver-prefill/decode 分段 metrics；普通 `generate` tensor 返回行为保持不变。
- CPU peak memory 明确为 unavailable；计时总和严格由四个阶段相加，token 长度与实际输入、virtual prompt、receiver 输出一致。
- smoke 核心函数使用注入的 tiny 模型/tokenizer/artifact 离线端到端运行，输出 receiver 解码文本及锁定配置、artifact provenance、shape、质量和 metrics。
- CLI 只在入口加载 Transformers 模型；导入、`--help` 和单元测试均不下载网络资源。
- artifact 加载必须核对 source/target tokenizer fingerprints；revision、transport tau/shift/top-m 和 generation 参数来自已验证 recipe，不允许混用通信温度与生成温度。
- JSON 使用临时文件原子替换；失败不得留下看似有效的最终产物，成功输出稳定、可 JSON 序列化且包含 code version。

实际结果：

- `python -m pytest -o addopts= test/transport/test_wrapper.py test/transport/test_metrics.py test/transport/test_smoke_stt.py test/transport/test_optional_torch_import.py -q`：19 passed（5.74s）。
- `python -m pytest -o addopts= -q`：69 passed（84.60s）；仅有本机 pandas 对既有 numexpr/bottleneck 版本的两条 warning。
- `python -m compileall -q rosetta/transport script/transport`：通过。
- Black 对 wrapper、smoke CLI 与对应测试共 4 个文件检查通过，均无需变更。
- `git diff --check`：通过；仅报告工作树 LF/CRLF 转换 warning。
- 当前仅验收 CPU tiny 离线诊断管线；真实模型 GPU smoke 尚未执行，不进入正式 latency 结果。

## 2026-09-01：candidate target-support rescue 实现单元

计划范围：

- source 主路径继续保持 special → exact-byte → span → ANN 优先级，不改变已有边选择。
- 若正质量 target 因 source exact 优先级而无入边，先从正质量 source 中增加反向 exact-byte 边，再使用 canonical 文本已观测的 byte-span overlap 补边。
- rescue 只使用 required source support，拒绝 special/ordinary 混合、越界 ID、零/非有限 evidence 和重复边。
- 无安全 exact/span/既有 ANN 证据的 target 仍显式失败，不用任意 token 静默兜底。
- 构造 source 单 token 与 target 多 token 的真实分词形态 toy case，验证原实现会缺 target、rescue 后 Sinkhorn 可行且 artifact 两侧边际通过审计。
- 保留所有既有 candidate graph、sparse Sinkhorn、artifact 与全量回归测试。

实际结果：

- 首轮定向测试 13 passed/1 failed；失败仅因既有错误消息正则要求复数 `target tokens`，新实现报告具体 `target token <id>`。更新正则后保持同一失败语义。
- `python -m pytest -o addopts= test/transport/test_candidate_graph.py test/transport/test_vocab_transport_facade.py test/transport/test_sparse_sinkhorn.py -q`：14 passed（5.35s）。
- `python -m pytest -o addopts= -q`：72 passed（24.94s）；仅有本机 pandas 对既有 numexpr/bottleneck 版本的两条 warning。
- `python -m compileall -q rosetta/transport`：通过。
- Black 对实现与对应测试共 3 个文件检查通过，均无需变更。
- `git diff --check`：通过；仅报告工作树 LF/CRLF 转换 warning。

## 2026-09-01：真实预览 Slurm 作业封装单元

计划范围：

- 作业脚本只从 `C2C` 提交目录运行，检查 Python venv 与输入 JSONL 存在，不直接修改 Git 源码。
- 候选构建、Sinkhorn、审计均在 Slurm allocation 内执行；登录节点仅允许 `git pull`、`sbatch`、`squeue/sacct` 和结果文件校验。
- source/target 名称与锁定 revisions、epsilon/tolerance/max-iter/smoothing/seed、artifact/audit/log 路径均显式记录；输出只进入被忽略的 `local/transport/`。
- `bash -n` 检查 shell 语法；离线 stub `sbatch` 环境验证路径检查、命令参数、失败传播和成功产物位置，不运行真实批量构建。
- 输入 canonical preview JSONL 由本地忽略目录提供并通过 `scp` 传输，不提交数据集/结果；网络恢复前不提交远端作业。

实际结果：

- 首轮 stub 测试 1 passed/2 failed：Codex Bash 启动提示混入 `cygpath` stdout，且 Windows 默认 GBK 无法解码提示。仅修正测试适配为取最后一个路径行并显式 UTF-8 replacement，不修改作业行为。
- `python -m pytest -o addopts= test/transport/test_preview_slurm.py -q`：3 passed（5.23s）。
- 首次全量测试 64 passed/11 setup errors：系统 `%TEMP%/pytest-of-asus` ACL 拒绝；第二次指定 `--basetemp` 时因父目录不存在得到相同 11 个 setup errors。创建忽略目录 `C2C/local/test-tmp` 后，未减少测试范围地重跑通过。
- `python -m pytest -o addopts= --basetemp=local/test-tmp/full_20260901_2058 -q`：75 passed（29.85s）；仅有本机 pandas 对既有 numexpr/bottleneck 版本的两条 warning。
- `bash -n script/transport/slurm/build_preview.sbatch`：通过。
- Black 对 Slurm 测试文件检查通过；`git diff --check` 通过，仅有 LF/CRLF warning。
- `git check-ignore -v --no-index`：确认 preview inputs、Slurm logs 与本地 pytest basetemp 均由精确 `.gitignore` 规则覆盖。

## 2026-09-01：ANN graph augmentation 语义单元

计划范围：

- special source 仍只走功能映射，ANN 不得连接 control/special。
- ordinary source 保留 exact-byte 优先于 observed-span 的基础边；提供 ANN 时，对每个 ordinary source 都追加 ANN 候选，而不只处理无基础边的 source。
- ANN 与已有 exact/span pair 重合时保留已有高优先级证据并跳过低优先级重复；ANN 返回内部重复 pair 仍显式失败。
- 没有 exact/span 且 ANN 为空的 source 仍失败；ANN target 必须在普通 target 词表内且 evidence 有限为正。
- 验证 ANN 增广可连接原本孤立的 exact 分量，并为双向 top-k candidate JSON 覆盖 target support 提供入口；保留完整回归。

实际结果：

- `python -m pytest -o addopts= --basetemp=local/test-tmp/ann_aug_target test/transport/test_candidate_graph.py test/transport/test_vocab_transport_facade.py test/transport/test_sparse_sinkhorn.py -q`：15 passed（4.72s）。
- `python -m pytest -o addopts= --basetemp=local/test-tmp/ann_aug_full -q`：76 passed（28.49s）；除既有 pandas optional dependency warning 外，pytest cache 因工作区 ACL 无法写入的 warning 不影响测试执行或结果。
- `python -m compileall -q rosetta/transport`：通过。
- Black 对实现与测试检查通过；`git diff --check` 通过，仅有 LF/CRLF warning。

网络经验与计划调整文档检查：`git diff --check` 通过。

## 2026-09-01：GPU 测试提交流程规范修订

计划范围：

- 检查临时验证提交、验收提交和正式分支的边界是否明确。
- 检查服务器仍只通过 `git pull` 同步受 Git 管理的源码，GPU 测试仍通过 Slurm 执行。
- 检查 Markdown 格式、文档路径和 Git diff。

实际结果：

- 语义检查通过：规范明确临时验证提交必须位于临时分支且标记为未验收，GPU 测试通过后才能形成验收提交或合并。
- 路径检查通过：`docs/agents/test.md`、`docs/agents/gpu.md`、`docs/agents/state.md` 和 `docs/agents/update.md` 均存在。
- `git diff --check`：通过；仅报告工作区既有的 LF/CRLF 转换 warning，无空白错误。

## 2026-09-01：候选图与边际实现单元

计划范围：

- special 功能映射优先于 exact/span/ANN，无法安全映射的 required special 显式失败。
- duplicate exact bytes 产生按 target ID 排序的确定候选，不任意挑选。
- ASCII、中文、emoji 与组合字符的 byte-span overlap 计数正确。
- ordinary source 按 exact→span→ANN 逐级 fallback；无安全 fallback 时失败。
- required source/target 正质量支撑缺边时显式失败。
- marginal 只调用 `add_special_tokens=False` 的 canonical 内容 tokenizer；平滑后 active 概率严格为正、归一化为 1，零质量 token 留在有效支撑外。
- 完整离线测试保留既有 27 个用例。

实际结果：

- `python -m pytest -o addopts= test/transport/test_candidate_graph.py test/transport/test_marginals.py -q`：6 passed（1.04s）。
- `python -m pytest -o addopts= -q`：33 passed（26.84s）。
- `python -m compileall -q rosetta/transport`：通过。
- `git diff --check`：通过。

## 2026-09-01：真实 tokenizer 审计（服务器集成）

计划范围：

- 服务器连接后首先 `git pull`，确认代码版本包含阶段 0 实现。
- 使用 Python 虚拟环境；若需新建，使用 `python -m venv` 并记录 Python/依赖版本。
- 按 recipe 的锁定 revision 加载两个 fast tokenizer，不下载模型权重。
- 输出全词表 bytes/special-control/exact-byte 覆盖与 sample 长度审计 JSON，并检查 revision 与 schema 字段。
- 该任务仅为轻量 tokenizer 元数据处理；若资源表现超出预期则改走 Slurm。

连接前文档检查：`docs/agents/gpu.md` 与锁定 recipe 路径存在，`git diff --check` 通过。服务器集成结果待执行。

首次服务器运行：通过 `hf-mirror.com` 下载缺失的 Mistral tokenizer 并完成全词表计算，但报告中的 resolved revision 为 `null`；原因是 Transformers 的 cache/mirror 路径未填写私有 `_commit_hash`。该报告不作为合格产物，已增加显式锁定 SHA 回退逻辑与回归测试，待重新运行。

revision 修复本地结果：

- `python -m pytest -o addopts= test/transport/test_tokenizer_audit.py -q`：2 passed（10.45s）。
- `python -m pytest -o addopts= -q`：26 passed（11.13s）。
- `python -m compileall -q script/transport`：通过。
- `git diff --check`：通过。

## 2026-09-01：baseline 快照实现单元

计划范围：

- canonical messages 与 source/target rendered prompts 分字段保存并计算输入指纹。
- 快照包含 schema version、构建配置、seed、代码版本、模型/tokenizer revision、生成参数、依赖和硬件状态。
- pending 或不存在的 C2C checkpoint 显式标记，不伪造成可用结果。
- 相同输入产生稳定 JSON；非法 message/prompt schema 显式失败。
- CLI 使用现有锁定 recipe 和预渲染输入生成离线 tiny snapshot。

实际结果：

- `python -m pytest -o addopts= test/transport/test_baseline.py -q`：3 passed（0.97s）。
- `python -m pytest -o addopts= -q`：24 passed（1.63s）。
- `python -m compileall -q rosetta/transport script/transport`：通过。
- `git diff --check`：通过。

## 2026-09-01：阶段 0 配置与 manifest 实现单元

计划范围：

- 合法配置稳定序列化/反序列化，保留模型 revision、seed、输出 schema 与生成参数。
- 缺少 revision、seed、输出路径，或使用 benchmark test split 构建 transport 时显式失败。
- `pending-new-projector-training` 只能作为 pending 状态，不可解析成可加载 checkpoint。
- 相同 seed/样本 ID 产生字节级一致 manifest；输入顺序变化不改变划分。
- 检测重复 sample ID，并验证 train/dev 无交集、无重复且数量符合约定。
- CLI 输入输出使用 JSONL/JSON，执行帮助与 tiny fixture smoke，不访问网络。

实际结果：

- `python -m pytest -o addopts= test/transport/test_config.py test/transport/test_manifest.py -q`：9 passed（0.59s）。
- `python -m pytest -o addopts= -q`：17 passed（0.67s），包含旧词表传输回归。
- `python -m compileall -q rosetta/transport script/dataset`：通过。
- 使用 `yaml.safe_load` 读取主 recipe 并交给 `TransportConfig.from_dict`：通过；source/target revision 为锁定 SHA，target checkpoint 状态为 unavailable/pending。
- `git diff --check`：通过。
- 首次 CLI smoke 采用文件路径直接执行，因项目未安装到当前解释器而无法导入 `rosetta`；改为文档统一的 `python -m script.dataset.build_transport_manifest` 后通过，未修改测试断言。

## 2026-09-01：token metadata 实现单元

计划范围：

- UTF-8、多字节字符与 GPT/Qwen byte-level BPE token 恢复为正确 raw bytes。
- fast tokenizer 的字符 offset 转换为 byte offset，覆盖中文、emoji 与组合字符。
- special/control token 分类明确，且不进入普通 exact-byte 索引。
- 相同 token ID 但不同 raw bytes 的两个 tokenizer 不产生 exact 匹配。
- 现有 `build_small_transport` exact/span 回归继续通过；比较脚本复用公共 metadata 逻辑并可静态导入。

实际结果：

- `python -m pytest -o addopts= test/transport/test_token_metadata.py test/test_vocab_transport.py -q`：5 passed（0.22s）。
- `python -m pytest -o addopts= -q`：21 passed（0.71s）。
- `python -m compileall -q rosetta/transport script/transport`：通过。
- `import script.transport.compare_tokenizers`：通过（13.45s）；本地 pandas 报告既有 numexpr/bottleneck 版本 warning，不影响导入。
- `git diff --check`：通过。
