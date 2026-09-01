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

第二次服务器运行：revision 字段已正确回填，但 completion audit 发现 JSON 缺少所有 artifact 通用的 schema/input fingerprint/build config/seed/code version，故仍不标记阶段 0 审计完成。已补 provenance 字段与输入敏感性测试，待最终重跑。

provenance 修复本地结果：

- `python -m pytest -o addopts= test/transport/test_tokenizer_audit.py -q`：3 passed（23.51s）。
- `python -m pytest -o addopts= -q`：27 passed（26.64s）。
- `python -m compileall -q script/transport`：通过。
- `git diff --check`：通过。

服务器同步异常：推送 `902ca9c` 后，服务器 HTTPS `git pull` 连续一次 GnuTLS 中断、两次超时，尚未同步 provenance 修复。已按三次失败规则记录经验，下一步改用 GitHub SSH transport；最终审计仍待执行。

SSH transport 结果：`git pull git@github.com:Mingzhe-Xuan/vocab_align.git main` 因服务器无 GitHub public-key 权限失败。已退出会话；不以 scp 覆盖源码，真实审计保持 pending。

网络经验与计划调整文档检查：`git diff --check` 通过。

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
