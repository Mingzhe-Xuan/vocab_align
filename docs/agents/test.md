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
