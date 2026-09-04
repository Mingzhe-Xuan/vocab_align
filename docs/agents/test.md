# 测试记录

## 2026-09-02：阶段 4 transport approximation 核心单元

计划范围：

- `approximations.py`：TH 必须等于精确 `Tp` 的确定性 argmax 对应 receiver embedding（tie 取最小原 token ID）；edge-chunk sparse accumulation 与预计算 `C_i=W_in^B T[:,i]` 均对齐现有 exact embedding oracle；source top-m 全词表退化为 exact，丢弃质量沿 m 单调不增。
- `orf.py`：按 `docs/assets/algo_detail.md`/用户提供的 `docs/assets/alignment.py` 使用 seeded block-orthogonal Gaussian directions和正特征；bias 与通信 tau 同时进入分子/分母，稀疏 T 分块构造 `S,z`，在线 row-vector 公式为 `u @ S.T / (u @ z)`。
- ORF 固定 seed 字节级可复现，不同 feature count 的 shape/内存估算正确；非法维度/tau/chunk、fingerprint/vocab 不匹配、非有限特征或非正 denominator 显式失败。
- cosine/relative error 对双零、单零向量给出明确有限/`inf` 语义，不产生 NaN；近似质量与 T 质量分开报告。
- 运行新增 approximation/ORF oracle、现有 soft transport/wrapper 定向回归、optional-torch import、完整 pytest、Black 与 `git diff --check`。

远端边界：Job 230 恢复连接 pull 先后因 GnuTLS `-110` 与 `net.sh` 后 443 timeout 失败，按 AGENTS 未执行后续查询；本单元完全本地、无需 GPU。

实际结果：

- approximation/ORF、现有 soft transport 与 optional-torch/public-export 定向回归：`26 passed in 2.46s`；新增 public export 单测独立复核：`2 passed in 1.88s`。
- 完整回归：`127 passed, 2 warnings in 55.66s`；两条 warning 仅为既有 pandas 对可选 `numexpr`/`bottleneck` 版本的提示。
- Black 首轮发现 5 个文件需格式化；完成格式化后复核 6 个相关 Python 文件全部保持不变。Windows ACL 阻止沙箱写入两个新测试文件时，按既有经验仅提升格式化写权限后成功复核，未缩减测试范围。
- `git diff --check`：通过，仅有工作树 LF→CRLF 提示，无 whitespace error。

## 2026-09-02：scaled-dual early-termination 修复单元

计划范围：

- 固定与 Job 230 相同的对偶、gauge 和 `sqrt(marginal)` 坐标，验证相对当前 Sinkhorn 点的增量目标及解析梯度与中心有限差分一致，并在大绝对 dual/小增量下避免原目标常数项抵消。
- 模拟 SciPy 首次仅消耗少量 evaluation 即返回未改善候选，验证求解器会在同一总 `acceleration_max_evaluations` 内确定性重启，而不是永久回到标准 scaling；每次候选必须使两侧原始 L1 residual 的最大值严格改善才可接受。
- 病态稀疏图必须在原 `1e-9` 两侧 residual 与总 `max_iter` 内收敛；`maxcor`、累计 evaluation、termination provenance 与非有限/预算失败语义显式且有界。
- 运行 sparse Sinkhorn/facade/audit 定向回归、完整 pytest、Black、compile 与 `git diff --check`；真实 2.3M-edge preview 只通过临时未验收提交和 Slurm 验证。

远端前置证据：Job 230 为 Exit 1，40:50.55，MaxRSS 1,847,076 KiB/0 swap；27 次 acceleration evaluations 后总 10,000 次仍为 row/column residual `4.6615468745e-4`/`6.0559911057e-14`，checkpoint `building`，artifact/audit 不存在。

本地阶段实际结果：

- 首次直接运行 pytest 因系统 Python 未安装项目配置引用的 pytest-cov 而在收集前失败；使用项目既有的 `-o addopts="--strict-markers --strict-config"` 离线测试协议重跑，未跳过任何用例。
- 首轮 sparse 回归 10/13 通过；3 个失败均为新协议 fixture 不一致（固定 gauge kernel、重启覆盖参数捕获、有限差分预算不足）。修正 fixture 后 sparse 13/13、facade/artifact/audit 24/24 通过。
- 完整本地回归：`129 passed, 2 warnings in 51.43s`；warnings 仍仅为 pandas 可选依赖版本提示。
- Black 首轮要求格式化 `sinkhorn.py`，沙箱内因 Windows ACL 无法原子替换；按既有经验仅提升两个明确文件的格式化权限，最终 `2 files would be left unchanged`。`compileall`、格式化后的 24 个定向回归和 `git diff --check` 均通过。
- 真实 2.3M-edge preview 尚未执行；本单元仍为未验收状态，只能创建临时验证分支提交，不得合并/形成 main 验收提交。
- 临时提交登记检查：branch `validation/job230-dual-increment`、commit `cfa1a87`、首条 pull、同配置 Slurm 边界与独立 job 后缀产物路径一致；相关文档 `git diff --check` 在连接前复核。

Job 232 实际结果：`39:06.15`、Exit 1、MaxRSS `1,847,136 KiB`、0 swap；21 attempts/1,000 evaluations，前 20 个 termination 均为 SciPy `RELATIVE REDUCTION OF F <= FACTR*EPSMCH`，最终 row/column residual `1.6915612104e-3`/`2.6332792027e-14`。checkpoint `building/fresh`，artifact/audit 不存在。本单元未验收；新增回归必须断言 `ftol=0`，防止严格 residual 前由函数值相对下降条件退出。

`ftol=0` 本地结果：sparse/facade/artifact/audit `24 passed in 5.80s`；完整回归 `129 passed, 2 warnings in 58.25s`，warnings 仍仅为 pandas 可选依赖版本提示；Black 两文件无需修改，`compileall` 与 `git diff --check` 通过。真实 Slurm 回归前仍保持 `[UNACCEPTED]`。

第二次临时验证登记检查：commit `463c3b9`、首条普通 pull、临时分支 pull、同输入/资源/数值参数、独立 `dual_ftol_validation` 路径及第三次失败 lessons 阈值均明确；连接前执行相关文档 `git diff --check`。

首次同步实际结果：首条 pull GnuTLS `-110`，`net.sh` 后 retry 为 GitHub 443 timeout；未同步/未提交 job。独立重试登记的首条 pull 与失败停止条件已复核，相关文档 `git diff --check` 通过。

Job 233 提交检查：服务器同步 `7482ef5`，输入哈希与 Job 230/232 一致，无同名作业，独立 `dual_ftol_validation` 路径；监控至 14:49 后 SSH reset 但未取消 Slurm。恢复连接用途/首条 pull/只读边界一致，相关文档 `git diff --check` 通过。

Job 233 第二次恢复检查：服务器同步 `4dbb598`，作业至至少 25:40 仍 RUNNING；会话关闭不等于作业终止。再次恢复用途、首条 pull 与只读产物验收边界一致，相关文档 `git diff --check` 通过。

Job 233 实际结果：`39:06.84`、Exit 1、MaxRSS `1,847,040 KiB`、0 swap；`ftol=0` 下仍有 20 个 `FACTR*EPSMCH` termination，21 attempts/1,000 evaluations 后 row/column residual 与 Job 232 相同，为 `1.6915612104e-3`/`2.6332792027e-14`；checkpoint `building/fresh`，artifact/audit 不存在。该方案未验收。

Newton-CG 调整测试计划：

- scaled dual Hessian-vector 必须与解析梯度中心有限差分一致，且 `D^-1 H D^-1`/对角 preconditioner 的 shape、有限性和 gauge-fixed 方向正确。
- monkeypatch CG 验证 `LinearOperator`、`rtol/atol/maxiter` 与 matvec 计数；CG matvec 加 backtracking 候选复验不得超过总 acceleration budget，非有限方向/无改善 step 显式记录并回到标准 scaling。
- Newton 候选只在原始 row/column L1 的最大值严格下降时接受；极端边际病态图仍在 `1e-9` 与总 1,500 预算内收敛，method/provenance 改为 `sinkhorn-scaled-newton-cg-sinkhorn`。
- 运行 sparse/facade/artifact/audit、完整 pytest、Black、compile 和 `git diff --check`；真实图继续只用临时未验收分支和 Slurm。

Newton-CG 本地实际结果：

- scaled Hessian-vector 中心有限差分、病态图严格收敛、CG LinearOperator/预条件器/预算与无改善重启通过；sparse/facade/artifact/audit `23 passed in 6.35s`，格式化后复核 `23 passed in 6.06s`。
- 完整回归 `128 passed, 2 warnings in 65.41s`；warnings 仍仅为 pandas 可选依赖版本提示。
- Black 首次要求格式化 `sinkhorn.py`，沙箱内 ACL 拒绝原子替换；仅提升明确文件权限后完成，最终两文件无需修改。`compileall` 与 `git diff --check` 通过。
- method/provenance 已改为 `sinkhorn-scaled-newton-cg-sinkhorn`，无 L-BFGS/FACTR 路径；真实 Slurm 未通过前仍为临时未验收实现。
- 最终代码形态再次复核：定向 `23 passed in 8.13s`，重定向 bytecode cache 后 compileall 通过；首次完整回归因既有 `%TEMP%/pytest-of-asus` ACL 导致一个 Slurm 包装测试失败（其余 127 通过）。按 `lessons.md` 改用忽略目录 `local/test-tmp/newton-full-2` 的全新 `--basetemp` 后，完整回归 `128 passed, 2 warnings in 75.88s`，未跳过任何测试。
- 上述代码与文档已形成并推送临时分支提交 `f62c540`（明确 `[UNACCEPTED]`）；远程真实图验收仍待 Slurm 执行。
- Job 234 真实图验收失败：37:11.54、Exit 1、MaxRSS 1,847,260 KiB、0 swap；12 attempts/1,000 evaluations 后 row/column residual `1.6915304665e-3`/`6.2669379185e-14`，method 为 `sinkhorn-scaled-newton-cg-sinkhorn`。checkpoint `building/fresh`，artifact/audit 不存在；该提交不得作为验收提交或进入正式分支。

Reduced row-dual Newton-CG 修复测试计划：

- 对消去 column dual 后的 Schur-complement Hessian-vector 做中心有限差分；验证 gauge anchor 选择最大 target marginal，变量 shape 只含 `n_rows - 1`，缩放/对角预条件均有限且为正。
- 每个 trial step 先改变 row dual，再按 source marginal 精确重归一化每一 column；断言候选 column residual 保持机器精度，而 row L1 residual 严格下降后才接受。
- 构造 truncated CG 会明显破坏 full-dual column residual、但 reduced feasible step 可接受非微小步的病态图回归；保留总 acceleration evaluation cap、无改善重启和显式失败语义。
- 运行 sparse/facade/artifact/audit 定向测试、完整 pytest、Black、compileall 与 `git diff --check`；真实 2.3M-edge 图仍只经新的临时 `[UNACCEPTED]` 提交和 Slurm 验收。

真实全词表 OT 精度需求调整文档单元：

- 将真实 2.3M-edge/full-vocabulary 构建的两侧最大 L1 边际残差验收阈值明确改为 `2e-3`，覆盖 Job 234 的 `1.6915304665e-3`；不得把该结果描述为在原 `1e-9` 要求下通过。
- toy/dense oracle、Hessian 有限差分和小图算法回归继续保留 `1e-9` 或各测试原有更严阈值，避免产品级近似容差降低数值单元测试标准。
- 同步 `docs/plan/T_plan.md`、`docs/plan/T_implementation_plan.md` 与 `assets/T_method.md`；检查相对链接、公式、命令、阈值分层和 Markdown 格式，并运行 `git diff --check`。本单元仅改文档，不运行代码单元测试。
- 上述 reduced row-dual 修复计划因用户明确接受当前真实图精度而取消，不实施代码改动；其诊断保留为历史经验，后续只有在精度需求重新收紧时再启用。

实际结果：

- `docs/plan/T_plan.md` 已定义真实 full-vocabulary `2e-3` 与 toy/dense `1e-9`（或原更严阈值）的精度分层，并明确 `delta_marginal=2e-3`、实际 residual/tolerance provenance 和旧 Job 234 不可直接转为有效 artifact。
- `docs/plan/T_implementation_plan.md` 与 `assets/T_method.md` 已同步测试边界、非单元验收、Job 234 结果和重跑要求；相对链接 `./T_plan.md` 目标存在，三份目标文档路径检查通过。
- `git diff --check` 通过，仅有既有 LF/CRLF 转换 warning；本单元只改文档，未运行代码测试。

真实 full-vocabulary tolerance 配置实现单元：

- 新增可序列化、严格校验的 transport construction 配置，固定 `epsilon=0.5`、`tolerance=2e-3`、`max_iter=10000`、`smoothing=1e-8`；缺省配置保持兼容，非法/非有限/非正数和布尔伪整数显式失败。
- 主 Qwen3→Mistral-Nemo recipe 与 schema 显式记录上述构建参数；配置 round-trip 和 pinned recipe 测试验证 `2e-3` 已进入结构化 provenance，而不是只存在于说明文字。
- `build_full_support_preview.sbatch` 默认转发 `2e-3`，环境变量 override 仍按原值转发；普通小语料 preview、库/CLI 默认和 toy/dense 测试继续使用 `1e-9`。
- artifact 保存/加载/独立 audit 从 `metadata.build_config.tolerance` 读取边际 L1 阈值，且拒绝高于当前预注册上限 `2e-3` 的 metadata；非负性和逐列归一化仍使用 dtype 级数值阈值，不能随边际容差放宽。
- 运行 config/full-support Slurm 定向测试、完整 pytest、Bash syntax、Black/compileall 与 `git diff --check`；真实 artifact 仍需临时分支 Slurm 重跑后验收。

实际结果：

- config/full-support 初始定向 `14 passed`；补齐 artifact tolerance 链后，config/full-support/artifact/audit/build CLI `25 passed`，最终 save→load→independent audit 相关集合 `21 passed`。
- 最终完整回归 `131 passed, 2 warnings in 157.33s`；warnings 仍仅为 pandas 可选 numexpr/bottleneck 版本提示。
- `TransportConstructionSpec` round-trip/缺省兼容/非法值、主 recipe `0.002`、Slurm 默认 `2e-3` 与 `4e-3` override、metadata 上限、边际近似通过且列和严格失败均有直接回归覆盖。
- Bash syntax、重定向 bytecode cache 的 compileall、Black（6 files unchanged）和 `git diff --check` 通过；compileall 临时缓存已在验证工作区边界后删除。
- 本地验收完成；真实 artifact 尚未生成，当前代码只能形成临时 `[UNACCEPTED]` 验证提交。
- 上述实现已形成并推送临时 `[UNACCEPTED]` 提交 `5207cc9`；真实验证将使用独立 `tolerance_2e3_validation` 路径，验收实际 residual、metadata tolerance、严格列和、save/load/audit 与 complete checkpoint。

Job 235 实际结果与 sparse audit 修复测试计划：

- Job 235 在约 22:36 被 signal 9 终止，MaxRSS `255,870,840 KiB`、0 swap；34 MiB partial 已产生但 checkpoint 仍为 `building/fresh`，无最终 artifact/audit。根因是独立 audit 对 full-vocabulary transport/coupling/entropy 做 dense 展开，该运行不验收。
- 用小矩阵手算对照 sparse audit 的列和、两侧边际、`Ta-b`、每列 entropy、transport cost 和正则目标；结果必须与现有定义一致。
- monkeypatch `transport_to_dense` 为失败并审计一个大 shape/低 nnz artifact，证明正式 audit 不依赖 dense helper，工作内存/中间数组仅为 O(nnz + source vocab + target vocab)。公开 dense helper 保留给 tiny oracle，不删除其行为。
- 运行 artifact/audit/build CLI 定向测试、完整 pytest、Black/compileall/diff；通过后仅形成新的临时 `[UNACCEPTED]` 提交，并用独立 `sparse_audit_validation` 路径经 Slurm 重跑。

实际结果：

- artifact/audit/facade/build CLI 定向 `18 passed in 9.57s`；2×2 手算覆盖列和、row/column/transported marginal、每列 entropy、candidate cost 和正则目标。
- 10,000×10,000、10,000 nnz 对角 artifact 在 monkeypatch dense helper 为强制失败时完成 audit，证明正式路径不调用 dense 转换；candidate cost 使用 NumPy pair key 排序/searchsorted，不再建立 230 万 Python edge/dict。
- 完整回归 `133 passed, 2 warnings in 55.63s`；warnings 仅为既有 pandas 可选依赖版本提示。Black 两文件 unchanged、compileall 和 `git diff --check` 通过。
- 真实 2.3M-edge artifact 尚未经新 audit 完成，当前仍只能形成临时 `[UNACCEPTED]` 验证提交。
- sparse audit 实现已形成并推送临时 `[UNACCEPTED]` 提交 `76ee480`；远程将使用全新 `sparse_audit_validation` 路径验收完整 audit、目标统计和 MaxRSS，不复用 Job 235 partial。
- Job 236 使用代码 `9036c70af78b09f3dad69d962731038b50e72350`、相同 preview/ANN 输入哈希、`epsilon=0.5`、`tolerance=0.002`、`max_iter=10000`、64G/8h Slurm 配置完成；GNU time 为 20:23.32、Exit 0、MaxRSS `2,113,980 KiB`、0 swap。
- 最终 artifact shape `131069×151669`、nnz/candidate edges `2,620,553`；audit `valid=true`，row/column/transported marginal L1 为 `1.9975102855e-3`/`8.5268617950e-14`/`1.9975102855e-3`，最大列和误差 `1.1883827256e-12`，transport cost `7.1692051605`、regularized objective `3.8293241068` 均有限，dangerous special mappings 为空。
- artifact metadata 保存 `tolerance=0.002`、真实 convergence 和输入 provenance；checkpoint 为 `complete/fresh`。最终 `.npz`、JSON、Markdown 均存在且无同名 partial；SHA-256 分别为 `b1ada569…18aca2`、`c8467f09…99d7d1`、`56ef61f7…fb739`，checkpoint 为 `88c9f4ff…c23501`，stderr 为 `3f9c2ee3…ec2784`。远程集成验收通过。
- 最终本地验收：`python -m pytest -o addopts="--strict-markers --strict-config" --basetemp=local/test-tmp/accept-20260903 -q` 为 `133 passed, 2 warnings in 126.61s`，warnings 仅为既有 pandas 可选 numexpr/bottleneck 版本提示；净变更的 10 个 Python 文件 Black 检查全部 unchanged，重定向 pycache 的 compileall、3 个相关 Slurm 脚本 `bash -n`、两份计划路径和 `git diff --check` 均通过。一次对整个 transport 目录的过宽 Black 探测只命中 9 个未被本分支修改的既有文件，未改写或纳入本次验收范围。

## 2026-09-02：scaled-dual Slurm 重跑登记检查

计划与实际结果：检查 `docs/agents/gpu.md` 锁定 `f5ba846`、相同输入/64G/8h/`1e-9` 对照、首项 `git pull` 与 Slurm-only 计算边界；关键字段检索和相关文档 `git diff --check` 在提交前执行并通过。

重连调整检查：记录 PowerShell 提前展开 Bash substitution 的失败边界，下一连接改为持久会话逐条字面命令；检查无新 job 的结论、首条 pull 和权限范围，相关文档 `git diff --check` 通过。

网络 retry 检查：记录首条 pull 60 秒、`net.sh` 成功、retry pull 90 秒和无新 job 边界；第三次独立连接用途/首条 pull/失败停止条件完整，相关文档 `git diff --check` 通过。

Job 230 恢复检查：记录第三次 pull/输入哈希/提交及会话关闭边界；新连接只读验收用途、首条 pull、成功/失败分支明确，相关文档字段与格式检查通过。

08:12 新连接登记检查：`gpu/state/update` 中的用途、提交 `06e9c7c`、首条 `git pull`、`net.sh` fallback、Job 230 只读范围与正式 500k 阻塞条件一致；`rg` 字段复核及 `git diff --check` 通过。

## 2026-09-02：sparse OT convergence follow-up 单元

计划范围：

- 从 Job 229 固定同一 Gibbs kernel/边际目标，复核 standard scaling、gauge-fixed dual gradient、预算计数与收敛检查，定位 column residual 达机器精度而 row residual 停滞的原因。
- 用小型病态稀疏图构造可由 dense oracle 验证的回归；新方案必须达到原 `1e-9` 两侧 L1 residual，不得提高容差或改用 feasibility coupling 伪装熵正则解。
- 约束内存仍为 O(edges + nodes + bounded history)，非法/不可行输入继续显式失败；运行 sparse Sinkhorn 定向测试、transport facade/audit 集成、完整 pytest、格式与 diff 检查。

远端前置证据：Job 229 `FAILED`/Exit `1:0`，standard 8,999 + acceleration 1,001，总 10,000；row/column residual `4.071621136e-4`/`4.996e-14`；GNU time MaxRSS `1,846,656 KiB`、0 swaps，checkpoint `building`，artifact/audit 不存在。

实际结果：

- dual 使用可逆 `sqrt(marginal)` 坐标缩放；scaled objective 解析梯度与中心有限差分在 `1e-8` 内一致，方法 provenance 为 `sinkhorn-scaled-lbfgs-sinkhorn`。
- 80×80、边际从 `1` 跨至 `1e-14` 的病态稀疏图：纯 scaling 100 次按预期失败；scaled dual 仅 60 次 evaluations、总 103 次即达到 row residual `9.3017e-10`，column residual `2.8356e-15`。
- sparse/facade/audit/CLI 定向回归：21 passed（9.88s）；强化 sparse 文件：11 passed（2.38s）。
- 完整 pytest：`118 passed, 2 warnings in 50.93s`，warnings 仍仅为既有 pandas 可选依赖版本提示。
- Black（独立仓库内 cache、单 worker）检查 sinkhorn 与测试：全部无需修改；`git diff --check` 提交前复核。

## 2026-09-02：Guqq telemetry 重连登记文档检查

计划与实际结果：检查 `docs/agents/gpu.md` 新条目的时间、用途、权限边界、首条 `git pull` 命令与目标提交；`git diff --check -- docs/agents/gpu.md` 通过，关键字段检索通过，路径和命令与当前仓库/AGENTS 规范一致。

后续检查：Job 229 提交结果与持久监控连接用途追加后，检查 gpu/state/update 的 job ID、commit、哈希和权限描述相互一致；相关字段检索及文档 `git diff --check` 均通过，无 whitespace error。

## 2026-09-02：OpenHermes 500k deterministic materialization 单元

计划范围：

- 从锁定 dataset revision/raw `train` split 流式保存前 500,000 个 source rows，严格复现 `OpenHermesChatDataset` 的 `select(range(num_samples))` 语义；不足 500,000 行必须失败。
- 单次扫描不把全量 conversation 常驻内存；使用 partial 文件原子发布 JSONL/manifest，验证或长度失败不留下看似完成的目标文件。
- manifest 继续绑定 selected JSONL SHA-256，并新增 prefix algorithm、source start、requested/selected rows、unique conversations、filtering 状态与 split seed provenance；canonical duplicate 不跨 99/1 split。
- CLI 支持正式 Hugging Face pinned-revision 模式和离线 `--input-jsonl` 测试模式，二者互斥；`datasets` 延迟导入，模块 import/help/离线测试不访问网络。
- Slurm 作业锁定 `teknium/OpenHermes-2.5@05c355...`、500k、seed 42、输出/缓存/log 忽略路径，无硬编码 partition；下载后的遍历/物化全部在 allocation 内。
- tiny fixtures 覆盖精确 source prefix、duplicate、limit 越界、test split/revision 拒绝、原子输出、CLI provenance、Bash syntax/stub failure propagation 与完整回归。

实际结果：

- 定向 pytest：`20 passed in 11.18s`。
- 完整 pytest：`117 passed, 2 warnings in 51.28s`；warnings 仅为既有 pandas 对可选 `numexpr`/`bottleneck` 版本提示。
- `bash -n script/transport/slurm/materialize_openhermes_500k.sbatch`：通过。
- Black（独立仓库内 cache、单 worker）检查 4 个新增/修改 Python 文件：全部无需修改。
- `git diff --check`：通过（仅 Git 的 LF→CRLF 工作树提示，无 whitespace error）。
- 远程集成验收计划：在 Guqq 通过 32G/4h Slurm 运行锁定 revision 的正式入口；要求 Exit 0、source prefix 恰为 500,000 rows、manifest 的 selected/unique/duplicate/train/dev 计数自洽、dataset/revision/raw split/selection/filtering/seed provenance 完整、records SHA-256 与 manifest 绑定一致、JSONL/manifest 均可解析且不存在 partial，并记录 GNU time/MaxRSS 与文件哈希。未满足任一项不得进入正式 T 构建。
- 首次远程环境门禁实际结果：`.venv/bin/python -c 'import datasets'` 在提交前以 `ModuleNotFoundError` 退出，队列为空且未生成/覆盖输出；按计划先补齐并记录精确依赖，不把环境失败计作物化结果，也不降低脚本的 `datasets==4.0.0` 检查。
- Job 239 远程实际结果：datasets 4.0.0/version/help 门禁通过；作业 2:00.06、Exit 0、MaxRSS `7,128,336 KiB`、0 swap。records 恰为 500,000 行/909,629,231 bytes，manifest 43,500,816 bytes；unique/duplicate/train/dev 为 500,000/0/495,000/5,000，split 内唯一且彼此无交叉；锁定 dataset/revision/raw split、identity、pinned-prefix selection、adapter filtering not-applied、seed 42 和输入指纹齐全。records SHA `539f2d30…5d485a` 与 manifest 绑定一致，manifest/SBATCH stderr SHA 为 `a50c0dca…7c60fa`/`c4a91c0d…728e65`，边界 JSON 可解析且无 partial；远程集成验收通过。

正式 manifest-bound T Slurm 入口实现单元测试计划：

- 新入口必须固定主 source/target 及两侧 revision、正式 records/manifest、`transport_train`、结构化 ANN、epsilon 0.5、tolerance `2e-3`、max_iter 10,000、smoothing `1e-8`、seed 42，并只写忽略的 artifact/audit/checkpoint/log 路径。
- 环境变量仅允许覆盖 Python、输入输出、构建参数和 code version；脚本必须在提交目录/解释器/输入存在时才启动，创建输出目录，保留 GNU time telemetry，支持显式 `RESUME=1`，不硬编码 partition。
- stub 集成测试捕获传给 builder 的参数与环境，验证 manifest 模式不混入 preview `--texts-jsonl`、默认值和 override 精确转发、失败码透传、resume 标志、SBATCH 资源/日志与 `bash -n`；完整 pytest、Black、compileall 和 diff 检查保持通过。
- 真实 500k 构建属于明显计算负载，只能在临时 `[UNACCEPTED]` 提交经 Guqq Slurm 验收；要求正式 artifact/audit/checkpoint 原子完整、metadata 绑定 manifest/records/ANN、两侧 residual `<=2e-3`、严格列和/非负/特殊映射/目标统计有效，并记录 MaxRSS、耗时和哈希。

实际结果：

- `python -m pytest ... test_formal_transport_slurm.py test_build_vocab_transport_cli.py test_corpus.py test_materialize_openhermes_slurm.py ... -q`：17 passed in 28.14s；覆盖脚本语法/资源/锁定值、manifest 模式、默认与 override、resume 和失败传播。
- 完整回归：`137 passed, 2 warnings in 68.64s`；warnings 仍仅为既有 pandas 可选 numexpr/bottleneck 版本提示。
- 新测试文件 Black unchanged；重定向 bytecode cache 的 compileall、`bash -n script/transport/slurm/build_formal_transport.sbatch` 和 `git diff --check` 通过。真实 formal artifact 尚未构建，当前实现仅可进入临时 `[UNACCEPTED]` Slurm 验证提交。
- Job 240 远程实际结果：兼容分支 C2C tree 与 main-based `5787a71` 一致；输入 records/manifest/ANN SHA 与 Job 239/既有 ANN 验收值一致，输出路径预先不存在。作业 51:57.26、Exit 0、MaxRSS `8,036,128 KiB`、0 swap。
- 正式 artifact 为 39,951,267 bytes、shape `131069×151669`、nnz/candidate edges `2,733,518`；metadata code version 为 main 验收 `f433000fa8514296dd5849c619ecd99a4e449bed`，data provenance 为锁定 OpenHermes revision、transport_train、495,000 selected samples、997,233 canonical messages、records SHA `539f2d30…5d485a`、manifest SHA `a50c0dca…7c60fa`、ANN SHA `260f9804…e91652`，构建参数 `0.5/0.002/10000/1e-8`。
- audit `valid=true`；row/column/transported L1 为 `1.9655245213e-3`/`1.0560509249e-13`/`1.9655245213e-3`，最大列和误差 `1.2299050667e-12`，minimum 0，transport cost `3.6869566806`、regularized objective `-1.2818040356` 均有限，dangerous special mappings 为空。checkpoint 为 `complete/fresh`，最终 artifact/JSON/Markdown 齐全且无 partial。
- artifact/checkpoint/audit JSON/audit Markdown/stderr SHA-256 为 `1495d522…0aba97`/`79c4ad38…5caf84`/`53b6a464…1948de`/`13bbda9b…c2457e`/`4deb7afd…517ad7`；远程正式 T 集成验收通过。
- Job 240 后最终本地回归：`137 passed, 2 warnings in 68.24s`，warnings 仍仅为 pandas 可选 numexpr/bottleneck 版本提示；新测试 Black unchanged、compileall、正式 Slurm 脚本 `bash -n`、两份计划路径与 `git diff --check` 全部通过。

## 2026-09-02：memory-bounded dual telemetry 单元

计划范围：

- sparse dual API 显式限制 L-BFGS `maxcor` history 和 evaluation budget；非法 history/budget 失败，convergence report 保持实际分阶段次数。
- monkeypatch SciPy optimizer 验证 `maxcor`、`maxfun` 与解析 jacobian 确实传入，不依赖实现默认值；病态图仍在 `1e-9` 收敛。
- full-support Slurm 作业在可用时通过 GNU `/usr/bin/time -v` 包装 builder，成功和 SIGKILL 都应在 stderr 记录 MaxRSS/elapsed/exit status；本地 stub 环境无 GNU time 时仍保持原命令与失败码传播。
- Bash syntax、stub 参数、failure propagation、Black/compile 与完整离线回归全部通过；不提高内存请求，先用遥测确认峰值。

远端依据：Job 226 在安装锁定 `scipy==1.15.3` 后运行 24:54，以 ExitCode `137:0` 被外部 SIGKILL；无 Python traceback，节点 swap 已满，64G 作业未留下有效 artifact/audit。当前没有 MaxRSS 证据，不能仅凭理论工作集盲目调整资源。

实际结果：

- `python -m pytest test/transport/test_sparse_sinkhorn.py test/transport/test_full_support_preview_slurm.py test/transport/test_vocab_transport_facade.py ...`：19 passed（16.34s）；验证 `maxcor/maxfun/jac` 透传、history 校验、GNU time 可用/缺失分支及失败码传播。
- `bash -n script/transport/slurm/build_full_support_preview.sbatch`：通过；Black 对 3 个实现/测试文件检查通过；production solver `compileall` 通过。
- `python -m pytest -o addopts="--strict-markers --strict-config" --basetemp=local/test-tmp/memory-full -q`：112 passed（44.79s）；仅有本机 pandas 对既有 numexpr/bottleneck 版本的两条 warning。
- `git diff --check`：通过；生成的 Black cache 由既有忽略规则覆盖。

## 2026-09-02：sparse OT dual acceleration 单元

计划范围：

- 保留标准 log-domain row/column Sinkhorn 更新；在大迭代预算中先 warm up，再对同一固定 kernel/边际的凸对偶使用 L-BFGS 加速，之后回到缩放更新并以原 residual 停止条件验收。
- 对偶变量固定 gauge，目标梯度必须等于 row/column marginal residual；用有限差分小图验证梯度，并与 dense oracle 的 coupling/目标一致。
- 构造病态但严格可行的稀疏图，证明纯缩放在限定 warm-up 内仍慢、混合求解在总 `max_iter` 内达到 `1e-9`；不可行图、NaN/Inf 与真实超预算仍显式失败。
- convergence report 记录方法、标准缩放次数和加速次数；artifact/audit 往返保留字段。SciPy 只在进入加速路径时延迟导入，普通导入与已快速收敛的小图不新增启动依赖。
- 完整离线测试保持 dense/sparse、极小 epsilon、极端边际和所有现有 artifact/wrapper 回归。

远端依据：Job 220 在 feasibility support 后将 row residual 从 Job 215 的 `0.2651238722` 降到 `0.0005577679`，column residual 为 `2.20e-14`，但标准缩放 10,000 次仍未达到 `1e-9`；说明支撑已可行但条件病态，单纯增加迭代会继续消耗约 40 分钟/万次。

实际结果：

- 对偶解析梯度与中心有限差分在 `1e-9` 内一致；50×50 极端几何边际图中，纯 scaling 100 次按预期失败，混合求解以 81 次 scaling + 802 次对偶 evaluation 在总预算 1,500 内达到 row residual `8.81e-10`。
- `python -m pytest ... test_sparse_sinkhorn.py/test_sinkhorn.py/test_vocab_transport_facade.py/test_artifact.py/test_audit.py`：24 passed（6.70s）。
- `python -m pytest -o addopts="--strict-markers --strict-config" --basetemp=local/test-tmp/dual-full -q`：110 passed（49.95s）；仅有本机 pandas 对既有 numexpr/bottleneck 版本的两条 warning。
- Black 对 solver/测试文件检查通过；production solver `compileall` 与 `git diff --check` 通过。测试文件因既有 Windows ACL 无法写相邻 `.pyc`，但 pytest 已完整导入执行该文件，未缩减测试范围。

## 2026-09-02：marginal-capacity feasibility support 单元

计划范围：

- 以 source/target 正边际构造确定性的 northwest-corner 稀疏可行耦合支撑，边数不超过 `n_source + n_target - 1`，总质量和两侧边际在浮点容差内严格一致。
- 只把原候选图缺少的可行支撑边标记为独立 `feasibility` 来源；保留 exact/span/ANN/special 边与证据，不覆盖或伪装语义证据。
- feasibility evidence 必须有限、为正且低于正常 ANN evidence；输入 shape、负值、非有限值、总质量不一致和重复 token IDs 显式失败。
- 构造一个“节点均有边且图连通、但违反容量 Hall 条件”的 toy graph：补边前 Sinkhorn 不收敛，补边后通过两侧 residual 与 artifact audit；已有可行图不新增边。
- artifact/build config 记录 feasibility edge count/method，候选来源 schema 可往返保存；完整离线回归不访问网络。

远端失败依据：Slurm Job 215 在 2,337,695-edge 图上运行 46:03 后退出 1；10,000 次迭代的 row residual `0.2651238722`、column residual `7.215e-12`，证明仅拓扑连通不足以保证当前 marginals 可行，不能通过增加迭代或放宽容差修复。

实际结果：

- `python -m pytest ... test_sparse_sinkhorn.py/test_candidate_graph.py/test_vocab_transport_facade.py/test_artifact.py/test_audit.py`：25 passed（5.93s）。
- facade 不平衡边际与 sparse capacity 用例最终 12 passed（5.88s）：补边前连通图在 100 次内按预期不收敛；补边后两侧 residual 小于 `1e-9`，artifact audit 记录独立 `feasibility` 来源与 edge count；已可行图不新增边。
- `python -m pytest -o addopts="--strict-markers --strict-config" --basetemp=local/test-tmp/feasibility-full -q`：108 passed（52.72s）；仅有本机 pandas 对既有 numexpr/bottleneck 版本的两条 warning。
- Black 对 5 个实现/测试文件检查通过；`python -m compileall -q -f -b ...` 与 `git diff --check` 通过。

## 2026-09-01：manifest-bound canonical corpus 单元

计划范围：

- recipe/DataSpec 必须锁定 40-character dataset revision；OpenHermes nullable `id/idx/hash` 不作为唯一身份来源。
- ShareGPT/OpenHermes `conversations[{from,value}]` 规范化为 system/user/assistant canonical messages，不使用 chat template/BOS/EOS；未知 role、空 value、损坏 schema 显式失败。
- sample ID 为 canonical messages 的 SHA-256；相同 conversation 去重并记录 duplicate count，防止相同内容跨 train/dev；输入行顺序不改变唯一 ID 集合和 split membership。
- manifest 记录 dataset、revision、raw split、identity scheme、raw JSONL SHA-256、unique/duplicate counts；materialization/build 时先复核 raw hash。
- builder 的正式模式要求 records JSONL + manifest + `transport_train`/`transport_dev`，拒绝 benchmark test、split 外样本、manifest 缺样本和 preview/formal 参数混用；artifact build config 纳入 manifest/raw/split provenance。
- tiny JSONL 覆盖稳定 manifest、去重、hash tamper、split 隔离、canonical text 提取和 builder toy integration；完整离线测试不访问网络。

数据 schema 依据：Hugging Face 官方 `teknium/OpenHermes-2.5` 页面显示 train 约 1M rows，列包含 nullable `id/idx/hash` 和 `conversations` list；当前仓库 revision 选择必须写入 recipe 后再用于下载/构建。

实际结果：

- `python -m pytest ... test_corpus.py/test_config.py/test_smoke_stt.py/test_build_vocab_transport_cli.py/test_vocab_transport_facade.py`：26 passed（19.35s）；覆盖 canonical role、content ID/去重、raw hash、split 完整性/隔离、CLI 与正式 builder provenance。
- 首次从仓库根目录无路径约束运行 pytest 时误收集历史 `local/test-tmp` 与 playground 脚本；改为 `C2C` 项目目录后，首次完整回归为 102 passed/2 failed，定位到旧 smoke fixture 缺少新增的 dataset revision。补齐同一锁定 SHA 后，目标测试 25/25 通过。
- 提交前复核补充“全部 raw canonical IDs 必须精确重现 manifest train/dev”校验与遗漏样本测试；最终 `python -m pytest -o addopts="--strict-markers --strict-config" --basetemp=local/test-tmp/corpus-final-repro -q`：105 passed（48.86s），仅有本机 pandas 对既有 numexpr/bottleneck 版本的两条 warning。
- `python -m compileall -q -f -b ...`：通过。设置任务本地 cache 后，Black 对最终变动的 corpus 实现、corpus 测试与 smoke fixture 检查通过（3 files unchanged）；此前本单元其余 Python 文件检查也已通过。

## 2026-09-01：全 source special 安全支撑修复单元

计划范围：

- 边际估计支持显式 allowed token IDs：source positive smoothing 覆盖完整 source vocab；target 只激活 ordinary token，BOS/EOS/UNK/pad/control 不因 smoothing 获得伪频率。
- 每个 source special/control 保留可用的同功能 special 边，同时必须通过 target tokenizer 的 literal-byte 分解获得 ordinary target 边；无 ordinary literal 支撑时显式失败，不映射到任意 target special/UNK。
- 新候选来源明确标记为 `special_literal` 并进入 artifact/audit provenance；普通 special/exact/span/ANN 优先级回归不变。
- tiny source 含 generic control/pad/eos、target 含不匹配 BOS/EOS/UNK 的正 smoothing facade 能构建 full source artifact；source IDs 等于完整连续词表，target IDs 仅 ordinary，Sinkhorn/audit 不变量通过。
- allowed IDs 越界或 special literal 只产生 target specials 时失败；完整离线测试保留。

远端复现：Job 214 在 Slurm 内运行 18 秒后 ExitCode `1:0`；`CandidateGraphError` 指向 Qwen source ID 151644 `<|im_start|>` 无唯一 generic-special target。真实 special 审计显示 Qwen 另有 pad/eos 与视觉 control，Mistral 仅 BOS/EOS/UNK；禁止通过伪 special 映射绕过。

实际结果：

- `python -m pytest -o addopts="--strict-markers --strict-config" test/transport/test_marginals.py test/transport/test_candidate_graph.py test/transport/test_vocab_transport_facade.py --basetemp=local/test-tmp/special-support-targeted -q`：15 passed（5.94s）。
- Black 对 6 个实现/测试文件检查通过；显式 workspace pycache 的 `compileall` 通过。
- `python -m pytest -o addopts="--strict-markers --strict-config" --basetemp=local/test-tmp/special-support-final -q`：97 passed（48.09s）；仅有本机 pandas 对既有 numexpr/bottleneck 版本的两条 warning。

Guqq 重跑审计检查：记录锁定提交 `0409679`、首条 `git pull`、Job 214 非有效 checkpoint、Slurm-only 计算边界和新 job 日志/产物验收范围；`git diff --check` 在提交前执行。

Guqq 重跑实际未开始：首次 pull 约 90 秒无响应；`bash net.sh` 成功后重试约 60 秒仍无响应。未同步修复、未执行 `sbatch`，因此没有新的测试结果或可验收 artifact。

## 2026-09-01：全词表支撑预览 Slurm 作业单元

计划范围：

- 新作业必须使用锁定 source/target revisions、canonical preview JSONL、Job 212 结构化 ANN candidates 和严格为正的 smoothing，产物命名明确为 full-support preview。
- 作业不硬编码未知 partition；候选构建/Sinkhorn/audit 全部由 Slurm 执行，登录节点只负责 `sbatch` 与状态/文件检查。
- Python venv、输入和 ANN JSON 缺失时在启动计算前失败；artifact、checkpoint、logs 和 audits 均位于 `local/transport/` 忽略目录。
- Bash 语法与 stub Python 参数传播测试覆盖 revisions、ANN 路径、smoothing、code version 和失败码；完整离线测试不访问网络。

前置 ANN 本地独立验收：scp 文件大小 134,332,695 bytes 且 SHA-256 与服务器一致；全 JSON 扫描得到 151,655 source、131,069 target、2,337,695 edges，0 个非法 evidence、0 个重复/乱序 source adjacency，evidence 范围 `[1e-6, 1.0000001192092896]`。最大值是 float32 余弦舍入产生的约 `1.2e-7` 上溢；raw evidence 在每个 source 内归一化后才转为代价，不作为概率直接使用。

实际结果：

- `python -m pytest -o addopts="--strict-markers --strict-config" test/transport/test_full_support_preview_slurm.py test/transport/test_preview_slurm.py --basetemp=local/test-tmp/full-preview-targeted -q`：6 passed（11.79s）。
- `bash -n script/transport/slurm/build_full_support_preview.sbatch`：通过；作业无硬编码 partition。
- Black 检查与显式本地 pycache 的 `compileall`：通过。
- `python -m pytest -o addopts="--strict-markers --strict-config" --basetemp=local/test-tmp/full-preview-final -q`：94 passed（50.18s）；仅有本机 pandas 对既有 numexpr/bottleneck 版本的两条 warning。

Guqq 输入/作业连接审计检查：canonical JSONL 存在、1,094 bytes、SHA-256 为 `05CA0628E57EADDA84F4D16968083D5BF12D8A9012B2A1081D9E372047207A3A`；记录区分单文件 scp 与首条 `git pull` 的持久 SSH 会话，并将实际计算限定在 Slurm。`git diff --check` 在提交前执行。

## 2026-09-01：ANN Slurm 连接审计文档

计划范围：

- 检查连接用途、登录节点/Slurm 边界、首条 `git pull`、`bash net.sh` 回退、锁定提交与产物目录均明确。
- 检查引用的 `script/transport/slurm/build_ann_candidates.sbatch` 路径存在，且文档无空白错误。

实际结果：

- Slurm 脚本路径存在；连接记录包含提交 `8f89fb4`、首条 `git pull`、网络失败时 `bash net.sh`、登录节点/批处理边界和忽略产物目录。
- `git diff --check`：通过；仅报告工作区 LF/CRLF 转换 warning。

Job 212 重连记录检查：用途限定为 `squeue/sacct`、日志与 JSON 只读校验；明确持久会话首条执行 `git pull`，避免后续每次监控建立未审计的新连接。相关脚本路径仍存在，提交前重新执行 `git diff --check`。

Job 212 实际结果：

- Slurm `scontrol`：`COMPLETED`、Reason None、ExitCode `0:0`、RunTime 00:00:52、4 CPU、32G memory request。
- JSON：134,332,695 bytes，schema version 1，code version `55825e4…`，锁定 source/target revisions 正确；151,655 source 与 131,069 target ordinary token 均有 candidates。
- SHA-256：`260f98048a3d50adb667a6c0b9d23126c7d0e533fd56791c6059001104e91652`；`.partial` 不存在。
- stderr 159 bytes，仅提示未安装 PyTorch/TensorFlow/Flax、只能使用 tokenizer/config/file utilities；与本作业 tokenizer-only 设计一致，不影响验收。

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

## 2026-09-01：双向 LSH ANN candidate 生成器单元

计划范围：

- 对 source/target ordinary token raw bytes 使用相同 seeded hashed byte-ngram 特征，归一化后以 LSH bucket 近似检索；special/control IDs 从候选中排除。
- 每个 ordinary source 至少有 forward top-k，每个 ordinary target 至少通过 reverse top-k 获得入边；候选 pair 去重且按 target ID 稳定排序。
- 增加低 evidence 的 source→target anchor 与 source anchor→all target bridge，保证普通二部图单连通；bridge evidence 必须严格小于正常 ANN evidence，并在 metadata 显式记录，不能伪装成语义近邻。
- 相同 token bytes/seed/config 产生字节级一致 JSON；改变输入、seed 或配置会改变 input fingerprint。
- 输出 schema、tokenizer fingerprints、method、ngram/dimension/signature bits/top-k/pool/bridge 参数、coverage 与代码版本完整；CLI 原子写入且 Transformers 仅在执行入口加载。
- builder 同时接受旧版纯 mapping JSON 与新版结构化 `{metadata,candidates}`，将候选 metadata 纳入 artifact build config；损坏 schema/重复/非法 evidence 显式失败。
- tiny tokenizers 验证双向覆盖、连通性、确定性和 builder 集成；全量测试不访问网络。

网络经验与计划调整文档检查：`git diff --check` 通过。

实际结果：

- `python -m pytest -o addopts="--strict-markers --strict-config" test/transport/test_ann_candidates.py test/transport/test_build_vocab_transport_cli.py --basetemp=local/test-tmp/ann-targeted-2 -q`：16 passed（12.04s），覆盖双向 support、图连通、确定性、参数边界、结构化/旧版 loader、原子 CLI 与 Slurm stub。
- 首轮完整回归在新增最后两个边界用例前为 87 passed（32.60s）；最终完整回归结果记录在本单元验收前的后续条目。
- 当前系统 `python` 未安装项目可选的 `pytest-cov`，首次 pytest 在收集前因 pyproject 中未知 `--cov` 参数退出；后续显式保留 `--strict-markers --strict-config` 并覆盖全部测试路径，不把启动环境问题记为用例失败。
- `$env:PYTHONPYCACHEPREFIX='local/pycache'; python -m compileall -q rosetta/transport script/transport test/transport/test_ann_candidates.py test/transport/test_build_vocab_transport_cli.py`：通过；显式缓存目录规避测试目录既有 `__pycache__` ACL。
- `$env:BLACK_CACHE_DIR='local/black-cache'; python -m black --check --workers 1 ...`：6 个本单元 Python 文件均无需修改；显式缓存目录规避用户级 Black cache 锁等待。
- `python -m pytest -o addopts="--strict-markers --strict-config" --basetemp=local/test-tmp/ann-final -q`：91 passed（34.97s）；仅有本机 pandas 对既有 numexpr/bottleneck 版本的两条 warning。
- `git diff --check`：通过；仅报告工作区 LF/CRLF 转换 warning。生成的 Black/pyc cache 已加入任务本地忽略路径。

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

## 2026-09-02：前后 sparse OT 加速方法说明文档

测试计划：

- 对照临时提交 `463c3b9` 的增量 scaled L-BFGS-B 与当前工作树的 residual-driven scaled Newton-CG，检查 `assets/T_method.md` 中的流程、变量缩放、停止/接受条件和预算描述与源码一致。
- 检查 Markdown 标题、公式、表格、仓库内路径以及方法标识；运行 `git diff --check`，预期无空白错误。
- 本单元仅新增/更新文档，不修改算法源码，因此不新增或运行代码单元测试。

实际结果：

- 已逐项对照临时提交 `463c3b9` 的 `_dual_increment_value_gradient`、L-BFGS-B 参数/外层接受逻辑，以及当前 `_scaled_dual_hessian_product`、CG、residual backtracking 和共享预算实现；说明与源码一致。
- `rg` 检查方法参数和 provenance 标识通过：旧版 `acceleration_history_size` / `sinkhorn-scaled-lbfgs-sinkhorn`，新版 `acceleration_cg_iterations` / `sinkhorn-scaled-newton-cg-sinkhorn`。
- 文档无外部链接；仓库路径 `assets/T_method.md` 存在，Markdown 标题、公式和表格人工检查通过。
- `git diff --check`：通过；仅报告工作树既有 LF/CRLF 转换 warning，无 whitespace error。

## 2026-09-03：真实模型短序列 smoke 实现单元

测试计划：

- 用 tiny stub 同一 prompt 运行 Receiver-only 与 STT，验证两路分别使用 target/source tokenizer，生成参数完全一致，报告包含两路 token IDs/文本、STT shape/transport quality/分段 metrics、锁定配置、artifact provenance 和输入指纹。
- smoke 专用运行必须把 `max_new_tokens` 限制为默认 2，并拒绝未知生成字段、空 prompt、缺失 artifact、tokenizer 指纹不匹配、无 CUDA、GPU 数量或显存不足、缺少精确依赖等情况；错误必须在加载 8B 权重前明确暴露。
- Slurm 入口不固定 partition，显式申请已核实的 GPU/CPU/内存/时限，运行前检查解释器、正式 artifact、CUDA、依赖和模型缓存，使用原子 JSON 输出与 GNU time telemetry，禁止覆盖既有正式结果。
- CLI help 和 tiny 单元测试不得访问网络或加载远程模型；完整 pytest、Black、compileall、Bash syntax、Markdown 路径/命令和 `git diff --check` 均须通过。
- 真实验收仅通过 Slurm：固定 Qwen3-8B/Mistral-Nemo revisions、正式 Job 240 artifact 和短 prompt，要求 Receiver-only/STT 均成功生成、报告可解析、provenance/shape/有限质量统计完整、无 `.partial`，并记录 Job ID、ExitCode、Elapsed、MaxRSS、GPU 和产物 SHA-256。该结果只证明功能正确性，不进入正式 latency 表。

本地实际结果：

- 首次从仓库根运行定向 pytest 在收集阶段因 `rosetta` 不在 Python 根失败；切换到 `C2C` 后未降低断言。系统 `%TEMP%` 路径随后触发既有 Bash/ACL 问题，按 lessons 改用工作区全新 `--basetemp` 后消除。
- `python -m pytest -o addopts= --basetemp=local/test-tmp/smoke-real-20260903c test/transport/test_smoke_stt.py test/transport/test_real_smoke_slurm.py -q`：12 passed（10.12s）。
- `python -m pytest -o addopts= --basetemp=local/test-tmp/full-smoke-20260903 -q`：146 passed（73.44s），仅有既存 pandas 对可选 numexpr/bottleneck 版本的 2 条 warning。
- Black 直接 CLI 在 Windows 现有目录 ACL 下遗留 worker 并无法替换新测试文件；只终止本轮启动的 worker 后，用同版本 Black `format_str` 内存比较 3 个变动 Python 文件，结果全部 unchanged。未终止昨日已有的两个 Python 进程。
- 3 个变动 Python 文件的内存 `compile(..., "exec")`、新 Slurm 脚本 `bash -n`、recipe 结构化解析/锁定字段、README 路径/命令和 `git diff --check` 均通过；compileall 仅因既有 `test/transport/__pycache__` ACL 无法写 `.pyc`，不是源码语法失败。

真实 Job 241 结果：

- 环境、artifact、cache、输出和 Slurm 提交门禁通过；作业加载两侧模型后在 Receiver-only 首个 CUDA generation kernel 失败，错误为 `no kernel image is available for execution on the device`。
- GNU time：0:26.39、Exit 1、MaxRSS 5,845,204 KiB、0 swap；无合格 JSON/验收结果。该失败不降低测试标准，代码继续位于 `[UNACCEPTED]` 分支。
- 初步归因是 torch 2.6.0/CUDA 12.4 wheel 不包含节点旧 GPU 架构；待 Slurm `nvidia-smi` capability 诊断确认。随后仅允许走计划已有的 CPU/offload 功能 smoke，结果不得进入正式 latency 表。

Job 241 兼容修复追加计划：

- Job 242 已确认硬件实际是新 GPU：RTX 5090 32,607 MiB、Blackwell `sm_120`、驱动 570.211.01/CUDA 12.8；修正“旧 GPU”初步判断，根因仍是 torch 2.6.0/cu124 compiled arch 不包含 `sm_120`。
- runtime 门禁增加命名 profile：项目默认继续精确要求 torch 2.6.0/accelerate 1.9.0/transformers 4.52.4；Guqq Blackwell profile 精确要求 torch 2.7.1+cu128，并把选择的 profile 写入报告。
- CUDA preflight 必须比较可见设备 compute capability 与 `torch.cuda.get_arch_list()`，缺少 `sm_120` 时在加载任何模型权重前明确失败；tiny mock 覆盖支持/不支持两种情况。
- Slurm stub 测试验证 `RUNTIME_PROFILE` override 被转发；Guqq 使用独立 `python3 -m venv .venv-smoke-cu128`，避免覆盖项目 venv。修复后重新运行定向、完整回归和真实 Job，不接受 CPU fallback 代替可用的兼容 GPU profile。

兼容修复本地实际结果：

- `python -m pytest -o addopts= --basetemp=local/test-tmp/smoke-profile test/transport/test_smoke_stt.py test/transport/test_real_smoke_slurm.py -q`：14 passed（9.71s），覆盖两个 runtime profile、Slurm override 和 `sm_120` 支持/拒绝门禁。
- `python -m pytest -o addopts= --basetemp=local/test-tmp/full-profile -q`：148 passed（74.19s），仅有既存 pandas 可选依赖 2 条 warning。
- Black `format_str` 对最终 3 个变动 Python 文件检查 unchanged；内存 compile 3/3、Slurm `bash -n` 与 `git diff --check` 通过。

真实 Job 243 结果与追加测试计划：

- cu128 profile 通过 arch 门禁、加载两模型并完成 Receiver-only；STT 因 Qwen3 LM head 151,936 rows 与 tokenizer/T 151,669 的尾部 padded rows 不等而在 exact support 门禁失败。0:14.99、Exit 1、MaxRSS 16,415,184 KiB、0 swap，无合格报告。
- 新增 explicit `source_vocab_size` 单元：当 artifact source IDs 恰为 `0..source_vocab_size-1` 且 logits 仅多连续尾部 padded rows 时，结果必须等于对 tokenizer logits 做精确 softmax/transport；统计质量为完整保留。
- 未显式给出 tokenizer vocab size、artifact 存在中间缺口、size 超过 logits、或 artifact 不完整覆盖 tokenizer vocab 时仍必须报错；不能把 `allow_partial_support` 静默用于 exact STT。
- wrapper/真实 loader 必须从 fingerprint 已验证的 source tokenizer `len()` 传递该值；tiny wrapper 回归继续验证默认严格路径。修复后重新跑定向、完整回归和真实 Slurm smoke。

LM-head padding 修复本地实际结果：

- `python -m pytest -o addopts= --basetemp=local/test-tmp/padded-vocab test/transport/test_soft_transport.py test/transport/test_wrapper.py test/transport/test_smoke_stt.py test/transport/test_real_smoke_slurm.py -q`：32 passed（9.94s）。
- `python -m pytest -o addopts= --basetemp=local/test-tmp/full-padded-vocab -q`：149 passed（74.57s），仅有既存 pandas 可选依赖 2 条 warning。
- Black `format_str` 对最终 5 个变动 Python 文件检查 unchanged；内存 compile 5/5、`git diff --check` 通过。测试明确覆盖尾部 padded rows 等价 oracle、未显式 size、超界 size 和中间缺口即使请求 partial 也失败。

真实 Job 244 结果与第三次失败后的追加计划：

- source-vocab 修复生效，但完整 receiver active embedding 表先 `index_select` 再由 BF16 升为 float32，额外请求 2.50 GiB；GPU 已被两模型占用约 30.8 GiB，故 OOM。0:15.08、Exit 1、MaxRSS 16,525,048 KiB、0 swap、无 JSON。
- 已按连续失败三次规则复查 `docs/agents/lessons.md` 并补充模型并行内存经验；不提高 GPU 资源、不降低模型/统计精度要求。
- 新测试以 float32 target probabilities + BF16 receiver weight 验证输出保持 receiver dtype、数值匹配高精度 oracle；连续 prefix token IDs 不复制完整权重表，非连续映射也按固定 target chunk 正确累加。
- chunk size 必须有正整数门禁；tiny vocab、非整除 chunk、原 token ID 映射和现有 dense oracle 全部回归。修复后重跑定向、完整测试和真实 Slurm。

chunked receiver embedding 本地实际结果：

- `python -m pytest -o addopts= --basetemp=local/test-tmp/chunked-embedding test/transport/test_soft_transport.py test/transport/test_wrapper.py test/transport/test_smoke_stt.py -q`：30 passed（4.04s）。
- `python -m pytest -o addopts= --basetemp=local/test-tmp/full-chunked-embedding -q`：150 passed（73.58s），仅有既存 pandas 可选依赖 2 条 warning。
- BF16 receiver/chunk size 2/non-divisible 5-row vocab 与 float32 oracle 在明确容差内一致，输出为 BF16；chunk size 0 拒绝，既有非连续 target ID oracle 继续通过。Black unchanged、内存 compile 2/2、`git diff --check` 通过。

真实 Job 245 最终结果：

- Job 245：0:16.67、Exit 0、MaxRSS 16,560,968 KiB、0 swap；JSON 8,581 bytes，SHA-256 `a14da4b15a368eefbd905d61ad4be71af143fdf2ab6df74071fa487c6b867c26`，stderr SHA-256 `6ecddddcca89383eeaf4c211c6b3fcbf412c4713c4dc301d2cc55401ff86eccc`，无 partial。
- schema v2；Receiver-only 输入 8/输出 2 tokens，STT source 7/virtual 7/output 2 tokens，virtual shape `[1,7,5120]`。两路均有 token IDs/解码文本。
- runtime 为 `blackwell-cu128`、torch 2.7.1+cu128、compiled arches 含 `sm_120`、RTX 5090 capability `[12,0]`；正式 artifact shape/nnz/provenance、锁定 revisions 和 code version `024beacd...` 完整。
- retained/active support mass 在 `0.9999999404..1.0000005960`，top-m dropped mass 全零；分段 metrics 有限，peak memory 31,074,283,520 bytes。真实功能 smoke 验收通过，其耗时不进入 latency 表。

最终本地验收：

- `python -m pytest -o addopts= --basetemp=local/test-tmp/final-stage2 -q`：150 passed（72.90s），仅有既存 pandas 可选依赖 2 条 warning。
- 最终 6 个变动 Python 文件经 Black `format_str` 检查全部 unchanged、内存 compile 6/6；`smoke_real_models.sbatch` Bash syntax、两份计划的 Job 245/脚本/Blackwell profile 路径和 `git diff --check` 均通过。

## 2026-09-03：阶段 3 统一 evaluator 核心单元

测试计划：

- R/S/T2T/C2C/STT 的 stub adapters 对同一 fixture 共享完全相同的 `sample_id`、canonical messages、prompt hash/metadata、true answer 和严格 answer parser；方法差异只出现在 generation/metrics/diagnostics。
- 逐样本成功记录采用版本化统一 schema，STT 分段 latency、source/virtual/output lengths、peak memory 和 transport quality 可原样保存；非有限 metrics 或缺字段明确失败。
- 单样本异常写入 status=`failed` 记录和独立 bad-sample JSONL，继续后续样本；summary 分母只含成功记录，失败不静默算错或丢弃。
- resume 跳过已有完整 success，重试 failed/incomplete；重复 success、输入 sample ID 重复或既有记录与当前 prompt fingerprint 不符时失败。
- 多 rank JSONL merge 按 subject/question index/sample ID 确定排序并拒绝重复；summary/category 计数守恒，逐题 records 是唯一统计输入。
- CLI 从 records JSONL 生成原子 summary，输入缺失/空 success/非法 JSON 显式失败；help 不加载模型或访问网络。完整 pytest、Black、compile、README/recipe/计划路径与 diff 检查均须通过。

本地实际结果：

- `python -m pytest ... test_evaluation.py test_summarize_transport.py test_transport_runner.py test_smoke_stt.py`：28 passed（31.07s），覆盖五类 method schema、STT adapter 指标/diagnostics、配置 factory、失败恢复、重复/漂移拒绝、rank merge、runner 输出和既有 smoke 回归。
- `python -m pytest -o addopts= --basetemp local/pytest-stage3-full`：168 passed（103.78s），仅有既存 pandas 可选依赖 2 条 warning；新增 Slurm/recipe 静态门禁通过。
- `python -m script.evaluation.unified_evaluator --help` 与 summary CLI help 通过；数学评测的可选 `math_verify` 已改为对应数据集才延迟导入。
- 生产 Python 文件已由 Black 格式化，测试格式通过 Black diff 校验；内存语法编译与 `git diff --check` 通过。Windows pytest/Black 的缓存原子替换受目录 ACL 限制，改用 workspace `local/` basetemp 与只读格式 diff，不改变测试标准。

真实集成待验收：

- 固定 `abstract_algebra` 前 5 题、greedy 64-token、正式 500k artifact、`blackwell-cu128`、单 RTX 5090；加载前必须通过 CUDA/锁定依赖/30GiB/compiled-arch/artifact 门禁。
- Slurm 成功后检查 5 条逐题 success 或显式 failure、canonical/prompt metadata、source/virtual/output lengths、分段 latency、support quality、summary 计数与 accuracy、无 `.partial`、作业 Exit/MaxRSS/日志哈希。
- 提交前 provenance 复核新增：每条 STT success diagnostics 必须绑定 git code version、runtime profile/package/CUDA arch/device、transport config、artifact path/SHA-256/shape/nnz/metadata；factory stub 与真实 adapter schema 均覆盖。修正后定向 26/26（6.04s）、完整 168/168（93.19s）通过，仅有既存 pandas 可选依赖 2 条 warning。

Job 246 OOM 修复测试计划：

- loader 默认调用保持 smoke recipe 原 `device_map`，evaluator factory 可显式把 `source_device_map=cpu`、`target_device_map=auto` 传入并写入 provenance；未配置 override 时行为不变。
- 固定评测 recipe 将 source CPU offload、receiver GPU auto 和 16-token greedy 答案边界写死；Slurm 保持 1 GPU/64G/30m 并增加 expandable segments，测试脚本、环境名、离线与无 uv。
- 重跑 adapter/smoke/Slurm 定向测试与完整回归；真实复验必须重试 Job 246 的 failed records，并至少产生 success summary。CPU source 推理仍必须在 Slurm 内，不能在登录节点验证。

Job 246 OOM 修复本地实际结果：

- 定向 `test_evaluation.py`、`test_smoke_stt.py`、`test_transport_evaluation_slurm.py`：25 passed（4.56s），覆盖 device-map override 传递/provenance、默认 smoke loader 回归、16-token recipe 和 expandable-segments Slurm 环境。
- 完整 `python -m pytest -o addopts= --basetemp local/pytest-stage3-offload-full`：168 passed（89.93s），仅有既存 pandas 可选依赖 2 条 warning。下一步以临时提交同步 Guqq，保留 Job 246 failed records并验证 resume 重试。

Job 247 真实断点复验结果：

- Job 247 使用 `d98a85e`、source CPU/receiver auto、同一正式 artifact 和原 Job 246 records；约 5 分钟完成，历史 5 failed 后追加 5 success，latest summary 为 5 success/0 failed/0 correct，证明 failed-only resume 与统计分母正确。集群 accounting disabled，故无 sacct MaxRSS；逐题 CUDA peak 平均 28,121,007,411 bytes，均低于 31.37GiB 设备容量。
- 平均 source/transport/receiver-prefill/decode/total 为 59.3623/0.7554/0.06735/0.04576/60.23085s；平均 source/virtual/output 为 154.4/154.4/2 tokens。5 条生成均为 receiver IDs `[1062,2]`、文本 `>`、prediction null，因此 accuracy 0.0 是如实质量结果，不作为阶段 3 功能通过阈值。
- provenance：runtime `blackwell-cu128`，source/target override `cpu`/`auto`，artifact `[131069,151669]`、2,733,518 nnz、SHA `1495d522...`，evaluator code `d98a85e5...`；active/retained mass 1.0、dropped 0。records/summary SHA 为 `5fd221487e7338bbe5ccfa95956e9eaa7012a6c1faf75508dcb1daa6e11c1644` / `5e7a2def5dffddf580d2c9b0c6ecb4ab1b78b18131d67bf8dd8f9513ba2cbbbf`，stderr/stdout SHA 为 `699e5ce4c173e9f87e5008e90dbff8d58620abd977b38c26619847547e20da74` / `a39b5b0767cfc2f8e472296c9435f07bca6e9caf2475a63865bf363ea340a81d`，无 partial。阶段 3 真实功能验收通过。

## 2026-09-03：阶段 4 消融计划与配对统计单元

测试计划：

- 版本化 ablation plan 只接受非空、有限、可 JSON 序列化的预注册维度；dev 按稳定键序做确定 Cartesian 展开，run ID 与 YAML 键顺序无关且无重复。
- frozen test 参数必须覆盖每个维度且值属于 dev search space；benchmark test 只能生成该冻结组合，不能用 CLI/YAML 额外 override 偷换 tau、epsilon、shift、T 来源或近似参数。
- 固定 recipe 明确 `transport_dev` 选择空间与 benchmark test 冻结值，包含 T 来源、local/Sinkhorn、epsilon、tau、shift、exact/hard/top-m/ORF；路径和输出均在忽略的 `local/transport/ablation/`。
- 配对统计以 sample ID 的 latest successful records 为输入，报告 reference/candidate 缺失 ID、成对数、两侧 accuracy 与 delta；存在缺失时仍显式输出但 `complete_pairing=false`，不得静默取交集宣称完整配对。
- `run_transport_ablation.py` 仅生成原子 plan JSON，不加载模型/访问网络；dev/test help 与生成命令、非法 frozen 配置、Black/AST、完整 pytest 和 README/路径/diff 均验证。

wrapper 近似模式测试计划：

- `exact` 保持现有 wrapper 输出逐位不变；`top_m` 要求正 `source_top_m` 并记录丢弃质量，full-vocab m 退化为 exact；`hard` 等于 `argmax(Tp)` 的 receiver embedding且 deterministic tie-break。
- `precomputed` 必须提供 shape/device 正确的 `W_in^B T` source values，并与 exact oracle 对齐；hard/precomputed/top-m 都接受 fingerprint-verified tokenizer vocab 对 LM-head 尾部 padding 的显式裁剪，中间缺口仍失败。
- `orf` 必须提供 fingerprint/维度匹配的 state，只调用 source backbone、不得调用 CausalLM LM head；virtual prompt stats 明确 unavailable/null，不伪造 retained mass。输出与 `apply_orf_transport` oracle、shift/mask/receiver prefill 对齐。
- 非法 mode、缺失/多余 precomputed 或 ORF state、top-m 无 m、backbone 无 hidden state均显式失败；receiver-only 路径不受模式影响。
- evaluator adapter 把 approximation mode 和 stats availability 写入统一 diagnostics；旧 exact Job 247 schema保持兼容。定向 wrapper/approximations/ORF/evaluation 后运行完整 pytest、Black/AST/diff。

## 2026-09-03：精确 STT 跨 benchmark 验证单元

测试计划：

- 数据加载器按 benchmark 显式选择 Hugging Face config/split：MMLU-Redux 使用 subject config，GSM8K 使用 `main`，MATH-500 不传 config，LongBench 使用具体 task；固定 `limit` 必须作用于原始稳定索引并保留可恢复 sample ID。
- MMLU/GSM8K/MATH 的参考答案和预测解析沿用同一 `UnifiedEvaluator` 协议；LongBench 保存原始 `answers/all_classes/length/_id` 与生成文本，明确标记为待官方 task scorer 的生成成功，不能伪造 boolean accuracy。
- 新建版本化的精确 STT multi-benchmark smoke recipes；四项均使用相同模型 revision、正式 T、source CPU/receiver GPU、greedy 解码和独立输出目录，不启用近似或消融参数。
- 单元测试覆盖四类 loader 调用、开放式无 boolean 标签的 summary、非法/空 task、limit 和 resume；本地完整 pytest 与静态检查通过后，使用临时 `[UNACCEPTED]` 分支在 Guqq 通过 Slurm 逐项运行。
- 远程验收逐 benchmark 检查成功/失败样本数、原始预测与参考、任务得分或“需官方 scorer”状态、分段 latency、token 长度、峰值显存、runtime/artifact/code provenance、文件 SHA 和无 `.partial`；小样本结果只作为功能与初步质量报告，不冒充完整 benchmark 主表。

本地实际结果：

- benchmark loader、开放式外部评分和 recipe/Slurm 定向测试 24/24 通过；MMLU-Redux/GSM8K/MATH-500/LongBench 的 config/split 调用、稳定 limit、Qasper 参考字段与 `external_required` 汇总均已覆盖。
- wrapper/approximation/ORF 定向测试 29/29、阶段 4 控制面与 adapter 合并定向测试 53/53 通过；ORF 仅调用 backbone，exact 旧接口保持兼容。按当前需求不提交任何近似/消融远程实验。
- 完整 `python -m pytest -o addopts= --basetemp local/pytest-benchmark-full -q`：192 passed，132.49s；仅有既存 pandas 对可选 numexpr/bottleneck 版本的 2 条 warning。
- 变更 Python 文件 Black 检查、`compileall`、通用 benchmark Slurm Bash syntax 与 `git diff --check` 通过；Black 默认缓存 ACL 问题通过仓库内 `local/black-cache-stage4` 规避。

数据文件 fallback 补充结果：

- 本地从固定 revision 下载 MATH-500 `test.jsonl`（446,564 bytes，SHA-256 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`）和 LongBench Qasper Parquet（1,863,050 bytes，SHA-256 `7c6bf3a2a402b557d001808ba345a23921a211c39bf2d36d925d1d70e21b3f03`）；均位于忽略的 `local/transport/datasets/`，不进入 Git。
- runner 对本地 `data_file` 在加载前校验存在性、64 位 SHA 和 `json/parquet` format；远程数据路径可绑定完整 40 位 revision。MATH/Qasper recipes 同时保存原 repo revision 与本地文件 SHA，GSM8K 保存现有 cache 的完整 revision。
- loader/recipe/evaluation 定向回归 25/25 通过；Black 写测试文件仍受 Windows ACL 限制，使用 `black --diff` 确认仅一处引号机械差异并由补丁修正，随后需再次执行 check。
- fallback 最终完整回归 `python -m pytest -o addopts= --basetemp local/pytest-benchmark-data-full -q`：193 passed，189.95s；仍只有既存 pandas 可选依赖 2 条 warning。
- Guqq Job 248（GSM8K exact smoke）：3 records、3 success、0 failed、3 scored、accuracy 0.0；平均总耗时 55.4335s，平均 source/transport/prefill/decode 为 52.9645/0.8113/0.0852/1.5725s，平均峰值显存 27,725,928,106.7 bytes；records/summary 均完成且无 `.partial`。
- Guqq Job 249（MATH-500 exact smoke）：3 records、3 success、0 failed、3 scored、accuracy 0.0；平均总耗时 62.9749s，平均 source/transport/prefill/decode 为 60.4815/0.8192/0.0867/1.5874s，平均峰值显存 28,172,192,085.3 bytes；records/summary 均完成且无 `.partial`。
- Jobs 248/249 产物经 scp 到本地忽略目录复核，SHA 与服务器一致。六条 record 均绑定 code `fb6c687...`、正式 artifact SHA `1495d522...`（131,069×151,669，2,733,518 nnz）、source CPU/target auto、`approximation.mode=exact`，active support mass 均为 1.0。GSM8K 三条输出为重复 `the`、重复短片段和 `>`；MATH-500 为重复 `and` 和两个 `>`；六条 parser prediction 均为 null，因此 0/3 是真实生成质量而不是 execution failure。

### LongBench 已渲染 prompt 单元

测试计划：

- LongBench sample 必须显式标记 formatter 输出已经由 source tokenizer 渲染；adapter 对该标记直接编码 `sample.prompt`，不得再次调用 `apply_chat_template`。
- MMLU/GSM8K/MATH 等普通 sample 继续由 adapter 对 canonical messages 单次应用 source chat template，保持既有 exact 行为和 Job 247–249 协议不变。
- 标记只接受严格布尔值；非法元数据显式失败。diagnostics 记录实际使用的是预渲染还是 adapter 渲染路径，便于远端审计。
- 运行 transport runner/adapter 定向测试、完整 pytest、Black、compileall、Slurm Bash syntax 和 diff 检查；通过后推送新的 `[UNACCEPTED]` 兼容提交，仅以独立输出路径重跑 Qasper exact。

实际结果：

- transport evaluation/runner 定向回归 25/25 通过；覆盖预渲染 prompt 不再调用 chat template、普通路径仍单次渲染、非布尔标记失败，以及 LongBench sample 标记与外部 scorer 字段。
- 完整 `python -m pytest -o addopts= --basetemp local/pytest-longbench-rendered-full -q`：195 passed，152.42s；仍只有既存 pandas 可选依赖 2 条 warning。
- 通用 benchmark Slurm Bash syntax 与四个变更 Python 文件的内存 AST 编译通过。当前 Black 版本不支持 `--cache-dir`，首次 `black --diff` 仅报告测试文件两处换行格式并已按输出修正；随后重新检查。

### Exact 长序列 query-chunk 单元

测试计划：

- exact/top-m soft transport 按 source sequence 维分成至多 32-token query chunks；每块继续调用同一完整词表 softmax、完整稀疏 T 和 receiver embedding 运算，再按原顺序拼接 embeddings 与全部 mass stats，数学语义必须与未分块 dense/sparse oracle 一致。
- 65-token toy prompt 必须观测到 `[32, 32, 1]`（按实际长度断言）的 transport 调用上界，输出、causal shift、mask 和 stats 与既有 exact 行为一致；短序列既有测试保持逐位通过。
- 该修复只缩小峰值中间张量，不使用 top-m、ORF 或预计算近似。Job 250 的 21.10GiB 申请应由约 `32/2048` 比例的 query-edge 中间块替代；真实验收仍以修复后 Qasper Job 的 Exit 0、1 success、无 partial 和峰值显存为准。
- 完成 wrapper/soft-transport/evaluator 定向回归、完整 pytest、Black、内存 AST、Bash 和 diff 检查后，推送新的 `[UNACCEPTED]` 兼容提交并仅重跑 Qasper exact。

实际结果：

- wrapper/soft-transport/evaluator/runner 定向回归 51/51 通过；65-token exact 输入严格分为 `[32, 32, 1]`，拼接后 embeddings `(1, 65, 2)` 和 mass stats `(1, 65)` 与既有 exact oracle 一致。
- 完整 `python -m pytest -o addopts= --basetemp local/pytest-exact-query-chunk-full -q`：196 passed，150.74s；仍只有既存 pandas 可选依赖 2 条 warning。
- 六个相关 Python 文件内存 AST 和通用 benchmark Slurm Bash syntax 通过；首次 `black --diff` 仅要求 wrapper 两行机械折叠，已按输出修正并待最终复核。
- Guqq Job 253（LongBench Qasper exact smoke）最终通过：1 record、1 success、0 failed、0 scored/1 unscored、accuracy null，scoring status `external_required`；生成 `>`，官方参考为 `Ground truth is not established in the paper`。source/virtual/output 为 2060/2060/2 tokens，source/transport/prefill/decode/total 为 864.2460/44.6731/0.3430/0.0745/909.3365s，CUDA peak 25,437,179,392 bytes（23.69GiB）。
- Job 253 diagnostics：`source_prompt_rendered=true`、virtual shape `[1,2060,5120]`、`approximation_mode=exact`、top-m null、transport stats available、retained/active mass `1.0000002384`、dropped mass 0；code `e02f4ad...`、正式 artifact SHA/shape/nnz、source CPU/target auto 和 Qasper revision/file SHA 齐全。records/summary SHA 为 `a8918457...`/`009ca3f4...`，stdout/stderr SHA 为 `d5a75c97...`/`7c3ab6b2...`，无 bad-samples 或 `.partial`。
- 兼容分支最终静态/文档检查：六个相关 Python 文件 `black --diff` unchanged、内存 AST 通过；两份模块 README、永久结果报告、两份计划路径存在，报告中的 Job 253/909.34s/23.69GiB/external-scorer/近似延期字段可检索，`git diff --check` 通过。`docs/assets/alignment.py` 保持未跟踪用户参考，不进入提交。
- main-based 最终移植树复验：C2C 与 Guqq 已测兼容树无差异，同时保留 main 的正式 Job 240/245 计划证据；完整 `python -m pytest -o addopts= --basetemp local/pytest-exact-mainbased-final -q` 为 196 passed、122.42s，仍仅有既存 pandas 可选依赖 2 条 warning。七个相关 Python 文件 `black --diff` unchanged、内存 AST 和通用 Slurm Bash syntax 通过。
# 2026-09-03：阶段 5 配对统计与结果聚合单元

测试计划：

- paired bootstrap 仅使用双方均成功且已评分的相同 sample ID；固定 seed 必须逐位可复现，置信区间边界有序，并拒绝非法重复次数、置信水平或不可配对输入。
- McNemar 输出 both-correct、reference-only、candidate-only、both-wrong 四格计数和 exact two-sided p-value；手工用例计数正确，零 discordant/all-identical 时返回 p=1，而不是除零或缺失。
- subject/category 切片必须同时报告样本数、双方正确数和 delta，所有切片计数之和分别与 paired 总数守恒；双方同一 sample 的切片标签不一致必须失败。
- latency 明确聚合 `source_prefill_seconds`、`transport_seconds`、`receiver_prefill_seconds`、`decode_seconds`，报告双方各段 count/mean；不得把缺失字段当作零。
- failure index 覆盖双方 latest 非成功记录，保留 sample ID、status、error type/message，按稳定键排序；CLI 在既有摘要 schema 上向后兼容地加入完整配对分析。
- 运行新增统计/CLI 定向 pytest、完整 pytest、Black、AST/compile 与 `git diff --check`；本单元不加载模型、不访问网络、不运行近似或消融实验。

实际结果：

- 新增统计与既有 summarizer/ablation 定向回归：`16 passed in 33.37s`；覆盖固定 seed bootstrap、手工 McNemar 四格表、零 discordant、切片守恒、缺失 latency、不完整/未评分配对、失败索引、标签漂移和 CLI 兼容输出。
- 完整回归：`202 passed, 2 warnings in 115.86s`；两条 warning 仍仅为既有 pandas 对可选 `numexpr`/`bottleneck` 版本的提示。
- 五个相关 Python 文件在任务专用 `BLACK_CACHE_DIR` 下 Black unchanged；`compileall`、CLI `--help` 与 `git diff --check` 通过。默认用户 Black 缓存连续卡住三次，未缩减测试范围，经验已写入 `docs/agents/lessons.md`。
# 2026-09-03：反向/第二模型对配置与方向安全单元

测试计划：

- 主、smoke、反向 Mistral-Nemo→Qwen3 与第二 Qwen3→DeepSeek recipes 均须通过同一 `TransportConfig` schema；模型/tokenizer revision 为 40 位提交，tokenizer fingerprint 为 64 位 SHA-256，artifact 输出路径互不相同。
- 反向 recipe 的 source/target 名称、revision、tokenizer fingerprint 和 artifact shape 语义必须相对主方向交换；source/target marginals 仍由该方向独立构建，recipe 不得声明转置或复用正向 artifact。
- 每个正式 recipe 显式冻结安全 special policy：完整 source tokenizer support、ordinary-only target support、exact-kind 后 literal-byte fallback、receiver-native boundary；非法或未知 policy 必须在加载配置时失败。
- wrapper 接受成对的 expected source/target fingerprints；用正向 artifact 初始化反向 wrapper 必须在任何模型 forward 前失败，匹配方向保持现有 exact oracle 结果不变；近似 wrapper 重建不得丢失方向门禁。
- 第二模型对使用既有 Qwen3→DeepSeek tokenizer 审计的锁定 revision/fingerprint，不下载模型、不构建 T、不运行远程实验。运行配置/artifact/wrapper 定向 pytest、完整 pytest、Black、compile 与 `git diff --check`。

实际结果：

- config/wrapper/smoke/evaluation 定向回归：`61 passed in 4.98s`；主/反向/DeepSeek recipes 均解析，方向交换、三组独立路径/shape/fingerprint、严格 special policy、wrapper 正反指纹拒绝与匹配路径通过。
- 完整回归：`206 passed, 2 warnings in 112.49s`；两条 warning 仍仅为既有 pandas 可选依赖提示。
- 七个相关 Python 文件 Black unchanged，六个文件内存 AST 通过，source `compileall` 通过，`git diff --check` 通过。测试文件 pycache 原子写受既有 Windows ACL 阻止，因此改用不写文件的 AST 检查；pytest 已实际导入并执行这些测试，未跳过范围。
- 本单元只验收配置和加载安全；反向/第二模型对尚未构建 full T 或运行 benchmark，recipe README 明确禁止把配置存在误报为实验完成。
# 2026-09-03：Planner→Thinker 双 CoT exact STT 单元

测试计划：

- adapter 从同一 canonical problem 构造不同角色的 sender/planner 与 receiver/thinker prompts；两次 native chat template 均显式收到 `enable_thinking=true`，recipe `use_cot=true`，生成保持 greedy。
- sender 先按独立 `sender_max_new_tokens` 生成 think；保存生成 token/text，并把原 sender prompt 与全部有效 think token 拼成完整 context 后重新 no-grad forward。prompt 或 think 任一段被遗漏、generate 未返回原 prompt 前缀、预算非法均显式失败。
- exact STT 对完整 sender context 每个有效 hidden/logit 位置产生一枚 aligned embedding，新主协议固定 no-shift；receiver prompt 由 receiver tokenizer 显式编码，其 native embeddings 严格拼在 aligned prefix 后。
- 拼接后的 attention mask、position IDs、receiver KV cache 与 decode position 连续且长度守恒；覆盖 batch、padding、receiver prompt 缺失/shape 不匹配和 EOS。receiver 不能收到 source token IDs。
- diagnostics/metrics 保存 sender/receiver rendered prompts、sender think、双 CoT 开关、sender prompt/think/context、aligned prefix、receiver prompt/output token 数，以及 planner generation/source alignment/transport/receiver prefill/decode 分段耗时。
- 更新四项 exact benchmark recipes 为新协议独立输出路径，防止与旧 prompt-only records resume 混用。本地运行定向 pytest、完整 pytest、Black、AST/Bash/diff；真实模型仅由临时验证分支经 Slurm 测试，近似/消融不运行。

实际结果（本地阶段）：

- adapter、wrapper、config、smoke、runner、Slurm recipe 与五阶段统计定向回归：`80 passed, 2 warnings in 8.27s`。测试明确断言两侧拿到相同题目、两个 chat template 均 `enable_thinking=true`、sender 先生成且返回原 prompt 前缀、完整 sender context 全位置对齐，以及 aligned prefix 后严格跟随 receiver native embeddings。
- 最终完整 `python -m pytest -o addopts= --basetemp .pytest-full-planner-final -q`：`207 passed, 2 warnings in 124.25s`；两条 warning 仍仅为 pandas 对可选 `numexpr`/`bottleneck` 版本的既存提示。
- 14 个新协议模块/测试文件 Black unchanged；保留既有风格的 legacy unified evaluator 通过内存 AST 与 25/25 adapter/runner 复验，避免把全文件机械格式化混入本单元。Windows 路径/临时目录的三次入口失败未进入业务断言；修正为从 `C2C/` 运行并使用内部相对 `--basetemp` 后完整通过，经验已追加到 `docs/agents/lessons.md`。
- 真实模型协议 smoke 与四项 benchmark 尚未运行；本地结果不能替代 Slurm/GPU 验收，当前提交只能标记为临时未验收。

# 2026-09-03：正式 T artifact 使用文档单元

测试计划：

- 从本地正式 `.npz` 只读核对文件大小、SHA-256、数组字段、shape、nnz、dtype 和 metadata fingerprints，文档不得混淆 recipe 构建目标路径与当前实际使用路径。
- 检查文档引用的本地文件、Python 模块、四个 benchmark recipes 和 Slurm 入口均存在；示例命令必须从 `C2C/` 目录运行，并明确真实推理只能在 Guqq 经 Slurm 执行。
- 检查方向固定为 Qwen3-8B→Mistral-Nemo、矩阵布局为 `[target, source]`，禁止转置、dense 化和 Git 提交；程序化示例使用 fingerprint-checked loader，不绕过 wrapper/adapter。
- 本单元仅修改 Markdown/`.gitignore`，无需新增代码单元测试；完成后运行链接/路径、命令、格式与 `git diff --check` 检查并记录实际结果。

实际结果：

- 本地正式 artifact 经 `load_transport_artifact(..., allow_pickle=False)` 路径加载成功：shape `(131069, 151669)`、`2,733,518` nnz、权重 `float64`；文件大小与 SHA-256 已与 scp 前远端值一致。
- 文档自身、本地 artifact、四个 benchmark recipes、通用 Slurm 入口、artifact loader、wrapper 和 adapter 共 10 个引用路径全部存在；PowerShell/Bash 示例的工作目录与换行语法已人工复核。
- `git diff --check` 通过；关键大小、SHA、shape、Slurm 命令和 fingerprint 警告均可检索。首次路径存在性命令从 `C2C/` 错用仓库根相对路径而返回 False，改从仓库根完整重跑后 10/10 均为 True，不属于文档链接失败。
- 审计发现正式 artifact metadata fingerprints `c39a.../12be...` 与当前 recipe `1a385.../8542...` 不一致；文档显式要求严格拒绝并先复核 provenance，未通过删除 fingerprint 检查降低标准。

# 2026-09-04：STT 伪代码文档单元

测试计划：

- 伪代码必须让同一 problem 分别进入 planner 与 thinker 的原生 chat template，并显式开启两侧 thinking；planner think 先生成，随后 `sender prompt + sender think` 全 context 重新 forward。
- exact STT 必须按 row-vector 语义表达 `softmax(LMHead_A(h)/tau)`、方向为 `[target, source]` 的稀疏 T 以及 receiver embedding expectation；不得写成转置方向、hard token 映射或 dense T。
- receiver 输入顺序必须是 aligned sender prompt、aligned sender think、receiver native prompt；mask 与 position 连续，receiver 再执行自己的 CoT 和自回归回答。
- 检查文档中引用的 artifact、loader、wrapper、adapter 路径存在，伪代码块/公式/Markdown 格式和 `git diff --check` 通过。本单元只修改文档，不要求新增代码测试。

实际结果：

- 正式 artifact 实测 `shape=(131069, 151669)`，source/target active supports 分别为 `151669/131069`，source IDs 连续覆盖完整 source tokenizer support，target 最大 ID 为 `131071`；伪代码据此使用 active token-ID gather，没有错误假设 T 两轴都等于完整 tokenizer 长度。
- `T_algo.md`、`T_artifact_usage.md`、artifact/soft-transport/wrapper/adapter 共 6 个引用路径全部存在；双 `enable_thinking`、sender full context、row-vector `p_A @ T^T`、三段 prefix、no-shift 与禁止 dense 等关键字段均可检索。
- Markdown 结构和伪代码人工复核通过，`git diff --check` 通过。本单元没有修改 Python 代码，因此未新增或运行无关单元测试。

# 2026-09-04：Mistral-Nemo→Qwen3 反向 transport artifact 单元

测试计划：

- toy 稀疏 artifact 的反向结果必须保持 `[target, source]` 布局，交换 active source/target token IDs，并使每个新 source 列严格归一化。
- 反演前后的联合质量逐边一致：`T_reverse[qwen, mistral] * p_realized(mistral)` 必须等于 `T_forward[mistral, qwen] * p(qwen)`；反向 source marginal 使用正向矩阵实际输运得到的 marginal，不能因正向允许的 `2e-3` 残差伪造联合质量。
- source/target fingerprint 字段只做方向交换且本单元不校验 live tokenizer fingerprint；candidate graph 坐标、special mappings、active supports 和派生 provenance 必须一并反转，不能把裸转置标记为可用 transport。
- 双重反演应在浮点容差内恢复原矩阵、marginals 和 token supports；序列化往返及独立 artifact audit 必须通过。
- CLI 拒绝输入输出为同一路径，以临时文件原子发布，并生成 JSON/Markdown audit；运行新增定向 pytest、完整 pytest、Black、compile/AST、CLI help 与 `git diff --check`。
- 正式 2,733,518-nnz artifact 的转换属于批量处理，只在 Guqq 通过 Slurm 执行；验收 shape 应为正向 active support 的严格交换 `(151669, 131069)`，并记录文件 SHA-256、大小、nnz、列和及 marginal 残差。fingerprint 一致性调查不属于本单元，不通过删除既有门禁来规避。

实际结果（本地阶段）：

- toy 反演、artifact/audit、config、wrapper 与构建 CLI 定向回归 `57 passed in 25.32s`；完整回归 `211 passed, 2 warnings in 244.38s`，两条 warning 仍仅为既有 pandas 可选依赖版本提示。
- 最终反演专属回归 `4 passed in 12.84s`：反向联合质量等于正向联合质量的转置，source marginal 使用正向实际 transported marginal，列和为 1；双重反演恢复原 transport/source marginal/support/candidate coordinates，special mapping 与 fingerprints 仅交换方向。
- 四个新增 Python 文件 Black unchanged、内存 AST 解析通过；CLI `--help`、CPU-only Slurm 脚本 `bash -n` 和 `git diff --check` 通过。默认 Black 写测试文件受既有 Windows ACL 阻止，按已有经验使用 `--diff` 后通过 `apply_patch` 应用两处机械格式，再由任务专用 cache 完成最终检查，未降低测试范围。
- 正式 Slurm 结果：Guqq CPU Job 324 Exit 0，耗时 17.45 秒、MaxRSS 1,435,048 KiB、0 swap；生成 artifact 40,694,539 bytes，SHA-256 `77905324ee9e063aef33c0e01a73c26bf4ac7907c8a48f972c463f5af3eb486f`，无 `.partial.npz`。
- 独立 audit：shape `[151669, 131069]`、2,733,518 nnz、candidate edges 2,733,518、nonnegative/valid true、最大列和误差 `4.907185768843192e-14`、column marginal L1 `1.9365990559674718e-15`、transported marginal L1 `1.052004005974113e-13`，无危险 special mappings。
- 本地 scp 副本大小和 SHA 与 Guqq 逐位一致；`load_transport_artifact` 全量验证成功，source/target active supports 为 `131069/151669`，metadata source/target fingerprints 为 `12be.../c39a...`，`fingerprint_validation=not-performed`，parent SHA 正确绑定正向正式 artifact `1495d522...aba97`。

# 2026-09-04：Mistral-Nemo→Qwen3 独立 OT 求解单元

测试计划：

- 反向 ANN Slurm 入口必须固定 source=`mistralai/Mistral-Nemo-Instruct-2407`、target=`Qwen/Qwen3-8B` 及各自 40 位 revision，输出独立候选文件；不得读取、转置或改写正向 ANN JSON。
- 正式反向 Slurm 入口必须消费同一 OpenHermes 500k records/manifest 的 `transport_train`、独立反向 ANN，调用通用 builder 重新估计 marginals、构图和求解 Sinkhorn；固定 epsilon `0.5`、tolerance `2e-3`、max iterations `10000`、smoothing `1e-8`。
- 入口必须使用独立 artifact/audit/checkpoint/log 路径，预期 active shape 为 `[Qwen ordinary target=151643, Mistral full source=131072]`；禁止把 Bayes 工件或正向 artifact 当作 resume 输入。
- Slurm wrapper 通过 Bash syntax、stub 参数转发、失败传播和无 partition/GPU 检查；recipe 的路径与 expected shape 必须与独立正式工件一致。
- 本地运行新增定向 pytest、相关 transport 回归、完整 pytest、Black、AST/Bash 和 `git diff --check`。正式 ANN 与 OT 只在 Guqq 经 Slurm 运行。
- 正式验收核对 ANN 方向 fingerprints/coverage/hash，artifact shape/nnz/source-target marginals/metadata code/data/ANN provenance、列归一化、marginal L1、dangerous specials、文件 SHA/大小、无 partial，并将独立工件 scp 到本地根目录再次加载验证。

实际结果（本地阶段）：

- special-token metadata、反向 ANN/formal Slurm、既有 ANN/formal 和 recipe 配置定向回归 `36 passed in 40.55s`；新增用例证明 backend `special=true` 的 added token 即使未出现在 `all_special_tokens` 也会从 ordinary support 排除。
- 完整回归 `215 passed, 2 warnings in 234.71s`；两条 warning 仍仅为既有 pandas 可选依赖版本提示。
- 三个相关 Python 文件 Black unchanged、内存 AST 通过；两个新 Slurm 脚本 Bash syntax、stub 参数转发、固定方向/revisions/solver 参数、独立路径、无 partition/GPU 和 `git diff --check` 均通过。
- 正式 Guqq ANN/Sinkhorn 尚未运行，本地测试不能替代真实 artifact 验收。
- Job 325 首次真实 ANN 诊断未通过正式门禁：source Mistral ordinary/source coverage `130072/130072` 正确，但 target Qwen ordinary/target coverage 为 `151655/151655`，与预注册 151643 不符；正式 Sinkhorn 未提交。新增回归要求 backend `special=false`、但 ID 位于 `tokenizer.vocab_size` 外的 reserved added token 仍从 ordinary support 排除。
- reserved-added 修复定向回归 `35 passed in 47.96s`，覆盖 metadata、ANN、candidate graph、facade 与两个反向 Slurm 入口；完整回归 `216 passed, 2 warnings in 362.05s`。三文件 Black unchanged、两个 Slurm Bash syntax 与 `git diff --check` 通过；warning 仍仅为既有 pandas 可选依赖版本提示。
- 正式 Job 326 ANN：Mistral ordinary source/coverage `130072/130072`、Qwen ordinary target/coverage `151643/151643`；JSON 133,772,725 bytes、SHA-256 `33fb337bea7c28d3d271f24ed02c336ae2f36942a645672392166c59dae6fc7d`，方向 revisions、fingerprints、code 和 input fingerprint 完整。
- 正式 Job 327 独立 OT：从 fresh checkpoint 运行 1:25:44，Exit 0、MaxRSS 8,389,548 KiB、0 swap；artifact 39,412,012 bytes、SHA-256 `78a01689490224824db6c54460c992091572694cd5c603327e5d84fe3efff84d`，无 partial。
- Job 327 audit：shape `[151643, 131072]`、2,693,524 nnz、nonnegative/valid true、最大列和误差 `7.178257988016412e-12`、column residual `1.371047512616176e-13`、row/transported marginal L1 `0.0019999667816998756 < 0.002`，无 dangerous special mappings；metadata 绑定 495,000 samples/997,233 canonical messages、manifest/ANN SHA，且不含 `derivation`。
- scp 后本地文件大小/SHA 与 Guqq 逐位一致；`load_transport_artifact` 全量验证 shape、nnz、support 131072/151643、fresh code/input provenance 和无 `derivation` 通过。
- recipe/Slurm 定向回归 `14 passed in 13.15s`；最终完整 `python -m pytest -o addopts= --basetemp local/pytest-independent-reverse-final -q` 为 `216 passed, 2 warnings in 182.56s`，warning 仍仅为既有 pandas 可选依赖提示。4 个相关 Python 文件在任务专用 Black cache 下 unchanged，两个 Slurm 脚本 Bash syntax、6 个文档引用路径和 `git diff --check` 通过。首次 pytest 因默认 coverage 插件缺失未收集测试，按既有记录显式清空 addopts 后完整重跑；默认 Black cache ACL 卡住后中止，使用任务专用 cache 完成相同检查，未降低范围。
