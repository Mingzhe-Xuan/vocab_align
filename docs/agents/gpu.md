# 服务器与 GPU 操作记录

## 2026-09-01 17:42 +08:00

- 连接用途：同步本地已提交源码，检查服务器 Python 环境与模型/tokenizer 缓存，并执行 Qwen3-8B → Mistral-Nemo-Instruct-2407 全词表 tokenizer 审计。
- 权限判断：`git pull`、环境检查、依赖检查和仅加载 tokenizer 的审计均属于轻量管理/诊断操作，不加载模型权重、不训练、不推理，可在登录节点执行。若发现需要明显计算、编译或批量预处理，将停止登录节点操作并改用 Slurm。
- 计划顺序：连接后先在仓库执行 `git pull`，再检查环境；不在服务器修改受 Git 管理源码。

## 2026-09-01 17:48 +08:00

- 连接用途：建立持久 SSH 会话，在已同步仓库中检查 Python/缓存并运行同一项真实 tokenizer 审计，避免反复建立连接。
- 权限判断：仍仅包含 `git pull`、只读检查和 tokenizer 元数据审计；不加载模型权重，不产生明显计算负载。
- 计划顺序：进入会话后的第一条命令仍为 `cd vocab_align && git pull`，随后所有操作留在该会话中。

## 2026-09-01 20:05 +08:00

- 连接用途：同步提交 `1c06cd3`，复用 `/home/xmz/vocab_align/C2C/.venv`，最终重跑 Qwen3-8B → Mistral-Nemo-Instruct-2407 的真实 tokenizer provenance 审计并校验产物。
- 权限判断：仅执行 `git pull`、Python venv/版本检查、tokenizer 元数据审计和文件哈希校验，属于登录节点允许的轻量操作；不加载模型权重、不训练、不推理。若实际负载超出预期则停止并改用 Slurm。
- 计划顺序：连接后的第一条命令为 `cd vocab_align && git pull`；如遇网络连接问题，执行 `bash net.sh` 后重试 HTTPS pull；不在服务器直接修改受 Git 管理源码。

## 2026-09-01 20:49 +08:00

- 连接用途：同步 `7922239`，使用 `sinfo` 查看 Guqq 当前 Slurm 分区与资源状态，为真实 tokenizer 预览 artifact 的 CPU 作业确定提交参数。
- 权限判断：仅执行 `git pull` 与 Slurm 状态查询，属于允许的登录节点轻量管理操作；本次不运行候选构建、数据处理、模型加载或推理。
- 计划顺序：连接后的第一条命令为 `cd vocab_align && git pull`；网络异常时先执行 `bash net.sh` 再重试；查询完成后退出，不修改服务器源码。

## 2026-09-01 21:03 +08:00

- 连接用途：同步 `e89818f`，查询 Slurm 状态，接收本地 canonical preview JSONL，并提交真实 tokenizer preview artifact 的 CPU Slurm 作业。
- 权限判断：连接后仅执行 `git pull`、`sinfo/squeue`、创建忽略的 input/log 目录、校验输入哈希与 `sbatch`；候选构建、Sinkhorn 和审计全部由 Slurm 作业执行，不在登录节点运行。
- 输入与产物：本地输入 `C2C/local/transport/inputs/preview_texts.jsonl` 共 16 行，SHA-256 `05CA0628E57EADDA84F4D16968083D5BF12D8A9012B2A1081D9E372047207A3A`；服务器输入、logs、artifact 和 audits 均位于任务专属 `C2C/local/transport/` 忽略目录。
- 计划顺序：新连接第一条命令仍为 `cd vocab_align && git pull`；成功后查询集群、创建任务目录，再由本地 `scp` 传输输入、远端校验哈希并执行 `sbatch`。网络异常时运行 `bash net.sh` 后重试。

## 2026-09-01 21:34 +08:00

- 连接用途：按用户最新确认同步 ANN 验收提交 `8f89fb4`，查询 Slurm 状态并提交 Qwen3-8B → Mistral-Nemo-Instruct-2407 全词表双向 ANN candidate CPU 批处理作业。
- 权限判断：连接后仅直接执行 `git pull`、`sinfo/squeue`、任务目录检查和 `sbatch`；全词表 byte-ngram hashing/LSH 属于批量处理，必须且只会通过 `script/transport/slurm/build_ann_candidates.sbatch` 交给 Slurm，不在登录节点运行。
- 输入与产物：作业仅加载锁定 revision 的 tokenizer；候选 JSON、partial 文件与 logs 位于任务专属 `C2C/local/transport/` 忽略目录，不覆盖其他环境、数据或结果。
- 计划顺序：新连接第一条命令为 `cd vocab_align && git pull`；如 pull 发生网络问题则执行 `bash net.sh` 后重试。同步成功后查询 Slurm、检查现有 Python venv 与依赖、提交作业并记录 job ID。

## 2026-09-01 21:37 +08:00

- 连接用途：首次查询 Job 212 状态时 SSH 在命令执行前被远端关闭；本次建立持久 SSH 会话，监控 ANN candidate Slurm 作业并在完成后只读检查日志、JSON schema、coverage、文件大小与 SHA-256。
- 权限判断：`git pull`、`squeue/sacct` 和产物只读校验均为轻量管理操作；不在登录节点执行候选生成、模型加载或其他批量处理。
- 计划顺序：持久会话的第一条远端命令为 `cd vocab_align && git pull`；成功后进入 `C2C` 查询 Job 212，并复用该连接直到本轮监控完成。若 pull 网络异常，按规范执行 `bash net.sh` 后重试。

## 2026-09-01 21:43 +08:00

- 连接用途：使用 `scp` 将 Job 212 生成的 ANN candidate JSON 从 Guqq 复制到本地任务专属忽略目录，供本地独立 schema/哈希检查与后续可复现记录。
- 权限判断：从服务器读取并复制当前任务结果属于许可操作；文件 134,332,695 bytes，超过 10 MiB，保持在 `.gitignore` 覆盖的 `C2C/local/transport/artifacts/`，不得通过普通 Git 提交。
- 已验证远端结果：Job 212 `COMPLETED`、ExitCode `0:0`、运行 52 秒；source/target ordinary coverage 为 151,655/131,069 且两侧全部覆盖；JSON SHA-256 为 `260f98048a3d50adb667a6c0b9d23126c7d0e533fd56791c6059001104e91652`，无 `.partial` 残留。stderr 仅有 tokenizer-only 环境缺少深度学习后端的提示。
- 计划顺序：本次连接只执行单文件 `scp`，不修改服务器文件或源码；传输后在本地复核字节数与 SHA-256。

## 2026-09-01 22:07 +08:00

- 连接用途 A（scp）：将本地 canonical preview JSONL 上传到 Guqq 任务专属忽略路径 `C2C/local/transport/inputs/preview_texts.jsonl`。
- 输入校验：文件 1,094 bytes、16 条 JSONL，SHA-256 `05CA0628E57EADDA84F4D16968083D5BF12D8A9012B2A1081D9E372047207A3A`；仅覆盖当前任务同名 canonical 输入，不触及其他数据或结果。
- 连接用途 B（持久 SSH）：同步 full-support preview 作业提交 `ff5af46`，校验上传输入哈希与既有 ANN JSON，查询 Slurm 后提交 `build_full_support_preview.sbatch` 并监控。
- 权限判断：scp 输入、`git pull`、哈希/Slurm 查询和 `sbatch` 属于许可操作；2.3M-edge graph、Sinkhorn 与 audit 只在 64G/8h Slurm 作业内执行，不在登录节点运行。
- 计划顺序：先完成单文件 scp；随后新持久 SSH 会话第一条命令为 `cd vocab_align && git pull`，同步成功后才校验输入并提交作业。网络异常时执行 `bash net.sh` 后重试 pull。

## 2026-09-01 22:26 +08:00

- 连接用途：同步 special-support 修复提交 `0409679`，复核 canonical JSONL/ANN JSON 哈希，通过 Slurm 重跑 full-support preview 并监控 convergence/audit 结果。
- 权限判断：登录节点只执行 `git pull`、SHA-256、`squeue/scontrol`、`sbatch` 与日志/产物只读检查；2.3M-edge graph、Sinkhorn 和 audit 仍全部在 64G/8h Slurm 作业中运行。
- 失败背景：Job 214 已安全失败且仅留下 48-byte building checkpoint；不把该 checkpoint 当有效 artifact。新作业使用独立 Slurm job ID，builder 按既有 checkpoint 语义从记录输入重新构建，不在服务器改源码。
- 计划顺序：持久会话第一条命令为 `cd vocab_align && git pull`；网络异常时执行 `bash net.sh` 后重试。同步/哈希成功后提交作业，保留并检查新的日志与 checkpoint。
- 实际结果：首次 `git pull` 等待约 90 秒无响应；中断后 `bash net.sh` 成功刷新网络登录，但第二次 pull 等待约 60 秒仍无响应。已中断并退出，未执行哈希查询、`sbatch` 或任何计算；服务器仍未获得 `0409679`。

## 2026-09-01 23:06 +08:00

- 连接用途：按用户确认同步 manifest-bound corpus 验收提交 `4876adb`（包含待重跑所需的 special-support 修复），复核既有 canonical preview/ANN 输入，并通过 Slurm 重跑 full-support preview。
- 权限判断：登录节点只执行 `git pull`、输入/环境轻量校验、`sinfo/squeue/sacct`、`sbatch` 与结果只读检查；候选图加载、Sinkhorn、artifact 构建和审计全部在 64G/8h Slurm 作业中运行。
- 作业边界：不复用 Job 214 的 48-byte building checkpoint 作为有效产物；新作业允许按记录输入重新构建，并使用新的 Slurm job ID/log。所有生成物继续位于 `C2C/local/transport/` 忽略目录。
- 计划顺序：新 SSH 会话的第一条远端命令为 `cd vocab_align && git pull`；如发生网络问题，才执行 `bash net.sh` 后重试。同步成功后再核对输入并提交/监控作业，不在服务器修改 Git 源码。
- 实际结果：首条 pull 等待 60 秒无输出；`bash net.sh` 成功后重试约 90 秒完成并 fast-forward 到 `5c37b39`。preview/ANN SHA-256 与既有记录一致，提交 Job 215；作业在 compute 节点运行 46:03 后 `FAILED`、ExitCode `1:0`。失败为 sparse Sinkhorn 10,000 次未收敛（row residual `0.2651238722`、column residual `7.215e-12`），checkpoint 仍为 71-byte `building`，无有效 artifact/audit。已退出服务器，源码修复转回本地。

## 2026-09-02 00:10 +08:00

- 连接用途：同步 marginal-capacity feasibility support 验收提交 `8f26fa8`，复核相同 preview/ANN 输入，通过 Slurm 再次构建 full-support preview 并监控收敛与 artifact audit。
- 权限判断：登录节点只执行首条 `git pull`、哈希/环境检查、`sinfo/squeue/scontrol`、`sbatch` 与结果只读验收；2.3M+ candidate graph、feasibility augmentation、Sinkhorn 和 artifact/audit 均只在 64G/8h Slurm 作业内执行。
- 计划顺序：新 SSH 会话第一条远端命令为 `cd vocab_align && git pull`；网络异常时执行 `bash net.sh` 后重试。同步到 `8f26fa8` 且输入哈希一致后提交新 job，不复用 Job 215 的 `building` checkpoint 作为成功产物，不在服务器修改源码。
- 实际结果：首条 pull 成功同步到 `1cc9163`，输入哈希一致，提交 Job 220。作业运行 40:33 后 `FAILED`、ExitCode `1:0`；10,000 次后的 row residual 为 `0.0005577679`、column residual 为 `2.199e-14`，较 Job 215 显著改善但仍未达到 `1e-9`。checkpoint 保持 `building` 且无有效 artifact/audit；已退出服务器并转回本地求解器加速，不提高远端 `max_iter` 掩盖问题。

## 2026-09-02 01:10 +08:00

- 连接用途：同步 sparse OT dual acceleration 验收提交 `cbf6f44`，使用完全相同的 preview/ANN 输入经 Slurm 重跑 full-support artifact，验收加速方法、迭代预算、严格 residual、artifact 与 audit。
- 权限判断：登录节点只执行首条 `git pull`、哈希/环境检查、Slurm 提交/监控和结果只读校验；候选构建、标准 scaling、L-BFGS 对偶加速及 artifact/audit 均只在 64G/8h compute allocation 内执行。
- 计划顺序：新会话第一条远端命令为 `cd vocab_align && git pull`；异常时运行 `bash net.sh` 后重试。同步/哈希确认后提交新 job，不复用 Job 220 的 `building` checkpoint，不在服务器修改源码或放宽 `1e-9`。
- 实际结果：首条 pull 成功同步到 `bade800`；发现 venv 缺少 SciPy，按 `environment.yml` 安装缓存 wheel `scipy==1.15.3` 并验证 NumPy 2.2.6/SciPy 1.15.3，输入哈希一致后提交 Job 226。作业运行 24:54 后被 SIGKILL，ExitCode `137:0`；stderr 仅记录 shell `Killed`，无 Python traceback，checkpoint 仍为 `building`。节点当时 swap 2G 已满但作业缺少 MaxRSS 遥测；已退出并转回本地限制 optimizer history、增加 GNU time 记录。

## 2026-09-02 01:46 +08:00

- 连接用途：同步 memory-bounded dual telemetry 验收提交 `0cf07e6`，在相同 64G/8h allocation 与相同输入上重跑 full-support preview，读取 GNU time 的 MaxRSS/elapsed/exit status 并验收 residual/artifact/audit。
- 权限判断：登录节点只执行首条 `git pull`、版本/哈希检查、Slurm 提交/监控及结果只读检查；候选构建、`maxcor=3` 对偶加速和 OT 审计全部在 compute allocation 内执行。
- 计划顺序：新会话第一条命令为 `cd vocab_align && git pull`；同步/哈希确认后提交新 job。若仍 SIGKILL，必须依据 GNU time MaxRSS 决定代码或资源调整；不复用 building checkpoint、不修改服务器源码、不放宽容差。
- 实际结果：首条 pull 60 秒无输出；中断后 `bash net.sh` 成功，但第二次 pull 等待 90 秒仍无输出。已中断并退出，未执行哈希检查或提交作业；服务器尚未同步 `0cf07e6/1830353`，MaxRSS 重跑保持 pending。

## 2026-09-02 02:08 +08:00

- 连接用途：按用户确认 Git pull 已恢复，同步至 OpenHermes 物化验收提交 `a4bd39b`（同时包含尚未同步的 memory-bounded telemetry），复核既有 preview/ANN 输入，并通过 Slurm 重跑 64G full-support preview。
- 权限判断：登录节点只执行首条 `git pull`、版本/输入/环境轻量检查、`sinfo/squeue/sacct`、`sbatch` 与结果只读验收；2.3M+ graph、Sinkhorn、artifact 构建和 audit 全部在 compute allocation 内运行。
- 计划顺序：新 SSH 会话第一条远端命令严格为 `cd vocab_align && git pull`；若出现网络连接问题，才执行 `bash net.sh` 后重试。同步成功后确认提交、输入哈希和无同名运行作业，再提交新的 preview job；不复用失败作业的 building checkpoint，不在服务器修改源码。
- 实际结果：pull 成功 fast-forward 到 `4e947f9`；canonical preview/ANN SHA-256 分别为 `05ca0628…207a3a`、`260f9804…e91652`，NumPy/SciPy 为 2.2.6/1.15.3，无同名运行作业。已提交 Job 229；首次查询为 `RUNNING`（node221），Slurm accounting storage disabled，需从作业 stderr 的 GNU time 获取 MaxRSS。

## 2026-09-02 02:12 +08:00

- 连接用途：建立持久 SSH 会话监控 Job 229，等待终态后读取 stdout/stderr、GNU time MaxRSS、checkpoint、artifact 与 audit；避免为每次轮询重复建立连接。
- 权限判断：登录节点仅执行首条 `git pull`、`squeue` 轮询和终态结果只读检查，不直接运行计算或修改服务器源码/产物。
- 计划顺序：持久会话第一条命令为 `cd vocab_align && git pull`；随后进入 `C2C`，每次间隔约一分钟查询 Job 229。完成后验证 artifact/audit；失败则保留原始日志与 building checkpoint 证据。
- 实际结果：Job 229 在 node221 运行 `00:38:24` 后 `FAILED`、ExitCode `1:0`；不是 SIGKILL。GNU time 记录 MaxRSS `1,846,656 KiB`、swaps `0`，证明 `maxcor=3` 已将内存远降至 64G 配额内。失败为总预算 10,000（standard 8,999 + acceleration evaluations 1,001）后 row residual `4.071621136e-4`、column residual `4.996e-14`，未达到 `1e-9`。checkpoint 保持 `building/restart-from-recorded-inputs`，artifact/audit 均不存在；已退出持久会话，下一步本地诊断收敛算法，不放宽容差。

## 2026-09-02 03:01 +08:00

- 连接用途：同步 marginal-scaled dual 验收提交 `f5ba846`，复核相同 preview/ANN 输入与无同名运行作业，通过 Slurm 在相同 64G/8h、`1e-9` 配置下重跑 full-support preview。
- 权限判断：远端连接中先 `git pull`，随后仅做哈希/队列轻量检查和 `sbatch`；候选加载、scaled L-BFGS、标准缩放、artifact/audit 全部在 compute allocation 内运行，不修改服务器源码。
- 计划顺序：连接后的第一项操作为 `cd vocab_align && git pull`；同步到 `f5ba846` 且输入哈希一致后提交新 job。失败的 building checkpoint 仅用于 restart provenance，不作为 resume artifact；不改变 epsilon、tolerance、max-iter 或资源以保证与 Job 229 可比。
- 实际结果：本地 PowerShell 在启动 SSH 时提前展开了 Bash `$(squeue ...)`，本地报 `squeue` 不存在，远端连接随即关闭；未完成提交/哈希检查，未提交 Slurm 作业。下一次不再把跨 shell substitutions 放入 `ssh` 命令字符串。

## 2026-09-02 03:02 +08:00

- 连接用途：使用持久 SSH 会话逐条同步并提交 scaled-dual preview，消除 PowerShell/Bash substitution 歧义；后续在同一会话监控新 job。
- 权限判断：第一条纯命令为 `cd vocab_align && git pull`；随后仅执行字面提交/哈希比较、`squeue`、`sbatch` 和状态/结果只读检查，计算仍只在 Slurm。
- 计划顺序：不使用本地可展开的 `$()`/反引号；pull 成功后逐条 `git rev-parse`、`sha256sum`、`squeue`，确认无误才 `sbatch`。
- 实际结果：持久会话首条 pull 等待 60 秒无输出；中断后 `bash net.sh` 成功刷新网络，retry pull 等待 90 秒仍无输出。已中断并退出，未执行版本/哈希检查，未提交新作业；服务器仍停留在旧 scaled-dual 前提交。

## 2026-09-02 03:07 +08:00

- 连接用途：在用户确认 pull 已恢复的当前窗口做第三次独立 scaled-dual 同步尝试；成功则复核输入并提交/持久监控 preview，失败则停止本轮网络重试并转回本地可推进事项。
- 权限判断与顺序：新会话第一条命令仍为 `cd vocab_align && git pull`；成功前不执行其他远端操作。成功后仅做轻量哈希/队列检查与 Slurm 提交/监控，计算不在登录节点运行。
- 实际结果：第三次 pull 约 90 秒后成功同步到 `ac689a4`；preview/ANN 哈希再次匹配，无同名作业，提交 Job 230。作业至少运行至 19:31 时保持 `RUNNING`，随后本地工具会话因新用户消息关闭；Slurm 作业未被中断，终态尚待只读验收。

## 2026-09-02 07:49 +08:00

- 连接用途：恢复检查 scaled-dual Job 230 的终态，读取 Slurm state/exit、stderr GNU time、checkpoint、artifact 和 audit；若成功则为正式 OpenHermes 物化准备环境/作业，若失败则保留证据回到本地修复。
- 权限判断：新连接首条命令为 `cd vocab_align && git pull`；之后仅做 Slurm 与结果文件只读检查。本次不在登录节点运行构建、评测或批量处理，也不修改服务器源码/结果。
- 计划顺序：pull 后进入 `C2C`，依次检查 Job 230、日志、checkpoint、artifact/audit；只在结果已证明成功时推进下一阶段。
- 实际结果：首次 SSH 在 shell 前被远端关闭；同用途 retry 建立会话后，首条 pull 约 90 秒以 GnuTLS `-110` 失败。`bash net.sh` 成功，但 retry pull 约 136 秒以 GitHub 443 timeout 失败。按权限约束未查询 Job 230，已退出；终态仍 pending，本地转入可独立实施的阶段 4 approximation 单元。

## 2026-09-02 08:12 +08:00

- 连接用途：按用户确认的 Git pull 恢复窗口，同步阶段 4 approximation/ORF 验收提交 `06e9c7c`，并恢复只读验收 scaled-dual Job 230 的终态、日志、checkpoint、artifact 与 audit。
- 权限判断：新连接首条远端操作必须为 `cd vocab_align && git pull`；之后仅执行 `git rev-parse`、`squeue/sacct` 与结果文件只读检查。不同步成功不继续查询；不在登录节点运行计算，不修改服务器源码或已有实验结果。
- 计划顺序：pull 成功后确认提交，再检查 Job 230 state/exit、GNU time、严格 residual、checkpoint 与 artifact/audit。网络异常时运行 `bash net.sh` 后重试；Job 230 验收完成前不提交正式 500k 后续作业。
- 实际结果：首次 pull 等待 90 秒无输出后中断；`bash net.sh` 成功，retry pull 同步到 `af3aadf`。Job 230 已从活动队列清除且 accounting disabled；stderr 证明作业运行 40:50.55、Exit 1、MaxRSS 1,847,076 KiB、0 swap。scaled L-BFGS 仅执行 27 次 evaluation，随后标准 scaling 至总预算 10,000，最终 row/column residual 为 `4.6615468745e-4`/`6.0559911057e-14`，未达到 `1e-9`。checkpoint 仍为 `building/restart-from-recorded-inputs`，正式 artifact/audit 均不存在；已退出服务器，不推进正式 500k 作业。

## 2026-09-02 08:29 +08:00

- 连接用途：验证临时分支提交 `cfa1a87`（明确 `[UNACCEPTED]`）的 stable incremental dual 与有界重启；在与 Job 230 相同输入、64G/8h、epsilon 0.5、`1e-9`/10,000 下通过 Slurm 重跑 full-support preview。
- 权限判断：建立会话后第一条远端操作仍为仓库内 `git pull`；随后仅用 `git pull origin validation/job230-dual-increment` 获取临时提交、复核版本/输入/队列并 `sbatch`。2.3M-edge 构建和 Sinkhorn 全部在 compute allocation，登录节点不运行计算、不编辑源码。
- 计划顺序：同步到 `cfa1a87` 后使用带新 job 后缀的 artifact/audit 路径，避免覆盖 Job 230 的 building checkpoint/partial；提交后持久监控 termination provenance、严格 residual、MaxRSS 和原子产物。未通过前不合并 main、不提交正式 500k 作业。
- 实际结果：首条普通 pull 成功获取远程临时分支，随后 `git pull origin validation/job230-dual-increment` fast-forward 到 `7ca1003`；输入哈希与 Job 230 一致，提交 Job 232。作业 39:06.15 后 Exit 1，MaxRSS 1,847,136 KiB、0 swap；21 次 acceleration 共严格消耗 1,000 evaluations，前 20 次 termination 均为 `RELATIVE REDUCTION OF F <= FACTR*EPSMCH`，最终 row/column residual `1.6915612104e-3`/`2.6332792027e-14`。独立 checkpoint 为 `building/fresh`，artifact/audit 不存在；已退出，临时提交保持未验收。

## 2026-09-02 09:16 +08:00

- 连接用途：验证临时分支第二个 `[UNACCEPTED]` 提交 `463c3b9`，确认 `ftol=0` 是否消除 Job 232 的 `FACTR*EPSMCH` early termination，并在同一真实图达到严格 `1e-9`。
- 权限判断：第一条远端操作为仓库内普通 `git pull`，随后仅 `git pull origin validation/job230-dual-increment`、版本/输入/队列轻量复核和 `sbatch`；所有 2.3M-edge 计算仍只在 64G/8h Slurm allocation 内。
- 计划顺序：使用新的 `dual_ftol_validation` artifact/audit 路径，不覆盖 Job 230/232 现场；验收 termination 不再由 FACTR、累计 evaluation 不越界、两侧 residual、MaxRSS、checkpoint 与原子产物。失败仍不合并 main，并因同一任务连续第三次失败阈值复核 `docs/agents/lessons.md`。
- 实际结果：首条 pull 约 90 秒后以 GnuTLS `-110` 失败；`bash net.sh` 成功，但 retry pull 约 135 秒后 GitHub 443 timeout。未同步临时提交、未执行版本/输入/队列查询，也未提交新 Slurm 作业；已退出。

## 2026-09-02 09:22 +08:00

- 连接用途：在用户确认 pull 可用的窗口做一次独立同步重试，目标仍为临时 `ftol=0` 提交 `463c3b9`/审计 `2e78433` 与 `dual_ftol_validation` Slurm 作业；若仍失败则停止本轮远端重试并保留 pending。
- 权限判断与顺序：新会话第一条操作仍为 `cd vocab_align && git pull`；成功前不做任何其他远端操作。成功后才 pull 临时分支、复核并 `sbatch`，所有计算只在 Slurm。
- 实际结果：首条 pull 成功获取临时分支更新，随后 fast-forward 到 `7482ef5`；输入哈希一致、无同名作业，提交独立 `dual_ftol_validation` Job 233。作业持续 RUNNING 至至少 14:49，随后 SSH 被远端 reset；Slurm 未被取消，终态待新审计连接恢复检查。

## 2026-09-02 09:43 +08:00

- 连接用途：恢复持久监控 `ftol=0` 临时验证 Job 233；等待终态后读取 termination provenance、严格 residual、GNU time、checkpoint 与独立 artifact/audit。
- 权限判断与顺序：新连接第一条操作为 `cd vocab_align && git pull`；随后只运行 `squeue` 和结果只读检查，不提交新作业、不运行登录节点计算、不修改服务器源码/产物。
- 实际结果：首次连接在 shell 前关闭；同用途 retry 建立会话后首条 pull 成功，并同步临时分支至 `4dbb598`。Job 233 持续 RUNNING 至至少 25:40；随后对话中断，恢复轮询时 SSH 已被远端 reset。Slurm 未被取消，终态仍待检查。

## 2026-09-02 10:02 +08:00

- 连接用途：再次恢复 Job 233 终态只读验收，读取 Slurm state/日志、termination provenance、严格 residual、GNU time、checkpoint 与独立 artifact/audit。
- 权限判断与顺序：新连接第一条操作为 `cd vocab_align && git pull`；之后只运行 `squeue` 及结果文件只读命令，不提交作业、不直接运行计算、不修改服务器源码或实验产物。
- 实际结果：首条 pull 与临时分支同步成功至 `bb84223`。Job 233 已终止并与 Job 232 得到相同严格失败：39:06.84、Exit 1、MaxRSS 1,847,040 KiB、0 swap；21 attempts/1,000 evaluations，前 20 次即使 `ftol=0` 仍报告 `RELATIVE REDUCTION OF F <= FACTR*EPSMCH`，最终 row/column residual `1.6915612104e-3`/`2.6332792027e-14`。checkpoint `building/fresh`，artifact/audit 不存在；已退出。

## 2026-09-02 10:36 +08:00

- 连接用途：验证第三个临时 `[UNACCEPTED]` 提交 `f62c540` 的 residual-driven scaled Newton-CG，在与 Jobs 230/232/233 相同的 2.3M-edge 输入、64G/8h、epsilon 0.5、`1e-9`/10,000 配置下经 Slurm 重跑 full-support preview。
- 权限判断与顺序：新会话第一条远端操作必须是 `cd vocab_align && git pull`；之后才拉取 `validation/job230-dual-increment`、复核 HEAD/输入/队列并 `sbatch`。真实图构建和求解全部在 compute allocation；登录节点只做轻量管理，不编辑受 Git 管理源码。
- 验收边界：使用独立 `newton_cg_validation` checkpoint/artifact/audit 路径，不覆盖前三次失败现场；检查严格两侧 residual、Newton/CG provenance、1,000 acceleration budget、GNU time/MaxRSS 和原子产物。若网络异常先运行 `bash net.sh` 再重试 pull；真实图通过前不进入 main。
- 实际结果：首条 pull 成功并同步到 `9c2ad04`，输入哈希保持 `05ca0628…207a3a`/`260f9804…e91652`，SciPy 1.15.3；提交 Job 234。作业 37:11.54 后 Exit 1，MaxRSS 1,847,260 KiB、0 swap；12 次 Newton 尝试共耗尽 1,000 evaluations，虽有少量极小步被接受，最终 row/column residual 仍为 `1.6915304665e-3`/`6.2669379185e-14`。checkpoint 为 `building/fresh`，artifact/audit 不存在；stderr/checkpoint SHA-256 分别为 `6f3807b4…785d65`/`a3a9acb8…5bdc7f`。已退出，临时提交保持未验收。

## 2026-09-02 12:12 +08:00

- 连接用途：验证临时 `[UNACCEPTED]` 提交 `5207cc9` 对新 full-vocabulary `2e-3` 需求的完整落盘链；以与 Job 234 相同输入、64G/8h、epsilon 0.5、max_iter 10,000 经 Slurm 重跑。
- 权限判断与顺序：新会话第一条远端操作必须是 `cd vocab_align && git pull`；之后才同步 `validation/job230-dual-increment`、复核 HEAD/输入/队列并 `sbatch`。所有 2.3M-edge 构建/求解在 compute allocation，登录节点只做轻量管理且不编辑源码。
- 验收边界：使用独立 `tolerance_2e3_validation` checkpoint/artifact/audit 路径，不覆盖 Job 234；要求 Exit 0、两侧最大 L1 residual `<=2e-3`、metadata `build_config.tolerance=0.002`、列随机性保持 dtype 精度、artifact save/load 与独立 JSON/Markdown audit 完整、checkpoint complete、MaxRSS/哈希可追溯。网络异常先 `bash net.sh`；通过前不进入 main。
- 实际结果：首条 pull 与临时分支同步成功至 `9ac437e`，输入哈希和 SciPy 1.15.3 一致，提交 Job 235。作业约 22:36 后在 partial artifact 已写出时被 signal 9 终止；GNU time 记录 MaxRSS `255,870,840 KiB`、0 swap，暴露 `audit_transport_artifact` 全矩阵 dense 展开。留下 34 MiB partial（SHA-256 `833a320a…e5e3e0`）和 `building/fresh` checkpoint（`a3a9acb8…5bdc7f`），无最终 artifact/audit；stderr SHA-256 `3e67cc70…438bf6`。已退出，该运行未验收。

## 2026-09-02 13:34 +08:00

- 连接用途：验证临时 `[UNACCEPTED]` sparse audit 提交 `76ee480`；复用 Job 235 相同输入、64G/8h、epsilon 0.5、tolerance `2e-3`、max_iter 10,000，经 Slurm 完成 full-vocabulary artifact 的保存/加载/独立审计。
- 权限判断与顺序：新会话第一条远端操作为 `cd vocab_align && git pull`；之后才同步临时分支、复核 HEAD/输入/队列并 `sbatch`。构建和 audit 都在 compute allocation；登录节点只做轻量管理/只读终态检查，不加载 partial、不编辑源码。
- 验收边界：使用全新 `sparse_audit_validation` artifact/audit/checkpoint 路径，不覆盖 Job 235 partial；要求 Exit 0、residual `<=2e-3`、metadata tolerance、完整 JSON/Markdown audit、checkpoint complete、目标统计有限、MaxRSS 显著低于 64G 且哈希可追溯。通过前不进入 main。

## 2026-09-02 16:21 +08:00

- 连接用途：此前用于监控 Job 236 的持久 SSH 会话被远端 reset；建立新连接恢复只读终态验收，查询 Slurm state/exit、GNU time/MaxRSS、独立 artifact/audit/checkpoint 和哈希。
- 权限判断与顺序：新连接的第一条远端操作仍为 `cd vocab_align && git pull`；成功后仅执行 `squeue`/`sacct`、日志和结果文件只读检查，不提交新计算、不在登录节点加载全量产物或修改服务器源码/实验结果。
- 验收边界：继续使用 Job 236 的 `sparse_audit_validation` 独立路径，核验 Exit 0、两侧 marginal residual、严格列归一化、metadata tolerance、有限目标统计、checkpoint complete、MaxRSS 与可追溯哈希；若失败则保留现场并回到本地修复。

## 2026-09-03 01:20 +08:00

- 连接用途：昨日恢复会话在多次 GitHub TLS/443 失败后被远端 reset；今天重新连接 Guqq，恢复 Job 236 的终态只读验收并收集 artifact/audit/checkpoint、调度资源和哈希证据。
- 权限判断与顺序：新连接第一条远端操作必须为 `cd vocab_align && git pull`；成功后才执行 `squeue`/`sacct`、日志及结果文件只读检查。网络异常时按规范运行 `bash net.sh` 后重试，不在登录节点执行构建、审计或全量产物加载，也不修改服务器文件。
- 验收边界：核验 Job 236 Exit 0、`2e-3` 两侧 marginal 要求、严格列归一化、metadata、有限目标统计、checkpoint complete、MaxRSS 和哈希；通过后回到本地补全测试/状态/进度记录并整理验收提交。
- 实际结果：首条 `git pull` 成功；Slurm accounting 已禁用，但 stderr 的 GNU time 给出 20:23.32、Exit 0、MaxRSS `2,113,980 KiB`、0 swap。最终 artifact 为 33,486,321 bytes，shape `131069×151669`、nnz `2,620,553`；audit `valid=true`，row/column/transported residual `1.9975102855e-3`/`8.5268617950e-14`/`1.9975102855e-3`，最大列和误差 `1.1883827256e-12`，目标统计有限且无危险 special mapping。metadata tolerance 为 `0.002`，checkpoint `complete/fresh`，JSON/Markdown 齐全、同名 partial 不存在。artifact/checkpoint/audit JSON/audit Markdown/stderr SHA-256 分别为 `b1ada569…18aca2`、`88c9f4ff…c23501`、`c8467f09…99d7d1`、`56ef61f7…fb739`、`3f9c2ee3…ec2784`；已退出服务器，远程验收通过。

## 2026-09-03 01:37 +08:00

- 连接用途：同步 main 验收提交 `f433000`，复核 Python venv/datasets 4.0.0、正式输出路径和队列后，通过 Slurm 物化锁定 `teknium/OpenHermes-2.5@05c3557e57b6dd1d0e0cb8369ba53b43e15fd10b` 的前 500,000 个 source rows，并在终态收集 corpus/manifest、资源和哈希证据。
- 权限判断与顺序：新连接第一条远端操作为仓库内 `git pull`；服务器本地 main 含先前临时验证历史，若与 squash 后的远端 main 分叉，仅使用 `git pull --no-rebase origin main` 完成 Git 管理的同步，不手工编辑/打补丁/覆盖源码。随后只做轻量环境/路径/队列检查和 `sbatch`；下载后的 500k 遍历、canonicalization、去重、划分及校验全部在 32G/4h allocation 内。
- 验收边界：默认独立输出 `local/transport/corpora/openhermes-500k.jsonl` 与 `local/transport/manifests/openhermes-500k.json` 必须预先不存在；要求 Exit 0、selected rows 500,000、计数与 provenance 自洽、records hash 与 manifest 绑定、无 partial、GNU time/MaxRSS 和产物哈希可追溯。失败则保留日志/现场，不进入正式 T 构建。
- 实际结果：首条普通 pull 遇到 GitHub TLS 中断；`net.sh` 后两次 main pull 分别 443 timeout，第三次获取远端后因 squash 与服务器临时验证历史在文档文件冲突。未手工解决；执行 `git merge --abort` 恢复工作树，再用 `git pull --ff-only origin validation/job230-dual-increment` 快进至 `5a0368f`，并确认 `C2C` 与 `origin/main` 无差异。环境预检发现 `.venv` 缺少 `datasets`，因此未提交作业并退出；下一步先按 `docs/agents/env.md` 补齐锁定依赖。

## 2026-09-03 01:43 +08:00

- 连接用途：同步已验证代码分支，按环境记录在既有 Python venv 中安装 `datasets==4.0.0` 并记录完整版本；环境门禁通过后复核正式输出路径/队列并提交 OpenHermes 500k 物化 Slurm 作业。
- 权限判断与顺序：第一条远端操作使用 `git pull --ff-only origin validation/job230-dual-increment`，避免服务器旧 main 与 squash main 冲突且只执行快进同步；随后仅执行允许的 venv 依赖安装与短时 import/version/help 检查。500k 数据遍历、规范化、去重、划分和校验全部通过 Slurm，不在登录节点运行。
- 验收边界：安装必须使用 wheel/普通包下载且精确为 datasets 4.0.0；若触发编译或异常资源负载则停止。作业沿用 01:37 条目的独立路径、计数/provenance/原子性/资源/哈希标准，提交前不得存在目标或 partial 文件。
- 实际结果：前三次 pull 分别为 TLS/443 失败，第四次 `net.sh` 后快进 pull 成功且已是 `5a0368f`。datasets 4.0.0 及 wheel 依赖安装/版本/help 门禁通过；正式输出与 partial 均不存在、队列为空，提交 Job 239。作业 2:00.06 后 Exit 0，MaxRSS `7,128,336 KiB`、0 swap；生成 500,000 行、909,629,231-byte records 和 43,500,816-byte manifest。manifest 为锁定 dataset/revision/raw train、pinned-source-prefix-v1、seed 42、adapter filtering not-applied，unique/duplicate/train/dev 为 500,000/0/495,000/5,000，split 唯一且无交叉，raw SHA 与 records 一致，边界 JSON 可解析且无 partial。records/manifest/stderr SHA-256 为 `539f2d30…5d485a`/`a50c0dca…7c60fa`/`c4a91c0d…728e65`；已退出服务器，正式物化验收通过。

## 2026-09-03 02:06 +08:00

- 连接用途：验证 main-based 临时提交 `5787a71` 的正式 T Slurm 入口；为避免 Guqq 旧验证历史与 squash main 分叉，服务器将通过从其当前 `5a0368f` 派生的 `validation/guqq-formal-transport-500k` 兼容分支做纯快进 pull，该分支的全部 C2C 文件必须与 `5787a71` 相同。
- 权限判断与顺序：新连接第一条远端操作为 `git pull --ff-only origin validation/guqq-formal-transport-500k`；随后以 `git diff --exit-code 5787a71 -- C2C` 等价检查、环境/输入哈希、正式输出路径和队列轻量复核。500k manifest 加载、tokenization、候选统计、Sinkhorn 和 audit 全部由 64G/24h Slurm 作业执行，不在登录节点直接运行。
- 验收边界：使用默认正式 artifact/audit/checkpoint 路径，提交时显式把 main 验收代码版本 `f433000b...` 写入 metadata；要求 Exit 0、manifest/records/ANN provenance、两侧 residual `<=2e-3`、严格列和/非负/特殊映射/有限目标、checkpoint complete、无 partial、MaxRSS/耗时/哈希完整。远程通过前不合并 main-based 临时分支。
- 实际结果：首条 ff-only pull 成功从 `5a0368f` 快进到兼容提交 `bb4bab6`；环境版本、Bash、records/manifest/ANN 哈希、空输出路径和队列通过。以 code version `f433000fa8514296dd5849c619ecd99a4e449bed` 提交 Job 240；作业 51:57.26 后 Exit 0，MaxRSS `8,036,128 KiB`、0 swap。正式 artifact 39,951,267 bytes、shape `131069×151669`、nnz/candidate edges `2,733,518`；metadata 绑定 transport_train 的 495,000 samples/997,233 canonical messages、锁定 dataset revision、records/manifest/ANN SHA、seed 42 和 `0.5/0.002/10000/1e-8`。audit `valid=true`，row/column/transported residual `1.9655245213e-3`/`1.0560509249e-13`/`1.9655245213e-3`，最大列和误差 `1.2299050667e-12`，目标统计有限、非负且无危险 special mapping；checkpoint `complete/fresh`，无 partial。artifact/checkpoint/audit JSON/audit Markdown/stderr SHA-256 为 `1495d522…0aba97`/`79c4ad38…5caf84`/`53b6a464…1948de`/`13bbda9b…c2457e`/`4deb7afd…517ad7`；已退出，正式 T 验收通过。

## 2026-09-03 03:22 +08:00

- 连接用途：为真实模型最短序列 smoke 做轻量资源预检；同步兼容验证分支后，仅查询 `sinfo`/GPU 类型与显存、现有 Hugging Face 模型缓存、Python venv 的 torch/accelerate/transformers 版本和正式 Job 240 artifact 是否存在，不加载模型、不运行推理。
- 权限判断与顺序：新连接第一条远端操作为 `git pull --ff-only origin validation/guqq-formal-transport-500k`；随后仅做允许的集群、环境、缓存和文件元数据检查。若网络异常先运行 `bash net.sh` 再重试；模型推理和 CUDA 验证留给后续 Slurm 作业。
- 验收边界：据实际集群资源冻结 smoke Slurm 的 GPU/CPU/内存/时限与预检门禁；不修改服务器源码或现有环境/产物，本次不提交计算作业。
- 实际结果：首项 ff-only pull 成功且兼容分支已是最新；随后 `sinfo -o` 的含管道格式被远端 shell 拆分，未取得资源表，后续环境/缓存检查因 `&&` 短路而未执行。会话已结束，未加载模型、未修改环境或产物。

## 2026-09-03 03:25 +08:00

- 连接用途：重试真实模型 smoke 的轻量资源预检；修正 `sinfo` 为不含 shell 管道符的 `--Format` 写法，并读取 venv 包元数据、正式 artifact 大小和两侧模型缓存目录。
- 权限判断与顺序：新连接第一条远端操作仍为 `git pull --ff-only origin validation/guqq-formal-transport-500k`；其余仅为 `sinfo`、`pip show`、`stat`、`ls` 等轻量只读检查，不执行 CUDA 初始化、模型加载或推理。
- 验收边界：取得足以冻结 Slurm 资源和依赖安装计划的证据后退出；不存在的包或缓存只记录，不在本连接安装/下载。
- 实际结果：首项 pull 成功；`sinfo` 确认 `compute` 为默认且可用、共 1 node、每节点 `gpu:1`，同时报告该版本不接受 `Timelimit` 字段。venv 路径误写成仓库根 `.venv`，实际为 `C2C/.venv`，因此后续 `&&` 短路；未修改任何远端状态。

## 2026-09-03 03:27 +08:00

- 连接用途：以已确认的 `C2C/.venv` 路径完成依赖/artifact/cache 只读预检，并用 `scontrol show node` 获取 GPU 型号/节点内存等 Slurm 资源细节。
- 权限判断与顺序：第一条远端操作继续为兼容分支 ff-only pull；之后仅运行 `scontrol`、`pip show`、`stat` 和缓存目录 `ls`，不存在项允许返回非零但不中断其他检查。
- 验收边界：本次只收集事实，不安装包、不下载或加载模型；根据结果在本地完成 smoke 入口后，另行登记环境安装及 Slurm 验证连接。
- 实际结果：首项 pull 成功；`node221` 为 48 CPU、257,787 MiB 总内存、当前约 161,250 MiB free、`compute` 默认分区、`Gres=gpu:1`，Slurm 未暴露 GPU 型号。`C2C/.venv` 有 transformers 4.52.4、无 torch/accelerate；正式 artifact 为 39,951,267 bytes，两侧模型 cache 目录均存在。全程只读，未初始化 CUDA 或加载权重。

## 2026-09-03 03:46 +08:00

- 连接用途：同步 `validation/guqq-real-model-stt-smoke` 兼容提交 `d559a6f`，确认其 C2C tree 与 main-based `[UNACCEPTED]` 提交 `036df80` 一致；检查磁盘/cache 后，在既有 Python venv 中安装 wheel-only `torch==2.6.0 accelerate==1.9.0` 并复核版本。
- 权限判断与顺序：第一条远端操作为 `git pull --ff-only origin validation/guqq-real-model-stt-smoke`；随后仅做 tree/disk/cache 检查和许可的环境依赖安装，不在登录节点初始化 CUDA、加载模型或推理。网络问题先运行 `bash net.sh` 再重试；若 pip 尝试源码编译或空间不足则停止。
- 验收边界：要求 C2C tree 等价、依赖精确、torch CUDA build 元数据可读、两侧 cache 大小和 snapshot 状态清晰；本连接不提交 smoke 作业，完成后退出并记录事实。
- 实际结果：第一项 ff-only pull 因 GitHub `GnuTLS recv error (-110)` 失败，后续命令全部由 `&&` 短路，未检查磁盘、未安装包、未改变环境。会话已结束。

## 2026-09-03 03:50 +08:00

- 连接用途：重试上一条环境安装；先 ff-only pull，若失败则执行 `bash net.sh` 后再次 pull，成功后才检查磁盘/cache 并安装精确 torch/accelerate wheel。
- 权限判断与验收边界：顺序、轻量环境操作和停止条件沿用 03:46 条目；本连接仍不加载模型、不初始化 CUDA、不提交推理作业。
- 实际结果：首次 pull 等待约 133 秒后 timeout，`bash net.sh` 成功续网，第二次 pull 将服务器从 `bb4bab6` 纯快进至 `d559a6f`。文件系统余量约 252G；Qwen cache 16G、Mistral-Nemo cache 9.1M。torch 2.6.0 与 accelerate 1.9.0 由缓存 wheel 安装成功，无编译；与 transformers 4.52.4 一同通过 `pip show` 版本门禁。Mistral 权重尚缺，本连接未提交作业。

## 2026-09-03 03:58 +08:00

- 连接用途：同步兼容分支后，使用 Hugging Face CLI 从 `hf-mirror.com` 下载锁定 `mistralai/Mistral-Nemo-Instruct-2407@04d8a90549d23fc6bd7f642064003592df51e9b3` 的缺失模型权重到既有用户 cache，并复核 snapshot 大小/文件列表。
- 权限判断与顺序：第一条远端操作为 `git pull --ff-only origin validation/guqq-real-model-stt-smoke`；下载属于许可的登录节点轻量资源操作，仅写任务使用的 Hugging Face cache，不修改 Git 源码。若网络失败先 `bash net.sh`；不加载模型、不初始化 CUDA、不运行推理。
- 验收边界：必须锁定 revision，磁盘余量足够，下载完成后 snapshot 不含 `.incomplete` 且权重 index/所有 shard 可见；失败则保留可续传 cache，不提交 smoke。
- 实际结果：首项 pull 成功；CLI 锁定 SHA 开始下载 18 个文件，metadata/tokenizer 和 `model-00001` 至 `model-00005-of-00005.safetensors` 五个分片均报告完成。额外 `consolidated.safetensors` 仍在下载时 SSH 被远端关闭，未执行尾部 `du/find/ls` 校验，预计留下可续传 `.incomplete`；未提交 smoke。

## 2026-09-03 04:10 +08:00

- 连接用途：同步兼容分支后，只读检查 Mistral snapshot/`.incomplete` 状态；为避免 SSH 生命周期再次中断大文件续传，提交一个 1 CPU/2G/2h、无 GPU 的轻量 Slurm 下载作业，继续锁定 revision 的 Hugging Face cache 下载。
- 权限判断与顺序：第一条远端操作为 ff-only pull；只读检查后通过 Slurm 运行下载 CLI，写入既有任务 cache 和独立日志，不加载模型、不修改源码。下载/续传本可在登录节点执行，改用 Slurm 仅为进程持久性；不申请 GPU。
- 验收边界：下载 job 必须 Exit 0，snapshot 的 5 分片/index/config 齐全且无 `.incomplete`；通过前不提交真实 smoke。
- 提交尝试：SSH/tool 调用在约 30 秒后结束且未返回 stdout/stderr、Job ID 或可轮询 session，无法证明 `sbatch` 是否执行。按未知状态处理，不盲目重复提交。

## 2026-09-03 04:15 +08:00

- 连接用途：先同步兼容分支，再用 `squeue`/`sacct`、`mistral-download-*.out/.err` 与 cache `.incomplete` 只读确认上一下载提交是否存在及其终态；仅在确定没有活动/已提交作业时补交一次相同轻量下载作业。
- 权限判断与验收边界：首项仍为 ff-only pull；只读状态检查优先，避免重复下载 job。若补交，资源与锁定 revision 沿用 04:10 条目；本连接不提交真实 smoke。
- 实际结果：工具再次在约 30 秒后返回空结果；本地检查发现 04:09 起的首个 SSH 客户端仍遗留运行且无可恢复输出通道，第二次调用未创建新 SSH 进程。仅终止该本轮客户端；若远端已执行 `sbatch`，Slurm 作业不受客户端终止影响。下载作业状态仍未知。

## 2026-09-03 04:20 +08:00

- 连接用途：在无遗留 SSH 客户端后进行一次干净恢复；首项 pull 后只读查询 `mistral-cache` 队列、日志和 `.incomplete`，确认是否已提交/完成。
- 权限判断与验收边界：本连接不补交作业，只取证后退出；取得明确结果再决定续传或进入 smoke，避免把 SSH 工具状态与 Slurm 状态混淆。
- 实际结果：带 `ConnectTimeout=10` 和远端 `git pull` 60 秒上限的连接正常给出可轮询 session，但终态 stdout/stderr 仍为空；本地无遗留 SSH 进程。因为命令未输出显式阶段 marker，无法区分 pull timeout、空队列/日志和无 `.incomplete`，仍不据此判定下载完成。

## 2026-09-03 04:23 +08:00

- 连接用途：使用相同连接/pull 超时，但为 pull 返回码、队列、日志、snapshot 大小、分片数和 incomplete 数分别输出显式 marker，消除空输出歧义。
- 权限判断与验收边界：仍为纯只读取证、不提交作业；首个实质远端命令是带 60 秒上限的 ff-only pull，随后无论 pull 结果均只读取状态并输出 marker。
- 实际结果：pull 达到 60 秒上限（rc 124），但状态 marker 完整返回。`mistral-cache` 队列为空且无下载日志，说明先前 `sbatch` 未执行；Mistral cache 已为 46G，锁定 snapshot 中 5 个 `model-0000*-of-00005.safetensors` 全部存在，`.incomplete` 为零。下载完整性门禁通过，可进入 smoke。

## 2026-09-03 04:24 +08:00

- 连接用途：同步 `validation/guqq-real-model-stt-smoke`；若 pull 网络失败，运行 `bash net.sh` 后重试。确认服务器 HEAD/C2C、锁定依赖、正式 artifact、空 smoke 输出路径和队列后，提交 `smoke_real_models.sbatch` 的 1-GPU/192G/4h 真实功能作业。
- 权限判断与顺序：首项为 ff-only pull；环境/cache/artifact/输出只做轻量门禁，模型加载、CUDA 检查、Receiver-only 与 STT 推理全部由 Slurm 作业执行。显式写入 main-based 未验收代码版本 `036df809c7816747cd5478a6a8b3b6376bf93337`。
- 验收边界：提交前输出及 `.partial` 必须不存在；作业需在加载权重前通过 torch/accelerate/transformers、CUDA、至少 20 GiB GPU、artifact 门禁。终态要求 Exit 0、schema v2、两路恰好最多 2 tokens、锁定 revisions/artifact metadata、有限 transport stats/metrics、runtime GPU 详情、无 partial、GNU time 与产物 SHA；功能耗时不进入正式 latency 表。
- 提交结果：首项 pull 成功且已最新；随后误用完整 `git rev-parse HEAD` 与短 SHA `d559a6f` 比较，门禁返回非零并由 `&&` 短路，未执行 `squeue` 或 `sbatch`，远端状态未变。

## 2026-09-03 04:26 +08:00

- 连接用途：修正 HEAD 门禁为 `git rev-parse --short HEAD == d559a6f`，其余同步、输入/输出/队列门禁和真实 smoke 提交完全沿用 04:24 条目。
- 权限判断与验收边界：第一项仍为 ff-only pull；只有所有轻量门禁通过才执行一次 `sbatch`，不在登录节点推理。
- 实际结果：pull 成功且短 SHA/正式 artifact/空输出及 partial/空队列门禁全部通过；以 `CODE_VERSION=036df809c7816747cd5478a6a8b3b6376bf93337` 提交真实 smoke 为 Job 241。提交成功后退出，模型加载和推理完全由 Slurm 执行。

## 2026-09-03 04:28 +08:00

- 连接用途：同步兼容分支后只读监控 Job 241 的 `squeue` 状态与增量 stdout/stderr；若已终态则收集 `sacct`、GNU time、报告/partial 状态和失败栈或成功摘要。
- 权限判断与验收边界：首项为 ff-only pull；仅只读观察，不取消、不重提交、不修改产物。RUNNING/PENDING 则退出并后续登记续查；终态按 04:24 验收边界处理。
- 实际结果：首项 pull 成功；Job 241 已离队并失败。stderr 显示两侧权重加载后，Receiver-only 首个生成 kernel 报 `CUDA error: no kernel image is available for execution on the device`；GNU time 为 0:26.39、Exit 1、MaxRSS 5,845,204 KiB、0 swap。失败发生在报告构建前，无合格 JSON；根因指向 torch 2.6.0 CUDA 12.4 wheel 与节点 GPU compute capability 不兼容，而非内存或模型/cache 缺失。

## 2026-09-03 04:31 +08:00

- 连接用途：同步兼容分支后提交一个 1 CPU/1G/5min 的 Slurm GPU 诊断，仅运行 `nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version`，确认 Job 241 的硬件兼容根因。
- 权限判断与顺序：首项为 ff-only pull；GPU 查询通过 Slurm allocation，不在登录节点初始化设备。该诊断无模型加载/推理，不覆盖 Job 241 日志或结果。
- 验收边界：记录 GPU 型号、显存、compute capability 和驱动；若确认锁定 torch wheel 不支持该架构，则依据计划允许的 CPU/offload 功能路径新增独立 Slurm fallback，仍不把其耗时纳入 latency 表。
- 提交结果：首项 pull 成功，但远端 `--wrap` 的引号未形成单一参数，`nvidia-smi --query-gpu=...` 被 sbatch 误解析并报 unrecognized option；诊断 job 未提交。

## 2026-09-03 04:33 +08:00

- 连接用途：改用无需引号的 `--wrap=nvidia-smi` 提交相同 1 CPU/1G/5min GPU 诊断；默认输出足以记录型号、显存和驱动，并据型号确认 compute capability。
- 权限判断与验收边界：其余顺序与只读硬件诊断边界沿用 04:31 条目；不加载模型或运行 smoke。
- 实际结果：pull 成功，轻量 GPU capability 诊断已提交为 Job 242。

## 2026-09-03 04:35 +08:00

- 连接用途：同步兼容分支后只读读取 Job 242 队列/终态与 `stt-gpu-cap-242.out/.err`，记录 GPU 硬件事实。
- 权限判断与验收边界：仅只读，不提交新作业；根据硬件结果回到本地设计 CPU/offload fallback。
- 实际结果：Job 242 已完成且 stderr 为空；GPU 为 NVIDIA GeForce RTX 5090、32,607 MiB，驱动 570.211.01、CUDA 12.8。RTX 5090 属 Blackwell `sm_120`，确认 torch 2.6.0/cu124 wheel 缺少其 kernel image，Job 241 根因成立。

## 2026-09-03 04:45 +08:00

- 连接用途：同步 Blackwell 兼容提交 `4fe0065`；创建独立 Python venv `C2C/.venv-smoke-cu128`，从 PyTorch 官方 cu128 index 安装 torch 2.7.1，并精确安装 smoke 依赖与记录最终版本。
- 权限判断与顺序：第一项为 `git pull --ff-only origin validation/guqq-real-model-stt-smoke`；随后仅执行 AGENTS 允许的 `python3 -m venv` 和 wheel 下载/安装，不初始化 CUDA、不加载模型或推理。网络失败用 `bash net.sh`；源码编译则停止。
- 验收边界：共享 `C2C/.venv` 保持不变；新环境需报告 torch `2.7.1+cu128`、accelerate 1.9.0、transformers 4.52.4 及锁定数值依赖。GPU arch/kernel 验证留给后续 Slurm 作业。
- 实际结果：首项 pull 从 `d559a6f` 快进至 `4fe0065`；独立 venv 创建成功，官方 cu128 wheel 安装 torch 2.7.1+cu128，其他精确依赖全部 wheel 安装并通过 `pip show`。共享 venv 未变，未在登录节点初始化 CUDA。

## 2026-09-03 04:55 +08:00

- 连接用途：同步兼容提交，复核 Job 241 未留下 JSON/partial 和队列后，使用 `.venv-smoke-cu128/bin/python`、`RUNTIME_PROFILE=blackwell-cu128`、`CODE_VERSION=17762a1...` 重提同一真实 smoke。
- 权限判断与顺序：第一项为 ff-only pull；其余输入/空输出/队列为轻量门禁，CUDA arch 检查、模型加载和推理全部在脚本的 Slurm GPU allocation 内。
- 验收边界：沿用 04:24 的 schema/两路/provenance/原子性/资源要求，并新增 runtime profile=`blackwell-cu128`、torch=`2.7.1+cu128`、compiled arches 包含 `sm_120`、device capability `[12,0]`。
- 实际结果：pull/HEAD/空输出及 partial/空队列门禁通过，Blackwell profile 真实重跑已提交为 Job 243；解释器、profile 与 code version 均通过 sbatch export 显式传入。

## 2026-09-03 04:57 +08:00

- 连接用途：同步兼容分支后只读监控 Job 243 队列和增量日志；终态则收集报告、runtime/profile/arch、两路输出、transport stats、GNU time、MaxRSS、原子性与 SHA。
- 权限判断与验收边界：仅只读、不取消或重提；验收标准完全沿用 04:55 条目。
- 实际结果：Job 243 使用 cu128 环境成功加载两模型并完成 Receiver-only，证明 Blackwell kernel 兼容修复生效；STT 在 source transport 前以 `exact transport requires artifact coverage of the full source vocabulary` 失败。Qwen3 LM head 为对齐填充的 151,936 rows，而 fingerprint-verified tokenizer/T source vocab 为 151,669，尾部 267 rows 不是可编码 token。作业 0:14.99、Exit 1、MaxRSS 16,415,184 KiB、0 swap，无合格 JSON。

## 2026-09-03 05:08 +08:00

- 连接用途：同步 padded-vocab 兼容提交 `f4de100`，确认 Job 243 无 JSON/partial、队列为空后，以 Blackwell venv/profile 和 `CODE_VERSION=b0bff17c9aa088ddad10fe98723a63b49f96e863` 重跑同一真实 smoke。
- 权限判断与顺序：首项 ff-only pull；其余轻量门禁后仅通过 Slurm 运行模型。环境、artifact、prompt、2-token、1 GPU/192G/4h 保持与 Job 243 相同，只改变已本地验收的 tokenizer-vocab padding 修复。
- 验收边界：沿用 04:55 全部标准，并要求 source input shape/virtual prompt 能跨过 151,936→151,669 显式裁剪，quality stats 有限且 retained/active mass 合理；无 JSON/partial 则失败。
- 实际结果：pull 从 `4fe0065` 快进到 `f4de100`，HEAD/空输出及 partial/空队列门禁通过；padding 修复真实 smoke 已提交为 Job 244。

## 2026-09-03 05:10 +08:00

- 连接用途：同步兼容分支后只读监控 Job 244 队列/日志；终态收集 schema v2 JSON 全字段、资源、原子性和 SHA。
- 权限判断与验收边界：仅只读、不取消或重提；标准沿用 05:08 条目。
- 实际结果：Job 244 跨过 source vocab 修复，在 `receiver_embedding_weight.index_select(...).to(logits.dtype)` 申请完整 float32 active embedding 表时 OOM；请求 2.50 GiB，而 31.37 GiB GPU 仅余 528.56 MiB。作业 0:15.08、Exit 1、MaxRSS 16,525,048 KiB、0 swap，无 JSON。三次真实失败后已复查 `lessons.md`，确认需新增模型并行内存约束并做 chunked receiver embedding 累加，而非提高资源或改精度标准。

## 2026-09-03 05:21 +08:00

- 连接用途：同步 chunked embedding 兼容提交 `6e08989`，确认空 JSON/partial/队列，以相同 Blackwell venv/profile、artifact、prompt 和资源重跑，code version=`024beacd250fd5e6a57e92aca1b055000b4992f2`。
- 权限判断与顺序：首项 ff-only pull；仅 Slurm 执行模型负载。除 O(vocab×hidden) 改为 O(chunk×hidden) 的本地验收修复外，所有输入/环境/资源不变。
- 验收边界：沿用 05:08 标准，并比较 Job 244 OOM 位置是否通过、报告 peak memory/GNU time/MaxRSS；不因连续失败降低双路径/schema/原子性要求。
- 实际结果：pull 快进至 `6e08989`，HEAD/空输出及 partial/空队列通过；chunked embedding 真实复验已提交为 Job 245。

## 2026-09-03 05:23 +08:00

- 连接用途：同步兼容分支后只读监控 Job 245 队列/日志；终态收集 schema v2 报告和资源/哈希证据。
- 权限判断与验收边界：仅只读；标准沿用 05:21 条目。
- 初步结果：Job 245 已离队，stderr 无异常栈且 GNU time 为 0:16.67、Exit 0、MaxRSS 16,560,968 KiB、0 swap；报告写入产生 72 个 filesystem output blocks，表明 chunked 路径跨过 Job 244 OOM。尚待独立读取 JSON/哈希/partial 后最终验收。

## 2026-09-03 05:26 +08:00

- 连接用途：同步兼容分支后只读收集 Job 245 JSON 全文、文件大小/SHA-256、stderr SHA 和 `.partial` 状态，逐项验收 runtime/profile/arch、artifact provenance、Receiver-only/STT shapes/outputs、质量统计与 metrics。
- 权限判断与验收边界：首项 ff-only pull；仅 `stat`/`sha256sum`/`test`/`cat` 读取，不修改或复制服务器结果。全部字段通过后才回本地形成验收提交。
- 实际结果：报告 8,581 bytes、SHA-256 `a14da4b15a368eefbd905d61ad4be71af143fdf2ab6df74071fa487c6b867c26`，stderr SHA-256 `6ecddddcca89383eeaf4c211c6b3fcbf412c4713c4dc301d2cc55401ff86eccc`，无 `.partial`。schema v2/code version `024beacd...`；runtime 为 Python 3.10.12、`blackwell-cu128`、torch 2.7.1+cu128/accelerate 1.9.0/transformers 4.52.4，compiled arches 含 `sm_120`，设备 RTX 5090 `[12,0]`/33,679,736,832 bytes。Receiver-only 输入 8、输出 2 tokens，文本 ` Gravity is`；STT source/virtual/output shapes `[1,7]`/`[1,7,5120]`/`[1,2]`，文本 `I'm`。retained/active mass 范围 `0.9999999404..1.0000005960`、dropped top-m 0；STT total 2.35798s、peak 31,074,283,520 bytes。模型/tokenizer/dataset revisions、正式 artifact 131,069×151,669/2,733,518 nnz 和 artifact provenance 完整。最终验收通过，时延仅作功能诊断。

## 2026-09-03 05:26 +08:00

- 连接用途：同步阶段 3 统一 evaluator 的 Guqq 兼容临时分支，检查 MMLU-Redux `abstract_algebra` 小子集是否已缓存；若未缓存，仅在登录节点预取锁定数据资源。随后检查正式 artifact、Blackwell venv、空输出/partial 与队列，再提交固定 5 题 STT Slurm 评测。
- 权限判断与顺序：连接后第一项必须 `git pull --ff-only`；源码只通过兼容分支快进同步，不在服务器编辑。数据下载/缓存检查属于轻量操作；模型加载和推理只经 `sbatch script/transport/slurm/evaluate_stt_mmlu_redux.sbatch`，不在登录节点运行。
- 验收边界：要求 Job Exit 0、5 条逐题 success 或显式失败记录、summary 只统计 success、canonical/prompt metadata、分段 latency/长度/显存、transport quality、正式 artifact/runtime provenance、无 `.partial`，并记录 GNU time/MaxRSS/日志与产物 SHA。真实通过前分支保持 `[UNACCEPTED]`。
- 首次连接结果：首项 `git pull --ff-only` 在 GitHub HTTPS 传输中以 `GnuTLS recv error (-110)` 失败，后续环境/队列检查因 `&&` 未执行，服务器源码未变更。下一次连接仍先尝试 pull；若失败则按规范在同一连接运行 `bash net.sh` 后重试 pull，成功前不执行其他任务。
- 第二次连接结果：首次 pull 超时后 `bash net.sh` 成功恢复网络，第二次 pull 将服务器从 `6e08989` 快进到 `8abf1cd`。随后版本检查的一行 Python 因远端 shell 引号转义错误而未执行，且 `&&` 阻止后续门禁；这不是环境或代码失败。下一次连接首项继续 pull，再改用 `pip show` 做无引号版本检查。
- 环境门禁结果：服务器已快进到 `b3457a1`；torch 2.7.1+cu128 与 transformers 4.52.4 正确，正式 artifact、空 records/partial 和空队列通过，MMLU-Redux/`abstract_algebra` cache 路径存在。但 `.venv-smoke-cu128` 缺少 evaluator 必需的 datasets，故未提交作业。下一次连接先 pull，再按 `env.md` 仅向隔离环境安装/核验 datasets 4.0.0。
- 依赖补齐结果：服务器先快进至 `e2a90f4`，随后隔离 venv 通过缓存 wheel 成功安装 datasets 4.0.0 及其依赖；锁定 torch/transformers/accelerate 未变，evaluator `--help` 通过。下一次连接首项 pull 后执行离线 `abstract_algebra` 5 题轻量加载检查、再次核对空输出与队列，全部通过才提交 Slurm。
- provenance 修正后的提交连接因 Windows→SSH→Bash 嵌套 `python -c` 引号被剥离，远端 Bash 在执行任何命令前 parse 失败，故本次没有 pull、检查或提交作业。连续远端连接/诊断失败达到三次后已查阅并补充 `lessons.md`；计划取消脆弱 inline Python，下一次首项 pull 后仅用无嵌套引号的 artifact/输出/队列门禁并直接 `sbatch`，数据离线加载由 Slurm 入口自身验证。
- 提交结果：服务器首项 pull 快进至 `98c5f85`；正式 artifact、空 records/partial 和同名空队列的纯 shell 门禁通过，固定 5 题 MMLU-Redux STT 评测已提交为 Job 246。模型加载/数据读取/推理均在 Slurm allocation 内；下一次连接仅监控终态并收集 records/summary/日志/资源证据，不取消或重提。
- 首次监控结果：首项 pull 再次因 GitHub 443 超时失败；后续只读 `squeue` 显示 Job 246 已不在队列，集群本次返回 accounting storage disabled，日志 tail 无输出。尚不能判定成功或失败；下一次连接仍先 pull，失败则运行 `net.sh` 后重试，再用 `scontrol`、文件状态与日志直接验收，不依赖 sacct。
- 第二次监控结果：首项 pull 与 `net.sh` 后重试均失败；Job 246 已从 controller 清除且 accounting disabled，预期相对路径下没有 246 日志或 evaluation 目录。该证据不足以区分 Slurm 在打开日志前失败与路径落在其他 submit directory。下一次连接仍先 pull/必要时 net.sh，再从 `/home/xmz/vocab_align` 范围只读查找 `*246*` 和 `stt.records.jsonl`，并查看 `slurmctld` 可提供的历史命令；找不到有效逐题产物即判定失败，不重提相同代码。
- 全仓定位结果：两次 pull 仍超时，但只读 find 确认 Job 246 的 stdout/stderr 和 `stt.records.jsonl` 均位于预期目录；未发现 summary。下一次连接首项以有界 `git pull` 尝试同步（服务器 C2C 已是运行提交且远端没有新源码），随后读取日志、records、partial 状态和哈希；若 5 题全失败或 summary 缺失原因来自代码/缓存，则判定本次真实验证失败并回本地修复。
- Job 246 最终结果：首项 pull 本次成功；stderr 2,539 bytes/SHA `047d3e82...`，stdout 200 bytes/SHA `91b6bdc1...`，records 11,716 bytes/SHA `823bc55b...`，无 partial/summary。离线 `abstract_algebra` cache 与两模型加载成功；5 条记录均显式 `failed`/`OutOfMemoryError`，首次需 1.74GiB 而仅余约 1004MiB，后续需 1.41—1.73GiB 而仅余约 1.08GiB。summary 正确拒绝零 success，未静默算错。计划经本地测试后以 source CPU、receiver auto 和 16-token 上限重试，资源仍为 RTX 5090/64G/30m。
- source-offload 修复首次同步连接的首项 pull 发生 GnuTLS recv (-110)，因此未执行 records/partial 门禁且未提交新作业。下一次连接首项仍 pull，失败则按既有经验运行 `net.sh` 后重试；服务器快进到 `d98a85e` 前不重提。
- 第二次同步连接的首次 pull 与 `net.sh` 后 pull 均在 GitHub 443 超时，仍未提交作业。下一次使用 20/30 秒有界 pull→net.sh→pull，避免单次网络尝试阻塞数分钟；只有 `git rev-parse` 为 `d98a85e` 后才执行断点门禁和 sbatch。
- 有界同步成功：服务器从 `6a57b07` 快进至含修复的 `d98a85e`，祖先校验、Job 246 records 存在、summary/records partial 不存在和空同名队列门禁通过；source CPU/receiver auto 的断点复验已提交为 Job 247。下一次连接首项有界 pull 后只读监控，不取消或重提。
- 2026-09-03 10:08 +08:00 exact benchmark 兼容同步用途：前四次连接均以 pull 为首项；三次短 timeout 后按 `lessons.md` 长轮询，最终确认网络 fetch 成功但服务器 `main` HEAD `d7cc4cf` 与 `origin/main` 分叉。只读状态仅含允许的 venv、评测结果和 `net.sh`。本地已从该精确 HEAD 构建并推送兼容分支 `validation/guqq-exact-benchmark-smokes`（`0834262`，其 C2C tree 与已测 `fec0eef` 相同）。下一次连接首项使用 `git pull --ff-only origin validation/guqq-exact-benchmark-smokes` 纯快进；成功后只检查 commit、数据缓存、空同名队列/输出，并提交 GSM8K、MATH-500、LongBench Qasper 三个 exact Slurm 作业，不切分支、不运行近似/消融。
- 2026-09-03 10:10 +08:00 同步/缓存结果与下载用途：服务器通过显式 pull 从 `d7cc4cf` 纯快进到 `a394e4e`，三个 recipe、空输出和空队列门禁通过；仅发现 GSM8K hub 与 Arrow cache，未发现 MATH-500/LongBench 缓存，故尚未提交作业。下一次连接首项继续 pull 同一兼容分支，随后只查看隔离 venv 的 `hf download` 能力，并把 MATH-500 与 LongBench Qasper 所需数据下载到 Hugging Face 缓存；下载属于许可的轻量资源操作，数据转换/模型加载/推理仍只在 Slurm 中执行。
- 2026-09-03 10:11 +08:00 下载工具检查结果与文件清单用途：服务器快进到 `23d8ea4`，隔离 venv 的 `hf download` 支持 dataset repo、revision、include/exclude 与 cache-dir。下一次连接首项 pull 后只列出 MATH-500/LongBench dataset repo 文件，确定 Qasper 的最小 include 模式，避免无依据下载整个 LongBench；随后下载小型 MATH-500 和 Qasper 所需文件到默认 Hugging Face cache，不做数据转换或推理。
- 2026-09-03 10:12 +08:00 文件清单尝试结果与 API 查询用途：服务器快进到 `aae99bf`，但当前 `hf repo-files` 仅提供 delete 子命令，`ls` 对两个 repo 均显式报参数错误，未下载文件。下一次连接首项 pull 后使用 Hugging Face 官方只读 tree API 查询 repo 文件名，以精确路径确定 MATH-500 与 Qasper 下载集合；查询与下载仍限于缓存管理，不运行数据处理或模型。
- 2026-09-03 10:14 +08:00 精确下载用途：本地从 Hugging Face 官方仓库页面确认 MATH-500 主分支为 `test.jsonl`（约 450KB 数据集），LongBench Qasper 为 `qasper/test-00000-of-00001.parquet`（约 1.86MB）。下一次连接首项 pull 后使用隔离 venv 的 `hf download --repo-type dataset` 精确下载两者及各自 README 到默认 cache；不下载整个 LongBench，不执行 Arrow 转换或模型推理。下载完成后只检查目标 cache 路径存在并退出。
- 2026-09-03 10:17 +08:00 下载连接结果与网络恢复用途：首项长 `git pull` 约 80 秒后以 GitHub HTTPS GnuTLS `recv error (-110)` 失败，因 `&&` 门禁未执行任何 `hf download`。下一次连接仍先 pull；若失败则运行 `bash net.sh` 后长 pull 重试，仅在成功同步兼容分支后执行上一条精确 MATH-500/Qasper 下载。不得因数据下载需要而跳过源码同步门禁。
- 2026-09-03 10:24 +08:00 网络恢复结果与继续用途：首次长 pull 在约 133 秒后 GitHub 443 超时，`bash net.sh` 成功后第二次长 pull 也在约 135 秒超时，精确数据下载仍未开始。服务器当前源码已包含完整 benchmark 实现，但仍不绕过 pull 门禁；下一次连接先 pull 兼容分支，成功后下载。为避免每次失败记录产生新的远端文档 commit 并扩大同步目标，本条先在本地保留，待一次有效同步/下载后与结果一并提交。
- 2026-09-03 10:28 +08:00 本地 ref 门禁与下载重试用途：`git pull --ff-only . refs/remotes/origin/validation/guqq-exact-benchmark-smokes` 成功确认已获取代码 ref 下工作树 up-to-date；随后 MATH-500 `hf download` 因 `huggingface.co` 临时 DNS 解析失败，LongBench 未执行，缓存无半成品声明。下一次连接仍以同一只含已测代码的本地 ref pull 为首项，随后运行 `bash net.sh` 恢复网络，再重试两个固定 revision 的精确文件下载；不提交作业直到缓存下载成功。
- 2026-09-03 10:35 +08:00 scp 结果与作业提交用途：`net.sh` 后 Hugging Face DNS 仍失败，已按许可在本地从固定 revision 下载并校验 MATH-500（446,564 bytes，SHA `35dc4108...`）与 Qasper（1,863,050 bytes，SHA `7c6bf3a2...`）；Guqq 先对已获取 ref 完成 pull 门禁并创建任务数据目录，随后两个文件 scp 均成功。下一次连接首项从 GitHub pull 兼容提交 `fb6c687`（失败则 `net.sh` 后重试）；成功后用 `sha256sum` 校验两文件、检查三个输出为空和同名队列为空，再分别 sbatch GSM8K、MATH-500、Qasper exact smoke。所有模型加载、数据转换和推理只在 Slurm allocation 内执行。
- 2026-09-03 10:37 +08:00 提交结果与首次监控：Guqq 成功快进到 `fb6c687`，scp 文件 SHA 门禁通过，Jobs 248/249/250 已提交。首次只读监控时 Job 248 在 node221 运行 0:52，source/receiver shards 与固定 revision GSM8K 离线 cache 均成功加载，尚无 records；Job 249 因 Resources、250 因 Priority 排队。下一次连接首项仍对本地已获取 ref pull，然后只读检查三作业终态/增量 records/日志；不取消或重提。
- 2026-09-03 10:37 +08:00 提交结果与监控用途：Guqq 从 GitHub 成功快进到 `fb6c687`，scp 两文件 SHA 分别为 `35dc4108...`/`7c6bf3a2...`，三个空输出门禁通过；exact smoke 已提交为 Job 248（GSM8K 3题/64 tokens）、249（MATH-500 3题/128 tokens）、250（Qasper 1题/输入截断2048/输出32 tokens）。下一次连接首项对已获取兼容 ref pull 后，仅用 `squeue` 和日志/records 行数监控，不取消、不重提、不在登录节点加载模型。
- 2026-09-03 10:40 +08:00 第二次 benchmark 监控用途：连接后首项对已获取的兼容分支 ref 执行 ff-only pull；随后只读检查 Jobs 248/249/250 的队列状态、records/summary/partial 与日志增量。若作业已完成，仅收集产物和资源证据；不取消、不重提、不运行近似或消融实验。
- 2026-09-03 10:41 +08:00 第二次 benchmark 监控结果：连接首项本地 ref pull 成功且服务器已是最新；`squeue` 显示 Jobs 248/249 已离队、Job 250 在 node221 运行 0:29。后续批量文件检查中的远端 shell 变量被本地 PowerShell 提前展开，故其 `records-missing` 与空日志标签无诊断效力。下一次连接仍先 pull，再改用无变量的显式路径只读验收 248/249，并监控 250。
- 2026-09-03 10:42 +08:00 第三次 benchmark 监控用途：连接后首项执行同一兼容 ref 的 ff-only pull；以显式路径核验 Jobs 248/249 的 records、summary、partial 和日志，并查看 Job 250 队列/日志增量。只读收集，不取消或重提作业。
- 2026-09-03 10:43 +08:00 第三次 benchmark 监控结果：pull 门禁通过；Jobs 248/249 已各写出 3 条 records 和 summary，均为 3 success/0 failed/accuracy 0.0，且 records/summary 均无 partial。Job 250 在 node221 运行至 1:10，已完成双模型 shard 和本地 Qasper 200-row split 加载，stderr 尚无异常。
- 2026-09-03 10:44 +08:00 第四次 benchmark 监控用途：连接后首项执行兼容 ref 的 ff-only pull；只读查看 Job 250 的队列终态、显式 Qasper records/summary/partial 和 stderr。若仍运行则保留作业并退出；若完成则收集本轮最终产物。
- 2026-09-03 10:45 +08:00 第四次 benchmark 监控结果：pull 门禁通过；Job 250 在 node221 运行至 2:37，输出目录尚空且无 partial，stdout/stderr 显示 recipe、GPU、双模型和 200-row Qasper split 均已正常加载，无异常栈。预计 source CPU 长上下文是主要耗时，保持自然运行。
- 2026-09-03 10:46 +08:00 Jobs 248/249 逐题验收用途：连接后首项执行兼容 ref 的 ff-only pull；只读计算 records/summary SHA-256，并用 `jq` 提取每题 status、prediction、reference、generation、metrics 和 diagnostics provenance，验证 exact mode、固定模型/artifact/data/code 与逐题形状/质量指标。同步只读查看 Job 250 队列，不修改任何服务器文件。
