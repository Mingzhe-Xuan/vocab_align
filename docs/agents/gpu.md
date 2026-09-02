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
