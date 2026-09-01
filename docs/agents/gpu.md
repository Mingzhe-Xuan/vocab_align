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
