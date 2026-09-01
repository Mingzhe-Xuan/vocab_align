# Agent 状态

## 当前状态

正在实施 Training-free Soft-Token Transport。阶段 2 的精确 soft transport、metrics 与 TrainingFreeTransportModel wrapper 已通过 65 项完整本地测试，Guqq 真实 tokenizer provenance 审计也已验收；下一步实现可复现的 STT smoke CLI，并准备按临时分支 + Slurm 规则进行真实模型短序列验证。

## 当前计划

1. 实现 `script/transport/smoke_stt.py` 的加载、短样本执行与 diagnostics 落盘，并用 tiny fixtures 离线验收。
2. 在临时验证分支提交真实模型 smoke 所需代码，通过 Guqq `git pull` 同步并使用 Slurm 验证。
3. 使用训练语料构建真实预览 transport artifact，审计所有正质量行列支撑与构建 provenance。

## 变更记录

- 2026-09-01 16:15 +08:00：开始任务，完成两份计划与当前工作树初审。下一步实现第一个可独立验收单元。
- 2026-09-01 16:22 +08:00：用户将服务器环境规范由 uv 更新为 Python venv；已同步环境记录，实施计划不变。
- 2026-09-01 16:35 +08:00：完成第一个实现单元并通过 8 个 CPU 测试；下一步进入阶段 0 配置与 manifest。
- 2026-09-01 16:42 +08:00：进入阶段 0 配置与 manifest 实现；先以离线 tiny fixtures 固定序列化和划分不变量。
- 2026-09-01 16:55 +08:00：阶段 0 配置与 manifest 完成，完整本地测试 17/17；下一步扩展 tokenizer metadata 与 baseline 快照。
- 2026-09-01 17:03 +08:00：进入 token metadata 实现单元；保留现有 exact/span 行为并消除脚本与库的 bytes 规则重复。
- 2026-09-01 17:18 +08:00：token metadata 单元完成，完整测试 21/21；下一步冻结 baseline schema。
- 2026-09-01 17:24 +08:00：进入 baseline 快照实现；不加载模型，快照只接受 canonical 与预渲染 prompt 作为冻结输入。
- 2026-09-01 17:36 +08:00：baseline 快照完成，完整测试 24/24；下一步同步服务器并执行真实 tokenizer 审计。
- 2026-09-01 17:42 +08:00：进入真实 tokenizer 审计阶段；先提交连接记录并推送代码，服务器连接后必须先 pull。
- 2026-09-01 18:02 +08:00：首次服务器审计计算成功但 revision 字段为 null，产物不合格；下一步本地修复、测试、提交后由服务器 pull 重跑。
- 2026-09-01 18:14 +08:00：第二次审计修复 revision，但通用 artifact provenance 不完整；继续修复，未降低验收标准。
- 2026-09-01 18:25 +08:00：服务器 HTTPS pull 连续三次失败；查阅 lessons 后新增网络经验，调整为 GitHub SSH transport，禁止 scp 覆盖受 Git 管理源码。
- 2026-09-01 18:30 +08:00：SSH transport 亦无权限，退出服务器；该外部验收保持 pending，计划调整为先推进阶段 1 本地模块。
- 2026-09-01 18:38 +08:00：进入候选图/边际实现；先固定 special/exact/span/ANN 优先级与正质量支撑失败语义。
- 2026-09-01 18:55 +08:00：候选图与边际完成，完整测试 33/33；下一步 sparse/log-domain Sinkhorn 与 dense oracle 对照。
- 2026-09-01 19:02 +08:00：进入 sparse Sinkhorn 实现；保持 `[target, source]` 方向与相同收敛报告协议。
- 2026-09-01 19:16 +08:00：sparse Sinkhorn 完成并与 dense oracle 对齐，完整测试 38/38；下一步 facade/audit/toy artifact。
- 2026-09-01 19:24 +08:00：进入 facade/audit 实现；修正零质量过滤与 artifact 全正边际约束的接口冲突。
- 2026-09-01 19:46 +08:00：facade/artifact graph/audit/atomic resume builder 完成，toy oracle 全不变量通过，完整测试 44/44；下一步阶段 2 精确 soft transport。
- 2026-09-01 19:54 +08:00：进入 exact soft transport/metrics 单元；完整 active support 是 exact 推理前置条件，截断或缺失质量必须显式报告。
- 2026-09-01 20:02 +08:00：阶段 2 精确 soft transport/top-m/metrics 单元完成，完整本地测试 51/51；用户确认 Guqq `git pull` 已恢复，网络异常时先执行 `bash net.sh`，并授权将 `AGENTS.md` 纳入提交。下一步提交并同步服务器审计。
- 2026-09-01 20:12 +08:00：Guqq 成功同步并完成真实 tokenizer provenance 审计，最终 JSON 的 schema、锁定 revisions、指纹与 SHA-256 均通过验收；用户补充 `docs/assets/alignment.py` 作为 align 实现参考。下一步先据此冻结 wrapper 测试协议，再实现 prefill/generate。
- 2026-09-01 20:18 +08:00：进入 TrainingFreeTransportModel wrapper 实现单元；冻结 receiver 起始 embedding + shifted source logits、mask/position/cache 与 receiver-only 独立路径的测试协议。
- 2026-09-01 20:27 +08:00：TrainingFreeTransportModel wrapper、独立 transport 配置块与 recipe 默认值完成，定向测试 25/25、完整测试 65/65；下一步形成验收提交后进入 smoke CLI 单元。
- 2026-09-01 20:19 +08:00：暂停 wrapper 实现并修订 GPU 测试提交流程；采用临时分支上的未验收验证提交供服务器 pull 和 Slurm 测试，正式分支仍只接受测试通过的验收提交。
- 2026-09-01 20:20 +08:00：GPU 测试提交流程修订完成；规范文本、相关文档路径与 Git diff 检查通过，恢复 TrainingFreeTransportModel wrapper 实现。
