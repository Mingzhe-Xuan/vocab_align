# Agent 状态

## 当前状态

正在实施 Training-free Soft-Token Transport。阶段 1 的候选图、边际、dense/sparse Sinkhorn 与 artifact 基础已通过测试；下一步实现 facade 构建与独立 audit CLI，产出 toy oracle artifact。

## 当前计划

1. 实现 vocab transport facade 与 audit 模块/CLI，产出 toy oracle artifact。
2. 完成阶段 1 resume/checkpoint 构建脚本后进入精确 STT 推理原型。
3. 服务器网络恢复后 pull 并最终重跑真实 tokenizer 审计。

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
