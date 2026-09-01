# 进度更新

## 2026-09-01

- 建立任务状态与测试计划文档；开始 transport 核心和 artifact 实现单元。
- 完成 dense log-domain Sinkhorn oracle、`T = Pi Diag(a)^-1`、版本化安全 CSC artifact 保存/加载及对应测试；完整本地单元测试 8/8 通过。
- 开始阶段 0 配置与确定性 manifest 实现单元。
- 完成阶段 0 版本化配置、锁定 revision 的主模型对 recipe、SHA-256 确定性 train/dev manifest 及 CLI；完整本地测试 17/17 通过。
- 开始抽取统一 token metadata/byte offset 模块，并增量改造 tokenizer 审计脚本。
- 完成公共 token metadata、byte offset、special/control 分类和 tokenizer fingerprint；现有 builder 与全词表比较脚本已复用该实现，完整测试 21/21 通过。
- 开始 baseline 快照 schema 与离线 CLI 实现。
- 完成 deterministic baseline snapshot 与 CLI，canonical messages/source prompt/target prompt 分离保存，硬件和 checkpoint 缺失显式标记；完整测试 24/24 通过。
- 准备服务器真实 tokenizer 审计，已记录连接用途与登录节点轻量操作边界。
- 服务器首次全词表审计完成计算但 revision 字段为空，判定产物不合格；回到本地修复报告 revision 回退逻辑后重跑。
- 服务器第二次审计 revision 已正确，但 completion audit 发现通用 artifact provenance 字段缺失；继续本地补齐 schema/input fingerprint/build config/seed/code version。
