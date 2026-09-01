# 进度更新

## 2026-09-01

- 建立任务状态与测试计划文档；开始 transport 核心和 artifact 实现单元。
- 完成 dense log-domain Sinkhorn oracle、`T = Pi Diag(a)^-1`、版本化安全 CSC artifact 保存/加载及对应测试；完整本地单元测试 8/8 通过。
- 开始阶段 0 配置与确定性 manifest 实现单元。
- 完成阶段 0 版本化配置、锁定 revision 的主模型对 recipe、SHA-256 确定性 train/dev manifest 及 CLI；完整本地测试 17/17 通过。
- 开始抽取统一 token metadata/byte offset 模块，并增量改造 tokenizer 审计脚本。
- 完成公共 token metadata、byte offset、special/control 分类和 tokenizer fingerprint；现有 builder 与全词表比较脚本已复用该实现，完整测试 21/21 通过。
