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
- provenance 修复已推送，但服务器 HTTPS pull 连续三次失败；已记录网络经验并切换到 SSH transport，未直接修改服务器源码。
- GitHub SSH transport 也因服务器无 public-key 权限失败；退出服务器并保留 audit pending，转入可离线推进的阶段 1。
- 开始阶段 1 候选图与边际实现单元，接口固定 required active support 与候选来源证据。
- 完成 special/exact/span/ANN 候选优先级、双侧正质量支撑验证和 canonical 内容边际估计；完整本地测试 33/33 通过。
- 开始 sparse/log-domain Sinkhorn 与候选证据代价实现，使用小图对照 dense oracle。
- 完成直接在候选边上运行的 sparse/log-domain Sinkhorn、连通分量可行性检查和稀疏条件矩阵转换；完整测试 38/38 通过。
- 开始 facade/artifact graph/audit 单元；采用 active-support 压缩坐标并保存原始 token ID 映射。
- 完成正式 vocab transport facade、候选图结构化 artifact、独立 JSON/Markdown audit CLI 与原子/resume 构建脚本；toy oracle artifact 审计全不变量通过，完整测试 44/44。
- 开始阶段 2 exact soft transport/top-m/metrics，实现前先修正全词表平滑与 active support 语义。
- 完成阶段 2 精确稀疏 `Tp`、embedding expectation、source top-m 质量报告和分段 metrics；补充无 PyTorch tokenizer-only 导入回归，完整本地测试 51/51 通过。
- 接受最新开发规范：服务器网络异常时先运行 `bash net.sh` 再重试 HTTPS `git pull`；`AGENTS.md` 纳入本次提交。
- Guqq HTTPS pull 已恢复并同步到 `36e6224`；真实 Qwen3-8B → Mistral-Nemo-Instruct-2407 tokenizer 审计完成，provenance 与产物哈希验收通过，JSON 已复制到本地忽略目录。
- 下一实现单元将参考 `docs/assets/alignment.py` 中 row-vector 约定、完整 logits（含 bias）温度缩放、分块计算与非有限值检查，接入 vocab transport，而不复制其等词表假设。
