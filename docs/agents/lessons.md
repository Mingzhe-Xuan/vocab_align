# 经验记录

## 病态稀疏支撑需要保持目标不变的对偶加速

即使候选图存在对所有 active edges 严格为正的可行 coupling，交替 row/column scaling 在极端边际和低证据 feasibility 边上仍可能以很慢的速率收敛；真实图中 column residual 已到机器精度而 row residual 在 10,000 次后仍为 `5.58e-4`。不能用提高 `max_iter`、放宽 tolerance 或直接采用可行 coupling 代替熵正则最优解。可先用标准 log-domain Sinkhorn warm up，再固定一个 dual gauge，对同一 Gibbs kernel 的凸缩放对偶做解析梯度 L-BFGS，最后继续标准缩放并用原始两侧 L1 residual 验收。正 smoothing 会使 dual Hessian 的边际尺度对角跨越许多数量级；直接优化 log scaling 会忽略稀有 token，必须用 `z = sqrt(marginal) * x` 的可逆坐标缩放把局部对角预条件到约 1，同时保持原目标不变。加速器必须有独立预算和方法记录；非有限变量、不可行支撑和总预算耗尽仍显式失败。

## 二部候选图连通不等于边际容量可行

每个正质量 source/target 都有边且整个二部图连通，只能排除孤立节点和分量总质量不等，不能满足所有子集的 Hall 型容量条件。此时 log-domain Sinkhorn 可能把一侧 residual 降到机器精度，另一侧仍长期停在大残差；增加迭代或放宽容差会掩盖结构错误。构图后应先在每条已有 active edge 上预留统一的小正质量，再用 source/target 剩余边际的 northwest-corner coupling 补充缺失 pair。这样新增边至多为两侧 active token 数之和减一，并提供一个对所有候选边严格为正的可行耦合作为存在性证明；新增 pair 必须标记为独立低证据 `feasibility` 来源，不能伪装成语义 ANN/exact 证据。

## 非对称 tokenizer special 集合与完整 source logits

正 smoothing 不能在 source/target 两侧机械激活相同类别的全部 special：两个 tokenizer 的 BOS/EOS/pad/UNK、chat 和多模态 control 集合通常不对称，强制按泛化 `special` kind 一一映射既不可行也不安全。STT source logits 又覆盖完整 source vocab，因此 source special 不能简单从 artifact 删除。安全策略是 source 边际保留完整词表，将每个 source special 的原始 token 字符串用 target tokenizer 分解为 ordinary literal-byte 候选；target 平滑边际只覆盖 ordinary token，receiver 原生 BOS/EOS 仍由 wrapper 的起始 embedding 和生成逻辑管理。若 literal 分解不能产生 ordinary target，构建应失败，不能任意落到 UNK 或无关 control token。

## Artifact 数值审计

稀疏 artifact 的边际与列和容差必须至少覆盖其存储 dtype 的机器精度；用固定的 float64 级容差审计 float32 数据会误拒绝合法 artifact。实现采用 `max(配置容差, 10 * dtype epsilon)`，同时仍拒绝非有限值和真实的归一化偏差。

## 项目脚本调用

未以 editable package 安装仓库时，直接执行 `python script/dataset/example.py` 只会把脚本目录加入 `sys.path`，可能无法导入顶层 `rosetta`。项目文档和作业入口统一使用 `python -m script.dataset.example`，从仓库根解析模块，避免在脚本中注入路径。

## 服务器 GitHub HTTPS 不稳定

Guqq 登录节点可能能解析 GitHub，却在 `git pull` 时出现 GnuTLS `recv error (-110)` 或长时间无响应。发生网络连接问题时，先在服务器运行 `bash net.sh`，再重试 HTTPS `git pull`；不要切换到 GitHub SSH transport，因为该服务器没有对应的 GitHub public key。若仍无法同步，则暂停需要新源码的服务器任务并保留已生成数据。不得用 `scp` 覆盖服务器受 Git 管理源码，因为服务器源码只能通过 `git pull` 同步。

## OT active support 与 artifact 坐标

零质量 token 必须在 `Diag(a)^-1` 前移出 OT active support，但 artifact 仍需保留原 tokenizer 方向。实现将正质量 source/target 压缩为连续矩阵坐标，同时保存唯一的 `source_token_ids`/`target_token_ids` 映射；候选边也使用压缩坐标结构化保存。不得假设压缩坐标等于原 token ID，也不得对零质量列做条件化除法。

## 跨词表 causal shift 与生成边界

source 位置 `t` 的 logits 预测下一 token，因此等长 virtual prompt 的首个有效位置必须由 receiver 原生起始 token embedding 注入，其余位置使用前一有效 source logits；padding 位置不能参与 transport 时序。source 与 receiver 的 token ID 空间不同，生成结果不得把 source prompt IDs 与 receiver token IDs 拼成一条伪序列；wrapper 只返回 receiver 新生成 token，receiver-only 基线则独立直通 receiver 原生 `generate`。模型并行时 source logits、receiver embedding 和 receiver 输出 logits 可能位于不同设备，索引与 transport 前必须显式对齐设备。

## GPU 分段计时与 padding 统计

CUDA kernel 异步执行，source、transport、receiver prefill 和 decode 的阶段边界若不显式 `synchronize`，计时会被错误归入后续阶段；CPU 路径不应伪造显存峰值。transport 的 retained/dropped mass 张量覆盖 batch 的物理 shape，但 smoke 汇总只能选择 attention mask 中的有效位置，否则 padding logits 会污染近似质量统计。

## 稀疏 OT 必须同时覆盖两侧 support

逐 source 的 special/exact/span/ANN 优先级只能保证每个正质量 source 有出边，不能保证每个正质量 target 有入边；尤其 source 存在未被语料实际使用的 exact target token 时，会遮蔽语料中真实出现的细粒度 target span。构图完成后必须对缺失 target 做反向 exact-byte 与 observed-span rescue，再执行两侧 support 和连通分量质量检查；没有安全证据时应失败，不能任意连边伪造可行性。

## Windows pytest 临时目录权限

沙箱内外混合运行 pytest/格式化工具后，系统 `%TEMP%/pytest-of-<user>` 可能出现 ACL 拒绝，导致所有使用 `tmp_path` 的测试在 setup 阶段统一失败。这不是业务断言失败；应在仓库忽略的 `local/test-tmp/` 下先创建父目录，再用全新的 `--basetemp` 完整重跑，不能通过跳过 `tmp_path` 用例降低测试范围。MSYS Bash 的启动提示还可能混入 stdout，跨平台路径测试应只解析 `cygpath` 的最后一个非空输出行并显式指定解码策略。

## ANN 是 OT 图增广而非仅缺边 fallback

仅在 source 没有 exact/span 时调用 ANN，虽然逐列有边，却会保留大量孤立 exact 分量并阻止 target-only 词获得覆盖，无法满足 Sinkhorn 的连通分量质量约束。ANN 应作为所有 ordinary source 的低优先级增广候选：pair 与 exact/span 重合时保留已有高优先级证据，其余 ANN 边用于连接分量；候选生成还必须做 source→target 与 target→source 双向 top-k，才能显式保证两侧 support。

## 冻结语料必须复现 adapter 的取样阶段

“确定性 500k”不等于可以从全量数据另做 seeded hash top-k。现有 `OpenHermesChatDataset` 先执行 `select(range(num_samples))`，再应用 token-length filter；若物化器改用全量 hash 抽样，即使 revision、seed 和数量都已记录，也会静默更换 C2C 训练语料。基础物化必须保存 pinned split 的相同 source prefix，并把“过滤尚未应用”写入 provenance；seed 只用于之后基于稳定 canonical ID 的 99/1 划分。消费层若应用长度过滤，C2C 与 STT 必须复用同一规则和过滤后的 manifest ID，不能各自重新随机切分。
