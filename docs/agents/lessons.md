# 经验记录

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
