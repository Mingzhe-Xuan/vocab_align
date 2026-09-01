# 经验记录

## Artifact 数值审计

稀疏 artifact 的边际与列和容差必须至少覆盖其存储 dtype 的机器精度；用固定的 float64 级容差审计 float32 数据会误拒绝合法 artifact。实现采用 `max(配置容差, 10 * dtype epsilon)`，同时仍拒绝非有限值和真实的归一化偏差。

## 项目脚本调用

未以 editable package 安装仓库时，直接执行 `python script/dataset/example.py` 只会把脚本目录加入 `sys.path`，可能无法导入顶层 `rosetta`。项目文档和作业入口统一使用 `python -m script.dataset.example`，从仓库根解析模块，避免在脚本中注入路径。

## 服务器 GitHub HTTPS 不稳定

Guqq 登录节点可能能解析 GitHub，却在 `git pull` 时出现 GnuTLS `recv error (-110)` 或长时间无响应。发生网络连接问题时，先在服务器运行 `bash net.sh`，再重试 HTTPS `git pull`；不要切换到 GitHub SSH transport，因为该服务器没有对应的 GitHub public key。若仍无法同步，则暂停需要新源码的服务器任务并保留已生成数据。不得用 `scp` 覆盖服务器受 Git 管理源码，因为服务器源码只能通过 `git pull` 同步。

## OT active support 与 artifact 坐标

零质量 token 必须在 `Diag(a)^-1` 前移出 OT active support，但 artifact 仍需保留原 tokenizer 方向。实现将正质量 source/target 压缩为连续矩阵坐标，同时保存唯一的 `source_token_ids`/`target_token_ids` 映射；候选边也使用压缩坐标结构化保存。不得假设压缩坐标等于原 token ID，也不得对零质量列做条件化除法。
