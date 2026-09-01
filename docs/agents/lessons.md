# 经验记录

## Artifact 数值审计

稀疏 artifact 的边际与列和容差必须至少覆盖其存储 dtype 的机器精度；用固定的 float64 级容差审计 float32 数据会误拒绝合法 artifact。实现采用 `max(配置容差, 10 * dtype epsilon)`，同时仍拒绝非有限值和真实的归一化偏差。

## 项目脚本调用

未以 editable package 安装仓库时，直接执行 `python script/dataset/example.py` 只会把脚本目录加入 `sys.path`，可能无法导入顶层 `rosetta`。项目文档和作业入口统一使用 `python -m script.dataset.example`，从仓库根解析模块，避免在脚本中注入路径。

## 服务器 GitHub HTTPS 不稳定

Guqq 登录节点可能能解析 GitHub，却在 `git pull` 时出现 GnuTLS `recv error (-110)` 或长时间无响应。连续三次同类失败后不得继续盲目重试：优先尝试 GitHub SSH transport；若 SSH transport 也不可用，则暂停需要新源码的服务器任务，保留已生成数据并等待网络恢复。不得用 `scp` 覆盖服务器受 Git 管理源码，因为服务器源码只能通过 `git pull` 同步。
