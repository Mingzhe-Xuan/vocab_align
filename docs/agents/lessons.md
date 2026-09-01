# 经验记录

## Artifact 数值审计

稀疏 artifact 的边际与列和容差必须至少覆盖其存储 dtype 的机器精度；用固定的 float64 级容差审计 float32 数据会误拒绝合法 artifact。实现采用 `max(配置容差, 10 * dtype epsilon)`，同时仍拒绝非有限值和真实的归一化偏差。

## 项目脚本调用

未以 editable package 安装仓库时，直接执行 `python script/dataset/example.py` 只会把脚本目录加入 `sys.path`，可能无法导入顶层 `rosetta`。项目文档和作业入口统一使用 `python -m script.dataset.example`，从仓库根解析模块，避免在脚本中注入路径。
