# 经验记录

## Artifact 数值审计

稀疏 artifact 的边际与列和容差必须至少覆盖其存储 dtype 的机器精度；用固定的 float64 级容差审计 float32 数据会误拒绝合法 artifact。实现采用 `max(配置容差, 10 * dtype epsilon)`，同时仍拒绝非有限值和真实的归一化偏差。
