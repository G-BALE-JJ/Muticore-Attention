# 2026-10-27 路径迁移审计

迁移映射、使用方式与验收说明见
[路径迁移记录](../../../docs/PATH_MIGRATION_2026-10-27.md)。

本地审计文件：

- `before.json`：原两个工作区的 HEAD、引用、完整 Git 状态与本地 diff。
- `primary_before.patch`、`latest_before.patch`：迁移前本地改动补丁。
- `primary_inventory.json`：59,904 个保留文件的原路径与 inode/大小/修改时间。
- `storage_result.json`：同文件系统迁移及原 Git 状态核对结果。
- `path_edits.json`：公共目录与最新原实验区的路径适配文件清单。

大型审计数据不纳入公共 Git；历史 JSON、日志和源码快照保持原始证据。
当前公共 Git 与原实验 Git 独立，原公共元数据保存在 `.git/legacy-experiments.git/`。
原主工作区已归档到 `test/history/legacy-primary/`，顶层旧目录已移除。
