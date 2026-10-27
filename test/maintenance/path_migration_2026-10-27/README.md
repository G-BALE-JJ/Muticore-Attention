# 2026-10-27 路径迁移审计

迁移映射、使用方式与验收说明见
[路径迁移记录](../../../docs/PATH_MIGRATION_2026-10-27.md)。

本地审计文件：

- `before.json`：原两个工作区的 HEAD、引用、完整 Git 状态与本地 diff。
- `primary_before.patch`、`latest_before.patch`：迁移前本地改动补丁。
- `primary_inventory.json`：59,904 个保留文件的原路径与 inode/大小/修改时间。
- `storage_result.json`：同文件系统迁移及原 Git 状态核对结果。
- `path_edits.json`：公共目录与最新原实验区的路径适配文件清单。
- `latest_changes_verified.json`：最新实验区仅发生预期路径适配，其余已有改动保持。
- `validation.json`：主机测试、新分支路径和历史 manifest 的核对结果。
- `retired_build_manifest.json`、`retired_build_result.json`：废弃构建缓存删除清单。

旧主工作区的独立构建缓存已删除，回收 **10,048,188,416 字节（9.36 GiB）**。
删除前从缓存中提取了与三个现存源码树不同的 41 份文本源码/配置文件，
共 991,580 字节，保存在本地 `retired_build_sources/`，便于追溯缓存中的独有版本。
原源码、安装库和历史结果保留；此回收量独立于此前的 772.33 GiB 清理。

大型审计数据不纳入公共 Git；历史 JSON、日志和源码快照保持原始证据。
当前公共 Git 与原实验 Git 独立，原公共元数据保存在 `.git/legacy-experiments.git/`。
原主工作区已归档到 `test/history/legacy-primary/`，顶层旧目录已移除。
