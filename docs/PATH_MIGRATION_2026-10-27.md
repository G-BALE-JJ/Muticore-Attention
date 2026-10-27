# baseline 与旧主工作区路径迁移

公共驱动、guest 和阶段入口已实际位于 `test/`，顶层 `baseline` 与
`Muticore-Attention` 均已移除。此次仅适配路径、Git 元数据与构建准备，
没有修改架构算法、硬件参数、数值契约或已选 prefill 的 guest/library 校验值。

| 原路径 | 当前维护位置 |
| --- | --- |
| `baseline/attention_cluster_8qk_8pv/` | `test/attention_cluster_8qk_8pv/` |
| `src/sst/elements/golem/tests/` | `test/golem/`；原路径仅保留原生构建兼容链接 |
| `Muticore-Attention/deps/` | `deps/`，实体目录 |
| `Muticore-Attention/` 的源码、安装及历史结果 | `test/history/legacy-primary/` |
| 原两个工作区的 Git 公共元数据 | `.git/legacy-experiments.git/` |
| `Muticore-Attention-all-gemm/` | 继续作为最新实验区；Git 和依赖指向新位置 |

旧主工作区通过同文件系统重命名归档，没有复制第二份历史数据。迁移时逐项核对
59,904 个非构建/依赖/Git 文件的 inode、大小与修改时间，全部一致。
原主工作区与最新实验区的 HEAD、引用和迁移前本地修改均已保存；
最新实验区只适配 11 份分析/驱动脚本中的旧主目录引用，不提交或覆盖原有架构改动。
原 Git 仓库已转换为独立 bare 公共仓库，两个原工作区都注册为 linked worktree，
因此删除旧顶层目录后，最新实验区的 `git status`、`git diff` 与分支历史仍正常使用。

## 日常入口

```bash
cd /data/jjgong/LLM
test/GEMM/run.sh --run
test/Attention/run.sh --run
test/GQA/run.sh --run
test/prefill/run.sh --run
scripts/new_experiment.sh exp-feature-name
```

四个入口直接使用 `test/golem/` 和 `test/attention_cluster_8qk_8pv/`，
输出进入各阶段自己的 `results/`。显式传入旧实验工作区时，runner 可以读取该工作区
自身的原生测试布局；公共目录不再依赖顶层 baseline。
新功能工作区共享 `deps/` 与 `test/history/`，有独立构建、安装和结果目录，
不会重新创建旧 `Muticore-Attention` 或 baseline 别名。

历史 JSON、日志、源码快照与测量证明不批量改写。
`test/attention_cluster_8qk_8pv/project_paths.py` 按完整路径段解析旧位置，
供输入 manifest 和验证证明读取使用；功能工作区也能解析公共目录的旧绝对路径。
这不会改变证据哈希校验，已退役张量仍须按清理记录重新生成。
历史分析脚本若直接读取压缩前日志，仍需先解压；迁移不代表重新验收所有历史实验。

## 验收与证据

路径迁移前的补丁、Git 状态、文件清单与存储核对保存在
`test/maintenance/path_migration_2026-10-27/`，大型清单仅保留本地。
44 个公共架构源文件及最新实验区对应文件的 SHA256 均与原快照一致。
共享驱动 284 项测试与四阶段主机检查通过；移除旧目录后重新执行四阶段正式仿真，
结果统一写入各阶段 `results/path_migration_20261027/`。
完整仿真与新工作区验收记录在本轮完成后补齐。
