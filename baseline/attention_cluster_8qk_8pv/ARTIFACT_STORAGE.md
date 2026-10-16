# 仿真产物保留策略

端到端 `run_sst.sh` / `run_decoder_stack.py`，以及 FFN 和 LM head 局部 runner
已接入 `artifact_storage.py`。这些操作只影响宿主文件，不改变模拟 GM/HBM
容量、带宽、数值计算或 cycle。新脚本在源码树生效；从 build 副本启动前需
刷新副本，避免继续执行旧脚本。

默认 `GOLEM_ARTIFACT_POLICY=compact`：

- 输入生成前检查目标文件系统至少有 24 GiB 剩余；每个局部实验开始前再次检查。
  外置 HBM backing 目录也单独检查。可用 `GOLEM_ARTIFACT_MIN_FREE_GIB` 调整。
- 仿真和入口规定的验证全部成功后，从 MPI rank stdout/stderr 按合并规则
  重建 SHA256；与 main/resources/causal 全部一致才删除重复 rank 日志和
  `.mpi-console-raw`。单 rank 的 `.raw` 同样核对后删除。
- 失败实验保留原始日志；日志不匹配时跳过日志删除。
- 对本实验 HBM `.bin`、顶层 `.bin` / `.npy` 回收全零磁盘块，逻辑长度和
  读取内容不变。保留规范化日志、配置、验证结果、来源哈希和张量。
- 操作记录在每个实验的 `artifact_cleanup.json`。清理时间不计入 SST wall。
  端到端 driver wall 包含成功后的清理时间。

调试时保留完整原始产物：

```bash
GOLEM_ARTIFACT_POLICY=full python3 baseline/attention_cluster_8qk_8pv/run_decoder_micro.py \
  --artifact-root results/my_debug_case
```

实验已完成并归档、确定无需再从 HBM 重新校验数值时，可显式删除该实验的
HBM 和顶层 `.bin`。必须指定单个已通过的 case，不能指定一组实验的父目录：

```bash
python3 baseline/attention_cluster_8qk_8pv/artifact_storage.py finalize \
  results/my_experiment/layout2 --discard-tensors
```

删除前写入文件大小和 SHA256；保留 `.npy`、规范化资源轨迹和共享 `inputs/`。
此操作没有设为自动默认，因为删除 HBM 输出后不能执行原来的完整数值复核。
只在仿真及验证进程全部结束后手动运行，不要与同一 case 的验证并发执行。

24 GiB 是启动门槛，不是本次运行占用上限，也没有预留给并发作业的空间。
长实验、并发实验应提高门槛并合理限制并发；默认保留的有效结果仍会随实验
累积，阶段验收后应归档并显式清理不再需要的张量。

`prepare_local_build.sh` 不复制任意层级的 `results/`、`build/`、`install/`
或自定义构建目标目录，并用 `rsync --sparse` 保留稀疏文件。刷新不会自动删除
旧副本，避免误删现有文件；本次遗留副本已单独核对并清理。

2026-10-16 清理记录位于 `results/disk_audit_20261016/`。此前删除约 769 GiB
build 实验副本；本次进一步删除约 754 MiB 的嵌套旧副本及约 513 MiB 的旧
smoke 产物，差异源码和小型记录已打包。build 剩余约 9.4 GiB，主要是有效的
对象文件、共享库及当前源码副本，予以保留。构建与安装的 golem/memHierarchy
库哈希在删除前后相同。
