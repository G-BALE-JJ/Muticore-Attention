# LLM 架构与四阶段测试

公共架构源码位于 `src/sst/elements/`，本项目的基线与实验测试位于 `test/`。
当前公共版本提取自 `Muticore-Attention-all-gemm` 的实际工作区，包含原有未提交改动；
默认 prefill 配置沿用 2026-10-26 已验证选择。硬件参数与数值契约未因目录整理而改变。

```text
LLM/
├── src/sst/elements/              公共架构源码（Golem、NoC、内存等）
├── config/                       SST 构建配置
├── scripts/                      构建、环境及功能工作区工具
├── test/
│   ├── GEMM/                     通用 GEMM / WCP
│   ├── Attention/                QK → softmax → PV
│   ├── GQA/                      多 Query head 共享 KV
│   ├── prefill/                  embedding → decoder → final norm → LM head
│   ├── golem/                    共享 guest、主机验证、SST 拓扑配置
│   ├── attention_cluster_8qk_8pv/ 共享驱动、审计及历史技术记录
│   ├── archive/                  更早的基线方案
│   ├── history/                  历史工作区与原始结果，只存一份
│   └── maintenance/              磁盘清理清单及核对记录
├── docs/                         项目进度材料与迁移记录
├── build/、install/               本工作区独立构建与安装产物
├── Muticore-Attention-all-gemm/    现有最新架构实验区，保留原分支和本地改动
└── exp-<功能>/                   今后新建的独立架构实验工作区
```

每个阶段都有 `baseline.json`、`run.sh`、`experiments/`；输出进入该阶段的
`results/<时间>/`，不会覆盖历史结果。共享实现只维护一份，四个阶段使用同一公共架构。
SST 上游组件自带测试仍保留其原生布局；本项目 Golem 测试已实际搬入 `test/golem/`。
`baseline/` 与 `Muticore-Attention/` 已移除，驱动直接使用 `test/`，依赖实体位于 `deps/`。
`archive`、`src/.../golem/tests` 与 `results` 仍保留必要的原生构建/历史证据兼容链接。
旧仓库源码与结果归档到 `test/history/legacy-primary/`；原 Git 历史独立保存在
`.git/legacy-experiments.git`，最新实验区无需依赖旧目录。
具体映射与验收见 [路径迁移记录](docs/PATH_MIGRATION_2026-10-27.md)。
移除旧路径后已再次完成四阶段正式回归，结果与迁移前一致，见
[迁移后回归报告](docs/PATH_MIGRATION_REGRESSION_2026-10-27.md)。

## 运行阶段测试

```bash
cd /data/jjgong/LLM/test/GEMM     # 可替换为 Attention、GQA、prefill
./run.sh --check                # 主机契约、数值与资源测试
./run.sh --show-config          # 查看实际运行命令
./run.sh --dry-run              # 参数检查；prefill 只生成调度
./run.sh --smoke                # 小规模 SST 与原有验证
./run.sh --run                  # 当前阶段 baseline.json 对应配置
```

Attention 与 GQA 支持 `-- --causal --rope` 等原 runner 参数。
`prefill --run` 使用 `SELECTED_PREFILL_2026-10-26.json`，保留 guest/library
SHA256 检查及原数值、资源验证；它是合成 workload，不是完整真实模型推理。
`prefill --smoke` 为独立小规模配置，其结果不能替代正式配置验收。
新延迟报告统一使用 **cycles**，宿主运行/验证时间单独标注。

2026-10-27 已完成四阶段正式重跑。完整两层 prefill 为 **6,084,167.75 cycles**，
与已选配置差值为 0；数值、DMA、资源与投影面板审计均通过。
配置、四阶段延迟及完整分解见 [正式回归记录](docs/FORMAL_REGRESSION_2026-10-27.md)。

公共架构重新编译：

```bash
cd /data/jjgong/LLM
scripts/build_and_install_local.sh
source scripts/env_local_install.sh
```

## 新功能开发与合并

仅调整测试参数时，在对应阶段 `experiments/<名称>/` 记录配置、基线引用及结论。
需要修改架构或数据流时，从公共 Git 版本建立新的文件夹与分支：

```bash
cd /data/jjgong/LLM
scripts/new_experiment.sh exp-new-feature
cd exp-new-feature
scripts/build_and_install_local.sh
test/GEMM/run.sh --check
test/Attention/run.sh --check
test/GQA/run.sh --check
test/prefill/run.sh --check
```

也可从公共测试入口指定 `--workspace /data/jjgong/LLM/exp-new-feature`。
各工作区使用独立 `build/`、`install/`，依赖与历史证据通过链接共享。
功能分支完成四阶段回归和相关 SST 数值/资源验证后，使用同构建、同硬件资源、
同数值契约的基线/候选成对端到端证据决定是否采纳，再在公共目录审阅并合并。

现有 `Muticore-Attention-all-gemm` 仍使用原分支；原有架构和本地改动已保留，
本次只对历史数据引用做路径适配，迁移前补丁保存在 `test/maintenance/path_migration_2026-10-27/`。
其架构源文件路径与公共目录一致，可先比较 `src/` 的差异，再将通过验证的
新改动移入公共仓库的功能分支。不要将旧测试目录整体覆盖回公共源码树。

磁盘清理明细见 `test/maintenance/cleanup_2026-10-27/`。
历史结果中的原始单位和测量边界保留原样，新报告遵守 cycles 约定。
