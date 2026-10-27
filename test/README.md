# 测试入口

四个阶段共用 `LLM/src/sst/elements/` 的公共架构：

| 阶段 | 入口 | 阶段内实验 |
| --- | --- | --- |
| GEMM | `GEMM/run.sh` | `GEMM/experiments/` |
| Attention | `Attention/run.sh` | `Attention/experiments/` |
| GQA | `GQA/run.sh` | `GQA/experiments/` |
| prefill | `prefill/run.sh` | `prefill/experiments/` |

进入阶段目录运行 `./run.sh --check`；查看配置用 `--show-config`。
完成本工作区构建后，使用 `--smoke` 或 `--run` 执行仿真。
输出位于各阶段 `results/`，每次调用单独建目录。

共享 guest、拓扑及验证位于 `golem/`；共享 prefill 驱动及技术记录位于
`attention_cluster_8qk_8pv/`。旧方案在 `archive/`，历史原始结果在 `history/`。
历史张量清理、日志解压方式及审计清单见
[清理记录](maintenance/cleanup_2026-10-27/README.md)。

需要修改架构或数据流时，使用 `LLM/scripts/new_experiment.sh exp-名称`
建立分支工作区。完整构建、回归和合并流程见 [项目说明](../README.md)。
