# GEMM 测试

默认正式测试生成独立 sample A/B 输入，执行 `--verify-c`；输入保存在本次结果目录的
`inputs/`，避免不同实验覆盖共享输入。四阶段汇总还会复核全部 C 元素。

Generic GEMM / WCP; 64x64x64 tiles, 16 workers。公共架构为 `LLM/src/sst/elements/golem`。

```bash
cd /data/jjgong/LLM/test/GEMM
./run.sh --check        # 主机契约、数值与资源检查
./run.sh --show-config  # 查看实际命令
./run.sh --dry-run      # 校验参数；prefill 仅生成调度
./run.sh --smoke        # 小规模 SST 与原有验证
./run.sh --run          # baseline.json 对应正式配置
```

参数通过 `--` 传给原 runner，例如 Attention/GQA 可加 `-- --causal --rope`。
结果默认保存在本目录 `results/<时间>/`，包括命令、工作区、Git 状态和库哈希。
硬件延迟统一报告 cycles，主机运行耗时单独记录。

四个公开测试均接受 `--sim-mode full-functional|full-timing`，可放在 `--` 前后。
例如 `./run.sh --run --sim-mode full-timing` 会去掉默认的 `--verify-c`，
并核实全部 16 个 worker 的阵列数值计算已关闭。显式追加 `-- --verify-c`
仍会被拒绝，因为 timing 结果不具有数值正确性。默认模式为 full-functional；
命令行优先于 `GOLEM_SIM_MODE`，`--` 后的模式优先于外层参数。

跨形状、头分组、层数、padding 和 MPI 的成对回归：
`python3 ../run_timing_regression.py --artifact-root /tmp/llm-timing-matrix --cpu-set 0-3`。
同构建 timing 快捷路径开关对照：
`python3 ../maintenance/mechanism_alignment/validate_timing.py my_timing --cpuset 4`。

阶段内配置/调度实验放在 `experiments/<名称>/`。如果需要修改架构或数据流，
在 `LLM` 下创建独立工作区，用 `--workspace /path/to/workspace` 运行本阶段测试；
每个工作区有独立 `build/` 和 `install/`，通过四阶段检查及同构建成对验证后合并。

共享测试实现位于 `../golem/` 与 `../attention_cluster_8qk_8pv/`，
历史证据位于 `../history/`。这些共享实现只维护一份。
