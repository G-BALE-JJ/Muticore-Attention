# GQA 测试

Grouped-query attention; 4 Q heads share 2 KV heads。公共架构为 `LLM/src/sst/elements/golem`。

```bash
cd /data/jjgong/LLM/test/GQA
./run.sh --check        # 主机契约、数值与资源检查
./run.sh --show-config  # 查看实际命令
./run.sh --dry-run      # 校验参数；prefill 仅生成调度
./run.sh --smoke        # 小规模 SST 与原有验证
./run.sh --run          # baseline.json 对应正式配置
```

参数通过 `--` 传给原 runner，例如 Attention/GQA 可加 `-- --causal --rope`。
结果默认保存在本目录 `results/<时间>/`，包括命令、工作区、Git 状态和库哈希。
硬件延迟统一报告 cycles，主机运行耗时单独记录。

阶段内配置/调度实验放在 `experiments/<名称>/`。如果需要修改架构或数据流，
在 `LLM` 下创建独立工作区，用 `--workspace /path/to/workspace` 运行本阶段测试；
每个工作区有独立 `build/` 和 `install/`，通过四阶段检查及同构建成对验证后合并。

共享测试实现位于 `../golem/` 与 `../attention_cluster_8qk_8pv/`，
历史证据位于 `../history/`。这些共享实现只维护一份。
