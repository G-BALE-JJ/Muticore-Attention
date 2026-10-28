# prefill 测试

Synthetic token embedding -> decoder layers -> final RMSNorm -> last-token LM head。公共架构为 `LLM/src/sst/elements/golem`。

```bash
cd /data/jjgong/LLM/test/prefill
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

使用 `./run.sh --run -- --sim-mode full-timing` 运行正式规模时序仿真；
`--smoke -- --sim-mode full-timing` 可运行小规模用例。默认仍为 `full-functional`。
Timing 模式只跳过阵列 GEMM 的数值 MAC（并默认跳过数值打包/解包），
保留 SFU 运算、计算延迟、DMA/NoC/HBM、任务依赖及最终输出 ACK 检查。
依赖 GEMM 结果的数值正确性校验标为 `SKIPPED`，prefill 不生成真实 logits 或 next token。
Timing 模式默认关闭额外的 CPU wait / scheduler 快捷路径，便于隔离比较数值 MAC 的影响。

原生精度与多实例矩阵算子是显式实验入口，例如：

```bash
./run.sh --smoke --native-operator qkv --sim-mode full-functional -- \
  --precision int8 --instances 2 --panel-mapping reuse-striped --replicate-b
./run.sh --smoke --native-operator qkv --sim-mode full-timing -- --precision int8
python3 ../run_native_llm_regression.py --artifact-root /tmp/native-llm-matrix --cpu-set 4-7
```

可选算子为 `qkv/wo/gate/up/down/lm_head`，精度为
`int4/int8/fp16/bf16/tf32/fp32/fp8e4m3/fp8e5m2`。默认通过固定 RISC-V guest、版本化描述符、生产 TE、
调度器及真实 NoC/HBM 执行，保持 16 workers、每 worker 64 阵列及四个数据 HBM 节点。
`--instances 1|2|4` 划分已有 worker 和 HBM，使用不同输入；`--panel-mapping`
和 `--replicate-b` 控制放置。完整 prefill 默认仍采用原有 FP16 数值契约。

该入口对单个矩阵算子使用 INT32/FP32 输出和原生累加语义，不运行完整 decoder，
也不输出 next token。小规模 WO/down 请加 `--reuse 1`，保证至少有 16 个宏任务。
所有维度补齐至 64，补齐计算计入 cycles；尤其 LM-head 的逻辑单行补齐成 64 行，
不能用其 cycles 直接比较现有优化过的 last-token LM-head。

默认验证输入为精确可表示的 -2..2；INT4/INT8、FP16/FP32 可通过
`--input-a a.npy --input-b b.npy` 提供与逻辑形状匹配的外部原始张量（单实例）。
整数输入须在原生位宽范围内；FP16 先量化到半精度。其余模式当前验证编码器只接受
-2..2，不包含模型权重量化、scale 或 tokenizer 支持。功能模式逐项核对输出，
timing 模式保留传输、计算完成事件及最后写回 ACK，数值检查标为 SKIPPED。

固定 guest 只在 runtime/ABI 源码变化时编译；修改算子、形状、精度或放置时，
主机重新生成描述符与输入数据，复用同一 guest 二进制。`--abi-selftest` 可额外检查
描述符拒绝和 submit/wait token 生命周期；`--launcher legacy` 可切回之前的实验入口。
具体 ABI、单槽限制和完成边界见 [runtime 文档](../../software/native/README.md)。
