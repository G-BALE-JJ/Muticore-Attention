# 六步优化执行状态（已纠正）

旧版本把步骤2–6的代码检查写成了执行结果，这是不准确的。当前状态如下，不能将未实施的方案记为“已经优化但无效”。

| 原步骤 | 实际完成内容 | 当前状态 |
|---|---|---|
| 1. 32-array WS | 原型、数值回归、小形状受控WS on/off比较 | fused packed配置正确但E2E变慢；unfused配置错误；用户决定舍弃，代码已移除 |
| 2. worker启动 | 阅读release代码、检查job起始时间 | 未做受控修改；启动guard归因不成立，release循环中打印是待验证线索 |
| 3. 命令批处理 | 阅读WCP及array-buffer API | 未实现/未A/B；单port必需服务与软件派发空档需区分 |
| 4. Norm→Projection | 检查GM布局及SFU生命周期 | 未实现/未A/B；既有256行输入区域可能重分配，不需要据此增加GM容量 |
| 5. per-head Projection→Attention | 检查completion与worker复用 | 未实现/未A/B；必须连同producer顺序和resource ownership设计 |
| 6. 直接消费者 | 检查HBM panel及descriptor | 未实现/未A/B，低优先级 |

旧77,966→79,911的WS比较改变了pipeline等条件，不能证明WS单变量代价。
严格隔离的小形状实验：S1024为91,109→91,828；S2048为201,067→201,755，详见 [隔离记录](WEIGHT_STATIONARY_ISOLATION_2026-10-11.md)。

后续执行次序以 [舍弃WS后的重分析](LATENCY_REANALYSIS_WITHOUT_WEIGHT_STATIONARY_2026-10-11.md) 为准。它优先处理64-array同一行组下一input pair预取与GM本地读取流水，而非继续WS。
