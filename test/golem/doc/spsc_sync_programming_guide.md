# SPSC 同步编程示例（Golem/NoC/GlobalMemory）

本文对应实现文件：
- [golem/tests/small/mvm_noc_int_array/test_noc_dma.cpp](golem/tests/small/mvm_noc_int_array/test_noc_dma.cpp)

目标：给出一个可复用的多进程核间同步模板，解决“消费者过早读取导致忙等浪费”的问题。

## 1. 核心思想

- `slot`：环形队列中的格子（固定大小数据位）。
- `seq`：每个 `slot` 的序列号，表示该格子是否包含当前轮次的新数据。
- `credit`：消费者回传给生产者的“已消费许可”，用于避免生产者覆盖未消费数据。

本示例采用 `core0 -> core1` 单生产者单消费者（SPSC）：
- `payload` 放在 `core1` 的 Data 区（消费者本地读取）。
- `prod_seq[slot]` 放在 `core1` 的 Mailbox（消费者本地轮询）。
- `cons_seq[slot]` 放在 `core0` 的 Mailbox（生产者本地轮询）。

这样两侧都只轮询本地 GM，减少远端轮询流量。

## 2. 同步协议

### 生产者（core0）
1. 等待 `cons_seq[slot] == tail`（说明该 slot 可写）。
2. `remote_store(payload -> core1.slot)`。
3. `fence` 保证 payload 先于发布可见。
4. 写 `prod_seq[slot] = tail + 1` 到 core1（发布事件）。
5. `tail++`。

### 消费者（core1）
1. 等待 `prod_seq[slot] == head + 1`（说明该 slot 就绪）。
2. `fence` 后读取本地 `payload`。
3. 处理数据。
4. 写 `cons_seq[slot] = head + RING_SIZE` 到 core0（返还 credit）。
5. `head++`。

## 3. 为什么比单 flag 更稳

- 单 `flag` 容易复用时误判新旧数据。
- `seq` 能区分“第几轮”的同一个 slot。
- `credit` 明确控制覆盖边界，避免生产者写爆。

## 4. 关键参数（示例默认）

- `RING_SIZE = 4`
- `NUM_MESSAGES = 8`
- `SLOT_BYTES = 8`
- 角色：`core0` 生产，`core1` 消费

## 5. 扩展方法

- 多阶段推理：将每个阶段拆成多个 `coreA -> coreB` SPSC 链路。
- 大数据块：保持 `seq/credit` 控制不变，`payload` 改为 DMA 传输；DMA 完成后再发布 `prod_seq`。
- 多对一：每个生产者独立一条 SPSC 队列，消费者做轮询仲裁。

## 6. 输出判定

示例运行时你会看到：
- 生产者输出 `[PUSH] msg/slot/publish_seq`
- 消费者输出 `[POP ] msg/slot/payload/expected/OK`

若全为 `OK`，表示同步协议与数据可见性正确。
