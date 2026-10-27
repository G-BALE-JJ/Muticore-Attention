# DMA 关键修改汇总

## 事件类型与回包
- 增加 `NetworkDataEvent::DMA_WRITE_COMPLETE`，用于 DMA 写完成回包。
- 修正 DMA 写回包 `size_in_bits` 计算。

## GlobalMemory 侧（DMA 发起与完成）
- `dma_write_to_host()`：
  - 以 64B burst 拆分发送 `DMA_WRITE`。
  - `dma_pending` 以 `host_addr` 记录 `WRITE_TO_HOST`。
  - 增加发送端数据打印（首 64B）。
- `dma_read_from_host_to_globalmem()`：
  - 以 64B burst 拆分发送 `READ`（`returnAddr` 携带 GM 目标）。
  - `dma_pending` 以 GM 目标地址记录 `READ_TO_GM`。
- `handle_receives()`：
  - 增加 `DMA_READ_COMPLETE` 数据打印（首 64B）。
  - 处理 `DMA_WRITE_COMPLETE`，对写回包完成回调。

## MemNIC 桥接（NetworkDataEvent → MemEvent）
- 在 MemNIC 接收路径将 `NetworkDataEvent` 转为 `MemEvent`：
  - `READ` → `GetS`（标记 `F_NONCACHEABLE`）。
  - `DMA_WRITE` → `Write`（标记 `F_NONCACHEABLE`）。
- 在 MemNIC 发送路径将 `MemEvent` 响应转回 `DMA_READ_COMPLETE`/`DMA_WRITE_COMPLETE`。
- 桥接时将 GM 源端点加入 `reachableNames`，避免 DC 回包不可达。

## 路由与目的端
- DMA 访问主存优先发往 `dirctrl_N` 或 `dirctrl_N.highlink`（MemNIC endpoint），找不到则回退到路由器编号。

## 日志验证
- `remote_ld` 已完成：日志显示 `DMA_READ_COMPLETE` 返回并写入 GM。

## 相关文件
- `src/sst/elements/golem/globalmemory/globalmemory.h`
- `src/sst/elements/golem/globalmemory/globalmemory.cc`
- `src/sst/elements/memHierarchy/memNICBase.h`
- `src/sst/elements/memHierarchy/memNIC.cc`
