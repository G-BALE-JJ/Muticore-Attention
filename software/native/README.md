# Fixed native LLM operator runtime

`make -C software/native` builds one static RISC-V launcher under
`build/native-guest/launcher`. Its dependencies are runtime sources and the
versioned ABI, not matrix dimensions, precision, inputs or placement.
`CXX` and `OUTPUT` may be overridden as make variables.

The public native matrix operator runner uses this launcher by default:

```bash
python3 test/run.py prefill --smoke --native-operator qkv --sim-mode full-functional -- \
  --precision int8 --instances 2 --panel-mapping reuse-striped --replicate-b
python3 test/run.py prefill --smoke --native-operator down --sim-mode full-timing -- \
  --precision fp16 --reuse 1
```

`--launcher legacy` retains the previous experimental GEMM-guest/TE-parameter
bridge. Default full decoder execution and its legacy descriptor path remain
separate. Native operator numerical, padding and input-encoding limits remain
those in `test/prefill/README.md`.

The host packs sixteen commands into `native_commands.bin` and preloads the
bundle into a reserved 1 MiB region at the top of existing Node0 memory.
Vanadis reserves the physical pages and maps this region read-only at
`0x50000000` in each process. All commands use the same physical TU/GM/NoC/HBM
resources. A launcher uses `sched_getcpu()` to select its physical worker's command
(process argument order is not physical-core order in Vanadis), checks bundle and
command integrity, relocates the descriptor start timestamp, copies it into
its reserved existing GM metadata area, and calls the runtime API. No guest
file reads, TileLang compiler or TileBin loader are required.

`GolemNative::submit(address, bytes)` uses the existing custom-0 funct7 0x15
with a nonzero byte count, and returns a positive per-core token or negative
status. `GolemNative::wait(token)` uses funct7 0x16 and returns the completion
status after the TE becomes idle following its final C write ACK. Submit has
release ordering and wait has acquire ordering. One token owns the native
slot until consumed; another submit returns QueueFull. Unknown positive or consumed
tokens return BadToken; zero is reserved for the historical wait encoding. Tokens are per-core, not inter-core dependency tokens.
Operator-wide completion in this one-operator experiment is established by
successful completion of all sixteen workers, checked in the host report.

The historical rs2=0 WCP submit and rs1=0 wait keep their existing semantics.
The new format is `llm.native-bundle-v1`; it is not donor TileBin wire format.
`nativeDescriptor.h` defines the 128-byte command header, four variable-length
node maps and checksums. Version 1 supports GEMM only and the existing fixed
machine: four managers, sixteen workers, 64x64 arrays, 2 MiB worker GM and four
128 MiB data nodes. Unsupported versions/kernel kinds and invalid descriptors
are rejected before TE launch. Each command is bounded to 32 KiB and the
complete bundle to the reserved region. The checksum detects corruption; it
is not an authentication mechanism.

`--abi-selftest` exercises malformed submission, bad token, occupied-slot and
consumed-token behavior through real RISC-V instructions. This adds guest
work and is separate from performance comparisons.

Extend this runtime with a new descriptor version for a new tensor/arithmetic
contract. A future multi-operator runtime must provide operator-wide memory
visibility or explicit producer/consumer dependencies before launching a
consumer. Per-core wait alone does not establish an inter-core barrier.
