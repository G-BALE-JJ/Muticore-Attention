.PHONY: all clean test scale base rmsnorm batch-release clean-startup

BASE_TEST_DIR := ../mvm_noc_int_array
ARCH ?= riscv64

RISCV_MUSL_TOOLCHAIN_BIN ?= /local/scratch/src/riscv64-linux-musl-cross/bin
CXX := $(RISCV_MUSL_TOOLCHAIN_BIN)/riscv64-linux-musl-g++
ATTENTION_EXE := $(ARCH)/fused_attention
RMSNORM_EXE := $(ARCH)/rmsnorm

all: scale

.PHONY: decoder-micro
decoder-micro: $(ARCH)/decoder-micro/ffn

$(ARCH)/decoder-micro/ffn: golem_decoder_micro.cpp golem_attention_runtime.h golem_decoder_runtime.h ../../../../src/sst/elements/golem/attention/decoderMemoryLayout.h ../../../../src/sst/elements/golem/attention/decoderOpAbi.h
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -static -no-pie -O2 -o $@ $<

base:
	$(MAKE) -C $(BASE_TEST_DIR) ARCH=$(ARCH)

scale: $(ATTENTION_EXE)

batch-release:
	$(MAKE) ATTENTION_EXE=$(ARCH)/batch_release/fused_attention CFLAGS="$(CFLAGS) -DGOLEM_PROJECTION_BATCH_RELEASE=1" scale

clean-startup:
	$(MAKE) ATTENTION_EXE=$(ARCH)/clean_startup/fused_attention CFLAGS="$(CFLAGS) -DGOLEM_GUEST_STARTUP_DIAGNOSTICS=0" scale

rmsnorm: $(RMSNORM_EXE)

$(RMSNORM_EXE): golem_rmsnorm_runtime.cpp
	mkdir -p $(ARCH)
	$(CXX) $(CFLAGS) -static -no-pie -O2 -o $@ $<

$(ATTENTION_EXE): golem_attention_runtime.cpp golem_attention_runtime.h golem_projection_runtime.h golem_decoder_runtime.h golem_tail_schedule.h ../../../../src/sst/elements/golem/attention/decoderMemoryLayout.h ../../../../src/sst/elements/golem/attention/decoderOpAbi.h ../../../../src/sst/elements/golem/attention/projectionJobAbi.h
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -DGOLEM_ATTENTION_SCALE=1 \
		-DGOLEM_GLOBAL_STRIDE_BYTES=2097152 \
		-DGOLEM_ATTENTION_QUERIES=256 -DGOLEM_ATTENTION_KEYS=1024 \
		-DGOLEM_ATTENTION_HEAD_DIM=128 -static -no-pie -O2 \
		-o $@ golem_attention_runtime.cpp

clean:
	rm -f $(ATTENTION_EXE) $(RMSNORM_EXE) $(ARCH)/batch_release/fused_attention $(ARCH)/clean_startup/fused_attention

test:
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s . -p 'test_*.py'

# Opt-in checkpoints; legacy scale binary and baseline hardware stay unchanged.
.PHONY: decoder-wo decoder-residual decoder-ffn decoder-layer
decoder-wo decoder-residual decoder-ffn decoder-layer:
	$(MAKE) ATTENTION_EXE=riscv64/$@/fused_attention CFLAGS="$(CFLAGS) -DGOLEM_GUEST_STARTUP_DIAGNOSTICS=0 -DGOLEM_DECODER_STAGE=$(if $(filter decoder-wo,$@),1,$(if $(filter decoder-residual,$@),2,$(if $(filter decoder-ffn,$@),3,4)))" scale

.PHONY: decoder-two-layer
decoder-two-layer:
	$(MAKE) ATTENTION_EXE=riscv64/$@/fused_attention CFLAGS="$(CFLAGS) -DGOLEM_GUEST_STARTUP_DIAGNOSTICS=0 -DGOLEM_DECODER_STAGE=4 -DGOLEM_DECODER_LAYERS=2" scale

.PHONY: decoder-final-norm
decoder-final-norm:
	$(MAKE) ATTENTION_EXE=riscv64/$@/fused_attention CFLAGS="$(CFLAGS) -DGOLEM_GUEST_STARTUP_DIAGNOSTICS=0 -DGOLEM_DECODER_STAGE=4 -DGOLEM_DECODER_LAYERS=2 -DGOLEM_DECODER_FINAL_NORM=1" scale

.PHONY: decoder-lm-head
decoder-lm-head:
	$(MAKE) ATTENTION_EXE=riscv64/$@/fused_attention CFLAGS="$(CFLAGS) -DGOLEM_GUEST_STARTUP_DIAGNOSTICS=0 -DGOLEM_DECODER_STAGE=4 -DGOLEM_DECODER_LAYERS=2 -DGOLEM_DECODER_FINAL_NORM=1 -DGOLEM_DECODER_LM_HEAD=1" scale

.PHONY: decoder-embedding
decoder-embedding:
	$(MAKE) ATTENTION_EXE=riscv64/$@/fused_attention CFLAGS="$(CFLAGS) -DGOLEM_GUEST_STARTUP_DIAGNOSTICS=0 -DGOLEM_DECODER_STAGE=4 -DGOLEM_DECODER_LAYERS=2 -DGOLEM_DECODER_FINAL_NORM=1 -DGOLEM_DECODER_LM_HEAD=1 -DGOLEM_DECODER_EMBEDDING=1" scale

.PHONY: decoder-embedding-tail decoder-lm-head-tail
decoder-embedding-tail decoder-lm-head-tail:
	$(MAKE) ATTENTION_EXE=riscv64/$@/fused_attention CFLAGS="$(CFLAGS) -DGOLEM_GUEST_STARTUP_DIAGNOSTICS=0 -DGOLEM_DECODER_STAGE=4 -DGOLEM_DECODER_LAYERS=2 -DGOLEM_DECODER_FINAL_NORM=1 -DGOLEM_DECODER_LM_HEAD=1 -DGOLEM_DECODER_EMBEDDING=$(if $(filter decoder-embedding-tail,$@),1,0) -DGOLEM_DECODER_TAIL_EXPERIMENT=1" scale

.PHONY: decoder-lm-micro
decoder-lm-micro: $(ARCH)/decoder-lm-micro/lm

$(ARCH)/decoder-lm-micro/lm: golem_decoder_lm_micro.cpp golem_attention_runtime.h golem_decoder_runtime.h ../../../../src/sst/elements/golem/attention/decoderMemoryLayout.h ../../../../src/sst/elements/golem/attention/decoderOpAbi.h
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -static -no-pie -O2 -o $@ $<
