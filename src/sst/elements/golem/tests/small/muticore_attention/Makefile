.PHONY: all clean test scale base

BASE_TEST_DIR := ../mvm_noc_int_array
ARCH ?= riscv64

RISCV_MUSL_TOOLCHAIN_BIN ?= /local/scratch/src/riscv64-linux-musl-cross/bin
CXX := $(RISCV_MUSL_TOOLCHAIN_BIN)/riscv64-linux-musl-g++
ATTENTION_EXE := $(ARCH)/fused_attention

all: scale

base:
	$(MAKE) -C $(BASE_TEST_DIR) ARCH=$(ARCH)

scale: $(ATTENTION_EXE)

$(ATTENTION_EXE): golem_attention_runtime.cpp golem_attention_runtime.h
	mkdir -p $(ARCH)
	$(CXX) $(CFLAGS) -DGOLEM_ATTENTION_SCALE=1 \
		-DGOLEM_GLOBAL_STRIDE_BYTES=2097152 \
		-DGOLEM_ATTENTION_QUERIES=256 -DGOLEM_ATTENTION_KEYS=1024 \
		-DGOLEM_ATTENTION_HEAD_DIM=128 -static -no-pie -O2 \
		-o $@ golem_attention_runtime.cpp

clean:
	rm -f $(ATTENTION_EXE)

test:
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s . -p 'test_*.py'
