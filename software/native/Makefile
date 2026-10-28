CXX := /local/scratch/src/riscv64-linux-musl-cross/bin/riscv64-linux-musl-g++
ROOT := $(abspath ../..)
OUTPUT ?= $(ROOT)/build/native-guest/launcher
SOURCES := apps/launcher.cpp runtime/runtime.cpp
HEADERS := include/runtime.h $(ROOT)/src/sst/elements/golem/isa/nativeDescriptor.h $(ROOT)/src/sst/elements/golem/isa/precision.h
.PHONY: all
all: $(OUTPUT)
$(OUTPUT): $(SOURCES) $(HEADERS)
	mkdir -p $(dir $@)
	$(CXX) -std=c++17 -O2 -static -no-pie -Iinclude -I$(ROOT)/src $(SOURCES) -o $@
