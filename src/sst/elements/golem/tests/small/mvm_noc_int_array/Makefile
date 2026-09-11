CC=$(ARCH)-linux-musl-gcc
RISCV_MUSL_TOOLCHAIN_BIN ?= /local/scratch/src/riscv64-linux-musl-cross/bin
CXX=$(RISCV_MUSL_TOOLCHAIN_BIN)/riscv64-linux-musl-g++

# The bundled musl toolchain defaults to PIE; static RISC-V worker binaries
# must disable it or ld rejects dynamic relocations in read-only segments.
LDFLAGS=-static -no-pie -fopenmp
LDFLAGS_PTHREAD=-static -no-pie -pthread
LDFLAGS_STATIC=-static -no-pie

PROG1=mvm_noc_int_array
PROG3=mvm_single_router
PROG4=mvm_pipeline_mainmem
PROG5=test_atomic_16cores_pthread
PROG6=test_hbm2
PROG7=test_noc_dma
PROG7_OUT?=$(ARCH)/$(PROG7)
PROG8=test_pipeline_mailbox_pthread
PROG9=test_pipeline_mailbox_software
PROG10=test_pipeline_mailbox_enhanced

.PHONY: all clean $(PROG7)

all: $(PROG7)

$(PROG7): $(PROG7_OUT)

$(PROG1) : $(PROG1).cpp
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $(ARCH)/$@ $<

$(PROG3) : $(PROG3).cpp
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $(ARCH)/$@ $<

$(PROG4) : $(PROG4).cpp
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $(ARCH)/$@ $<

$(PROG5) : $(PROG5).cpp
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $(ARCH)/$@ $<

$(PROG6) : $(PROG6).cpp
	$(CXX) $(CFLAGS) $(LDFLAGS_STATIC) -o $(ARCH)/$@ $<

$(PROG7_OUT) : $(PROG7).cpp golem_matmul_runtime.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) $(LDFLAGS_STATIC) -o $@ $(PROG7).cpp golem_matmul_runtime.cpp

$(PROG8) : $(PROG8).cpp
	$(CXX) $(CFLAGS) $(LDFLAGS_STATIC) -o $(ARCH)/$@ $<

$(PROG9) : $(PROG9).cpp
	$(CXX) $(CFLAGS) $(LDFLAGS) $(EXTRA_CFLAGS) -o $(ARCH)/$@ $<

$(PROG10) : $(PROG10).cpp
	$(CXX) $(CFLAGS) $(LDFLAGS_STATIC) $(EXTRA_CFLAGS) -o $(ARCH)/$@ $<

clean:
	rm -f $(PROG7_OUT)
