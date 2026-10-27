#!/usr/bin/env python3
"""Source contracts for the generic GEMM RISC-V toolchain."""

import os
import subprocess
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
RUNNER = TESTS_DIR / "run_noc_dma_pipeline.sh"
MAKEFILE = TESTS_DIR / "small" / "mvm_noc_int_array" / "Makefile"
TOOLCHAIN_BIN = "/local/scratch/src/riscv64-linux-musl-cross/bin"
COMPILER = "riscv64-linux-musl-g++"


class RiscvToolchainContractTest(unittest.TestCase):
    def test_generic_gemm_runner_uses_overridable_absolute_sst_binary(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn(
            'SST_CORE_HOME="${SST_CORE_HOME:-/local/sstcore}"',
            source,
        )
        self.assertIn(
            'REAL_SST_BIN="${REAL_SST_BIN:-$SST_CORE_HOME/bin/sst}"',
            source,
        )
        self.assertIn('SST_BASE_CMD=("$REAL_SST_BIN")', source)
        self.assertNotIn("SST_CMD=(sst)", source)

    def test_generic_gemm_runner_exports_default_toolchain_to_make(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn(
            f'RISCV_MUSL_TOOLCHAIN_BIN="${{RISCV_MUSL_TOOLCHAIN_BIN:-{TOOLCHAIN_BIN}}}"',
            source,
        )
        self.assertIn("export RISCV_MUSL_TOOLCHAIN_BIN", source)
        self.assertIn("make ARCH=riscv64", source)

    def test_generic_gemm_runner_sets_sst_loader_paths_with_build_fallback(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn(
            'SST_ELEMENTS_HOME="${SST_ELEMENTS_HOME:-$REPO_ROOT/install}"',
            source,
        )
        self.assertIn(
            'SST_BUILD_LIB_PATH="${SST_BUILD_LIB_PATH:-$REPO_ROOT/build/sst-elements/src/sst/elements/golem/.libs}"',
            source,
        )
        self.assertIn(
            'SST_INSTALL_LIB_PATH="${SST_INSTALL_LIB_PATH:-$REPO_ROOT/install/lib/sst-elements-library}"',
            source,
        )
        self.assertIn('if [[ -z "${SST_LIB_PATH+x}" ]]; then', source)
        self.assertIn('if [[ -f "$SST_BUILD_LIB_PATH/libgolem.so" ]]; then', source)
        self.assertIn('SST_LIB_PATH="$SST_BUILD_LIB_PATH"', source)
        self.assertIn('SST_LIB_PATH="$SST_INSTALL_LIB_PATH"', source)
        self.assertIn(
            'CONDA_LIB_DIR="${CONDA_LIB_DIR:-/usr/lib/x86_64-linux-gnu}"',
            source,
        )
        self.assertIn(
            'export SST_SOFTMAX_LD_LIBRARY_PATH="${SST_SOFTMAX_LD_LIBRARY_PATH:-$CONDA_LIB_DIR:$SST_LIB_PATH:$SST_INSTALL_LIB_PATH:$SST_CORE_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"',
            source,
        )
        self.assertIn('export LD_LIBRARY_PATH="$SST_SOFTMAX_LD_LIBRARY_PATH"', source)

    def test_generic_gemm_runner_restores_caller_architecture_after_default_preset(self):
        archive_script = "architecture/archive/ncores_selfcom_dma.py"
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("CALLER_SET_GOLEM_ARCH_SCRIPT=0", source)
        self.assertIn('CALLER_GOLEM_ARCH_SCRIPT="$GOLEM_ARCH_SCRIPT"', source)
        self.assertIn('GOLEM_ARCH_SCRIPT="$CALLER_GOLEM_ARCH_SCRIPT"', source)
        self.assertLess(
            source.index('source "$DEFAULT_PRESET_FILE"'),
            source.index('GOLEM_ARCH_SCRIPT="$CALLER_GOLEM_ARCH_SCRIPT"'),
        )

        env = os.environ.copy()
        env.update(
            {
                "GOLEM_ARCH_SCRIPT": archive_script,
                "GOLEM_CTRL_LINK_ENABLE": "0",
                "GOLEM_GROUP_MANAGER_ENABLE": "0",
                "GOLEM_SKIP_BUILD": "1",
                "REAL_SST_BIN": "/tmp/golem-test-sst",
            }
        )
        result = subprocess.run(
            [
                str(RUNNER),
                "--groups",
                "4",
                "--num-cores",
                "16",
                "--gemm-cores",
                "16",
                "--num-mem-nodes",
                "9",
                "--mesh-dim-x",
                "8",
                "--gemm-m",
                "128",
                "--gemm-n",
                "128",
                "--gemm-k",
                "128",
                "--gemm-block-m",
                "64",
                "--gemm-block-n",
                "64",
                "--gemm-block-k",
                "64",
                "--dry-run",
            ],
            cwd=TESTS_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("/tmp/golem-test-sst", result.stdout)
        self.assertIn("--num-threads=1", result.stdout)
        self.assertIn(archive_script, result.stdout)

    def test_int_array_makefile_uses_absolute_overridable_riscv_gxx(self):
        source = MAKEFILE.read_text(encoding="utf-8")

        self.assertIn(f"RISCV_MUSL_TOOLCHAIN_BIN ?= {TOOLCHAIN_BIN}", source)
        self.assertIn(
            f"CXX=$(RISCV_MUSL_TOOLCHAIN_BIN)/{COMPILER}",
            source,
        )
        self.assertNotIn("CXX=$(ARCH)-linux-musl-g++", source)


if __name__ == "__main__":
    unittest.main()
