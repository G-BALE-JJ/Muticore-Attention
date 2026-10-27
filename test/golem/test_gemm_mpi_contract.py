#!/usr/bin/env python3
"""Contracts for MPI-parallel SST execution of the generic GEMM workload."""

import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
RUNNER = TESTS_DIR / "run_noc_dma_pipeline.sh"
ARCH = TESTS_DIR / "architecture" / "ncores_selfcom_dma_ctrl.py"
MEMORY_SUMMARY = TESTS_DIR / "stats" / "extract_memory_summary_csv.py"


class GemmMpiContractTest(unittest.TestCase):
    def _dry_run(self, *extra_args):
        env = os.environ.copy()
        env.update(
            {
                "GOLEM_ARCH_SCRIPT": "architecture/ncores_selfcom_dma_ctrl.py",
                "GOLEM_CTRL_LINK_ENABLE": "0",
                "GOLEM_GROUP_MANAGER_ENABLE": "0",
                "GOLEM_SKIP_BUILD": "1",
                "REAL_SST_BIN": "/tmp/golem-test-sst",
            }
        )
        return subprocess.run(
            [
                str(RUNNER),
                "--groups", "4",
                "--num-cores", "16",
                "--gemm-cores", "16",
                "--num-mem-nodes", "5",
                "--mesh-dim-x", "4",
                "--gemm-m", "128",
                "--gemm-n", "128",
                "--gemm-k", "128",
                "--gemm-block-m", "64",
                "--gemm-block-n", "64",
                "--gemm-block-k", "64",
                *extra_args,
                "--dry-run",
            ],
            cwd=TESTS_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_runner_builds_mpi_sst_command(self):
        result = self._dry_run(
            "--mpi-ranks", "2",
            "--mpi-launcher", "/bin/echo",
            "--mpi-args", "",
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("GOLEM_MPI_PARTITIONING=1", result.stdout)
        self.assertIn("/bin/echo -np 2 /tmp/golem-test-sst", result.stdout)
        self.assertIn("--num-threads=1", result.stdout)
        self.assertIn("--partitioner=sst.self", result.stdout)
        self.assertIn("architecture/ncores_selfcom_dma_ctrl.py", result.stdout)

    def test_runner_rejects_being_launched_once_per_rank(self):
        env = os.environ.copy()
        env["OMPI_COMM_WORLD_SIZE"] = "2"
        result = subprocess.run(
            [str(RUNNER), "--help"],
            cwd=TESTS_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("Do not launch run_noc_dma_pipeline.sh under mpirun", result.stdout)

    def test_architecture_enables_partitioning_and_isolates_dramsim_outputs(self):
        source = ARCH.read_text(encoding="utf-8")

        self.assertIn('MPI_PARTITIONING = _env_flag("GOLEM_MPI_PARTITIONING", False)', source)
        self.assertIn('backend_params["output_dir"] = node_output_dir', source)
        self.assertGreaterEqual(source.count("if not MPI_PARTITIONING:"), 3)

    def test_runner_rejects_unsupported_control_topology(self):
        env = os.environ.copy()
        env.update(
            {
                "GOLEM_ARCH_SCRIPT": "architecture/ncores_selfcom_dma_ctrl.py",
                "GOLEM_CTRL_LINK_ENABLE": "1",
                "GOLEM_GROUP_MANAGER_ENABLE": "1",
                "GOLEM_EXPLICIT_PARTITION": "0",
                "GOLEM_SKIP_BUILD": "1",
                "REAL_SST_BIN": "/tmp/golem-test-sst",
            }
        )
        for groups, cores, mesh_dim_x in ((4, 20, 8), (2, 10, 4)):
            with self.subTest(groups=groups, mesh_dim_x=mesh_dim_x):
                result = subprocess.run(
                    [
                        str(RUNNER),
                        "--groups", str(groups),
                        "--num-cores", str(cores),
                        "--gemm-cores", str(cores),
                        "--num-mem-nodes", "3",
                        "--mesh-dim-x", str(mesh_dim_x),
                        "--dry-run",
                    ],
                    cwd=TESTS_DIR,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("total_groups=4 且 mesh_dim_x=4", result.stdout)

    def test_runner_rejects_unsupported_explicit_partition_mesh(self):
        result = self._dry_run("--mesh-dim-x", "8")

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("显式分区布局要求 mesh_dim_x=4", result.stdout)

    def test_memory_summary_combines_multiple_memory_nodes(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            json_paths = []
            txt_paths = []
            for node, (latency, reads) in enumerate(((10.0, 1), (20.0, 3))):
                json_path = tmp / f"node{node}.json"
                txt_path = tmp / f"node{node}.txt"
                json_path.write_text(
                    json.dumps(
                        {
                            "0": {
                                "average_read_latency": latency,
                                "average_bandwidth": 4.0 + node,
                                "num_reads_done": reads,
                                "num_writes_done": node,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                txt_path.write_text(
                    "## Statistics of Channel 0\nread_latency[0-10] = 1\n",
                    encoding="utf-8",
                )
                json_paths.append(json_path)
                txt_paths.append(txt_path)

            output = tmp / "summary.csv"
            command = ["python3", str(MEMORY_SUMMARY)]
            for path in json_paths:
                command.extend(("--json", str(path)))
            for path in txt_paths:
                command.extend(("--txt", str(path)))
            command.extend(("--output", str(output)))
            result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            with output.open(newline="", encoding="utf-8") as summary_file:
                metrics = dict(csv.reader(summary_file))
            self.assertEqual(metrics["channel_count"], "2")
            self.assertEqual(metrics["total_reads_done"], "4")
            self.assertEqual(metrics["total_writes_done"], "1")
            self.assertEqual(metrics["mem_avg_read_latency_cycles"], "17.500000")


if __name__ == "__main__":
    unittest.main()
