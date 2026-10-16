#!/usr/bin/env python3
"""Run and archive the synthetic Llama 32:8/D64 prefill baseline."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GM_PROFILE = {
    "GOLEM_LOCAL_GM_MODEL": "banked",
    "GOLEM_LOCAL_GM_BYTES_PER_CYCLE": "256",
    "GOLEM_LOCAL_GM_BASE_LATENCY_CYCLES": "1",
    "GOLEM_LOCAL_GM_READ_PORTS": "1",
    "GOLEM_LOCAL_GM_WRITE_PORTS": "1",
    "GOLEM_LOCAL_GM_MAX_REQUEST_BYTES": "4096",
    "GOLEM_LOCAL_GM_QUEUE_DEPTH": "256",
    "GOLEM_MPI_RANKS": "4",
    "VANADIS_CPU_CLOCK": "1.0GHz",
    "GOLEM_ARRAY_CLOCK": "1.0GHz",
    "GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE": "1",
    "GOLEM_ARRAY_PIPELINE_DEPTH": "2",
}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def summarize(case):
    names = ("projection_e2e_result", "fused_attention_result",
             "attention_hbm_layout", "attention_mpi_partition", "sst_qk_bridge_result")
    reports = {name: json.loads((case / f"{name}.json").read_text()) for name in names}
    if any(report.get("status") != "PASS" for report in reports.values()):
        raise ValueError(f"{case}: incomplete or failed verification")
    pipeline = reports["projection_e2e_result"]
    shape = pipeline["shape"]
    if (shape["Hq"], shape["Hkv"], shape["head_dim"]) != (32, 8, 64):
        raise ValueError(f"{case}: requires Llama 32:8/D64 shape")
    gm = pipeline["stages"]["projection"].get("local_gm_by_worker", pipeline["stages"]["projection"].get("local_gm_by_manager", {}))
    if len(gm) != (16 if pipeline.get("resource_topology_valid") else 4) or any(x["timed"] != 1 for x in gm.values()):
        raise ValueError(f"{case}: missing timed GM evidence")
    if any(x["reuse_block"] and not x.get("paired_weights") for x in gm.values()):
        raise ValueError(f"{case}: missing paired-weight reuse evidence")
    rope = pipeline.get("attention_rope_table_first_load_by_worker", {})
    if not any(x["rope"] for x in pipeline.get("fusion_by_worker", pipeline.get("fusion_by_manager", {})).values()) and (len(rope) != 8 or any(x["local_read_cycles"] <= 0 for x in rope.values())):
        raise ValueError(f"{case}: missing timed RoPE table evidence")
    errors = {}
    for node in reports["attention_hbm_layout"]["nodes"]:
        for name, metrics in node["numerical_errors"].items():
            errors[name] = max(errors.get(name, 0), metrics["max_abs_error"])
    return {
        "artifact_root": str(case),
        "pipeline": pipeline,
        "max_abs_errors": {**errors, "O": reports["fused_attention_result"]["max_abs_error"]},
        "output_verification": reports["fused_attention_result"],
        "attention": reports["sst_qk_bridge_result"],
        "layout_checked_bytes": reports["attention_hbm_layout"]["checked_bytes"],
        "provenance": json.loads((case / "baseline_provenance.json").read_text()),
    }


def compare_reference(case, reference):
    """Compare measured cycles and byte-exact projection layouts."""
    current, previous = summarize(case), summarize(reference)
    if current["pipeline"]["shape"] != previous["pipeline"]["shape"]:
        raise ValueError("reference shape mismatch")
    for name in ("projection_x.bin", "projection_gamma.bin", "projection_weights.bin"):
        if digest(case / name) != digest(reference / name):
            raise ValueError(f"reference input mismatch: {name}")
    current_layout = json.loads((case / "attention_hbm_layout.json").read_text())
    prior_layout = json.loads((reference / "attention_hbm_layout.json").read_text())
    checked = []
    names = {"RMSNorm", "Q", "K", "V", "Q_panels", "K_panels", "V_panels", "O"}
    for node, prior in zip(current_layout["nodes"], prior_layout["nodes"], strict=True):
        if node["node"] != prior["node"]:
            raise ValueError("reference HBM node mismatch")
        prior_regions = {r["name"]: r for r in prior["regions"]}
        filename = f"hbm_out_node{node['node']}.bin"
        with (case / "hbm" / filename).open("rb") as actual, (reference / "hbm" / filename).open("rb") as expected:
            for region in node["regions"]:
                if region["name"] not in names:
                    continue
                actual.seek(region["begin"])
                prior_region = prior_regions[region["name"]]
                expected.seek(prior_region["begin"])
                length = region["end"] - region["begin"]
                wanted = expected.read(length)
                if (region["name"] in {"Q_panels", "K_panels"} and
                        current_layout.get("rotated_panels") and not prior_layout.get("rotated_panels")):
                    import numpy as np
                    import sys
                    sys.path.insert(0, str(ROOT / "src/sst/elements/golem/tests/small/muticore_attention"))
                    from verify_fused_attention_scale_output import rotate_rope
                    shape = current["pipeline"]["shape"]
                    rows, dim = shape["query_length"] // 4, shape["head_dim"]
                    heads = shape["Hq"] if region["name"] == "Q_panels" else shape["Hkv"]
                    raw_region = prior_regions[region["name"][0]]
                    expected.seek(raw_region["begin"])
                    values = np.frombuffer(expected.read(raw_region["end"] - raw_region["begin"]),
                                           dtype="<f2").reshape(heads, rows, dim)
                    table = np.fromfile(reference / f"rope_{dim}.bin", dtype="<f2").reshape(-1, dim)
                    table = table[(node["node"] - 1) * rows:node["node"] * rows]
                    rotated = np.asarray([rotate_rope(v, rows, dim, table, simulator_storage=True) for v in values], dtype="<f2")
                    wanted = rotated.reshape(heads, rows // 64, 64, dim // 64, 64).transpose(0, 1, 3, 2, 4).tobytes()
                if actual.read(length) != wanted:
                    raise ValueError(f"reference byte mismatch: node {node['node']} {region['name']}")
                checked.append({"node": node["node"], "region": region["name"], "bytes": length})
    def reduction(old, new):
        return {"before": old, "after": new, "saved_cycles": old - new,
                "reduction_percent": 100 * (old - new) / old}
    comparison = {
        "status": "PASS", "reference_root": str(reference),
        "performance_comparable": bool(current["pipeline"].get("resource_topology_valid") and
                                       previous["pipeline"].get("resource_topology_valid") and
                                       current["pipeline"].get("array_operand_banks") is not None and
                                       current["pipeline"].get("array_operand_banks") ==
                                       previous["pipeline"].get("array_operand_banks")),
        "reference_resource_topology_valid": bool(previous["pipeline"].get("resource_topology_valid")),
        "byte_exact_regions": checked,
        "end_to_end": reduction(previous["pipeline"]["end_to_end_cycles"], current["pipeline"]["end_to_end_cycles"]),
        "projection": reduction(previous["pipeline"]["stages"]["projection"]["elapsed_cycles"], current["pipeline"]["stages"]["projection"]["elapsed_cycles"]),
    }
    if not comparison["performance_comparable"]:
        comparison["comparison_scope"] = "numerical compatibility; resource topology or operand storage differs or is unverified"
        for stage in ("end_to_end", "projection"):
            comparison[stage].pop("saved_cycles")
            comparison[stage].pop("reduction_percent")
    (case / "reference_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--sequences", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--reference-root", type=Path,
                        help="prior s1024/s2048 artifacts; check exact outputs and compare cycles only for valid resource topologies")
    args = parser.parse_args()
    env = {**os.environ, **GM_PROFILE}
    env.setdefault("GOLEM_PROJECTION_ARRAYS", "64")
    env.setdefault("GOLEM_PROJECTION_WEIGHT_PREFETCH", "1")
    cases = []
    for sequence in args.sequences:
        if sequence not in (1024, 2048):
            parser.error("supported sequences: 1024, 2048")
        case = args.artifact_root.resolve() / f"s{sequence}"
        if args.reference_root:
            env["GOLEM_PROJECTION_INPUT_REFERENCE"] = str(args.reference_root.resolve() / f"s{sequence}")
        if not args.summarize_only:
            case.mkdir(parents=True, exist_ok=True)
            if (case / "baseline_provenance.json").exists():
                raise ValueError(f"{case}: choose a new artifact root to retain prior evidence")
            command = [str(HERE / "run_sst.sh"), "--projection",
                       "--query-length", str(sequence), "--kv-length", str(sequence),
                       "--num-query-heads", "32", "--num-kv-heads", "8",
                       "--head-dim", "64", "--artifact-root", str(case)]
            sources = [ROOT / "src/sst/elements/golem/attention/projectionJob.h",
                       ROOT / "src/sst/elements/golem/array/computeArray.h",
                       ROOT / "src/sst/elements/golem/attention/attentionCluster.h",
                       ROOT / "src/sst/elements/golem/attention/projectionJobAbi.h",
                       ROOT / "src/sst/elements/golem/tests/small/muticore_attention/golem_projection_runtime.h",
                       ROOT / "src/sst/elements/golem/tests/small/muticore_attention/golem_attention_runtime.cpp",
                       ROOT / "src/sst/elements/golem/rocc/roccAnalog.h",
                       ROOT / "src/sst/elements/golem/workercmdproc/workercmdproc.h",
                       ROOT / "src/sst/elements/golem/array/mvmComputeArray.h",
                       ROOT / "src/sst/elements/golem/tests/architecture/cpu_builder.py",
                       ROOT / "src/sst/elements/golem/tests/small/muticore_attention/projection_case.py"]
            binaries = [ROOT / "install/lib/sst-elements-library/libgolem.so",
                        ROOT / "src/sst/elements/golem/tests/small/muticore_attention/riscv64/fused_attention"]
            provenance = {
                "git_revision": git("rev-parse", "HEAD"),
                "git_status": git("status", "--short"),
                "tracked_diff_sha256": hashlib.sha256(
                    subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
                "command": command,
                "environment_overrides": {k: v for k, v in env.items()
                                          if k.startswith("GOLEM_")},
                "sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources + binaries},
                "input_seed": 1742,
                "scope": "synthetic FP16 X/gamma/weights; RMSNorm -> QKV -> RoPE -> causal GQA -> O HBM; excludes Wo/residual/MLP",
            }
            (case / "baseline_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
            print(f"Running S={sequence}: {case}", flush=True)
            with (case / "baseline_driver.log").open("w") as handle:
                subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, check=True)
            configs = list((case / "stats").rglob("run_config.env"))
            if len(configs) != 1:
                raise ValueError(f"{case}: missing resolved run config")
            provenance["resolved_run_config"] = configs[0].read_text()
            # The wrapper rebuilds the guest before SST starts. Archive the
            # binaries that actually ran rather than the pre-build guest.
            provenance["sha256"].update({str(p.relative_to(ROOT)): digest(p)
                                         for p in binaries})
            provenance["input_sha256"] = {
                p.name: digest(p) for p in sorted(case.glob("*.bin"))}
            (case / "baseline_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        cases.append(summarize(case))
        if args.reference_root:
            cases[-1]["reference_comparison"] = compare_reference(
                case, args.reference_root.resolve() / f"s{sequence}")
        print(f"S={sequence}: {cases[-1]['pipeline']['end_to_end_cycles']:,} cycles, PASS", flush=True)
    output = args.artifact_root.resolve() / "summary.json"
    output.write_text(json.dumps({"status": "PASS", "cases": cases}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
