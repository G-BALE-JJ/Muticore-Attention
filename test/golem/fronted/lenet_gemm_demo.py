#!/usr/bin/env python3

import os
import sys
import subprocess

if __package__ in {None, ""}:
    _fronted_dir = os.path.dirname(os.path.abspath(__file__))
    _tests_dir = os.path.dirname(_fronted_dir)
    if _tests_dir not in sys.path:
        sys.path.insert(0, _tests_dir)

from fronted.lenet_pipeline.plan import REAL_BIN_DIR, REAL_DATASET_DIR, TESTS_DIR
from fronted.lenet_pipeline.reference import (
    conv1_ref_direct,
    conv2_ref_direct,
    conv_max_abs_diff,
    fc1_ref_direct,
    fc2_ref_direct,
    fc3_ref_direct,
    load_real_lenet5_case,
    vec_max_abs_diff,
)
from fronted.lenet_pipeline.runtime import run_conv12_single_sim_from_prebuilt_bins


def main() -> None:
    image_index = int(os.environ.get("LENET5_IMAGE_INDEX", "0"))
    weights_source = os.environ.get("LENET5_WEIGHT_SOURCE", "onnx")

    subprocess.run(
        [
            "python3",
            os.path.join(
                TESTS_DIR, "fronted", "lenet_pipeline", "prepare_real_lenet5_bins.py"
            ),
            "--dataset-dir",
            REAL_DATASET_DIR,
            "--image-index",
            str(image_index),
            "--out-dir",
            REAL_BIN_DIR,
            "--weights-source",
            weights_source,
        ],
        cwd=TESTS_DIR,
        check=True,
    )

    (
        input_28x28,
        weight_6x25,
        bias_6,
        weight_16x150,
        bias_16,
        weight_256x120,
        bias_120,
        weight_120x84,
        bias_84,
        weight_84x10,
        bias_10,
    ) = load_real_lenet5_case(
        REAL_DATASET_DIR,
        image_index,
        weights_source=weights_source,
    )

    pool1_gemm, pool2_gemm, fc1_gemm, fc2_gemm, fc3_gemm = (
        run_conv12_single_sim_from_prebuilt_bins(REAL_BIN_DIR)
    )

    pool1_ref = conv1_ref_direct(input_28x28, weight_6x25, bias_6)
    pool2_ref = conv2_ref_direct(pool1_ref, weight_16x150, bias_16)
    fc1_ref = fc1_ref_direct(pool2_ref, weight_256x120, bias_120)
    fc2_ref = fc2_ref_direct(fc1_ref, weight_120x84, bias_84)
    fc3_ref = fc3_ref_direct(fc2_ref, weight_84x10, bias_10)

    diff1 = conv_max_abs_diff(pool1_gemm, pool1_ref)
    diff2 = conv_max_abs_diff(pool2_gemm, pool2_ref)
    diff3 = vec_max_abs_diff(fc1_gemm, fc1_ref)
    diff4 = vec_max_abs_diff(fc2_gemm, fc2_ref)
    diff5 = vec_max_abs_diff(fc3_gemm, fc3_ref)

    ok = (
        diff1 <= 1e-5
        and diff2 <= 1e-5
        and diff3 <= 1e-5
        and diff4 <= 1e-5
        and diff5 <= 1e-5
    )
    print(f"conv1_pool max_abs_diff={diff1:.8f}")
    print(f"conv2_pool max_abs_diff={diff2:.8f}")
    print(f"fc1 max_abs_diff={diff3:.8f}")
    print(f"fc2 max_abs_diff={diff4:.8f}")
    print(f"fc3 max_abs_diff={diff5:.8f}")
    print(f"python_lenet_gemm_demo: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
