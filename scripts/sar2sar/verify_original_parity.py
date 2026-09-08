"""Compare saved-graph inference against the authors' complete test procedure.

The authors' files remain unchanged. For the reference execution only, this
script redirects TensorFlow imports to compat.v1 and disables an unused
tf.contrib regularizer declaration; every forward layer and the complete
normalization, patch scan, averaging and inverse transform are original code.
"""

from __future__ import annotations

import ast
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import types

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tensorflow
from run_local import infer, validate_official_package
from download_official import ROOT, sha256

tf = tensorflow.compat.v1
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


class CompatibilityImports(ast.NodeTransformer):
    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "tensorflow":
                alias.name = "tensorflow.compat.v1"
                alias.asname = alias.asname or "tensorflow"
        return node

    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id == "regularizer" for target in node.targets):
            # Declared in u_net.py but never passed to any convolution.
            node.value = ast.Constant(None)
        return node


def load_original(name, path):
    tree = CompatibilityImports().visit(ast.parse(path.read_text(encoding="utf-8")))
    ast.fix_missing_locations(tree)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    package, provenance = validate_official_package()
    folder = (arguments.output or ROOT / "output/sar2sar_local" /
              datetime.now().strftime("parity_validation_%Y%m%d_%H%M%S_%f")).resolve()
    inputs, reference_out = folder / "input", folder / "official_reference"
    inputs.mkdir(parents=True, exist_ok=True)
    reference_out.mkdir(parents=True, exist_ok=True)
    source = package / "test_data/lely.npy"
    full = np.load(source, allow_pickle=False)
    cases = {"center256": full[122:378, 122:378], "full500": full}
    if any((reference_out / f"denoised_{name}.npy").exists() for name in cases):
        raise FileExistsError("Parity outputs already exist; preserve them and choose another output folder for a rerun")
    for name, array in cases.items():
        np.save(inputs / f"{name}.npy", array)
    load_original("utils", package / "utils.py")
    load_original("u_net", package / "u_net.py")
    model = load_original("model", package / "model.py")
    config = tf.ConfigProto(device_count={"GPU": 0}, intra_op_parallelism_threads=4,
                            inter_op_parallelism_threads=1)
    with tf.Graph().as_default(), tf.Session(config=config) as session:
        reference = model.denoiser(session)
        reference.test([str((inputs / f"{name}.npy").as_posix()) for name in cases],
                       str((package / "checkpoint").as_posix()), str(reference_out.as_posix()),
                       str(inputs.as_posix()), stride=64)
    results = []
    for name, amplitude in cases.items():
        expected = np.load(reference_out / f"denoised_{name}.npy", allow_pickle=False)
        actual, execution = infer(amplitude.astype(np.float64), package, stride=64, threads=4)
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        passed = np.allclose(actual, expected, rtol=2e-5, atol=1e-5)
        results.append({"case": name, "shape": list(amplitude.shape),
                        "patch_count": execution["patch_count"], "passed": bool(passed),
                        "bitwise_identical": bool(np.array_equal(actual, expected)),
                        "max_absolute_amplitude_difference": float(error.max()),
                        "max_relative_amplitude_difference": float(np.max(error / np.maximum(np.abs(expected), 1e-12)))})
        if not passed:
            raise AssertionError(json.dumps(results[-1]))
    report = {
        "all_cases_passed": True, "source_commit": provenance["source_commit"],
        "reference": "Authors' u_net.py + model.denoiser.test + utils.py, original checkpoint",
        "reference_compatibility_changes": ["TensorFlow imports -> tensorflow.compat.v1",
                                              "Unused tf.contrib regularizer declaration -> None"],
        "source_files_sha256": {name: sha256(package / name) for name in ("u_net.py", "model.py", "utils.py")},
        "tensorflow_version": tensorflow.__version__, "device": "CPU", "cases": results,
    }
    (folder / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
