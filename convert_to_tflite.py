#!/usr/bin/env python3
"""
Convert the trained Keras model into a small TensorFlow Lite file.

You only need this to DEPLOY to Vercel (TensorFlow is too big to run there). For local
use, app.py can load model/xray_model.keras directly.

Run it on your Mac, in the environment where you trained the model:

    python convert_to_tflite.py --model model/xray_model.keras

Writes:
    model/xray_model.tflite
    model/classes.json        (label -> index mapping saved by the training script)

TensorFlow's converter hard-crashes (not a Python exception) on some Keras 3 models on
some machines, e.g. "error: missing attribute 'value'" / "LLVM ERROR". So each conversion
method runs in its own subprocess, and the next one is tried if it fails. Every result is
checked against the original model before it is accepted.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

STRATEGIES = ["freeze", "export", "direct"]
DEFAULT_CLASSES = {"NORMAL": 0, "PNEUMONIA": 1}  # Keras assigns indices alphabetically
TOLERANCE = 1e-3


# --------------------------------------------------------------------------- worker
def convert_and_verify(model_path: Path, tflite_path: Path, strategy: str) -> None:
    """Runs in a subprocess: convert with one strategy, then compare with the original."""
    import keras
    import numpy as np
    import tensorflow as tf

    try:  # preferred: the same runtime the deployed app uses
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter

    print(f"  TensorFlow {tf.__version__}, Keras {keras.__version__}")
    model = tf.keras.models.load_model(model_path)
    input_shape = (1, *model.input_shape[1:])

    with tempfile.TemporaryDirectory() as tmp:
        if strategy == "freeze":
            # Bake the weights into the graph as constants before converting, so the
            # converter never has to read variables (the step that crashes).
            from tensorflow.python.framework.convert_to_constants import (
                convert_variables_to_constants_v2,
            )

            concrete = tf.function(lambda x: model(x, training=False)).get_concrete_function(
                tf.TensorSpec(input_shape, tf.float32)
            )
            frozen = convert_variables_to_constants_v2(concrete)
            converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
        elif strategy == "export":
            model.export(tmp)  # Keras 3 -> SavedModel
            converter = tf.lite.TFLiteConverter.from_saved_model(tmp)
        elif strategy == "direct":
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
        else:
            raise ValueError(f"unknown strategy {strategy}")

        tflite_bytes = converter.convert()

    tflite_path.write_bytes(tflite_bytes)

    # Compare the converted model with the original on a few random inputs.
    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    max_diff = 0.0
    for _ in range(5):
        sample = np.random.rand(*input_shape).astype("float32")
        expected = np.asarray(model(sample, training=False))
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()
        actual = interpreter.get_tensor(output_index)
        max_diff = max(max_diff, float(np.abs(expected - actual).max()))

    print(f"  Max difference between Keras and TFLite output: {max_diff:.2e}")
    if max_diff > TOLERANCE:
        print("  Outputs differ more than expected - rejecting this result.")
        sys.exit(3)


# ---------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Keras model to TFLite.")
    parser.add_argument("--model", type=Path, default=None,
                        help="path to the trained .keras file "
                             "(default: model/xray_model.keras or xray_model.keras)")
    parser.add_argument("--classes", type=Path, default=None,
                        help="class mapping JSON (default: <model name>_classes.json next to the model)")
    parser.add_argument("--out-dir", type=Path, default=Path("model"),
                        help="where to write the TFLite model (default: ./model)")
    parser.add_argument("--strategy", choices=["auto", *STRATEGIES], default="auto",
                        help="conversion method; 'auto' tries them in turn (default)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)  # internal
    parser.add_argument("--tflite-out", type=Path, help=argparse.SUPPRESS)  # internal
    args = parser.parse_args()

    if args.model is None:
        args.model = next(
            (p for p in (Path("model/xray_model.keras"), Path("xray_model.keras")) if p.exists()),
            Path("xray_model.keras"),
        )

    if args.worker:
        convert_and_verify(args.model, args.tflite_out, args.strategy)
        return

    if not args.model.exists():
        sys.exit(f"Model file not found: {args.model}\nPass its location with --model.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.out_dir / "xray_model.tflite"
    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2", "PYTHONWARNINGS": "ignore"}

    for strategy in (STRATEGIES if args.strategy == "auto" else [args.strategy]):
        print(f"\nConverting {args.model} with method '{strategy}' ...")
        partial = args.out_dir / f".xray_model.{strategy}.tmp"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker",
             "--model", str(args.model), "--tflite-out", str(partial), "--strategy", strategy],
            env=env,
        )
        if result.returncode == 0 and partial.exists():
            os.replace(partial, final_path)
            print(f"\nSaved {final_path} ({final_path.stat().st_size / 1_000_000:.1f} MB)"
                  f" using method '{strategy}'")
            break
        partial.unlink(missing_ok=True)
        print(f"  Method '{strategy}' failed (exit code {result.returncode}).")
    else:
        sys.exit(
            "\nNone of the conversion methods worked on this machine.\n"
            "  * For local use you don't need a .tflite file: app.py loads the .keras file directly.\n"
            "  * For Vercel, run this script somewhere else, for example Google Colab (Linux),\n"
            "    and download model/xray_model.tflite and model/classes.json.\n"
            "  * Or send the output of  pip list | grep -i -E 'tensorflow|keras|numpy'  and\n"
            "    python --version  so the cause can be tracked down."
        )

    classes_path = args.classes or args.model.with_name(f"{args.model.stem}_classes.json")
    if classes_path.exists():
        classes = json.loads(classes_path.read_text())
    else:
        classes = DEFAULT_CLASSES
        print(f"Note: {classes_path} not found, using the default mapping {classes}.")
    (args.out_dir / "classes.json").write_text(json.dumps(classes, indent=2))
    print(f"Saved {args.out_dir / 'classes.json'}")


if __name__ == "__main__":
    main()
