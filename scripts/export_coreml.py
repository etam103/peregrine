"""Convert exported Peregrine model weights to Core ML format.

Reads a manifest.json + binary weight files (from peregrine::serial::export_for_coreml)
and produces a .mlpackage that can run on CPU, GPU, or Apple Neural Engine (ANE).

Usage:
    python scripts/export_coreml.py <export_dir> [--output model.mlpackage]

Requirements:
    pip install coremltools torch numpy
"""

import argparse
import json
import os
import numpy as np

def load_weights(export_dir):
    """Load weights from manifest.json + binary files."""
    manifest_path = os.path.join(export_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    weights = {}
    for param in manifest["parameters"]:
        filepath = os.path.join(export_dir, param["file"])
        data = np.fromfile(filepath, dtype=np.float32)
        shape = param["shape"]
        weights[param["name"]] = data.reshape(shape)
        print(f"  Loaded {param['name']}: {shape}")

    return weights


def build_mlp_coreml(weights, input_dim, output_path):
    """Build a Core ML MLP model from exported weights.

    Assumes Linear layers with weight shape [in, out] and bias shape [1, out].
    """
    try:
        import coremltools as ct
        import torch
        import torch.nn as tnn
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install coremltools torch numpy")
        return

    # Discover layers from weight names
    layer_names = sorted(set(
        name.rsplit(".", 1)[0] for name in weights.keys()
    ))

    print(f"Layers: {layer_names}")

    # Build PyTorch model to trace
    layers = []
    for name in layer_names:
        w_key = f"{name}.weight"
        b_key = f"{name}.bias"
        if w_key not in weights:
            continue
        w = weights[w_key]
        b = weights[b_key].flatten() if b_key in weights else None

        in_f, out_f = w.shape
        linear = tnn.Linear(in_f, out_f)
        # Peregrine stores weight as [in, out], PyTorch as [out, in]
        linear.weight.data = torch.from_numpy(w.T.copy())
        if b is not None:
            linear.bias.data = torch.from_numpy(b.copy())

        layers.append(linear)
        # Add ReLU between hidden layers (not after last)
        if name != layer_names[-1]:
            layers.append(tnn.ReLU())

    model = tnn.Sequential(*layers)
    model.eval()

    # Trace and convert
    example_input = torch.randn(1, input_dim)
    traced = torch.jit.trace(model, example_input)

    ct_model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, input_dim))],
        compute_units=ct.ComputeUnit.ALL,  # CPU + GPU + ANE
    )

    ct_model.save(output_path)
    print(f"Saved Core ML model: {output_path}")
    print(f"Compute units: ALL (CPU + GPU + Neural Engine)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Peregrine weights to Core ML")
    parser.add_argument("export_dir", help="Directory with manifest.json + .bin files")
    parser.add_argument("--output", default="model.mlpackage", help="Output .mlpackage path")
    parser.add_argument("--input-dim", type=int, default=784, help="Input dimension")
    args = parser.parse_args()

    print("Loading weights...")
    weights = load_weights(args.export_dir)
    print(f"Building Core ML model (input_dim={args.input_dim})...")
    build_mlp_coreml(weights, args.input_dim, args.output)
