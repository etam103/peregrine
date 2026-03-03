#!/usr/bin/env python3
"""Convert Grok-1 JAX checkpoint to Peregrine binary format.

Usage:
    python scripts/convert_grok1.py <checkpoint_dir> <output.bin>
    python scripts/convert_grok1.py --random <output.bin>

The Grok-1 checkpoint is a directory containing distributed pickle files
with QuantizedWeight8bit tensors (int8 weights + float32 scales).

Example:
    python scripts/convert_grok1.py ~/Desktop/grok-1/checkpoints/ weights/grok1.bin
"""
import sys
import os
import struct
import pickle
import numpy as np


def save_peregrine_model(params: dict, path: str):
    """Save parameters in Peregrine binary format.

    Format: [num_tensors: u32]
    Per tensor: [name_len: u32][name: utf8][ndim: u32][shape: u32*ndim][data: f32*N]
    """
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(params)))

        for name, (shape, data) in params.items():
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            f.write(struct.pack('<I', len(shape)))
            for s in shape:
                f.write(struct.pack('<I', s))

            f.write(np.array(data, dtype=np.float32).tobytes())


def ffn_size(emb_size, widening_factor=8):
    """Compute FFN hidden dim matching model.py."""
    _ffn_size = int(widening_factor * emb_size) * 2 // 3
    _ffn_size = _ffn_size + (8 - _ffn_size) % 8
    return _ffn_size


class QuantizedWeight8bit:
    """Placeholder for unpickling the checkpoint."""
    def __init__(self, weight, scales):
        self.weight = weight
        self.scales = scales

    @property
    def shape(self):
        return self.weight.shape


def dequantize(qw):
    """Dequantize QuantizedWeight8bit to float32."""
    if isinstance(qw, QuantizedWeight8bit):
        return qw.weight.astype(np.float32) * qw.scales.astype(np.float32)
    return np.array(qw, dtype=np.float32)


def load_jax_checkpoint(checkpoint_dir):
    """Load all tensor files from the JAX checkpoint directory."""
    ckpt_dir = os.path.join(checkpoint_dir, "ckpt-0")
    if not os.path.isdir(ckpt_dir):
        # Maybe the user passed the ckpt-0 dir directly
        if os.path.basename(checkpoint_dir) == "ckpt-0":
            ckpt_dir = checkpoint_dir
        else:
            raise FileNotFoundError(f"Cannot find checkpoint at {ckpt_dir}")

    # Register the QuantizedWeight8bit class for unpickling
    sys.modules['__main__'].QuantizedWeight8bit = QuantizedWeight8bit

    # Find all tensor files (single-shard: tensor00000_000, etc.)
    tensor_files = sorted([
        f for f in os.listdir(ckpt_dir)
        if f.startswith("tensor") and "_" in f
    ])

    print(f"Found {len(tensor_files)} tensor files in {ckpt_dir}")

    tensors = []
    for tf in tensor_files:
        path = os.path.join(ckpt_dir, tf)
        with open(path, 'rb') as f:
            tensor = pickle.load(f)
        tensors.append(tensor)

    return tensors


def map_jax_to_peregrine(jax_path, jax_tensor):
    """Map a JAX parameter path to Peregrine naming convention.

    JAX checkpoint paths look like:
        language_model/~/in_out_embed/embeddings
        language_model/~/transformer/~/rms_norm/scale
        language_model/~/transformer/~/layer_stack/~/decoder_layer_{i}/~/rms_norm/scale
        language_model/~/transformer/~/layer_stack/~/decoder_layer_{i}/~/multi_head_attention/query/w
        language_model/~/transformer/~/layer_stack/~/decoder_layer_{i}/~/router/w
        language_model/~/transformer/~/layer_stack/~/decoder_layer_{i}/~/moe/linear/w
        language_model/~/transformer/~/layer_stack/~/decoder_layer_{i}/~/moe/linear_v/w
        language_model/~/transformer/~/layer_stack/~/decoder_layer_{i}/~/moe/linear_1/w

    Returns (peregrine_name, numpy_array) or None to skip.
    """
    data = dequantize(jax_tensor)

    # Embedding
    if "in_out_embed" in jax_path and "embeddings" in jax_path:
        return "embedding", data  # [vocab_size, model_dim]

    # Final RMSNorm
    if "transformer" in jax_path and "rms_norm" in jax_path and "layer_stack" not in jax_path:
        return "final_norm.weight", data  # [model_dim]

    # Layer-level parameters
    import re
    layer_match = re.search(r'decoder_layer_(\d+)', jax_path)
    if not layer_match:
        return None, None

    layer_idx = int(layer_match.group(1))
    prefix = f"layers.{layer_idx}"

    # RMSNorm parameters (4 per layer)
    # JAX uses rms_norm, rms_norm_1, rms_norm_2, rms_norm_3
    # Mapping: rms_norm -> pre_attn_norm, rms_norm_1 -> post_attn_norm,
    #          rms_norm_2 -> pre_moe_norm, rms_norm_3 -> post_moe_norm
    if "/rms_norm_3/" in jax_path:
        return f"{prefix}.post_moe_norm.weight", data
    elif "/rms_norm_2/" in jax_path:
        return f"{prefix}.pre_moe_norm.weight", data
    elif "/rms_norm_1/" in jax_path:
        return f"{prefix}.post_attn_norm.weight", data
    elif "/rms_norm/" in jax_path:
        return f"{prefix}.pre_attn_norm.weight", data

    # Attention projections
    if "multi_head_attention" in jax_path:
        if "/query/" in jax_path:
            # JAX shape: [emb, heads, key_size] -> flatten to [emb, heads*key_size]
            # Then transpose for Peregrine: [in, out]
            orig_shape = data.shape
            if data.ndim == 3:
                data = data.reshape(orig_shape[0], -1)
            if data.ndim == 2:
                data = data.T  # [out, in] -> [in, out]
            return f"{prefix}.attention.q_proj", data
        elif "/key/" in jax_path:
            orig_shape = data.shape
            if data.ndim == 3:
                data = data.reshape(orig_shape[0], -1)
            if data.ndim == 2:
                data = data.T
            return f"{prefix}.attention.k_proj", data
        elif "/value/" in jax_path:
            orig_shape = data.shape
            if data.ndim == 3:
                data = data.reshape(orig_shape[0], -1)
            if data.ndim == 2:
                data = data.T
            return f"{prefix}.attention.v_proj", data
        elif "/linear/" in jax_path:
            # Output projection: [heads*key_size, emb]
            orig_shape = data.shape
            if data.ndim == 3:
                data = data.reshape(-1, orig_shape[-1])
            if data.ndim == 2:
                data = data.T
            return f"{prefix}.attention.o_proj", data

    # Router
    if "/router/" in jax_path:
        # JAX: [emb, num_experts], already [in, out] format
        return f"{prefix}.moe.router", data

    # MoE expert weights
    # JAX stores experts with a leading expert dimension: [num_experts, in, out]
    if "/moe/" in jax_path:
        if "/linear_v/" in jax_path:
            # [num_experts, emb, ffn] -> per-expert [emb, ffn]
            if data.ndim == 3:
                num_experts = data.shape[0]
                results = []
                for e in range(num_experts):
                    expert_w = data[e].T  # [out, in] -> [in, out]
                    results.append((f"{prefix}.moe.experts.{e}.linear_v", expert_w))
                return results, None  # Special multi-return
            else:
                data = data.T if data.ndim == 2 else data
                return f"{prefix}.moe.experts.0.linear_v", data
        elif "/linear_1/" in jax_path:
            # linear_1 is the output projection
            if data.ndim == 3:
                num_experts = data.shape[0]
                results = []
                for e in range(num_experts):
                    expert_w = data[e].T
                    results.append((f"{prefix}.moe.experts.{e}.linear_out", expert_w))
                return results, None
            else:
                data = data.T if data.ndim == 2 else data
                return f"{prefix}.moe.experts.0.linear_out", data
        elif "/linear/" in jax_path:
            # linear is the gate projection
            if data.ndim == 3:
                num_experts = data.shape[0]
                results = []
                for e in range(num_experts):
                    expert_w = data[e].T
                    results.append((f"{prefix}.moe.experts.{e}.linear_gate", expert_w))
                return results, None
            else:
                data = data.T if data.ndim == 2 else data
                return f"{prefix}.moe.experts.0.linear_gate", data

    return None, None


def convert_grok1(checkpoint_dir, output_path):
    """Convert Grok-1 JAX checkpoint to Peregrine binary format."""
    print(f"Loading checkpoint from {checkpoint_dir}...")

    # We need to reconstruct the parameter tree from the flat tensor files
    # The JAX checkpoint stores tensors in a flattened order matching the model structure
    # For now, we try loading and inspecting
    tensors = load_jax_checkpoint(checkpoint_dir)

    # The full mapping requires knowing the JAX tree structure.
    # For a practical approach, we iterate through the flattened state and use
    # the path information embedded in the checkpoint structure.

    print(f"Loaded {len(tensors)} tensors from checkpoint")
    print("Note: Full conversion requires reconstructing the JAX parameter tree.")
    print("Use --random flag for testing with random weights instead.")

    # TODO: Complete mapping once we can inspect the actual checkpoint tree structure.
    # The checkpoint uses pickle files that encode the nested dict structure.
    # Each tensor file corresponds to a leaf in the flattened JAX pytree.


def generate_random_weights(output_path):
    """Generate random weights for the small test config."""
    print("Generating random weights for small test config...")

    vocab_size = 1024
    model_dim = 256
    num_layers = 2
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    num_experts = 4
    ffn_dim = 512

    params = {}

    # Embedding
    emb = np.random.randn(vocab_size, model_dim).astype(np.float32) * 0.02
    params["embedding"] = (list(emb.shape), emb.flatten().tolist())

    # Final norm
    params["final_norm.weight"] = ([model_dim], [1.0] * model_dim)

    for i in range(num_layers):
        p = f"layers.{i}"

        # RMSNorm weights (all ones)
        for norm_name in ["pre_attn_norm", "post_attn_norm", "pre_moe_norm", "post_moe_norm"]:
            params[f"{p}.{norm_name}.weight"] = ([model_dim], [1.0] * model_dim)

        # Attention projections
        def rand_weight(rows, cols):
            w = np.random.randn(rows, cols).astype(np.float32) * 0.02
            return (list(w.shape), w.flatten().tolist())

        params[f"{p}.attention.q_proj"] = rand_weight(model_dim, num_q_heads * head_dim)
        params[f"{p}.attention.k_proj"] = rand_weight(model_dim, num_kv_heads * head_dim)
        params[f"{p}.attention.v_proj"] = rand_weight(model_dim, num_kv_heads * head_dim)
        params[f"{p}.attention.o_proj"] = rand_weight(num_q_heads * head_dim, model_dim)

        # Router
        params[f"{p}.moe.router"] = rand_weight(model_dim, num_experts)

        # Experts
        for j in range(num_experts):
            ep = f"{p}.moe.experts.{j}"
            params[f"{ep}.linear_gate"] = rand_weight(model_dim, ffn_dim)
            params[f"{ep}.linear_v"] = rand_weight(model_dim, ffn_dim)
            params[f"{ep}.linear_out"] = rand_weight(ffn_dim, model_dim)

    print(f"Generated {len(params)} parameter tensors")
    total = sum(np.prod(s) for s, _ in params.values())
    print(f"Total parameters: {total:,}")

    save_peregrine_model(params, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved to {output_path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python scripts/convert_grok1.py <checkpoint_dir> <output.bin>")
        print("       python scripts/convert_grok1.py --random <output.bin>")
        print()
        print("Options:")
        print("  --random    Generate random weights for small test config")
        sys.exit(1)

    if sys.argv[1] == "--random":
        generate_random_weights(sys.argv[2])
    else:
        convert_grok1(sys.argv[1], sys.argv[2])
