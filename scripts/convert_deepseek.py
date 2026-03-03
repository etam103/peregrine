#!/usr/bin/env python3
"""Convert DeepSeek-V3/R1 HuggingFace SafeTensors weights to Peregrine binary format.

Usage:
    # Convert real weights from HuggingFace checkpoint
    python3 scripts/convert_deepseek.py --hf-path /path/to/DeepSeek-R1/ --output weights/deepseek_r1.bin

    # Generate random weights for small test config
    python3 scripts/convert_deepseek.py --random --output weights/deepseek_small.bin
"""

import argparse
import struct
import sys
import os
import numpy as np

def save_peregrine_model(tensors, output_path):
    """Save tensors in Peregrine binary format.
    Format: [num_tensors: u32]
            [per tensor: name_len: u32, name: utf8, ndim: u32, shape: u32*ndim, data: f32*N]
    """
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', len(tensors)))
        for name, array in tensors:
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            shape = array.shape
            f.write(struct.pack('<I', len(shape)))
            for s in shape:
                f.write(struct.pack('<I', s))
            data = array.astype(np.float32).tobytes()
            f.write(data)
    print(f"Saved {len(tensors)} tensors to {output_path}")
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")


# HuggingFace -> Peregrine name mapping
HF_NAME_MAP = {
    "embed_tokens": "embed",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
    "q_proj": "wq",
    "q_a_proj": "wq_a",
    "q_a_layernorm": "q_norm",
    "q_b_proj": "wq_b",
    "kv_a_proj_with_mqa": "wkv_a",
    "kv_a_layernorm": "kv_norm",
    "kv_b_proj": "wkv_b",
    "o_proj": "wo",
    "gate": "gate",
    "gate_proj": "w1",
    "down_proj": "w2",
    "up_proj": "w3",
    "lm_head": "head",
}

# Which weights need transposing (2D linear weights: PyTorch [out, in] -> Peregrine [in, out])
TRANSPOSE_KEYS = {
    "wq", "wq_a", "wq_b", "wkv_a", "wkv_b", "wo",
    "w1", "w2", "w3", "head", "embed",
}


def map_hf_name(hf_name):
    """Convert HuggingFace weight name to Peregrine name."""
    name = hf_name
    if name.startswith("model."):
        name = name[len("model."):]

    # model.layers.N.self_attn.X.weight -> layers.N.attn.X
    name = name.replace("self_attn", "attn")
    name = name.replace("mlp", "ffn")
    name = name.replace("e_score_correction_bias", "bias")

    # Remove .weight suffix for linear layers (keep for norms)
    parts = name.split(".")
    # Map the key part
    for i, part in enumerate(parts):
        if part in HF_NAME_MAP:
            parts[i] = HF_NAME_MAP[part]

    name = ".".join(parts)

    # Remove trailing .weight for most things
    if name.endswith(".weight"):
        # Keep .weight for norm layers
        parent = name.rsplit(".", 1)[0]
        if parent.endswith("_norm") or parent == "norm":
            pass  # keep .weight
        else:
            name = name[:-len(".weight")]

    return name


def convert_hf_weights(hf_path, output_path):
    """Convert HuggingFace SafeTensors checkpoint to Peregrine format."""
    try:
        from safetensors import safe_open
    except ImportError:
        print("Error: safetensors package required. Install with: pip install safetensors")
        sys.exit(1)

    import glob

    safetensor_files = sorted(glob.glob(os.path.join(hf_path, "*.safetensors")))
    if not safetensor_files:
        print(f"No .safetensors files found in {hf_path}")
        sys.exit(1)

    print(f"Found {len(safetensor_files)} safetensor files")

    tensors = []
    seen_names = set()

    for filepath in safetensor_files:
        print(f"  Reading {os.path.basename(filepath)}...")
        with safe_open(filepath, framework="numpy") as f:
            for hf_name in f.keys():
                # Skip scale tensors (FP8 quantization)
                if "weight_scale_inv" in hf_name or "scale" in hf_name:
                    continue
                # Skip layer 61 (unused in some checkpoints)
                if "model.layers.61" in hf_name:
                    continue

                peregrine_name = map_hf_name(hf_name)

                if peregrine_name in seen_names:
                    continue
                seen_names.add(peregrine_name)

                tensor = f.get_tensor(hf_name).astype(np.float32)

                # Determine if we should transpose
                key_part = peregrine_name.split(".")[-1]
                if key_part in TRANSPOSE_KEYS and tensor.ndim == 2:
                    tensor = tensor.T.copy()

                tensors.append((peregrine_name, tensor))

    # Sort by name for deterministic output
    tensors.sort(key=lambda x: x[0])
    print(f"\nConverted {len(tensors)} tensors")

    save_peregrine_model(tensors, output_path)


def generate_random_weights(output_path):
    """Generate random weights for the small test config."""
    np.random.seed(42)
    scale = 0.02

    # Small config values
    vocab_size = 1024
    dim = 256
    inter_dim = 512
    moe_inter_dim = 128
    n_layers = 2
    n_dense_layers = 1
    n_heads = 4
    n_routed_experts = 8
    n_shared_experts = 1
    q_lora_rank = 64
    kv_lora_rank = 32
    qk_nope_head_dim = 32
    qk_rope_head_dim = 16
    v_head_dim = 32
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    tensors = []

    # Embedding [vocab_size, dim]
    tensors.append(("embed", np.random.randn(vocab_size, dim).astype(np.float32) * scale))

    # Output head [dim, vocab_size]  (already transposed for Peregrine)
    tensors.append(("head", np.random.randn(dim, vocab_size).astype(np.float32) * scale))

    # Final norm
    tensors.append(("norm.weight", np.ones(dim, dtype=np.float32)))

    for i in range(n_layers):
        prefix = f"layers.{i}"

        # Norms
        tensors.append((f"{prefix}.attn_norm.weight", np.ones(dim, dtype=np.float32)))
        tensors.append((f"{prefix}.ffn_norm.weight", np.ones(dim, dtype=np.float32)))

        # MLA attention
        attn = f"{prefix}.attn"
        tensors.append((f"{attn}.wq_a", np.random.randn(dim, q_lora_rank).astype(np.float32) * scale))
        tensors.append((f"{attn}.q_norm.weight", np.ones(q_lora_rank, dtype=np.float32)))
        tensors.append((f"{attn}.wq_b", np.random.randn(q_lora_rank, n_heads * qk_head_dim).astype(np.float32) * scale))
        tensors.append((f"{attn}.wkv_a", np.random.randn(dim, kv_lora_rank + qk_rope_head_dim).astype(np.float32) * scale))
        tensors.append((f"{attn}.kv_norm.weight", np.ones(kv_lora_rank, dtype=np.float32)))
        tensors.append((f"{attn}.wkv_b", np.random.randn(kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim)).astype(np.float32) * scale))
        tensors.append((f"{attn}.wo", np.random.randn(n_heads * v_head_dim, dim).astype(np.float32) * scale))

        # FFN
        ffn = f"{prefix}.ffn"
        if i < n_dense_layers:
            # Dense MLP
            tensors.append((f"{ffn}.w1", np.random.randn(dim, inter_dim).astype(np.float32) * scale))
            tensors.append((f"{ffn}.w2", np.random.randn(inter_dim, dim).astype(np.float32) * scale))
            tensors.append((f"{ffn}.w3", np.random.randn(dim, inter_dim).astype(np.float32) * scale))
        else:
            # MoE
            tensors.append((f"{ffn}.gate", np.random.randn(n_routed_experts, dim).astype(np.float32) * scale))
            for j in range(n_routed_experts):
                tensors.append((f"{ffn}.experts.{j}.w1", np.random.randn(dim, moe_inter_dim).astype(np.float32) * scale))
                tensors.append((f"{ffn}.experts.{j}.w2", np.random.randn(moe_inter_dim, dim).astype(np.float32) * scale))
                tensors.append((f"{ffn}.experts.{j}.w3", np.random.randn(dim, moe_inter_dim).astype(np.float32) * scale))

            # Shared experts
            shared_inter = n_shared_experts * moe_inter_dim
            tensors.append((f"{ffn}.shared_experts.w1", np.random.randn(dim, shared_inter).astype(np.float32) * scale))
            tensors.append((f"{ffn}.shared_experts.w2", np.random.randn(shared_inter, dim).astype(np.float32) * scale))
            tensors.append((f"{ffn}.shared_experts.w3", np.random.randn(dim, shared_inter).astype(np.float32) * scale))

    tensors.sort(key=lambda x: x[0])
    print(f"Generated {len(tensors)} random tensors (small config)")
    save_peregrine_model(tensors, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DeepSeek weights to Peregrine format")
    parser.add_argument("--hf-path", type=str, help="Path to HuggingFace checkpoint directory")
    parser.add_argument("--output", type=str, required=True, help="Output .bin path")
    parser.add_argument("--random", action="store_true", help="Generate random weights for small test config")
    args = parser.parse_args()

    if args.random:
        generate_random_weights(args.output)
    elif args.hf_path:
        convert_hf_weights(args.hf_path, args.output)
    else:
        print("Error: specify --hf-path or --random")
        sys.exit(1)
