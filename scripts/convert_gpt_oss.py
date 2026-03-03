#!/usr/bin/env python3
"""Convert GPT-OSS safetensors checkpoint to Peregrine binary format.

Usage:
    python scripts/convert_gpt_oss.py <safetensors_dir> <output.bin>
    python scripts/convert_gpt_oss.py --random <output.bin>

The GPT-OSS checkpoint uses safetensors with MXFP4 quantized MoE weights.
Non-MoE weights are stored in bfloat16.

Example:
    python scripts/convert_gpt_oss.py ~/Desktop/gpt-oss/original/ weights/gpt_oss.bin
    python scripts/convert_gpt_oss.py --random weights/gpt_oss_small.bin
"""
import sys
import os
import struct
import numpy as np


# FP4 lookup table (16 representable values)
FP4_LUT = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=np.float32)


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


def decompress_mxfp4(blocks: np.ndarray, scales: np.ndarray, shape: tuple) -> np.ndarray:
    """Decompress MXFP4 quantized tensor.

    blocks: uint8 array with pairs of 4-bit indices packed per byte
    scales: uint8 array with per-group exponents (biased by 127)
    shape: target output shape

    Block size is 32 elements. Each block has 16 bytes (32 nibbles)
    and one scale value.
    """
    # Unpack nibbles
    low_nibbles = blocks & 0x0F
    high_nibbles = (blocks >> 4) & 0x0F

    # Interleave: for each byte, low nibble comes first, then high nibble
    nibbles = np.stack([low_nibbles, high_nibbles], axis=-1).reshape(-1)

    # Look up FP4 values
    values = FP4_LUT[nibbles]

    # Compute scale factors: 2^(scale - 127)
    scale_factors = np.ldexp(np.ones_like(scales, dtype=np.float32),
                             scales.astype(np.int32) - 127)

    # Each scale applies to a block of 32 elements
    block_size = 32
    n_elements = values.shape[0]
    n_blocks = n_elements // block_size

    # Broadcast scales to elements
    scale_expanded = np.repeat(scale_factors.reshape(-1)[:n_blocks], block_size)
    if scale_expanded.shape[0] < n_elements:
        scale_expanded = np.pad(scale_expanded, (0, n_elements - scale_expanded.shape[0]))

    result = values[:n_elements] * scale_expanded[:n_elements]

    total = int(np.prod(shape))
    return result[:total].reshape(shape).astype(np.float32)


def convert_safetensors(checkpoint_dir, output_path):
    """Convert GPT-OSS safetensors checkpoint to Peregrine binary format."""
    try:
        import safetensors
        from safetensors import safe_open
    except ImportError:
        print("Error: safetensors package required. Install with: pip install safetensors")
        sys.exit(1)

    print(f"Loading checkpoint from {checkpoint_dir}...")

    # Find all safetensor files
    st_files = sorted([
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith('.safetensors')
    ])
    if not st_files:
        print(f"No .safetensors files found in {checkpoint_dir}")
        sys.exit(1)

    print(f"Found {len(st_files)} safetensors files")

    # Configuration
    num_layers = 36
    hidden = 2880
    num_q_heads = 64
    num_kv_heads = 8
    head_dim = 64
    intermediate = 2880
    num_experts = 128

    params = {}

    for st_file in st_files:
        print(f"  Processing {os.path.basename(st_file)}...")
        with safe_open(st_file, framework="numpy") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                # Map safetensors name to Peregrine name
                mapped = map_weight(key, tensor, num_layers, hidden, num_q_heads,
                                    num_kv_heads, head_dim, intermediate, num_experts)
                if mapped is not None:
                    for name, data in mapped:
                        shape = list(data.shape)
                        params[name] = (shape, data.flatten().tolist())

    print(f"Converted {len(params)} parameter tensors")
    total = sum(int(np.prod(s)) for s, _ in params.values())
    print(f"Total parameters: {total:,}")

    save_peregrine_model(params, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved to {output_path} ({size_mb:.1f} MB)")


def map_weight(key, tensor, num_layers, hidden, num_q_heads, num_kv_heads,
               head_dim, intermediate, num_experts):
    """Map safetensors key to Peregrine name(s), transposing 2D weights.

    Returns list of (name, numpy_array) or None to skip.
    """
    # For MXFP4 tensors (.blocks/.scales), keep raw uint8; otherwise convert to f32
    is_raw_uint8 = (key.endswith('.blocks') or key.endswith('.scales'))
    if is_raw_uint8:
        data = tensor  # keep as uint8
    else:
        data = tensor.astype(np.float32) if tensor.dtype != np.float32 else tensor

    # === gpt-oss-120b naming (model.layers.N.*) ===

    # Embedding
    if key == "model.embed_tokens.weight":
        return [("embedding", data)]

    # Unembedding
    if key == "model.lm_head.weight":
        # Transpose [vocab, hidden] -> [hidden, vocab]
        return [("unembed", data.T)]

    # Final norm
    if key == "model.norm.weight":
        return [("final_norm.weight", data)]

    # === gpt-oss-20b naming (block.N.*, embedding.*, etc.) ===

    if key == "embedding.weight":
        return [("embedding", data)]

    if key == "unembedding.weight":
        return [("unembed", data.T)]

    if key == "norm.scale":
        return [("final_norm.weight", data)]

    # Per-layer weights
    import re

    # Match gpt-oss-20b naming: block.{n}.*
    m20 = re.match(r'block\.(\d+)\.(.*)', key)
    if m20:
        layer_idx = int(m20.group(1))
        rest = m20.group(2)
        prefix = f"layers.{layer_idx}"

        # Attention norm
        if rest == "attn.norm.scale":
            return [(f"{prefix}.attn_norm.weight", data)]

        # Attention QKV
        if rest == "attn.qkv.weight":
            return [(f"{prefix}.attn.qkv_weight", data.T)]
        if rest == "attn.qkv.bias":
            return [(f"{prefix}.attn.qkv_bias", data)]

        # Attention sinks
        if rest == "attn.sinks":
            return [(f"{prefix}.attn.sinks", data)]

        # Attention output
        if rest == "attn.out.weight":
            return [(f"{prefix}.attn.o_weight", data.T)]
        if rest == "attn.out.bias":
            return [(f"{prefix}.attn.o_bias", data)]

        # FFN norm
        if rest == "mlp.norm.scale":
            return [(f"{prefix}.ffn_norm.weight", data)]

        # MoE gate
        if rest == "mlp.gate.weight":
            return [(f"{prefix}.moe.gate", data.T)]
        if rest == "mlp.gate.bias":
            return [(f"{prefix}.moe.gate_bias", data)]

        # MXFP4 expert weights (20b uses these names)
        if rest == "mlp.mlp1_weight.blocks":
            results = []
            raw = tensor
            for e in range(raw.shape[0]):
                expert_bytes = raw[e].reshape(-1).astype(np.uint8)
                packed = np.frombuffer(expert_bytes.tobytes(), dtype=np.float32)
                results.append((f"{prefix}.moe.experts.{e}.mlp1_blocks", packed))
            return results
        if rest == "mlp.mlp1_weight.scales":
            results = []
            raw = tensor
            for e in range(raw.shape[0]):
                expert_bytes = raw[e].reshape(-1).astype(np.uint8)
                packed = np.frombuffer(expert_bytes.tobytes(), dtype=np.float32)
                results.append((f"{prefix}.moe.experts.{e}.mlp1_scales", packed))
            return results
        if rest == "mlp.mlp2_weight.blocks":
            results = []
            raw = tensor
            for e in range(raw.shape[0]):
                expert_bytes = raw[e].reshape(-1).astype(np.uint8)
                packed = np.frombuffer(expert_bytes.tobytes(), dtype=np.float32)
                results.append((f"{prefix}.moe.experts.{e}.mlp2_blocks", packed))
            return results
        if rest == "mlp.mlp2_weight.scales":
            results = []
            raw = tensor
            for e in range(raw.shape[0]):
                expert_bytes = raw[e].reshape(-1).astype(np.uint8)
                packed = np.frombuffer(expert_bytes.tobytes(), dtype=np.float32)
                results.append((f"{prefix}.moe.experts.{e}.mlp2_scales", packed))
            return results

        # Expert biases (20b)
        if rest == "mlp.mlp1_bias":
            results = []
            for e in range(data.shape[0]):
                results.append((f"{prefix}.moe.experts.{e}.mlp1_bias", data[e]))
            return results
        if rest == "mlp.mlp2_bias":
            results = []
            for e in range(data.shape[0]):
                results.append((f"{prefix}.moe.experts.{e}.mlp2_bias", data[e]))
            return results

        return None

    # Match gpt-oss-120b naming: model.layers.{n}.*
    m = re.match(r'model\.layers\.(\d+)\.(.*)', key)
    if not m:
        return None

    layer_idx = int(m.group(1))
    rest = m.group(2)
    prefix = f"layers.{layer_idx}"

    # Attention norms
    if rest == "input_layernorm.weight":
        return [(f"{prefix}.attn_norm.weight", data)]
    if rest == "post_attention_layernorm.weight":
        return [(f"{prefix}.ffn_norm.weight", data)]

    # Attention QKV (fused)
    if rest == "self_attn.qkv_proj.weight":
        # [total_qkv, hidden] -> transpose to [hidden, total_qkv]
        return [(f"{prefix}.attn.qkv_weight", data.T)]
    if rest == "self_attn.qkv_proj.bias":
        return [(f"{prefix}.attn.qkv_bias", data)]

    # Attention output
    if rest == "self_attn.o_proj.weight":
        # [hidden, num_q_heads * head_dim] -> transpose to [num_q_heads * head_dim, hidden]
        return [(f"{prefix}.attn.o_weight", data.T)]
    if rest == "self_attn.o_proj.bias":
        return [(f"{prefix}.attn.o_bias", data)]

    # Attention sinks
    if rest == "self_attn.sinks":
        return [(f"{prefix}.attn.sinks", data)]

    # MoE gate
    if rest == "mlp.gate.weight":
        # [num_experts, hidden] -> transpose to [hidden, num_experts]
        return [(f"{prefix}.moe.gate", data.T)]
    if rest == "mlp.gate.bias":
        return [(f"{prefix}.moe.gate_bias", data)]

    # MoE expert weights (MXFP4 quantized) — pack raw uint8 into f32 slots
    if rest == "mlp.mlp1_weight.blocks":
        # tensor is uint8 [num_experts, intermediate*2, hidden/32 * 16]
        # Split per expert and pack bytes as f32
        results = []
        raw = tensor  # keep as uint8
        for e in range(raw.shape[0]):
            expert_bytes = raw[e].reshape(-1).astype(np.uint8)
            packed = np.frombuffer(expert_bytes.tobytes(), dtype=np.float32)
            results.append((f"{prefix}.moe.experts.{e}.mlp1_blocks", packed))
        return results
    if rest == "mlp.mlp1_weight.scales":
        results = []
        raw = tensor
        for e in range(raw.shape[0]):
            expert_bytes = raw[e].reshape(-1).astype(np.uint8)
            packed = np.frombuffer(expert_bytes.tobytes(), dtype=np.float32)
            results.append((f"{prefix}.moe.experts.{e}.mlp1_scales", packed))
        return results
    if rest == "mlp.mlp2_weight.blocks":
        results = []
        raw = tensor
        for e in range(raw.shape[0]):
            expert_bytes = raw[e].reshape(-1).astype(np.uint8)
            packed = np.frombuffer(expert_bytes.tobytes(), dtype=np.float32)
            results.append((f"{prefix}.moe.experts.{e}.mlp2_blocks", packed))
        return results
    if rest == "mlp.mlp2_weight.scales":
        results = []
        raw = tensor
        for e in range(raw.shape[0]):
            expert_bytes = raw[e].reshape(-1).astype(np.uint8)
            packed = np.frombuffer(expert_bytes.tobytes(), dtype=np.float32)
            results.append((f"{prefix}.moe.experts.{e}.mlp2_scales", packed))
        return results

    # MoE expert biases (bf16 -> f32)
    if rest == "mlp.mlp1_bias":
        # [num_experts, intermediate*2] -> per-expert
        results = []
        for e in range(data.shape[0]):
            results.append((f"{prefix}.moe.experts.{e}.mlp1_bias", data[e]))
        return results
    if rest == "mlp.mlp2_bias":
        results = []
        for e in range(data.shape[0]):
            results.append((f"{prefix}.moe.experts.{e}.mlp2_bias", data[e]))
        return results

    # MoE expert weights (bfloat16 / direct, non-quantized path)
    if rest == "mlp.mlp1.weight":
        # Batched: [num_experts, intermediate*2, hidden] -> per-expert [hidden, intermediate*2]
        results = []
        for e in range(data.shape[0]):
            expert_w = data[e].T  # [out, in] -> [in, out]
            results.append((f"{prefix}.moe.experts.{e}.mlp1", expert_w))
        return results
    if rest == "mlp.mlp1.bias":
        # [num_experts, intermediate*2] -> per-expert
        results = []
        for e in range(data.shape[0]):
            results.append((f"{prefix}.moe.experts.{e}.mlp1_bias", data[e]))
        return results
    if rest == "mlp.mlp2.weight":
        # Batched: [num_experts, hidden, intermediate] -> per-expert [intermediate, hidden]
        results = []
        for e in range(data.shape[0]):
            expert_w = data[e].T
            results.append((f"{prefix}.moe.experts.{e}.mlp2", expert_w))
        return results
    if rest == "mlp.mlp2.bias":
        results = []
        for e in range(data.shape[0]):
            results.append((f"{prefix}.moe.experts.{e}.mlp2_bias", data[e]))
        return results

    return None


def generate_random_weights(output_path, quantized=False):
    """Generate random weights for the small test config.

    If quantized=True, generate MXFP4-packed expert weights (blocks/scales as u8 in f32 slots).
    """
    mode = "quantized MXFP4" if quantized else "f32"
    print(f"Generating random weights for small test config ({mode})...")

    vocab_size = 1024
    model_dim = 256
    num_layers = 2
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    num_experts = 8
    top_k = 2
    intermediate = 256

    params = {}

    # Embedding
    emb = np.random.randn(vocab_size, model_dim).astype(np.float32) * 0.02
    params["embedding"] = (list(emb.shape), emb.flatten().tolist())

    # Final norm
    params["final_norm.weight"] = ([model_dim], [1.0] * model_dim)

    # Unembed (separate from embedding)
    unembed = np.random.randn(model_dim, vocab_size).astype(np.float32) * 0.02
    params["unembed"] = (list(unembed.shape), unembed.flatten().tolist())

    total_qkv = (num_q_heads + 2 * num_kv_heads) * head_dim

    for i in range(num_layers):
        p = f"layers.{i}"

        # RMSNorm weights
        params[f"{p}.attn_norm.weight"] = ([model_dim], [1.0] * model_dim)
        params[f"{p}.ffn_norm.weight"] = ([model_dim], [1.0] * model_dim)

        def rand_weight(rows, cols):
            w = np.random.randn(rows, cols).astype(np.float32) * 0.02
            return (list(w.shape), w.flatten().tolist())

        def rand_bias(n):
            b = np.random.randn(n).astype(np.float32) * 0.02
            return ([n], b.tolist())

        # Attention (fused QKV)
        params[f"{p}.attn.qkv_weight"] = rand_weight(model_dim, total_qkv)
        params[f"{p}.attn.qkv_bias"] = rand_bias(total_qkv)
        params[f"{p}.attn.o_weight"] = rand_weight(num_q_heads * head_dim, model_dim)
        params[f"{p}.attn.o_bias"] = rand_bias(model_dim)
        params[f"{p}.attn.sinks"] = ([num_q_heads], [0.0] * num_q_heads)

        # MoE gate
        params[f"{p}.moe.gate"] = rand_weight(model_dim, num_experts)
        params[f"{p}.moe.gate_bias"] = rand_bias(num_experts)

        # Experts
        for j in range(num_experts):
            ep = f"{p}.moe.experts.{j}"
            if quantized:
                # MXFP4: generate random uint8 blocks/scales packed into f32 slots
                # mlp1: weight [intermediate*2, model_dim]
                mlp1_out = intermediate * 2
                mlp1_bpr = model_dim // 32  # blocks per row
                mlp1_block_bytes = np.random.randint(0, 256, size=mlp1_out * mlp1_bpr * 16, dtype=np.uint8)
                mlp1_scale_bytes = np.random.randint(100, 140, size=mlp1_out * mlp1_bpr, dtype=np.uint8)
                # Pack u8 as f32
                mlp1_blocks_f32 = np.frombuffer(mlp1_block_bytes.tobytes(), dtype=np.float32)
                mlp1_scales_f32 = np.frombuffer(mlp1_scale_bytes.tobytes(), dtype=np.float32)
                params[f"{ep}.mlp1_blocks"] = (list(mlp1_blocks_f32.shape), mlp1_blocks_f32.tolist())
                params[f"{ep}.mlp1_scales"] = (list(mlp1_scales_f32.shape), mlp1_scales_f32.tolist())

                # mlp2: weight [model_dim, intermediate]
                mlp2_out = model_dim
                mlp2_bpr = intermediate // 32
                mlp2_block_bytes = np.random.randint(0, 256, size=mlp2_out * mlp2_bpr * 16, dtype=np.uint8)
                mlp2_scale_bytes = np.random.randint(100, 140, size=mlp2_out * mlp2_bpr, dtype=np.uint8)
                mlp2_blocks_f32 = np.frombuffer(mlp2_block_bytes.tobytes(), dtype=np.float32)
                mlp2_scales_f32 = np.frombuffer(mlp2_scale_bytes.tobytes(), dtype=np.float32)
                params[f"{ep}.mlp2_blocks"] = (list(mlp2_blocks_f32.shape), mlp2_blocks_f32.tolist())
                params[f"{ep}.mlp2_scales"] = (list(mlp2_scales_f32.shape), mlp2_scales_f32.tolist())
            else:
                params[f"{ep}.mlp1"] = rand_weight(model_dim, intermediate * 2)
                params[f"{ep}.mlp2"] = rand_weight(intermediate, model_dim)

            params[f"{ep}.mlp1_bias"] = rand_bias(intermediate * 2)
            params[f"{ep}.mlp2_bias"] = rand_bias(model_dim)

    print(f"Generated {len(params)} parameter tensors")
    total = sum(int(np.prod(s)) for s, _ in params.values())
    print(f"Total parameters: {total:,}")

    save_peregrine_model(params, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved to {output_path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python scripts/convert_gpt_oss.py <safetensors_dir> <output.bin>")
        print("       python scripts/convert_gpt_oss.py --random [--quantized] <output.bin>")
        print()
        print("Options:")
        print("  --random      Generate random weights for small test config")
        print("  --quantized   Generate MXFP4 packed MoE weights (with --random)")
        sys.exit(1)

    if sys.argv[1] == "--random":
        quantized = "--quantized" in sys.argv
        output_path = [a for a in sys.argv[2:] if not a.startswith("--")]
        if not output_path:
            print("Error: missing output path")
            sys.exit(1)
        generate_random_weights(output_path[0], quantized=quantized)
    else:
        convert_safetensors(sys.argv[1], sys.argv[2])
