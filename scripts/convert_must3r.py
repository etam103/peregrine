#!/usr/bin/env python3
"""Convert MUSt3R PyTorch checkpoint to Peregrine binary format.

Usage:
    python scripts/convert_must3r.py <input.pth> <output.bin>

Example:
    python scripts/convert_must3r.py MUSt3R_224_cvpr.pth weights/must3r_224.bin

The MUSt3R checkpoints can be downloaded from:
    https://github.com/naver/must3r
"""
import sys
import struct
import torch
import numpy as np


def save_peregrine_model(params: dict, path: str):
    """Save parameters in Peregrine binary format.

    Format: [num_tensors: u32]
    Per tensor: [name_len: u32][name: utf8][ndim: u32][shape: u32*ndim][data: f32*N]
    """
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(params)))

        for name, (shape, data) in params.items():
            # Name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            # Shape
            f.write(struct.pack('<I', len(shape)))
            for s in shape:
                f.write(struct.pack('<I', s))

            # Data (f32 little-endian)
            f.write(np.array(data, dtype=np.float32).tobytes())


def convert_must3r(input_path: str, output_path: str):
    """Convert a MUSt3R PyTorch checkpoint to Peregrine format."""
    print(f"Loading {input_path}...")

    # Load checkpoint
    ckpt = torch.load(input_path, map_location='cpu', weights_only=False)

    # The checkpoint may have the model under different keys
    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt

    print(f"Found {len(state_dict)} parameters in checkpoint")

    # Map PyTorch parameter names to Peregrine names
    # MUSt3R state dict has keys like:
    #   patch_embed.proj.weight [1024, 3, 16, 16]
    #   patch_embed.proj.bias [1024]
    #   enc_blocks.{i}.norm1.weight [1024]
    #   enc_blocks.{i}.attn.qkv.weight [3072, 1024]
    #   enc_blocks.{i}.attn.proj.weight [1024, 1024]
    #   enc_blocks.{i}.norm2.weight [1024]
    #   enc_blocks.{i}.mlp.fc1.weight [4096, 1024]
    #   enc_blocks.{i}.mlp.fc2.weight [1024, 4096]
    #   enc_norm.weight [1024]
    #   dec_blocks.{i}.norm1.weight [768]
    #   dec_blocks.{i}.attn.qkv.weight [2304, 768]
    #   ...
    #   dec_blocks.{i}.cross_attn.proj{q,k,v}.weight [768, 768]
    #   ...
    #   dec_norm.weight [768]
    #   downstream_head1.proj.weight [1792, 768]
    #   downstream_head2.proj.weight [1792, 768]

    # Some checkpoints use different naming, let's handle the common patterns
    params = {}
    skipped = []

    # Detect naming pattern
    has_blocks_enc = any(k.startswith('blocks_enc.') for k in state_dict.keys())
    has_enc_blocks = any(k.startswith('enc_blocks.') for k in state_dict.keys())

    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        data = tensor.float().detach().numpy().flatten().tolist()
        shape = list(tensor.shape)

        # Normalize key names to our convention
        name = key

        # Handle different naming conventions
        if has_enc_blocks:
            # DUSt3R/older naming: enc_blocks -> blocks_enc
            name = name.replace('enc_blocks.', 'blocks_enc.')
            name = name.replace('dec_blocks.', 'blocks_dec.')
            name = name.replace('enc_norm.', 'norm_enc.')
            name = name.replace('dec_norm.', 'norm_dec.')

        # Handle head naming
        if 'downstream_head1' in name:
            name = name.replace('downstream_head1', 'head_dec')
        elif 'downstream_head2' in name:
            # Skip head2 (we only need one head for basic inference)
            skipped.append(key)
            continue

        # Handle encoder-decoder projection
        if name == 'decoder_embed.weight':
            name = 'feat_embed_enc_to_dec.weight'
        elif name == 'decoder_embed.bias':
            name = 'feat_embed_enc_to_dec.bias'

        # Skip non-parameter entries
        if '_orig_mod.' in name:
            name = name.replace('_orig_mod.', '')

        params[name] = (shape, data)

    print(f"Converted {len(params)} parameters ({len(skipped)} skipped)")

    # Print parameter summary
    total_params = sum(np.prod(shape) for shape, _ in params.values())
    print(f"Total parameters: {total_params:,}")

    # Show some sample keys for verification
    print("\nSample parameter keys:")
    shown = 0
    for key, (shape, _) in params.items():
        if shown < 10 or 'norm_enc' in key or 'head_dec' in key or 'feat_embed' in key:
            print(f"  {key}: {shape}")
            shown += 1
        elif shown == 10:
            print(f"  ... ({len(params) - 10} more)")
            shown += 1

    # Save
    print(f"\nSaving to {output_path}...")
    save_peregrine_model(params, output_path)

    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done! Output: {size_mb:.1f} MB")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python scripts/convert_must3r.py <input.pth> <output.bin>")
        print()
        print("Download checkpoints from: https://github.com/naver/must3r")
        print("Example: python scripts/convert_must3r.py MUSt3R_224_cvpr.pth weights/must3r_224.bin")
        sys.exit(1)

    convert_must3r(sys.argv[1], sys.argv[2])
