# RT-DETR Example

## Architecture

```
Input Image [1, 3, 256, 256]
          │
    ┌─────▼────┐
    │  ResNet  │  4-stage backbone with residual connections
    │ Backbone │  1x1 stem → 3 stages of 3x3 conv blocks + skip + pool
    └─┬──┬──┬──┘
      s2 s3 s4    multi-scale features [128², 64², 32²]
      │  │  │
    ┌─▼──▼──▼──┐
    │ Channel  │  1x1 conv projections to embed_dim
    │  Project │  pool all scales to common 32×32
    └─────┬────┘
          │  [batch, 3072, embed_dim]
    ┌─────▼────┐
    │   Xfmr   │  self-attention + FFN with pre-norm
    │ Encoder  │
    └─────┬────┘
          │  encoder memory
    ┌─────▼────┐
    │   Xfmr   │  learned object queries attend to memory
    │ Decoder  │  self-attn → cross-attn → FFN
    └─────┬────┘
          │  [batch, num_queries, embed_dim]
    ┌─────▼────┐
    │  Heads   │  classification (softmax) + bbox regression (sigmoid)
    └──────────┘
```
