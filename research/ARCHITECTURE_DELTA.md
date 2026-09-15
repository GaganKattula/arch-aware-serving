# Architecture Delta — Baseline Server → Qwen3.8-27B

> What the existing inference server (A0–A4, committed) assumes, what Qwen3.8-27B actually is,
> and every place those two disagree.
>
> Written 2026-09-08. Sources: `model-configs/*.json` pulled from HuggingFace.

---

## 1. What the target model actually is

`Qwen/Qwen3.8-27B` — released 2026-08-14, Apache 2.0, 27.78B params, 18 safetensors
shards (~55.6 GB bf16), `Qwen3_5ForConditionalGeneration`.

It is **dense**, but it is **not a uniform transformer**. The decoder is a *hybrid*:

```
num_hidden_layers:       64
full_attention_interval:  4
layer_types:             [linear, linear, linear, full] × 16
```

- **16 layers** are conventional full attention with a KV cache.
- **48 layers** are gated linear attention (Gated DeltaNet lineage) carrying a
  **fixed-size recurrent state** plus a short causal conv state
  (`linear_conv_kernel_dim: 4`, `mamba_ssm_dtype: float32`).

This is the reason a 27B model is newly practical to serve on one GPU. It is not
quantization and it is not MoE. It is that **three quarters of the layers have
O(1) memory in sequence length.**

### Full config, text side

| Field | Value | Baseline server has it? |
|---|---|---|
| `hidden_size` | 5120 | yes |
| `num_hidden_layers` | 64 | yes |
| `num_attention_heads` | 24 | yes |
| `num_key_value_heads` | 4 | yes (GQA) |
| `head_dim` | 256 | yes |
| `intermediate_size` | 17408 | yes |
| `hidden_act` | silu (SwiGLU) | yes |
| `rms_norm_eps` | 1e-6 | yes |
| `vocab_size` | 248320 | yes |
| `max_position_embeddings` | 262144 | yes |
| `layer_types` | 48 linear / 16 full | **NO** |
| `full_attention_interval` | 4 | **NO** |
| `linear_num_key_heads` | 16 | **NO** |
| `linear_num_value_heads` | 48 | **NO** |
| `linear_key_head_dim` | 128 | **NO** |
| `linear_value_head_dim` | 128 | **NO** |
| `linear_conv_kernel_dim` | 4 | **NO** |
| `output_gate_type` | swish | **NO** |
| `attn_output_gate` | true | **NO** |
| `partial_rotary_factor` | 0.25 | **NO** |
| `mrope_section` | [11, 11, 10] | **NO** |
| `mrope_interleaved` | true | **NO** |
| `mtp_num_hidden_layers` | 1 | **NO** (free spec-decode draft) |
| `rope_theta` | 1e7 | yes |
| `tie_word_embeddings` | false | check |

Plus a 27-layer ViT vision tower (`patch_size 16`, `spatial_merge_size 2`,
`out_hidden_size 5120`). **Out of scope** — see ROADMAP_C.

---

## 2. The memory number that drives every design decision

### KV cache (16 full-attention layers only)

```
per token, per full-attn layer = 2 (K,V) × 4 kv_heads × 256 head_dim × 2 B = 4,096 B
per token, all 16 full layers                                              = 65,536 B = 64 KB
```

Had all 64 layers been full attention it would be **256 KB/token**. The hybrid is
a straight 4× reduction in KV.

### Linear-attention recurrent state (48 layers)

```
per layer = 48 value_heads × 128 key_head_dim × 128 value_head_dim × 4 B (fp32) = 3.15 MB
all 48 layers                                                                   ≈ 151 MB
```

Plus a small causal-conv state (kernel width 4) per layer — negligible by comparison.

**This is constant. It does not grow with context length.**

### The crossover — the central insight of this project

```
151 MB / 64 KB per token ≈ 2,300 tokens
```

- **Below ~2,300 tokens of context**, a sequence's memory cost is dominated by a
  *flat* 151 MB that the baseline scheduler has no concept of.
- **Above ~2,300 tokens**, the paged KV cache dominates and the baseline model
  applies again.

The baseline `Scheduler` budgets admission purely as "blocks needed ∝ sequence
length." That is wrong here in both directions: it under-charges short requests
(missing 151 MB of fixed cost) and over-charges long ones (KV is 4× smaller than
a uniform model of this depth).

### Fitting on one GPU (A100/H100 80 GB, bf16)

```
weights                       55.6 GB
headroom                     ~24   GB
÷ 64 KB/token                ≈ 390k tokens of KV budget total
− 151 MB × concurrent seqs   ← the term the baseline scheduler doesn't model
```

At 16 concurrent sequences the linear state alone is ~2.4 GB — 10% of headroom
before a single KV block is allocated.

### The same math on the dev model (Qwen3.5-0.8B)

```
full layers 6, kv_heads 2, head_dim 256  →  12 KB/token
linear state 18 layers × 1.05 MB          →  ~19 MB/seq
crossover                                 ≈ 1,540 tokens
```

The crossover is reachable on MPS. **The scheduling insight is testable locally
before spending a dollar on cloud.**

---

## 3. Component-by-component gap list

### `src/config.py` — ModelConfig

- Everything now lives under `config.text_config`, not top-level.
  `from_hf()` will `AttributeError` on the first field.
- Needs new fields: `layer_types`, `full_attention_interval`, the four
  `linear_*` dims, `linear_conv_kernel_dim`, `attn_output_gate`,
  `output_gate_type`, `partial_rotary_factor`, `mrope_section`,
  `mrope_interleaved`, `mamba_ssm_dtype`.
- `num_blocks` / `block_size` now describe only 16 of 64 layers. A second
  budget field is needed for linear state.

### `src/model.py`

| Class | Change |
|---|---|
| `RoPE` | Two changes. (a) `partial_rotary_factor: 0.25` — rotate only the first 64 of 256 head dims, pass the rest through unrotated. (b) **mRoPE**: positions become 3D (t, h, w) split `[11, 11, 10]` across the rotary dims, interleaved. Current class is 1D full-rotation. For text-only, all three sections take the same position value — worth confirming against HF's implementation rather than assuming. |
| `AttentionGQA` | Add output gating (`attn_output_gate: true`). head_dim 256 with hidden 5120 and 24 heads means `24 × 256 = 6144 ≠ 5120` — the q projection is **not** square. Baseline may assume `n_heads × head_dim == d_model`. **Check this first; it is the kind of silent-shape bug the A0 postmortem warns about.** |
| *new* `GatedDeltaNet` | The whole 48-layer path. Causal conv (width 4) → gated delta recurrence → swish output gate. Needs both a chunked/parallel form (prefill) and a recurrent step form (decode). This is the single largest piece of new work. |
| `TransformerBlock` | Must dispatch on `layer_types[i]`. |
| `Decoder` | Carries two cache structures, not one. |
| `load_hf_weights` | 18 shards + `model.safetensors.index.json`. Must stream shard-by-shard or OOM. |

### `src/paged_cache.py`

`BlockAllocator` / `BlockTable` / `attention_kernel` remain **correct but now
apply to only 16 of 64 layers.** Nothing here breaks; it just becomes half the
memory story. Needs a sibling manager for linear state — fixed-size slab
allocation, no paging, no block table, different eviction semantics.

### `src/scheduler.py`

The deepest changes.

- **Admission control**: budget becomes two-dimensional — KV blocks *and* linear
  state slots. A request can be admissible on one axis and not the other.
- **Preemption**: baseline weighs swap-to-CPU vs recompute for KV. For linear
  state there is no partial recompute — the recurrence must be replayed over the
  entire prefix from scratch. Cost curve differs; the A2 decision needs
  re-measuring, not assuming.
- **Chunked prefill (A3)**: still valid, but the linear layers must thread
  recurrent state across chunks correctly. This is a likely source of silent
  correctness bugs — the equivalence gate matters more here than anywhere.

### `src/server.py`

Does not exist. No HTTP layer in the baseline at all.

---

## 4. Validation ladder

The linear-attention GQA broadcast ratio scales with model size, so the dev model
does **not** exercise the target's code path:

| Model | Layers | lin k/v heads | Broadcast | bf16 | Role |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 24 (18/6) | 16 / 16 | 1:1 | 1.6 GB | MPS dev loop, parity gate |
| Qwen3.5-2B | 24 (18/6) | 16 / 16 | 1:1 | 4 GB | larger local check |
| Qwen3.5-9B | 32 | 16 / 32 | **2:1** | 18 GB | **validates broadcast** |
| Qwen3.8-27B | 64 (48/16) | 16 / 48 | **3:1** | 55.6 GB | target |

**Risk:** a parity test that passes on 0.8B and 2B proves nothing about the
broadcast. 9B must pass before the first 27B cloud run.

Every other structural field is identical across all four — `head_dim 256`,
`full_attention_interval 4`, `linear_*_head_dim 128`, `linear_conv_kernel_dim 4`,
`partial_rotary_factor 0.25`, `mrope_section [11,11,10]`, `rope_theta 1e7`,
`vocab_size 248320`. The 0.8B is a faithful structural proxy in every respect
except the broadcast.

---

## 5. Free wins in the config

- `mtp_num_hidden_layers: 1` — a trained multi-token-prediction head ships in the
  weights. That is a speculative-decoding draft you do not have to train or host
  separately. Roadmap B3 gets substantially cheaper.
- `language_model_only` flag exists in the 27B config — text-only operation looks
  supported, supporting the decision to cut the vision tower.

---

## 6. Verified against the HF reference implementation (C0, 2026-09-08)

Read from `transformers==5.16.1`,
`transformers/models/qwen3_5/modeling_qwen3_5.py` (2033 lines). This section
supersedes §3 where they disagree — §3 was written from `config.json` alone.

### Confirmed

- **State shape.** `last_recurrent_state` is `(batch, num_v_heads, head_k_dim,
  head_v_dim)` and the chunked path casts to **fp32**. The §2 arithmetic
  (48 × 128 × 128 × 4 B × 48 layers ≈ 151 MB) is exact, not an estimate.
- **Both forms ship as readable PyTorch**: `torch_chunk_gated_delta_rule` (prefill,
  chunk_size=64) and `torch_recurrent_gated_delta_rule` (decode). These are the
  C2 reference implementations.
- **Text-only entry point exists**: `Qwen3_5ForCausalLM` and `Qwen3_5TextModel`,
  separate from `Qwen3_5ForConditionalGeneration`. Cutting the vision tower is
  supported by the library, not a hack.
- `o_proj` is `(num_heads × head_dim) → hidden_size` = 6144 → 5120. Non-square, as predicted.

### Corrections to §3

**`AttentionGQA` needs more than output gating.** Three changes, not one:

1. **`q_proj` is double width.** `nn.Linear(hidden_size, num_heads × head_dim × 2)`.
   The output is `torch.chunk`'d into `query_states` and `gate`. For the 27B that's
   5120 → 12288, not 5120 → 6144.
2. **The gate is `sigmoid`, not swish.** `attn_output * torch.sigmoid(gate)`, applied
   *after* the attention interface and *before* `o_proj`. `output_gate_type: swish`
   in the config refers to the GatedDeltaNet path, not this one. Easy to conflate.
3. **QK-RMSNorm — missing from §3 entirely.** `q_norm` and `k_norm` are
   `Qwen3_5RMSNorm(head_dim)` applied to q and k **before RoPE**. The baseline
   `AttentionGQA` has no equivalent. This is a silent-numerics bug waiting to happen:
   shapes are unaffected, so nothing will error — parity will just fail by a margin
   that looks like a tolerance problem.

**`GatedDeltaNet` projection layout** (was unspecified in §3):

```
conv_dim    = key_dim × 2 + value_dim        # 27B: 2048×2 + 6144 = 10240
conv1d      = Conv1d(conv_dim, conv_dim, kernel_size=4, groups=conv_dim, bias=False)  # depthwise
in_proj_qkv = Linear(hidden, key_dim×2 + value_dim, bias=False)
in_proj_z   = Linear(hidden, value_dim, bias=False)     # output gate
in_proj_b   = Linear(hidden, num_v_heads, bias=False)   # beta
in_proj_a   = Linear(hidden, num_v_heads, bias=False)   # alpha / decay
dt_bias     = Parameter(num_v_heads)
A_log       = Parameter(num_v_heads)
norm        = Qwen3_5RMSNormGated(head_v_dim)
out_proj    = Linear(value_dim, hidden, bias=False)
```

The conv is **depthwise** (`groups=conv_dim`) — one filter per channel, not a full
convolution. Conv state per layer is `conv_dim × (kernel-1)` ≈ 123 KB for the 27B,
so ~5.9 MB across 48 layers. Small next to 151 MB but not zero; budget it.

### New risk: the FLA kernel swap

Both delta-rule functions are decorated:

```python
@use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
```

HF will use a **flash-linear-attention** kernel from the Hub when one is available
and silently fall back to the pure-PyTorch path when it isn't. On MPS you get the
torch path. **On CUDA you may get the FLA kernel instead** — different numerics.

Implication for the validation ladder: a parity gate that passes on MPS against the
torch path is not automatically a parity gate against what HF runs on the A100. Pin
the reference explicitly when capturing targets, and re-verify on first cloud
bring-up rather than assuming the tolerance transfers.

### Still to confirm in C1

- `apply_interleaved_mrope` (line 149) — does it collapse to standard 1D RoPE for
  text-only input, as §3 assumes? Read `get_rope_index` / `compute_3d_position_ids`.
- `partial_rotary_factor` handling in `Qwen3_5TextRotaryEmbedding`.
