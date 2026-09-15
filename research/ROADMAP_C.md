# Roadmap — C Phases: Serving Qwen3.8-27B

> Continuation of `baseline-docs/ROADMAP.md`.
> Lettered C to avoid clashing with the B phases, which are unstarted.
>
> Written 2026-09-08.

## Actual baseline state (verified from git, not from the session docs)

| Phase | State |
|---|---|
| A0 decoder + weight parity | **committed** (`6b3e39b`) |
| A1 paged KV-cache | **committed** (`6b3e39b`, `336d3b4`) |
| A2 continuous batching scheduler | **committed** (`827bc67`) |
| A3 prefill/decode separation + chunked prefill | **committed** (`64e772e`) |
| A4 benchmarks + lambda sweep + README | **committed** (`d3cd6c7`, `f7c524c`) |
| A5 technical analysis document | **SKIPPED** (decided 2026-09-08) — superseded by C7 |
| B1–B5 (Triton, prefix caching, spec decoding, quantization, A100 bench) | **never started — roadmap only** |

Everything the C phases build on is committed and working code. Every B phase
referenced below is greenfield, not resumable work.

### A5 is dropped

The A5 technical analysis document was written for the dense server and is
superseded by C7. It will not be finished. Consequences:

- **C7 is the only writeup**, so it must carry a short A-phase foundation section —
  paged cache, continuous batching, chunked prefill — because no other document
  explains that work at depth. The committed `README.md` is the only existing
  account and it is a feature list, not an analysis.
- The A4 benchmark results (`bench/results.json`, the λ sweep, the λ=8 paged
  inversion) stay valuable as the **dense baseline C6 measures against**. Skipping
  A5 drops the *document*, not the data.
- Untracked A5-era working files still have value independent of the document:
  `bench/plot_styled.py` and `bench/cover_image.py` are working code, and
  `bench/BUG_SUMMARY.md` is a debugging record. They exist only in the original
  working directory and in `baseline/`. Committing them upstream is a loose end.

## Scope

Extend the from-scratch inference server to serve **Qwen3.8-27B, text-only**, on a
rented A100/H100 80GB — and in doing so, build the thing the baseline server has
no concept of: a **scheduler that manages two cache types with opposite cost
curves.**

The differentiator is unchanged from the A phases: the decision trail. What is new
is that the central design problem is genuinely novel rather than a reimplementation
of vLLM's known-good answer. Hybrid linear/full attention serving is new enough that
"how should admission control work when 3/4 of layers have O(1) memory" does not
have a settled textbook answer.

### Explicitly out of scope
- **Vision tower** (27-layer ViT). Adds an encoder, projector, and image-token
  accounting; teaches nothing about serving. Stretch phase only.
- **Training / fine-tuning.** Inference only.
- **Multi-GPU.** 27B bf16 fits on one 80GB card. Tensor parallelism is a different project.

---

## Collaboration contract (carried forward from ROADMAP.md, unchanged)

**Gagan codes. Claude reviews and documents.**

- Claude provides directional guidance, derivation prompts, and code review.
- Claude does **NOT** write implementation code — narrows the problem when stuck,
  never solves it.
- Claude asks questions that force derivation before implementation.
- Claude maintains `log.md` and `CONTEXT.md` (see below).
- Gagan writes the technical analysis document.
- Every design decision measured and documented.

## Session-continuity files — REQUIRED

Two files, different jobs. Both live in `research/`. This split exists to survive
Claude Code context compaction: when context is summarized mid-project, `log.md`
is the durable record and `CONTEXT.md` is the fast re-entry point.

| File | Discipline | Purpose |
|---|---|---|
| **`log.md`** | **APPEND-ONLY.** Never edit or delete prior entries. Never overwrite. Newest at the bottom, one `## Session N` heading per session. | Chronological trace with full reasoning: what was decided and *why*, what was tried and failed, corrections made. The durable record. |
| **`CONTEXT.md`** | **OVERWRITTEN every session.** | Current state only. Where the project stands, what's blocking, what's next, gotchas not to re-learn. Read this first on any new session. |

Rules:
- Update both at the end of every working session, and before any likely compaction.
- Never delete from `log.md` — if something recorded there turns out to be wrong,
  append the correction as a new entry rather than editing the old one. The record
  of having been wrong is part of the decision trail.
- `CONTEXT.md` carries a "Gotchas discovered" section. Anything that cost more than
  a few minutes to figure out goes there so it is never re-derived.

## Validation discipline (carried forward)

Every phase has an equivalence gate. No proceeding until it passes. The lesson from
the A0 positional-encoding-offset bug applies with more force here: the gated delta
recurrence has many ways to be shape-correct and numerically wrong.

---

## Repository layout

```
~/Documents/skill_dev/model_serving/
├── .venv/
├── baseline/          # clone of inference-server @ main — A0–A4, installed as a package
│   ├── pyproject.toml #   declares package `inference_baseline`
│   ├── inference_baseline/  # was src/ — renamed so the import name isn't generic
│   │                  #   config.py, model.py, paged_cache.py, scheduler.py, utils.py
│   ├── bench/         #   load_gen, measure, run, plots
│   └── tests/         #   parity + equivalence suites
├── research/          # notes, configs, derivations, this file
│   ├── ARCHITECTURE_DELTA.md
│   ├── ROADMAP_C.md
│   ├── NEOVIM_PLAN.md
│   ├── baseline-docs/     # private A-phase decision trail (gitignored in the public repo)
│   └── model-configs/     # qwen3.5-{0.8b,2b,9b}, qwen3.8-27b config.json
└── hybrid/            # NEW WORK — created in C0
```

`baseline/` keeps its own git history and both remotes (`origin` → GitHub,
`local` → the original working dir). All new code lands in `hybrid/`.

**Baseline is an installed package** (decided 2026-09-09, resolves Open Question #1).
`uv pip install -e baseline/` puts `inference_baseline` on the path, importable from
anywhere in the venv:

```python
from inference_baseline.paged_cache import BlockAllocator, BlockTable
from inference_baseline.scheduler import Scheduler, Request
```

`src/` was renamed to `inference_baseline/` because internal imports were absolute
(`from src.paged_cache import ...`), so a `package-dir` mapping alone would have
broken at runtime. `src` is also too generic to claim as a global import name once
two packages exist. Rejected alternative: forking the modules into `hybrid/` —
rejected in favour of a real dependency edge.

---

## Phases

### C0 — Environment, repo, and dev-model bring-up
**Goal:** everything in place to iterate, nothing built yet.

- `hybrid/` package skeleton; decide the import path to `baseline/` (see Open Questions)
- Pull `Qwen/Qwen3.5-0.8B` locally; confirm it runs under HF transformers on MPS
- Reference forward pass captured: fixed prompt → logits, saved to disk as the
  parity target for every subsequent phase
- Read HF's `modeling_qwen3_5.py` for the linear-attention layer — this is the
  reference implementation you are validating against
- **Neovim from scratch** (parallel track, see `NEOVIM_PLAN.md`)

**Gate:** `pytest` runs green against the ported baseline test suite; HF reference
logits for 0.8B saved and reproducible.

#### C0 status (2026-09-08)

Done:
- `.venv` provisioned: **torch 2.14.0, transformers 5.16.1**, MPS available.
  `qwen3_5` + `qwen3_5_text` are registered in the released library — no dev
  install needed, despite the 27B config declaring `5.8.0.dev0`.
- `Qwen/Qwen3.5-0.8B` pulled (1.7 GB). Note it ships a
  `model.safetensors.index.json` **even as a single shard** — so the C1 streaming
  loader can be developed and tested locally, not only against the 27B.
- HF reference implementation read; findings in `ARCHITECTURE_DELTA.md §6`,
  which supersedes §3 in three places (QK-norm, double-width `q_proj`,
  sigmoid-not-swish gate).

**Blocker found — `transformers` 5.x moved `rope_theta`.** It now lives in a
`rope_parameters` dict, not as a config attribute:

```python
c.rope_theta                        # AttributeError on 5.x
c.rope_parameters["rope_theta"]     # 5.x location
```

This breaks exactly one line — `baseline/src/config.py:54` — which cascades to
`bench/test_bench.py` failing at *collection* time, so the baseline suite won't
even enumerate. Everything else in `from_hf()` survives the upgrade.

Decision needed (see Open Questions #5). This matters beyond C0: **C6 measures the
hybrid scheduler against the dense baseline**, which requires the baseline to
actually run.

**Model verified working on MPS.** `Qwen3_5ForCausalLM` loads 0.752B params and
runs a forward pass in ~12 s cold, ~1 s warm. `"The capital of France is"` →
`' Paris'`, identical top-5 on CPU and MPS. All five load paths tested clean
(eager / sdpa / default × `.to("mps")` / `device_map="mps"`), each under 1.5 s.

**One unexplained incident, recorded for the cloud run.** The first invocation
burned ~40 min at 393% CPU and produced nothing. It has not reproduced. Ruled out:
network hang (it was CPU-bound, not idle), `device_map` vs `.to()`, attention
implementation, and kernel-hub compilation (`kernels`, `fla-core`, and `triton`
are all absent, so the delta-rule decorators can only take the pure-torch fallback;
no cache was written during the incident). Cause unknown. If a first-run stall
recurs on the A100, do not assume it is the same thing — instrument before
theorising, and run with `python -u`, because piping to `tail` buffers all output
and hides which stage is stuck.

Remaining:
- Decide the `hybrid/` → `baseline/` import path (Open Questions #1)
- Decide the `rope_theta` fix (Open Questions #5)
- Write the reference-capture script and save 0.8B parity targets
- `hybrid/` package skeleton
- Neovim Stages 0–1

**Working recipe for the reference-capture script:**
```
Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16)
# then .to("mps"); pin attn_implementation explicitly rather than taking the default
```
Pin the attention implementation when saving parity targets — the default is
resolved at load time and is not guaranteed stable across environments.

---

### C1 — Config + sharded weight loading + hybrid decoder skeleton
**Goal:** the model loads and dispatches per-layer. No linear attention yet.

**What to derive before coding:**
- [ ] Why `num_attention_heads × head_dim` (24 × 256 = 6144) ≠ `hidden_size` (5120).
      What does that mean for the shapes of `W_q`, `W_o`? Does the baseline
      `AttentionGQA` silently assume they're equal?
- [ ] `partial_rotary_factor: 0.25` — which 64 of the 256 head dims get rotated,
      and why would a model rotate only a fraction?
- [ ] mRoPE `[11, 11, 10]` interleaved — what are the three sections, and what do
      they collapse to for text-only input?
- [ ] Streaming shard load: why does `torch.load` of a full state_dict OOM when
      the model itself would fit?
- [ ] Which config fields are determined by the *checkpoint* and which by your
      *hardware*? Draw the line before writing `CacheConfig` — getting it wrong is
      what caused the silent weight drop.

**What to build:**
- **`CacheConfig` dataclass — do this first.** Split runtime/cache parameters
  (`num_blocks`, `block_size`, and in C3 the linear-state budget) out of
  `ModelConfig`, so `from_hf` can fill its object completely and honestly.
  This is the root-cause fix for the silent weight drop — see
  `POSTMORTEM_silent_weight_drop.md`. It also unblocks the dormant A0 parity gate.
- **Re-run the A0 parity gate** immediately after, and find out what else three
  phases of a non-running gate concealed.
- `load_hf_weights` reports key mismatches — diff expected vs provided keys. Keep
  `strict=False` for the RoPE buffers, but stop it silencing unexpected keys.
- Bias becomes **config-driven** (`attention_bias`) rather than hardcoded `bias=False`
- `ModelConfig` reading from `text_config`, with all `linear_*` fields
- Streaming loader over `model.safetensors.index.json` — one shard resident at a time
- `TransformerBlock` dispatching on `layer_types[i]`
- Full-attention path updated: QK-norm, output gating, partial RoPE, mRoPE

**Gate:** on 0.8B, a forward pass through **only the 6 full-attention layers**
(linear layers stubbed to identity) matches a correspondingly-stubbed HF model to
atol=1e-4. Proves RoPE and attention changes before the hard part.

---

### C2 — Gated linear attention (the big one)
**Goal:** the 48-layer path. Full-model parity.

**What to derive before coding:**
- [ ] The delta rule as a recurrence. What is the state, and why is it
      `(value_heads × key_dim × value_dim)` rather than growing with sequence length?
- [ ] Why linear attention has O(1) memory and O(N) compute where softmax attention
      has O(N) memory and O(N²) compute. Where does the softmax non-linearity block
      the same trick?
- [ ] The two forms: chunked/parallel for prefill vs recurrent step for decode.
      Why do you need both, and what must be identical between them?
- [ ] The causal conv (width 4) before the recurrence — what is it for, and what
      state does it carry across a chunk boundary?
- [ ] The 16-key / 48-value head broadcast in the 27B. How do 16 key heads serve
      48 value heads?

**What to build:**
- `GatedDeltaNet` module: causal conv → gated delta recurrence → swish output gate
- Chunked (prefill) and recurrent-step (decode) paths
- Weight-key mapping for the linear layers

**Gates:** see the staged parity ladder below — C2 owns gates 4a/4b/4c and 5,
then the model ladder (0.8B → 2B → **9B**) on gate 5.

---

## The staged parity ladder (C1–C4)

> Added 2026-09-15. Replaces the per-phase gate lists with one ladder, because the
> gates only isolate failure if they are ordered and each rung changes exactly one
> variable.

There are not two failure surfaces, there are ten: weight loading; individual
modules; the full-attention layer; the chunked delta rule; the recurrent delta
rule; layer-type dispatch; decoder composition; the paged KV cache; the linear
state slab; the scheduler. A whole-model parity failure cannot tell you which.

**Principle: each rung differs from the one below by exactly one variable.**

| Gate | What runs | Compared against | Tolerance | Phase |
|---|---|---|---|---|
| **0** | HF only | — | *produces* the fixture | C0 |
| **1** | the weight loader | expected vs provided keys | zero missing, **zero unexpected** | C1 |
| **2** | each module alone | HF's same module, same input | atol | C1 |
| **3** | one full-attn layer | HF layer *i*, same hidden state | atol | C1 |
| **4a** | your chunked vs your recurrent | **each other** | **bit-identical** | C2 |
| **4b** | your delta rule | `torch_chunk_gated_delta_rule` | atol | C2 |
| **4c** | your `GatedDeltaNet` layer | HF's layer | atol | C2 |
| **5** | **full decoder, no cache, no scheduler** | fixture, **per layer** | atol | C2 |
| **6** | + paged KV + linear slab | gate 5's own output | **bit-identical** | C3 |
| **7** | + batching / chunked prefill | single-stream generation | **identical tokens** | C4 |

### Gate 5 is the cut between the two surfaces

It runs the model with no cache and no scheduler.

- **Fail 5** → the model implementation.
- **Pass 5, fail 6** → the cache.
- **Pass 6, fail 7** → the scheduler.

This is why the C0 fixture captures per-layer `hidden_states` and not just logits.
Gate 5 then reports *"correct through layer 2, diverges at layer 3"* rather than a
single unusable number — and on the 0.8B, layers 0–2 are `linear_attention` while
layer 3 is the first `full_attention`, so the layer index names the suspect.

### Gate 4a before 4b — the one that gets skipped

If the chunked and recurrent forms disagree with **each other**, comparing either
to HuggingFace cannot tell you which one is wrong. Establish internal consistency
before external correctness.

### Two tolerances, used deliberately

- **atol** — comparing *your implementation* against *a different implementation*.
  Different op order, different rounding; some error is expected.
- **bit-identical** — comparing *the same maths against itself*. Paged vs
  contiguous, chunked vs recurrent, batched vs single-stream are **refactors, not
  reimplementations**. Any difference is a bug.

Getting this backwards is how a cache bug survives: give the paged path an atol and
a genuine indexing error disappears into the rounding noise.

### bf16 error compounds — record the curve, not the endpoint

A per-layer atol of 1e-4 does **not** give 1e-4 at the logits after 24 layers.
Record per-layer drift as a series. Smooth growth is arithmetic; a jump at one
layer is a bug. **The shape of the curve is the diagnostic, not the final value.**

### Nothing hardcodes the layer count

64 layers at 3:1 is the 27B. The dev model is 24 (18 linear / 6 full), the 9B is
32. Same ratio, same `full_attention_interval: 4`. Iterate `config.layer_types`
and dispatch per layer — then one implementation serves 0.8B → 2B → 9B → 27B, and
gate 5 runs up the model ladder unchanged.

---

### C3 — Hybrid cache manager
**Goal:** two coexisting memory managers.

**What to derive before coding:**
- [ ] Why the linear state must NOT be paged. What would blocks buy you when the
      state is fixed-size?
- [ ] Recompute vs swap for linear state: recomputing means replaying the recurrence
      over the whole prefix. How does that cost curve compare to KV recompute?
- [ ] Verify the crossover arithmetic in `ARCHITECTURE_DELTA.md §2` independently.

**What to build:**
- Keep `BlockAllocator`/`BlockTable` for the 16 full-attn layers, unchanged
- New fixed-slab allocator for the 48 linear layers
- A unified per-sequence memory handle owning both

**Gate:** hybrid cache path produces bit-identical logits to the C2 no-cache path.

---

### C4 — Scheduler v2: two-dimensional admission
**Goal:** the intellectual centerpiece.

**What to derive before coding:**
- [ ] Admission when cost is `151 MB flat + 64 KB/token`. Which axis binds first,
      and at what concurrency and context length?
- [ ] Under memory pressure with mixed short and long requests, which do you evict?
      The baseline answer (evict the biggest) may invert below the crossover.
- [ ] Chunked prefill: how does recurrent state thread across chunks without
      corrupting the sequence?

**What to build:**
- Two-axis admission control
- Re-measured preemption policy (do not assume the A2 answer transfers)
- Chunked prefill with correct recurrent-state threading

**Gate:** continuous batching produces identical output tokens to single-stream
generation, for every request, including across preemption and chunk boundaries.

---

### C5 — `server.py` — the actual serving layer
**Goal:** it is a server, not a script. This is the "build a system around it" part.

- Async HTTP layer, OpenAI-compatible `/v1/chat/completions`
- Token streaming (SSE)
- Request queueing, cancellation, timeouts
- Health/metrics endpoint exposing live KV utilization, linear-slot utilization,
  in-flight sequences, queue depth
- Tokenizer/chat-template handling (`chat_template.jinja` ships with the model)

**Gate:** sustained concurrent load through the HTTP layer with no correctness
regression against C4.

---

### C6 — Cloud bring-up + benchmarks
**Goal:** real numbers on real hardware.

- 9B on a cheap GPU first — proves the broadcast path and the loader at scale
- Then 27B on A100/H100 80GB
- Port the `bench/` harness (Poisson load gen, TTFT/TPOT/throughput)
- **The crossover experiment**: sweep mean context length across the ~2,300-token
  boundary and show admission behaviour inverting. This is the headline result.
- Head-to-head vs vLLM and/or SGLang on identical hardware and workload
- Memory-utilization-over-time plots, split by cache type

**Cost discipline:** every cloud session has a written objective and a spend cap.
Nothing gets debugged at $2/hr that could have been debugged on 0.8B.

---

### C7 — Technical analysis document
The artifact, and — since A5 is dropped — the **only** writeup covering any of this
work. Sections:

0. **The dense baseline** (absorbed from A5): paged KV-cache, continuous batching,
   chunked prefill, and the A4 λ-sweep results including the λ=8 paged inversion.
   Short, but it has to be here; nothing else documents it beyond a README feature list.
1. Hybrid attention: what changes for a serving system
2. The two-cache memory model and the crossover
3. Scheduler design: two-dimensional admission, measured
4. Benchmark analysis and the gap vs vLLM/SGLang
5. What I'd build next, ranked by impact

Section 0 is deliberately compressed — it is context for sections 1–4, not a
co-headline. The crossover is the result this document is about.

---

## Stretch — the B phases, re-scoped

All of these are **unstarted greenfield work**, not resumable. Listed in the order
their value changed under the hybrid architecture, not the order the old roadmap
assumed. None are prerequisites for C0–C7.

| Phase | How the hybrid model changes its value |
|---|---|
| **B3 spec decoding** | Cheapest it will ever be — `mtp_num_hidden_layers: 1` ships a trained draft head inside the weights. No draft model to source, train, or host. |
| **B1 Triton kernels** | Value went up. The A4 benchmark already showed the Python block loop losing to contiguous at λ=8 on MPS; at 27B on CUDA it dominates. But it now splits into *two* kernels — paged softmax attention and the delta recurrence — so it is roughly double the work the old roadmap scoped. |
| **B2 prefix caching** | Genuinely novel here, and harder. A shared prefix means a shared *recurrent state*, not just shared KV blocks — and recurrent state can't be partially shared the way a block table can. No published answer to copy. |
| **B4 quantization** | Value went down as a learning exercise, up as cost control: FP8 (~28 GB) moves the target to a 48 GB card and roughly halves the hourly rate. |
| Vision tower | Only if the text path is fully done. |

---

## Open questions

1. ~~**Import path from `hybrid/` to `baseline/`.**~~ **RESOLVED 2026-09-09** —
   baseline is a package. `src/` → `inference_baseline/`, `pyproject.toml` added,
   editable-installed. See Repository layout above.
2. **Untracked A5-era files.** `COMPANION.md`, `INTERVIEW_PREP.md`,
   `bench/plot_styled.py`, `bench/cover_image.py`, `bench/BUG_SUMMARY.md`, and
   `content/` are untracked in the original repo. Copied into `baseline/` by hand.
   A5 is skipped, but the plotting scripts and bug record are working artifacts —
   decide whether to commit them upstream or let them lapse. Currently they exist
   in exactly two uncommitted places.
3. **Schedule.** Deliberately unset. The A-phase roadmap's 3-week sprint was tied
   to an application deadline; no equivalent forcing function is assumed here.
4. **mRoPE for text-only.** Assumed to collapse to standard 1D RoPE. Must be
   confirmed against HF's implementation in C1, not assumed. Read
   `apply_interleaved_mrope`, `get_rope_index`, `compute_3d_position_ids`.

5. **How to keep the dense baseline runnable under transformers 5.x.** One line
   (`baseline/src/config.py:54`) blocks the whole suite. Options:
   (a) fix the line in `baseline/` — it is a copy, the upstream original at
   `~/Documents/studysessions/.../inference-server` stays untouched;
   (b) a second venv pinned to `transformers<5` purely for baseline runs;
   (c) accept the baseline as frozen reference and never run it again — but this
   forfeits the dense comparison C6 depends on.
   *(a) is the obvious call unless there's a reason to keep the clone byte-identical
   to `origin/main`.*
