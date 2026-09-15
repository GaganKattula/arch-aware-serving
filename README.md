# arch-aware-serving

## What is this?

An inference engine built from scratch with compounding support -- it takes on one model architecture at a time, and exposes the unique inference support for each.

Assuming a dense transformer with a growing KV cache simplifies the scheduler and admission patterns. Newer architectures like MoE and hybrid attention require special handling to deliver the efficiency they were designed for.

## What's here right now

First architecture: hybrid linear/full attention, the Qwen3.5 family and Qwen3.8-27B. 48 of its 64 layers carry a fixed-size recurrent state instead of a KV cache, so per-sequence memory stops being proportional to sequence length -- which breaks how a scheduler decides what to admit.

So far that means a reference fixture and the start of the model. Not a working server.

## Setup

```bash
uv venv && uv pip install -e hybrid/
python hybrid/tests/capture_reference.py
```

Captures HF reference logits and per-layer hidden states for Qwen3.5-0.8B. Byte-identical across runs.

## Background

The dense GQA server this builds on: [inference-server](https://github.com/GaganKattula/inference-server) -- paged KV-cache, continuous batching, chunked prefill, benchmarked three ways.
