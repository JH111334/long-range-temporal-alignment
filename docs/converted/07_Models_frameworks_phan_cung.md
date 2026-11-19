---
title: "Models, frameworks và tối ưu phần cứng"
lang: vi-VN
---

# Nguyên tắc chọn model

Chọn model sau khi problem/data/evaluator ổn định, không theo leaderboard. Ưu tiên deterministic solver và retrieval baseline trước frozen model; PEFT chỉ mở khi resource probe và held-out signal đạt.

# Candidate sau Ticktack

| Candidate | Vai trò | Bằng chứng hiện có | Gate còn thiếu |
|---|---|---|---|
| `Qwen/Qwen3-0.6B` | Probe candidate | Model card: Apache-2.0, 0,6B, 32.768 context, Transformers ≥4.51 | Pin revisions, tokenizer/date probe, RAM/VRAM |
| `Qwen/Qwen2.5-0.5B-Instruct` | Control | Apache-2.0, nhỏ | Pin revisions, resource probe |
| Qwen2.5-3B/7B | Paper-scale comparison | Ticktack dùng 3B/7B | GPU ngoài, exact config |
| LLaMA2-7B/13B | Reproduction-only | Ticktack dùng | Gated terms, GPU ngoài |

Qwen3 chỉ là candidate. Model card không thay measurement trên compute profile đích ([Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)).

# Compute profiles

## `local_validation`

- Calendar/normalization, schema, split, dedup probe.
- BM25/in-memory retrieval, date arithmetic deterministic.
- Scalar/Fourier/Ticktack-year/multi-granularity math, interval retrieval và evaluator.
- Dữ liệu synthetic chỉ cho contract.

- Tokenizer-only → bounded frozen forward → hidden-state extraction → lightweight probe.
- Giới hạn sample, batch, index, workers và duration; ghi peak RAM/VRAM và latency.

## `external_gpu`

Full training, large embedding/index hoặc GPU-heavy probe dùng immutable request có
approval và public/licensed input. Cùng source package, config và artifact với local.

## `full_experiment`

Paper-scale 3B–13B, multi-seed và long ablation chỉ sau probe có tín hiệu, frozen
evaluator, budget và local replay. Ticktack dùng tám A100 chỉ là prior-art resource.

# Resource probe contract

| Nhóm | Trường bắt buộc |
|---|---|
| Environment | OS, Python, package lock, driver/CUDA, device |
| Identity | Model/tokenizer ID và commit/revision, license/terms |
| Numerics | Precision, quantization, attention implementation |
| Shape | Sequence length, micro-batch, accumulation, trainable params |
| Measurement | Load time, forward/backward latency, peak process RAM, peak allocated/reserved VRAM |
| Outcome | Pass/OOM/invalid, warnings và release decision |

# Dependency plan

Core: Python ≥3.11, không runtime dependency. `requirements.txt` cài extra dev; `requirements-research.txt` cài `accelerate`, `datasets`, `numpy`, `peft`, `psutil`, `safetensors`, `torch`, `transformers`. Các range trong `pyproject.toml` là kế hoạch tương thích, chưa phải environment lock đã benchmark.

Không thêm Hydra/MLflow/DVC/vector DB/GraphRAG khi JSON manifest, local artifacts và baseline retrieval còn đủ. Thêm công cụ chỉ khi nhu cầu nhiều run, nhiều người/máy hoặc corpus lớn được đo.

# Tối ưu theo thứ tự

1. Sửa data contract, granularity coverage, qrels và split trước khi tăng model.
2. Giảm corpus/top-k và dùng deterministic operation.
3. Cache canonicalization/tokenization theo revision hash.
4. Batch nhỏ theo shape tương đồng; tránh padding lãng phí.
5. Frozen features/lightweight probe trước PEFT.
6. Mixed precision sau compatibility probe.
7. Gradient checkpointing/accumulation theo measurement.
8. Quantization/offload như ablation có control; chuyển lab/cloud nếu local vượt gate, không hạ evaluator để vừa máy.
