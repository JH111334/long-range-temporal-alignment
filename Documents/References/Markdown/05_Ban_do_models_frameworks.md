# Bản Đồ Models Và Frameworks

## Tầng 0: Không model

- Python standard library.
- Kiểm thử calendar, encoder, objective, split, parser và schema.
- Đầu ra: smoke artifact; không phải benchmark.

## Tầng 1: Probe đóng băng

- Candidate chính: `Qwen/Qwen3-0.6B`; control: `Qwen/Qwen2.5-0.5B-Instruct`.
- Framework: PyTorch, Transformers; Datasets chỉ sau data gate.
- Task: deterministic date-arithmetic control, tokenizer fragmentation, hidden-state extraction, linear probe hoặc nearest-neighbor.
- Gate: khóa revision, precision, sequence length; đo RAM/VRAM trên profile thực thi đích.

Qwen3 cần Transformers ≥4.51 theo model card. Apache-2.0 và 0,6B không tự chứng minh khả năng fine-tune trên 4 GB VRAM.

## Tầng 2: PEFT nhỏ

- PEFT LoRA trên model nhỏ nếu bounded resource probe đạt; chọn compute profile từ
  resource contract thay vì hardware tier.
- Bắt buộc ghi rank, alpha, dropout, target modules, optimizer, precision, seed, micro-batch, accumulation, checkpointing và resource peak.
- Không mặc định LoRA là memory-efficient: activation vẫn có thể vượt 4 GB VRAM.

## Tầng 3: Tái lập scale paper

- Qwen2.5-3B/7B, LLaMA2-7B/13B và multi-seed.
- Cần GPU lab/cloud; Llama có điều khoản riêng.
- Chỉ mở khi dataset/split/evaluator đã ổn định ở tầng thấp.

## Khi nào thêm công cụ quản trị

| Nhu cầu | Baseline | Nâng cấp khi cần |
|---|---|---|
| Config/run | JSON + SHA-256 | Hydra khi có nhiều tổ hợp cấu hình |
| Experiment tracking | `registries/experiments.json` + artifacts | MLflow khi có nhiều người/máy chạy |
| Data version | Revision/checksum + manifest | DVC khi data lớn và storage remote ổn định |
| Packaging | `pyproject.toml` | Lockfile theo môi trường thí nghiệm |
| CI | unittest + validator | Matrix GPU riêng, không chạy model download trong CI public |
| Retrieval | BM25/in-memory exact baseline | Vector DB/GraphRAG chỉ khi corpus và ablation chứng minh cần |

## Nguyên tắc harmonize

Tất cả adapter phải nhận `TemporalExample`, `ResourceBudget` và manifest; trả artifact có model revision, config/data hash, code revision, metric, resource và warnings. Framework không được thay đổi semantics của split hoặc evaluator.
