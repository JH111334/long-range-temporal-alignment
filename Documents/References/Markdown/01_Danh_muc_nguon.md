---
loại: "tham-chiếu"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: false
nguồn: "Papers, repositories và datasets được dẫn trực tiếp trong registry"
độ_tin_cậy: "medium"
chủ_đề: "Temporal alignment, retrieval và evaluation"
cập_nhật_lần_cuối: "2026-09-06"
---

# Danh Mục Nguồn

Ngày rà soát: 2026-08-26.

## Nguồn trọng tâm

| ID | Nguồn | Loại bằng chứng | Vai trò | Trạng thái |
|---|---|---|---|---|
| S01 | [Temporal Alignment through Cycle Encoding, arXiv:2503.04150v3](https://arxiv.org/abs/2503.04150) | Paper/preprint | Phương pháp Ticktack khởi điểm | Nộp 2025-03-06, v3 ngày 2025-10-21; chưa thấy release code/TempLS có license |
| S02 | [Set the Clock](https://arxiv.org/abs/2402.16797) | Paper | Temporal alignment theo mốc thời gian | Đã xác minh |
| S03 | [Repository Set the Clock](https://github.com/yizhongw/llm-temporal-alignment) | Nguồn tác giả | Code/data metadata | Public, Apache-2.0; README nói code đang được tổ chức lại |
| S04 | [TAQA](https://huggingface.co/datasets/ROIM/temporal-alignment-qa) | Nguồn tác giả | Dataset 20.148 câu hỏi | Candidate; khóa revision/checksum trước ingest |
| S05 | [TempUN](https://arxiv.org/abs/2402.11997) | Paper | Temporal multiple-choice QA | Hold; availability hiện tại cần xác minh lại |
| S06 | [Back to the Future / TimeLLaMA](https://arxiv.org/abs/2310.01074) | Paper | Temporal reasoning có giải thích | Related work; external compute candidate, không phải local default |
| S07 | [Time Is Encoded in the Weights](https://aclanthology.org/2024.acl-long.141/) | Paper đã xuất bản | Time vectors và nội suy | Baseline/related method |
| S08 | [Repository time vectors](https://github.com/KaiNylund/lm-weights-encode-time) | Nguồn tác giả | Mã nguồn related method | Candidate tham khảo |

## Benchmark và error analysis

| ID | Nguồn | Giá trị cho dự án |
|---|---|---|
| S09 | [TimeBench](https://aclanthology.org/2024.acl-long.66/) | Taxonomy và benchmark temporal reasoning |
| S10 | [Test of Time](https://arxiv.org/abs/2406.09170) | Benchmark tổng hợp kiểm soát contamination |
| S11 | [ToT dataset](https://huggingface.co/datasets/baharef/ToT) | Candidate test; phải kiểm tra license/provenance file |
| S12 | [Date Fragments / DateAugBench](https://aclanthology.org/2025.emnlp-main.159/) | Rủi ro tokenizer phân mảnh ngày tháng |
| S13 | [ETRQA](https://aclanthology.org/2025.findings-acl.1198/) | Lỗi timespan, câu hỏi ghép và độ hạt mịn |
| S14 | [ChronoSense](https://aclanthology.org/2025.acl-short.46/) | Quan hệ khoảng thời gian theo Allen |
| S15 | [TRD](https://aclanthology.org/2026.iwsds-1.19/) | Synthetic multilingual, nhiều mức khó |
| S16 | [TRD repository](https://github.com/amazon-science/temporal-reasoning-dataset) | Nguồn tác giả cho data/code |
| S22 | [VLSP 2025 TemporalQA](https://aclanthology.org/2025.vlsp-1.35/) | Shared task tiếng Việt; date arithmetic và duration; dataset terms cần xác minh trước ingest |
| S23 | [HisGraphRAG](https://aclanthology.org/2025.paclic-1.49/) | Historical QA Việt Nam; temporal retrieval/reranking có ablation âm khi đứng riêng |
| S24 | [HistoryBankQA](https://aclanthology.org/2026.starsem-conference.33/) | 10M+ historical events, 10 ngôn ngữ, sáu task; không có tiếng Việt |
| S25 | [ComplexTempQA](https://aclanthology.org/2025.emnlp-main.463/) | Temporal multi-hop/comparison/counting quy mô lớn; chỉ dùng metadata/taxonomy ở pilot |
| S26 | [TempAnswerQA](https://aclanthology.org/2025.findings-emnlp.637/) | Cho thấy exact match không phản ánh độ lớn sai số thời gian |
| S27 | [Survey Temporal QA](https://arxiv.org/abs/2505.20243) | Taxonomy retrieval, normalization, reasoning và evaluation |
| S29 | [MultiTempBench](https://arxiv.org/abs/2603.19017) | Tokenization so với representation trên nhiều ngôn ngữ, format và hệ lịch |
| S30 | [TDBench](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2cbeff05cd646fcc09f6c3c48ba56f74-Abstract-Conference.html) | Chẩn đoán năng lực temporal theo task; chỉ chuyển giao taxonomy/evaluator phù hợp |
| S31 | [When Silence Is Golden](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7718914dfe7d5a657bf6261b5f431021-Abstract-Conference.html) | Abstention trong temporal QA; không chuyển RL/LLM lớn vào P1 |
| S32 | [QDoRA tại VLSP TemporalQA](https://aclanthology.org/2025.vlsp-1.40/) | Parser/solver/canonicalization tiếng Việt; cấu hình 7B/2×5090 không phù hợp local |

## Model và framework

| ID | Nguồn | Ghi chú |
|---|---|---|
| S17 | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Apache-2.0; candidate probe, chưa xác nhận vừa 4 GB VRAM |
| S18 | [LoRA](https://arxiv.org/abs/2106.09685) | Base weights đóng băng, adapter low-rank trainable |
| S19 | [EWC](https://doi.org/10.1073/pnas.1611835114) | Regularization theo Fisher cho continual learning |
| S20 | [PEFT LoRA](https://huggingface.co/docs/peft/main/package_reference/lora) | API triển khai; version phải được khóa khi chạy thật |
| S21 | [PyTorch Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) | Seed/determinism không đảm bảo tuyệt đối giữa release/nền tảng |
| S28 | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Apache-2.0; candidate probe mới, Transformers ≥4.51; chưa resource probe |

## Quy tắc sử dụng

- HyperAI và Papers with Code chỉ dùng để khám phá. Không tìm thấy trang chính thức của Ticktack ở hai nguồn này trong lần rà soát; không suy ra paper không tồn tại.
- Số liệu, giấy phép và availability phải lấy từ paper/trang tác giả/trang dataset chính thức.
- URL truy cập được không đồng nghĩa dữ liệu được phép ingest hoặc phân phối lại.
