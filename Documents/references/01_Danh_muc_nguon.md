# Danh Mục Nguồn

Ngày rà soát: 2026-08-07.

## Nguồn trọng tâm

| ID | Nguồn | Loại bằng chứng | Vai trò | Trạng thái |
|---|---|---|---|---|
| S01 | [Ticktack, arXiv:2503.04150v3](https://arxiv.org/abs/2503.04150) | Paper/preprint | Phương pháp khởi điểm | Đã đọc PDF 12 trang; chưa thấy code/TempLS chính thức |
| S02 | [Set the Clock](https://arxiv.org/abs/2402.16797) | Paper | Temporal alignment theo mốc thời gian | Đã xác minh |
| S03 | [Repository Set the Clock](https://github.com/yizhongw/llm-temporal-alignment) | Nguồn tác giả | Code/data metadata | Public, Apache-2.0; README nói code đang được tổ chức lại |
| S04 | [TAQA](https://huggingface.co/datasets/ROIM/temporal-alignment-qa) | Nguồn tác giả | Dataset 20.148 câu hỏi | Candidate; khóa revision/checksum trước ingest |
| S05 | [TempUN](https://arxiv.org/abs/2402.11997) | Paper | Temporal multiple-choice QA | Hold; availability hiện tại cần xác minh lại |
| S06 | [Back to the Future / TimeLLaMA](https://arxiv.org/abs/2310.01074) | Paper | Temporal reasoning có giải thích | Related work; không phù hợp 4 GB VRAM |
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

## Model và framework

| ID | Nguồn | Ghi chú |
|---|---|---|
| S17 | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Apache-2.0; candidate probe, chưa xác nhận vừa 4 GB VRAM |
| S18 | [LoRA](https://arxiv.org/abs/2106.09685) | Base weights đóng băng, adapter low-rank trainable |
| S19 | [EWC](https://doi.org/10.1073/pnas.1611835114) | Regularization theo Fisher cho continual learning |
| S20 | [PEFT LoRA](https://huggingface.co/docs/peft/main/package_reference/lora) | API triển khai; version phải được khóa khi chạy thật |
| S21 | [PyTorch Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) | Seed/determinism không đảm bảo tuyệt đối giữa release/nền tảng |

## Quy tắc sử dụng

- HyperAI và Papers with Code chỉ dùng để khám phá. Không tìm thấy trang chính thức của Ticktack ở hai nguồn này trong lần rà soát; không suy ra paper không tồn tại.
- Số liệu, giấy phép và availability phải lấy từ paper/trang tác giả/trang dataset chính thức.
- URL truy cập được không đồng nghĩa dữ liệu được phép ingest hoặc phân phối lại.
