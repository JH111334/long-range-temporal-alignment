---
title: "Phân tích Ticktack và ranh giới tái lập"
subtitle: "arXiv:2503.04150v3 — Nghiên cứu sâu, hoàn thiện toán học và tối ưu triển khai"
lang: vi-VN
---

# Nguồn và trạng thái

Paper hiện có tên *Temporal Alignment of LLMs through Cycle Encoding for Long-Range Time Representations*, nộp ngày 2025-03-06 và sửa lần cuối thành v3 ngày 2025-10-21 ([arXiv:2503.04150](https://arxiv.org/abs/2503.04150)). Đây là preprint. Rà soát ngày 2026-08-26 chưa xác minh được release code hoặc TempLS có license/revision/checksum.

# Mental Model: Căn chỉnh biểu diễn nội tại thay vì truy xuất (Retrieval)

Khác biệt bản chất với các bài toán truy xuất văn bản (LegalQA):

```text
Truy xuất ngữ cảnh (LegalQA):
Tài liệu -> Chunks -> BM25/FAISS Index -> Retriever -> Evidence -> Generator

Căn chỉnh thời gian nội tại (Ticktack):
Mốc thời gian -> Chu kỳ thời gian -> Tọa độ cực/Descartes -> Temporal Encoding (TE)
             -> Inject vào Input Embedding h' = h + TE
             -> Post-train LoRA qua Loss Alignment + EWC
             -> Không gian trọng số LLM đã căn chỉnh -> Generator
```

Ticktack không sử dụng cơ sở dữ liệu vector hay retriever ngoài. Mục tiêu là thay đổi biểu diễn thời gian bên trong LLM và post-training không gian trọng số LoRA để gắn kết tri thức với niên đại.

# Quy trình hệ thống hai pha

- **Phase A (Offline Build):** Trích xuất năm $t_{\text{AD}} \rightarrow$ Chu kỳ Lục thập hoa giáp $t_{\text{cycle}} \in \{0, \dots, 59\} \rightarrow$ Tọa độ cực $(\theta, r) \rightarrow$ Descartes $(x, y) \rightarrow$ Mã hóa $\text{TE}(x), \text{TE}(y) \rightarrow$ Hòa trộn embedding tại span thời gian $h' = h + \text{TE}(x) + \text{TE}(y) \rightarrow$ Post-train với $\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{NTP}} + \sigma [\mathcal{L}_T + \mathcal{L}_{\text{EWC}}]$.
- **Phase B (Online Inference):** Câu hỏi $\rightarrow$ Trích xuất mốc thời gian $\rightarrow$ Chuyển đổi chu kỳ/Descartes $\rightarrow$ Tính vector $\text{TE} \rightarrow$ Cộng vào embedding $h' \rightarrow$ Đưa qua LLM đã căn chỉnh $\rightarrow$ Sinh câu trả lời.

# Phân tích phản biện phương án chính: Mở rộng sang độ hạt tháng

- **Bản chất dữ liệu và nhu cầu con người:** Độ hạt Ngày thiếu >85% dữ liệu trong lịch sử trước hiện đại. Độ hạt Năm làm mất khả năng phân biệt quan hệ trước–sau của các biến cố dồn dập trong cùng một năm (ví dụ năm 1945). Độ hạt Tháng là biên tối ưu Pareto: vừa giải quyết trọn vẹn nhu cầu tra cứu nhân quả của con người, vừa đủ dồi dào dữ liệu để học máy.
- **Tính khả thi theo compute profile:** Kênh tọa độ tháng có chi phí lý thuyết nhỏ,
  nhưng model 0,5B–1,5B/QLoRA chưa được xác nhận vừa local host. Phải đo peak RAM/VRAM,
  batch và sequence length; workload nặng chuyển sang approved external profile mà không
  đổi pipeline hoặc evaluator.
- **Chu kỳ kép (Dual-Cycle):** Vòng vĩ mô 60 năm $(x_{\text{year}}, y_{\text{year}})$ kết hợp vòng trung mô 12 tháng $(x_{\text{month}}, y_{\text{month}})$, tạo độ phân tách $30^\circ$ giữa các tháng liền kề.
- **Precision Masking:** $\mathbf{m}_{\text{precision}} = [m_{\text{year}}, m_{\text{month}}]$. Khi dữ liệu chỉ có năm, $m_{\text{month}} = 0$; tuyệt đối không gán mặc định tháng 1 (nguyên tắc fail-closed).

# Năm điểm cải thiện kỹ thuật phụ từ bài báo gốc (Ticktack v3)

1. **Khắc phục bán kính chống va chạm alias:** Sửa $r = \alpha + \beta \theta$ thành hàm đơn điệu theo kỷ nguyên $r(t_{\text{AD}}) = \alpha + \beta \cdot \lfloor (t_{\text{AD}} - t_0)/60 \rfloor$.
2. **Chuẩn hóa mất mát theo batch:** Thay Eq. 8 unnormalized bằng Mean Unique-Pair Loss có margin.
3. **Khôi phục lực đẩy chu kỳ ($\delta < 1$):** Đặt $\delta = 0.5$ thay vì $\delta = 1$ để không triệt tiêu $(1 - \delta)\mathcal{L}_{\text{inter}}$.
4. **Tiêm thời gian cấp span:** Chỉ tiêm $\text{TE}$ tại token thuộc biểu thức thời gian xác định.
5. **Định vị Fisher EWC trên LoRA:** Tính ma trận Fisher trên các tham số LoRA trainable.

# Tích hợp kỹ thuật và mô hình 2025–2026

- Ứng dụng Qwen2.5-0.5B hoặc Qwen3-0.6B (2025–2026).
- RoPE/Attention modulation (Date Fragments EMNLP 2025).
- Temporal hard negative sampling (ETRQA ACL 2025).

# Các phương án đề xuất phụ, đối chứng và mở rộng khác

1. **Lexical Retrieval + Interval Table:** Baseline đối chứng bắt buộc trên CPU để bóc tách gain của retriever với representation.
2. **Tokenizer Probe & Lịch Việt Nam:** Mapping âm lịch/niên hiệu; chỉ mở khi có dữ liệu tự nhiên chứng minh lỗi đủ lớn.
3. **Historical GraphRAG:** Đa chặng thực thể thời gian; tạm hoãn vì chưa có qrels và
   paired error evidence chứng minh relation/multi-hop là blocker. Nếu gate đạt, chạy cùng
   contract trên `external_gpu` hoặc `full_experiment`.
