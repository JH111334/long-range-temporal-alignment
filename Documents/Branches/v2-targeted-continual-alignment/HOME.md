---
loại: "tổng-quan"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
cập_nhật_lần_cuối: "2026-11-19"
---

# Nhánh V2: Nhánh Làm Việc Hiện Hành (Targeted Continual Temporal Alignment)

Nhánh V2 là không gian làm việc tích cực (**Active Working Branch**) cho **Goal `G1` (YEAR-only Closed-Answer Temporal QA)** với mô thức **Zero-Shot Cross-Task Transfer** (Train NWP 104k $\to$ Eval MCQA/Rel). **Ràng buộc bất biến:** chỉ can thiệp ở các tầng **MLP Linear** (`gate_proj`, `up_proj`, `down_proj`, 108 modules), **tuyệt đối không can thiệp Attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) và không dùng DoRA/DLoRA**. Tập cải tiến hiện hành gồm: (1) `asinh` cho $z$ + FoNE đa chu kỳ, (2) `MULTI_YEAR_SPAN` token-level injection đồng bộ train/eval, và (3) `AdamW + LR Scheduling (Warmup + Cosine/WSD)` cho MLP LoRA.

---

## 1. Bài toán cốt lõi và bài học chuyển giao từ V1

Thực nghiệm đo đạc chuẩn tắc trên Nhánh V1 ($M_0\text{-FC40}$) cho thấy:
- Quá trình huấn luyện Next-Word Prediction (NWP) thông thường làm hàm mất mát giảm mạnh ($1.866 \to 1.720$), chứng tỏ mô hình học tốt văn bản lịch sử.
- Tuy nhiên, tỷ lệ lỗi cú pháp (Invalid Syntax Rate) trên tập đánh giá MCQA tăng vọt lên $93.33\%$, kéo Strict Accuracy giảm xuống $2.22\%$.
- **Bản chất vấn đề**: Mô hình Instruct bị trôi dạt phân phối (distribution drift) do hiện tượng đánh đổi căn chỉnh (**Alignment Tax / Task Interference**). Việc tối ưu cục bộ hàm mất mát causal LM trên ngữ liệu mở đã triệt tiêu cấu trúc sinh câu trả lời ngắn của mô hình suy luận.

Nhánh V2 ra đời để giải quyết trực tiếp mâu thuẫn này: làm thế nào để adapter tiếp nhận biểu diễn thời gian (`FIXED_FONE_CANCHI_V1`) mà không làm suy thoái năng lực định dạng sinh của mô hình nền.

---

## 2. Nguyên lý vận hành và Bộ Cổng Phòng Ngừa Thất Bại (Project Failure Gates)

Để tránh lặp lại các bài học thất bại trong lịch sử nghiên cứu (như tối ưu cục bộ không mang lại cải thiện đầu cuối tại DSC LegalQA, hay sai lệch môi trường runtime tại AIC2026), Nhánh V2 áp dụng triệt để chuỗi kiểm soát:

$$\boxed{\text{Problem} \longrightarrow \text{Baseline} \longrightarrow \text{Controlled Experiment} \longrightarrow \text{Evidence} \longrightarrow \text{Decision} \longrightarrow \text{Verified Delivery}}$$

Mọi bước triển khai trong Nhánh V2 bắt buộc phải vượt qua 8 cổng kiểm định độc lập:

| Mã Cổng | Tên Cổng Kiểm Định | Tiêu chuẩn nghiệm thu tối thiểu (Evidence Receipt) |
| :--- | :--- | :--- |
| **G0** | **Problem & Success Contract** | Đặc tả bài toán rõ ràng, tiêu chí thành công định lượng (khắc phục invalid rate $< 25\%$, tăng strict accuracy $> 25\%$), giới hạn ranh giới bất biến. |
| **G1** | **Data & Evaluation Audit** | Đóng băng tập dữ liệu, split kiểm toán rò rỉ (leakage-free), evaluator bất biến (`GENERATIVE_STRICT_PARSE_V1`), mẫu số thống nhất 180 ca. |
| **G2** | **Baseline Reproducibility** | Khóa mốc đối chứng $M_0\text{-FC40}$ từ Nhánh V1 với đầy đủ config hash, data manifest hash và kết quả đo đạc 3 seed. |
| **G3** | **Experimental Validity** | Cô lập duy nhất một biến độc lập trong mỗi thử nghiệm ghép cặp (Paired Comparison); có nhóm chứng (control) và kiểm định thống kê McNemar. |
| **G4** | **Integration Contract** | Kiểm thử đầu cuối (E2E vertical-slice) tại mọi ranh giới giữa các module; kiểm tra tính tương thích tensor và dữ liệu trước khi chạy quy mô lớn. |
| **G5** | **Reliability & Observability** | Checkpoint theo chặng, đo đạc VRAM/loss hữu hạn fail-fast, lưu trữ vết vi mô (OpenTelemetry spans) và khả năng phục hồi khi gián đoạn. |
| **G6** | **Collaboration & Governance** | Phân định ranh giới rõ ràng: Nhánh Learning không can thiệp code hình học của Partner; mọi quyết định đều có căn cứ bằng chứng kiểm chứng được. |
| **G7** | **Delivery & Claims** | Không suy diễn từ kết quả trung gian; mọi tuyên bố cải thiện đều kèm độ tin cậy thống kê, mã băm artifact và khả năng tái lập độc lập. |

---

## 3. Không gian giải pháp kỹ thuật ưu tiên trong Nhánh V2

1. **Instruction-Preserving Regularization (Điều hòa giữ gìn định dạng)**:
   - Kết hợp hàm mục tiêu học thời gian với thành phần mất mát bảo toàn định dạng chỉ dẫn (ví dụ: huấn luyện kết hợp song song các mẫu instruction định dạng chuẩn để chống trôi dạt cú pháp).
2. **Contrastive Temporal Objective (Hàm mất mát tương phản thời gian)**:
   - Buộc mô hình phân biệt sự khác biệt giữa các mốc thời gian thông qua phép đo khoảng cách biểu diễn thay vì chỉ dựa vào causal LM next-token prediction thuần túy.
3. **Balanced Generative Replay (Phát lại mẫu sinh cân bằng)**:
   - Duy trì một tỷ lệ nhỏ các ca mẫu đại diện từ các chặng thời gian trước nhằm triệt tiêu hiện tượng quên thảm khốc (Catastrophic Forgetting) mà không làm tăng dung lượng bộ nhớ.
4. **Adaptive Seam Gating**:
   - Tối ưu hóa động lực học của hệ số cổng $\alpha$ trong Safe Seam, ngăn chặn việc adapter lấn át hoàn toàn biểu diễn ngữ cảnh gốc của mô hình nền.

---

## 4. Liên kết điều hướng

- [[Documents/Branches/v1-dual-cycle-mat-lora/HOME|Nhánh V1: Mốc xuất phát nền tảng]]
- [[Documents/Branches/v2-targeted-continual-alignment/Current-State|Trạng thái chi tiết V2]]
- [[Documents/Decision-Making/FIXED_FONE_CANCHI_LEARNING_TRACK_105|Khế ước biểu diễn 40D]]
- [[Documents/General/RULE|Quy tắc kiểm định và ghi nhận nhật ký]]
