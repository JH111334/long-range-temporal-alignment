---
loại: "quyết-định"
mã: "DECISION-V2-TARGETED-CONTINUAL-ALIGNMENT-106"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
thay_thế: []
bị_thay_thế_bởi: []
tham_chiếu: ["DECISION-FIXED-FONE-CANCHI-LEARNING-TRACK-105"]
được_xác_thực_bởi: []
cập_nhật_lần_cuối: "2026-10-13"
---

# Khế Ước Khoa Học Nhánh V2: Bảo Toàn Cú Pháp và Căn Chỉnh Thời Gian Dài Hạn

Văn kiện này xác lập khế ước chính thức cho Nhánh V2 (**Targeted Continual Temporal Alignment**), kế thừa kết quả đo đạc từ [[Branches/v1-dual-cycle-mat-lora/HOME|Nhánh V1 (Starting Anchor M0-FC40)]] và tích hợp toàn diện phương pháp luận phòng ngừa thất bại dự án.

---

## 1. Mốc xuất phát và ranh giới thừa kế từ Nhánh V1

1. **Khóa mốc đối chứng (Starting Anchor)**: Nhánh V1 sử dụng Periodic Dual-Cycle MaT-LoRA trên biểu diễn `FIXED_FONE_CANCHI_V1` 40D, Safe Seam token 14 prefill, huấn luyện Causal LM `MASKED_TARGET` qua 3 chặng TempUN Small ($2.461$ mẫu $\times$ 3 seeds).
2. **Hiện trạng bằng chứng đo đạc M0-FC40 chính thức (3 hạt giống, 180 ca, $1.620$ dự đoán)**:
   - Tính toàn vẹn số học: ĐẠT $100\%$ hữu hạn qua $7.383$ bước đào tạo trên GPU Tesla T4 (9/9 checkpoints hữu hạn, không có `NaN`/`Inf`).
   - Hàm mất mát: Hội tụ qua các chặng ($1.866 \to 1.720 \to 1.739$; không phải giảm đơn điệu).
   - Năng lực suy luận sinh (Strict MCQA, giao thức chuẩn $P_{11}$): **Strict Accuracy $= 12.78\% \pm 5.77\%$** (Seed 1: $19.44\%$, Seed 2: $9.44\%$, Seed 3: $9.44\%$), **Invalid Syntax Rate $= 0.00\% \pm 0.00\%$**, **BWT $= +3.36\%$**. (Số liệu cũ $2.22\%$ Acc và $93.33\%$ Invalid chính thức bị thay thế do xác nhận lỗi bộ phân tích cú pháp chữ hoa và thiếu ChatML).
3. **Quy kết nguyên nhân khoa học**:
   - Hiện tượng **Instruction Drift / Alignment Tax** được phân loại là **`HYPOTHESIS / PARTIALLY_SUPPORTED`**: Mô hình bảo toàn tuyệt đối tính hợp lệ cú pháp ($0\%$ Invalid) và Seed 1 đạt $19.44\%$ (tương đương mốc đối tác $20.51\%$). Tuy nhiên, tại Seed 2 và 3 xuất hiện hiện tượng sụp đổ phân phối đầu ra co cụm về nhãn an toàn 'Yes' (đúng 17/24 câu Yes/No nhưng đúng 0/156 câu MCQA).
   - **Mục tiêu can thiệp Nhánh V2**: Bổ sung hàm mục tiêu điều hòa bảo tồn chỉ dẫn để triệt tiêu hiện tượng sụp đổ phân phối sang nhãn 'Yes' trên mọi hạt ngẫu nhiên, ổn định hiệu năng trên mức $\ge 25.00\%$.

---

## 2. Mục tiêu kỹ thuật và Tiêu chí Thành công của Nhánh V2

Nhánh V2 tập trung giải quyết bài toán cốt lõi: **Tiếp nhận biểu diễn thời gian dài hạn mà vẫn duy trì tính toàn vẹn cú pháp chỉ dẫn của mô hình suy luận**.

### Tiêu chí định lượng (Quantitative Success Targets)

- **Strict Accuracy**: Tăng từ $2.22\%$ lên $\ge 25.00\%$ trên 180 ca MCQA đóng băng.
- **Invalid Syntax Rate**: Giảm từ $93.33\%$ xuống $\le 25.00\%$.
- **Backward Transfer (BWT)**: Triệt tiêu hiện tượng quên thảm khốc, duy trì $\text{BWT} \ge -2.00\%$.
- **Ranh giới bất biến (Invariants)**: Giữ nguyên $100\%$ mô hình nền `Qwen2.5-0.5B-Instruct` đóng băng; giữ nguyên khế ước 40D (`FoNE 32D + Can-Chi 8D`); không can thiệp code hình học của đối tác.

---

## 3. Bộ Cổng Phòng Ngừa Thất Bại (Project Failure Gates G0 -> G7)

Nhánh V2 bắt buộc tuân thủ chuỗi điều phối 8 cổng độc lập trước khi kết luận bất kỳ cải tiến nào:

```
[G0: Contract] -> [G1: Data/Eval] -> [G2: Baseline] -> [G3: Validity]
      -> [G4: Integration] -> [G5: Reliability] -> [G6: Governance] -> [G7: Delivery]
```

1. **G0 (Problem & Success Contract)**: Bài toán khắc phục alignment tax và tiêu chí định lượng được phê duyệt trước khi viết code.
2. **G1 (Data & Evaluation)**: 180 ca MCQA và bộ phân tích cú pháp `GENERATIVE_STRICT_PARSE_V1` giữ nguyên vẹn mẫu số.
3. **G2 (Baseline Reproducibility)**: Khóa kết quả 3 seed của V1 làm mốc so sánh ghép cặp (paired comparison).
4. **G3 (Experimental Validity)**: Mỗi thí nghiệm chỉ thay đổi đúng một biến độc lập; đánh giá ý nghĩa thống kê bằng kiểm định Edwards McNemar ($p < 0.05$).
5. **G4 (Integration Contract)**: Kiểm thử đầu cuối tensor và pipeline trước khi chạy GPU quy mô lớn.
6. **G5 (Reliability & Observability)**: Lưu trữ checkpoint theo từng chặng, giám sát VRAM và số học fail-fast, lưu vết OpenTelemetry spans.
7. **G6 (Collaboration & Governance)**: Tôn trọng ranh giới phân công sở hữu; mọi nhận định đều có biên bản mã băm kiểm chứng.
8. **G7 (Delivery & Claims)**: Chỉ công bố kết luận khi toàn bộ bằng chứng được tái lập độc lập.

---

## 4. Không gian thí nghiệm kiểm soát trong Nhánh V2

Nhánh V2 sẽ thực hiện tuần tự các phép thử nghiệm sau:

1. **$V_{2.1}$ — Instruction-Preserving Regularization (Mất mát bảo tồn chỉ dẫn)**:
   - Thêm thành phần tổn thất điều hòa dựa trên một tập nhỏ các mẫu định dạng chuẩn chạy xen kẽ, nhằm neo giữ không gian biểu diễn cú pháp Instruct.
2. **$V_{2.2}$ — Contrastive Temporal Margin Objective**:
   - Tối ưu hóa phân tách vector biểu diễn giữa các mốc năm khác nhau thông qua hàm khoảng cách tương phản.
3. **$V_{2.3}$ — Balanced Generative Replay**:
   - Lưu giữ và phát lại có kiểm soát một tỷ lệ mẫu từ các chặng lịch sử trước đó nhằm hạn chế quên thảm khốc.
