# Khế Ước Cắt Bỏ và Đánh Giá Thẩm Định Qwen2.5-3B (Ablation & Evaluation Contract)

**Dự án**: `long-range-temporal-alignment` — Learning Track — Stage 03  
**Nhánh thực hiện**: `v2-targeted-continual-alignment`  
**Thời điểm ban hành**: `2026-10-20`  
**Trạng thái khế ước**: `PRE_REGISTERED_FROZEN`

---

## 1. Ma Trận Cắt Bỏ Đăng Ký Trước (Pre-Registered Ablation Arms)

Để tránh hiện tượng tìm kiếm mò mẫm tổ hợp ($2^k$ brute-force combinations), nghiên cứu phân tầng thành 5 cánh tay thực nghiệm có đối chứng ghép cặp chặt chẽ:

| Mã Cánh Tay (Arm ID) | Mô Hình Nền | Trạng Thái Thích Ứng | Biểu Diễn Thời Gian | Cơ Chế Seam | Mục Đích Khoa Học Cốt Lõi |
|---|---|---|---|---|---|
| **`Q3_A0_BASE`** | Qwen2.5-3B-Instruct | Đóng băng 100% (Không train) | Không áp dụng (OFF) | Không áp dụng (OFF) | **Mốc đối chứng tự nhiên (Anchor)**: Đo lường thiên kiến tiên nghiệm và năng lực suy luận gốc của mô hình 3B trên tập kiểm thử nguyên bản. |
| **`Q3_A1_STATIC_MLP_LORA`** | Qwen2.5-3B-Instruct | Huấn luyện LoRA tĩnh (Rank 8) | Không áp dụng (OFF) | Không áp dụng (OFF) | **Tách biệt can thiệp phi thời gian**: Đo lường tác động thuần túy của hàm mục tiêu thích ứng đa nhiệm lên mô hình 3B mà không có vector thời gian. |
| **`Q3_A2_TEMPORAL_CORE_NO_SEAM`** | Qwen2.5-3B-Instruct | Huấn luyện MaT-LoRA (Rank 8) | Lõi động $\Delta W(t) = B F_t A$ | Vô hiệu hóa (OFF) | **Đo lường đòn bẩy lõi động**: Xác định phần đóng góp riêng rẽ của việc điều biến trọng số MLP theo góc pha Can-Chi và thế kỷ mà không tiêm token. |
| **`Q3_A3_FULL_TEMPORAL`** | Qwen2.5-3B-Instruct | Huấn luyện MaT-LoRA (Rank 8) | Lõi động $\Delta W(t) = B F_t A$ | Kích hoạt (Gated Seam Tầng 21) | **Phương pháp đề xuất hoàn chỉnh**: Kết hợp đồng thời cả biến thiên trọng số liên tục và neo chặn kích hoạt token tại cùng ngân sách tính toán. |
| **`Q3_A4_TICKTACK_INSPIRED`** | Qwen2.5-3B-Instruct | Checkpoint do đối tác cung cấp | Biểu diễn đối tác (Polar/Sexagenary) | Theo thiết kế đối tác | **Đối chứng độc lập từ đối tác**: Chỉ đánh giá khi có checkpoint được phân phối chính thức kèm tài liệu phả hệ rõ ràng. |

---

## 2. Quy Tắc Đánh Giá và Khế Ước Dữ Liệu Bất Biến

1. **Bảo tồn nguyên vẹn tập dữ liệu kiểm toán (Zero Data Mutation)**:
   - Tập phản thực nghiệm 54 cặp: [`data/counterfactual/COUNTERFACTUAL_GOLD_AUDIT.jsonl`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/data/counterfactual/COUNTERFACTUAL_GOLD_AUDIT.jsonl) (Mã băm chuẩn tắc SHA-256: `49eea7fe6fad69d4f3695e7cd8fbeac2c383e65b76d0802801666f9982cdeba5`).
   - Tập kiểm định TempUN Small: [`data/tempun/eval_mcqa.jsonl`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/data/tempun/eval_mcqa.jsonl) ($N=180$).
   - Tuyệt đối không sinh dữ liệu tổng hợp bổ sung, không đảo thứ tự câu hỏi, không thay đổi nhãn mốc thời gian.
2. **Khóa chức năng hoán vị phương án (No Option Permutation)**:
   - Theo chỉ đạo trực tiếp của GVHD: **Giữ nguyên định dạng dữ liệu gốc, không xáo trộn hay hoán vị thứ tự các phương án A/B/C/D trong huấn luyện và đánh giá**.
   - Cờ `OnlineOptionPermutation.enabled = False` được khóa chặt ở mọi cấp độ.
3. **Giao thức giải mã chuẩn tắc (Decoding Protocol)**:
   - Giao thức: `RAW_GREEDY_V1`.
   - `do_sample`: `False` (Greedy argmax).
   - `num_beams`: `1`.
   - `max_new_tokens`: `4` (Yes/No) hoặc `2` (MCQA single letter).
   - `temperature`: `1.0`.

---

## 3. Danh Mục Các Chỉ Số Đo Lường Bắt Buộc

Mọi đợt chạy đánh giá trên Qwen2.5-3B bắt buộc phải xuất đầy đủ các chỉ số đo lường sau vào tệp JSON báo cáo:

1. **Chỉ số phân loại và thiên kiến nhãn**:
   - Tỷ lệ chính xác nghiêm ngặt (Strict Accuracy).
   - Tỷ lệ cú pháp không hợp lệ (Invalid Format Rate).
   - Phân bố dự đoán nhãn thô và Entropy phân bố (Prediction Entropy in bits).
   - Độ chính xác cân bằng (Balanced Accuracy) và Recall trên từng lớp (Recall_Yes, Recall_No).
2. **Chỉ số nhân quả và độ ổn định**:
   - Tỷ lệ đúng cả hai mốc thời gian trên 36 cặp Gold-Flip (`BothCorrect`).
   - Tỷ lệ ổn định trên 18 cặp đối chứng Gold-Same (`GoldSameStability`).
   - Ma trận 4 ô nhân quả 2x2 ($T_1 Z_1, T_1 Z_2, T_2 Z_1, T_2 Z_2$) và độ dịch chuyển biên độ $E_z(t_1), E_z(t_2), E_{z\text{\_pair}}$.
   - Tỷ lệ chuyển dịch đồng thuận hai chiều (`PositiveBothSidesRate`).
3. **Chỉ số suy luận thống kê**:
   - Kiểm định McNemar nhị thức hai phía chính xác so với mô hình nền: $p = \text{ExactBinomial}(k=\min(b, c), n=b+c, p=0.5)$.
   - Khoảng tin cậy Bootstrap cụm 95% lấy mẫu lại theo 23 cụm sự kiện lịch sử (Clustered Bootstrap CI, 2000 replicates, seed=42).
