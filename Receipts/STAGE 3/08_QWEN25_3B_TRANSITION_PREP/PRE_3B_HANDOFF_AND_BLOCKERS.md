# Báo Cáo Bàn Giao và Danh Mục Rào Chắn Tiền Triển Khai Qwen2.5-3B (Handoff & Blockers)

**Dự án**: `long-range-temporal-alignment` — Learning Track — Stage 03  
**Nhánh thực hiện**: `v2-targeted-continual-alignment`  
**Thời điểm bàn giao**: `2026-10-20`  
**Trạng thái hệ thống**: **`PRE_3B_READY`**  
**Rào chắn mặc định**: **`NO_GO_LARGE_TRAINING`**

---

## 1. Danh Mục Các Hạng Mục Đã Hoàn Thành Toàn Diện (Ready Artifacts)

Hệ thống đã hoàn tất 100% công tác kiểm toán khoa học và chuẩn bị tiền triển khai:

1. **Khóa bất biến toàn bộ bằng chứng Qwen2.5-0.5B**:
   - Xác thực đối soát $488 / 488$ tệp snapshot trong `Receipts/` và `JSONS/` đạt $100\%$ mã băm SHA-256 (`ARCHIVE_VERIFY_20261020T013844Z.json`).
   - Tái tính toán toán học chính xác dữ liệu thô 432 bản ghi: Exact Binomial McNemar $p = 0.167444$ (không có ý nghĩa thống kê), Clustered Bootstrap $95\%$ CI $[-23.15, -3.70]\text{ pp}$.
   - Định vị sụp đổ nhãn All-No của Arm B ($108/108$ No, Entropy $= 0.0$ bits) và thiên kiến chiều $100\%$ Yes $\to$ No trên 36 cặp Gold-Flip.
   - Hoàn thành ma trận 4 ô nhân quả 2x2 trên 54 cặp ($E_z(t_1) = -0.0580$, $E_z(t_2) = -0.0584$, PositiveBothSides $= 36.11\%$).
2. **Thẩm định mô hình nền Qwen2.5-3B-Instruct**:
   - Pinned revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
   - Giấy phép `Qwen RESEARCH LICENSE AGREEMENT`: Thẩm định đầy đủ, xác nhận hợp lệ cho nghiên cứu học thuật của trường Đại học, cấm thương mại hóa.
   - Tokenizer 3B: Xác thực `Yes` $\to [9454]$, `No` $\to [2753]$, không có tokenizer drift.
3. **Thiết kế ma trận cắt bỏ 5 cánh tay chuẩn tắc**:
   - Phân tầng `Q3_A0_BASE` (mốc đối chứng tự nhiên), `Q3_A1_STATIC_MLP_LORA` (can thiệp phi thời gian), `Q3_A2_TEMPORAL_CORE_NO_SEAM` (đòn bẩy lõi động), `Q3_A3_FULL_TEMPORAL` (phương pháp đề xuất), `Q3_A4_TICKTACK_INSPIRED` (đối chứng đối tác).
4. **Bảo tồn dữ liệu và ranh giới kiến trúc**:
   - Tập dữ liệu 54 cặp và 180 câu TempUN được bảo tồn $100\%$ không đổi byte.
   - Khóa chức năng hoán vị lựa chọn `OnlineOptionPermutation.enabled = False` theo lệnh GVHD.
   - Xác lập nguyên tắc cấm tải chéo trọng số 0.5B sang 3B (`Zero Cross-Scale Weight Loading`).
   - Ánh xạ độ sâu tương đối Gated Seam tại Tầng 21 trên 36 tầng.

---

## 2. Danh Mục Các Rào Chắn và Điều Kiện Khóa Dừng (Active Blockers & Prohibitions)

Mọi hoạt động vượt quá phạm vi chuẩn bị đều bị phong tỏa theo các cổng kiểm soát:

1. **Rào chắn phần cứng GPU (`G11_GPU_VRAM_PRECISION_MEASUREMENT: NOT_VERIFIED`)**:
   - Chưa có GPU được cấp phát trong phiên chuẩn bị hiện tại.
   - Ngân sách VRAM dự kiến cho FP16 huấn luyện ($11.43\text{ GB}$) tiệm cận giới hạn $16\text{ GB}$ của GPU Tesla T4; phương án QLoRA 4-bit NF4 ($4.90\text{ GB}$) là giải pháp dự phòng khả thi nhưng đòi hỏi kiểm thử tương thích với lớp tuyến tính lượng tử hóa.
   - **Rào chắn**: Tuyệt đối không khởi chạy bất kỳ script huấn luyện nào trước khi đo lường VRAM thực tế trên GPU mục tiêu.
2. **Khóa dừng huấn luyện quy mô lớn (`NO_GO_LARGE_TRAINING`)**:
   - Dự án duy trì lệnh khóa dừng huấn luyện quy mô lớn. Không cấp phát tài nguyên Kaggle GPU cho việc huấn luyện đa bước khi chưa có sự phê duyệt chính thức từ người dùng.
3. **Ranh giới workspace đối tác (`G15: STRICT_READ_ONLY`)**:
   - Thư mục đối tác `C:\Users\GF63\Documents\Long-Range-Representation` duy trì trạng thái READ-ONLY tuyệt đối ($0$ byte can thiệp). Hợp đồng bàn giao giao diện được xuất bản độc lập tại namespace của nhánh hiện tại.

---

## 3. Lệnh Thao Tác Kế Tiếp Được Đề Xuất (Recommended Next Command)

Khi người dùng quyết định kích hoạt bước thực nghiệm đầu tiên trên Qwen2.5-3B, lệnh hợp lệ duy nhất được phép thực thi là:

```bash
# CHỈ THỰC THI SUY LUẬN MỐC CƠ SỞ (ZERO-SHOT INFERENCE) — KHÔNG HUẤN LUYỆN, KHÔNG GPU
py -3.12 scripts/run_stage03_3b_base_inference.py --arm Q3_A0_BASE --device cpu
```

*(Lệnh này chỉ thực hiện suy luận đánh giá cánh tay `Q3_A0_BASE` trên 54 cặp phản thực nghiệm để ghi nhận năng lực và thiên kiến tự nhiên của mô hình 3B, không can thiệp trọng số).*
