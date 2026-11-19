# STAGE 01 — MLP-Only LoRA và Các Đường Cơ Sở PEFT Tĩnh (Static PEFT Baselines)

**Thời gian thực hiện:** 04/10/2026 – 07/10/2026  
**Chủ trì:** Nhánh Cơ chế học tập (Learning Track)  
**Mô hình cơ sở:** `Qwen/Qwen2.5-0.5B-Instruct` (Revision `7ae557604adf67be50417f59c2c2f167def9a775`)  
**Trạng thái tri thức:** `VALIDATED / ANCHOR_ESTABLISHED` (Đã hiệu chỉnh đối soát dữ liệu gốc)

---

## 1. Bối Cảnh và Mục Tiêu Nghiên Cứu

Giai đoạn đầu của đề tài tập trung giải quyết câu hỏi nền tảng: **Lựa chọn hàm mất mát tối ưu và ranh giới can thiệp tham số PEFT tĩnh nào trên mô hình nhỏ (`Qwen2.5-0.5B-Instruct`) vừa hạn chế suy thoái cú pháp sinh văn bản, vừa tối ưu hóa tài nguyên tính toán trước khi tiếp nhận thông tin thời gian?**

Trong các mô hình sinh tự hồi quy căn chỉnh chỉ dẫn (Instruct models), việc huấn luyện tinh chỉnh tham số đối mặt với hai thách thức kỹ thuật:
1. *Hiệu ứng của hàm mất mát giám sát*: So sánh giữa huấn luyện trên toàn bộ chuỗi token (`UNMASKED_FULL_SEQUENCE`) và chỉ tính mất mát trên các token mục tiêu (`MASKED_TARGET`, đánh nhãn $-100$ cho toàn bộ tiền tố prompt).
2. *Ranh giới tham số can thiệp*: So sánh giữa việc cập nhật toàn bộ các module tuyến tính (`ALL_LINEAR`) và chỉ can thiệp vào các module MLP (`MLP_ONLY`: `gate_proj`, `up_proj`, `down_proj`).

---

## 2. Thiết Kế Ma Trận Nhân Quả 2×2 Gốc (WEEK1TARGET)

Thực nghiệm `WEEK1TARGET` thiết lập ma trận giai thừa $2 \times 2$ qua 3 hạt giống độc lập (Seed 1, 2, 3) trên tập dữ liệu TempUN Substrate (2,461 mẫu huấn luyện qua 3 chặng 1947, 1984, 2022). Đánh giá trên 180 ca trắc nghiệm TempUN Small dưới bộ phân tích cú pháp sơ khởi (Legacy Parser):

| Hàm Mục Tiêu \\ Đường Dẫn | ALL_LINEAR ($1,099,776$ tham số trainable) | MLP_ONLY ($829,440$ tham số trainable, 0 attention) |
| :--- | :--- | :--- |
| **UNMASKED_FULL_SEQUENCE** | **B1**<br>Valid Accuracy: $25.38\% \pm 2.42\%$<br>Strict Accuracy: **$13.15\% \pm 1.64\%$**<br>Invalid Rate: **$48.15\% \pm 2.36\%$** | **B1M**<br>Valid Accuracy: $30.44\% \pm 3.60\%$<br>Strict Accuracy: **$16.11\% \pm 4.55\%$**<br>Invalid Rate: **$47.59\% \pm 9.53\%$** |
| **MASKED_TARGET** | **B2**<br>Valid Accuracy: $32.95\% \pm 2.80\%$<br>Strict Accuracy: **$19.07\% \pm 1.64\%$**<br>Invalid Rate: **$42.04\% \pm 2.36\%$** | **B3**<br>Valid Accuracy: $34.85\% \pm 4.02\%$<br>Strict Accuracy: **$20.93\% \pm 3.85\%$**<br>Invalid Rate: **$40.19\% \pm 3.35\%$** |

*(Nguồn chứng thực gốc: `artifacts/releases/WEEK1TARGET/summary/01_factorial_2x2_results.json` và `report/WEEK1_Ablation_Objective_x_Trainable_Path.md`)*

### Hiệu chỉnh số liệu báo cáo cũ:
Văn bản sơ thảo trước đây từng ghi nhận các con số "B1: 35.00%/18.33% invalid; B2: 38.89%/0.00% invalid; B3: 31.11%/0.00% invalid". Kiểm toán pháp y xác nhận:
- Con số $38.89\%$ thực tế là tỷ lệ lỗi cú pháp (Invalid Rate) của mô hình nền $F_0$ Baseline trong tài liệu kiểm toán cũ.
- Con số $35.00\%$ và $32.78\%$ là độ chính xác trên lát cắt riêng biệt (Overlap-prone slice 60 ca của Seed 1).
- Việc ghi nhận "0.00% Invalid" ở B2 và B3 trong báo cáo sơ thảo là nhầm lẫn giao thức đánh giá. Dưới bộ phân tích cú pháp ban đầu, tỷ lệ Invalid thực tế của B2 là **$42.04\%$** và B3 là **$40.19\%$**. Toàn bộ các giá trị không có căn cứ này đã bị thu hồi và thay thế bằng bảng số liệu 3 seed chuẩn tắc ở trên.

---

## 3. Phân Rã Hiệu Ứng Chính và Hiệu Ứng Tương Tác

1. **Hiệu ứng của hàm mục tiêu (Objective Effect):**
   - Chuyển từ `UNMASKED` sang `MASKED_TARGET` giúp tăng Strict Accuracy $+5.93\text{ pp}$ trên nền All-Linear ($B2 - B1$) và $+4.81\text{ pp}$ trên nền MLP-Only ($B3 - B1M$).
   - Invalid Rate giảm từ $-6.11\text{ pp}$ đến $-7.41\text{ pp}$.
   - *Lưu ý kỹ thuật:* Việc che mặt nạ prompt (`MASKED_TARGET`) chỉ không tính mất mát tại các token câu hỏi, nhưng **không ngăn chặn luồng gradient lan truyền ngược qua các trạng thái ẩn của prompt** trong quá trình tính toán self-attention của toàn chuỗi.
2. **Hiệu ứng của đường dẫn thích ứng (Trainable Path Effect):**
   - Chuyển từ `ALL_LINEAR` sang `MLP_ONLY` giúp giảm từ $1,099,776$ xuống $829,440$ tham số thích ứng, tương ứng mức cắt giảm **$24.58\%$** trong khế ước đối chuẩn (matched contract), không phải $57\%$ như ước tính thô ban đầu.
   - Strict Accuracy tăng nhẹ $+2.96\text{ pp}$ trên nền Unmasked và $+1.85\text{ pp}$ trên nền Masked.
3. **Kiểm định hiệu ứng tương tác (Interaction):**
   - $\text{INTERACTION} = (B3 - B1M) - (B2 - B1) = -1.11\text{ pp}$ Strict Accuracy (std $9.09\text{ pp}$).
   - **Bác bỏ giả thuyết hiệp đồng số học dương (super-additive synergy):** Không có bằng chứng thống kê cho thấy sự kết hợp giữa MLP-Only và Masked Target tạo ra bước nhảy vọt vượt trên tổng hai hiệu ứng độc lập.

---

## 4. Các Thử Nghiệm Đối Chứng Biểu Diễn và Giải Cứu Cú Pháp

### 4.1. Thử nghiệm FoNE 3-Seed (`LOGGING102_FONE`)
- Đánh giá phương pháp cộng thô vector sóng Fourier 32 chiều trực tiếp vào token nhúng thời gian ($F_1$) so với đường cơ sở ($F_0$).
- Kết quả: $F_0$ đạt $34.76\%$ Acc ($38.89\%$ Invalid), $F_1$ đạt $28.15\%$ Acc ($43.33\%$ Invalid), chênh lệch $\Delta = -6.61\text{ pp}$.
- **Hiệu chỉnh kiểm định McNemar:** Kiểm định McNemar đạt ý nghĩa thống kê ($p < 0.05$) trên **2/3 seeds** (Seed 1: $p = 0.0104$, Seed 2: $p = 0.0101$; Seed 3 không đạt $p < 0.05$).
- **Phán quyết khoa học:** Kết luận âm bản được **khu biệt chặt chẽ vào phương pháp cộng thô (additive FoNE setup)** trên mô hình nhỏ, không quy kết cho toàn bộ họ phương pháp nhúng Fourier.

### 4.2. Thử nghiệm Tiêm Token-0 (`REP_ABLATION_102`)
- Thử nghiệm tiêm FoNE và Can-Chi trực tiếp vào vị trí token đầu tiên (`token-0`) của chuỗi ChatML.
- Kết quả: $R_{11}$ bị sụp đổ cú pháp nghiêm trọng (Invalid Rate $> 96.8\%$, Strict Acc $< 1.3\%$).
- *Phân định ranh giới lý luận:* Nhận định "làm nhiễu loạn attention router của token `<|im_start|>`" được xác định là **giả thuyết cơ chế ứng viên (candidate hypothesis)**, do chưa thực hiện kỹ thuật truy vết kích hoạt nơ-ron (activation tracing) chi tiết.

### 4.3. Giải cứu Cú pháp bằng Gated Residual Projection (`RESCUE_103`)
- Nhánh $B_2$ đưa vector thời gian vào pha prefill qua cổng số dư liên tục tại token index 14 với $\alpha_0 = 0.0$:
  $$\mathbf{h}_{14} \leftarrow \mathbf{h}_{14} + \tanh(\alpha) \cdot \operatorname{LayerNorm}(\mathbf{W}_{\text{proj}} \mathbf{z}_{\text{time}})$$
- Kết quả: Khôi phục hoàn toàn tính hợp lệ cú pháp (Invalid Rate $40.19\%$, tương đương đường cơ sở không tiêm $C_0$ $40.19\%$).

---

## 5. Làm Rõ Quan Hệ Đồng Danh (Aliases): B3 = R00 = C0

Kiểm toán phả hệ xác nhận cấu hình cơ sở của Stage 01 được tái sử dụng qua các chiến dịch với các tên gọi đại diện:
- **$B_3$ (WEEK1TARGET):** Cấu hình gốc MLP-Only LoRA ($r=8, \alpha=16$) + MASKED_TARGET.
- **$R_{00}$ (REP_ABLATION_102):** Nhánh đối chứng không tiêm vector thời gian ở token-0 (kế thừa $100\%$ cấu hình và dữ liệu của $B_3$).
- **$C_0$ (RESCUE_103):** Nhánh đối chứng không điều kiện hóa thời gian (kế thừa $100\%$ cấu hình của $R_{00}$).
- Cả ba ký hiệu đều đại diện cho **cùng một trạng thái cấu hình chuẩn tắc (Anchor Baseline)** với Strict Accuracy $20.93\% \pm 3.85\%$ và Invalid Rate $40.19\% \pm 3.35\%$.
