---
loại: "quyết-định"
mã: "DECISION-TEMPORAL-CONDITIONING-RESCUE-103"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
thay_thế: []
bị_thay_thế_bởi: []
tham_chiếu: ["DECISION-REP-ABLATION-102-EVALUATION"]
được_xác_thực_bởi: []
cập_nhật_lần_cuối: "2026-10-07"
---

# Quyết Định Đánh Giá Khoa Học: Giải Cứu Biểu Diễn Thời Gian Qua Phân Nhánh B1 và B2 (TEMPORAL_CONDITIONING_RESCUE_103)

## 1. Bối Cảnh Và Câu Hỏi Nghiên Cứu

Thực nghiệm tiền nhiệm `REP_ABLATION_102` đã xác lập kết quả âm bản chuẩn tắc: việc tiêm vector biểu diễn trực tiếp bằng phép cộng số học vào vị trí token đầu tiên (`token-0`) của chuỗi định dạng ChatML trên mô hình `Qwen/Qwen2.5-0.5B-Instruct` gây nhiễu loạn phân phối embedding nghiêm trọng, dẫn đến sụp đổ định dạng cú pháp (`Invalid Rate` lên tới 84.4% - 97.2%).

Chiến dịch `TEMPORAL_CONDITIONING_RESCUE_103` được thiết kế nhằm kiểm định xem hiện tượng suy thoái này có thể được cứu vãn thông qua hai cơ chế điều kiện hóa an toàn hơn hay không:
- **Nhánh B1 (Textual Prompt Conditioning)**: Biểu diễn thời gian tự nhiên (năm Dương lịch và năm Can-Chi theo chu kỳ Lục Thập Hoa Giáp) được đưa trực tiếp vào tiền tố câu hỏi người dùng dưới dạng văn bản tự nhiên, giữ nguyên cấu trúc phân phối token của tokenizer, $0$ tham số gia tăng.
- **Nhánh B2 (Virtual Temporal Token với Zero-Init Gated Residual)**: Vector đặc trưng Can-Chi ($8\text{D}$) được chiếu qua mạng tuyến tính kết hợp chuẩn hóa và cổng vô hướng học được khởi tạo bằng $0$ ($\alpha=0.0$), chỉ tiêm vào pha tiền tính toán (prefill) sau tiêu đề hệ thống, bảo toàn tuyệt đối tính bất biến tại các bước sinh tự hồi quy từng token đơn lẻ.

Toàn bộ công thức học tập cốt lõi được giữ cố định theo đúng khế ước Week 1: mô hình nền `Qwen/Qwen2.5-0.5B-Instruct` (revision `7ae557604adf67be50417f59c2c2f167def9a775`), mục tiêu `MASKED_TARGET`, kiến trúc `MLP_ONLY` LoRA ($r=2, \alpha=4$, 72 khối MLP), dữ liệu TempUN Small qua 3 chặng lịch sử ($1947 \to 1984 \to 2022$), bộ đánh giá $180$ mẫu MCQA cố định tuyệt đối qua 3 hạt nhân độc lập (Seed 1, 2, 3).

---

## 2. Bằng Chứng Thực Nghiệm Đo Đạc (Empirical Evidence)

Dữ liệu thực nghiệm được lưu trữ chuẩn tắc tại `artifacts/releases/TEMPORAL_CONDITIONING_RESCUE_103/`:
- Khế ước thực nghiệm: `artifacts/releases/TEMPORAL_CONDITIONING_RESCUE_103/contract.json`
- Tóm tắt tổng hợp: `artifacts/releases/TEMPORAL_CONDITIONING_RESCUE_103/aggregate/final_summary.json`
- Báo cáo kiểm định McNemar: `artifacts/releases/TEMPORAL_CONDITIONING_RESCUE_103/aggregate/mcnemar.json`

### Bảng đối chiếu hiệu năng qua 3 seed độc lập (Trung bình ± Độ lệch chuẩn):

| Nhánh / Cấu hình | Cơ chế Điều kiện hóa | Trainable Params | Strict Accuracy | Valid Accuracy | Invalid Rate | Trạng thái Định dạng |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$C_0$ (Baseline)** | Không điều kiện hóa thời gian | $829{,}440$ | $20.93\% \pm 3.26\%$ | $34.85\% \pm 3.61\%$ | $40.19\% \pm 3.39\%$ | Chuẩn tắc |
| **$N_0$ ($R_{11}$ Cũ)** | Tiêm trực tiếp token-0 | $916{,}352$ | $1.30\% \pm 2.25\%$ | $46.67\% \pm 0.00\%$* | $97.22\% \pm 4.81\%$ | **Sụp đổ định dạng** |
| **$B_1$ (Textual)** | Tiền tố văn bản tự nhiên | $829{,}440$ | $20.93\% \pm 2.51\%$ | $26.58\% \pm 2.91\%$ | **$21.30\% \pm 3.06\%$** | **Phục hồi tối ưu** |
| **$B_2$ (Gated Residual)** | Token ảo + Cổng $\alpha=0.0$ | $838{,}401$ | **$21.48\% \pm 4.63\%$** | **$35.67\% \pm 4.65\%$** | $40.19\% \pm 5.28\%$ | **Phục hồi hoàn toàn** |

*\*Valid Accuracy của $N_0$ chỉ tính trên số ít mẫu sống sót do $97.22\%$ mẫu bị loại bỏ vì lỗi định dạng.*

---

## 3. Kết Quả Kiểm Định Ghép Cặp (Paired McNemar & Holm-Bonferroni)

Kiểm định McNemar được thực hiện ghép cặp trên từng mẫu suy luận ($N=180$ cases $\times 3$ seeds) với mức ý nghĩa điều chỉnh theo Holm-Bonferroni ($\alpha=0.05$):

### 3.1. So sánh với Đối chứng Âm bản $N_0$ (Failed Token-0 Injection)
- **$M(N_0 \text{ vs } B_1)$**: Nhánh $B_1$ vượt trội hoàn toàn so với $N_0$ trên toàn bộ 3 seeds:
  - Seed 1: $p = 2.54 \times 10^{-8}$, Holm-Bonferroni Significant (`True`).
  - Seed 2: $p = 1.10 \times 10^{-7}$, Holm-Bonferroni Significant (`True`).
  - Seed 3: $p = 1.95 \times 10^{-9}$, Holm-Bonferroni Significant (`True`).
  - **Kết luận**: Hướng ưu thế $B_1$ SUPERIOR nhất quán $100\%$ ($p < 10^{-7}$).
- **$M(N_0 \text{ vs } B_2)$**: Nhánh $B_2$ vượt trội hoàn toàn so với $N_0$ trên toàn bộ 3 seeds:
  - Seed 1: $p = 1.17 \times 10^{-11}$, Holm-Bonferroni Significant (`True`).
  - Seed 2: $p = 4.16 \times 10^{-6}$, Holm-Bonferroni Significant (`True`).
  - Seed 3: $p = 4.25 \times 10^{-8}$, Holm-Bonferroni Significant (`True`).
  - **Kết luận**: Hướng ưu thế $B_2$ SUPERIOR nhất quán $100\%$ ($p < 10^{-5}$).

### 3.2. So sánh với Đối chứng Chuẩn $C_0$ (Baseline B3)
- **$M(C_0 \text{ vs } B_1)$**: Giá trị $p \in [0.253, 0.894]$ trên 3 seeds, không có sự khác biệt có ý nghĩa thống kê (HB-significant: `False`, hướng: `MIXED`).
- **$M(C_0 \text{ vs } B_2)$**: Giá trị $p \in [0.286, 1.000]$ trên 3 seeds, tương đương hoàn toàn về mặt thống kê với baseline (HB-significant: `False`, hướng: `MIXED`).

### 3.3. So sánh Trực tiếp Giữa Hai Nhánh Giải Cứu ($B_1 \text{ vs } B_2$)
- **$M(B_1 \text{ vs } B_2)$**: Giá trị $p \in [0.068, 0.488]$ trên 3 seeds, hướng `MIXED` không có phân nhánh nào áp đảo tuyệt đối về độ chính xác phân loại. Tuy nhiên:
  - $B_1$ đạt tỷ lệ tuân thủ định dạng vượt trội (`Invalid Rate` chỉ $21.30\%$, giảm một nửa so với $C_0$ và $B_2$).
  - $B_2$ duy trì độ chính xác câu trả lời hợp lệ cao hơn ($35.67\%$ so với $26.58\%$).

---

## 4. Phân Tích Cơ Chế Học Và Động Lực Học Cổng $\alpha$ (Gate Dynamics)

Trong phân nhánh $B_2$, tham số cổng vô hướng $\alpha$ được khởi tạo tại $0.0$ và tối ưu hóa cùng mô hình:
- **Seed 1**: $\alpha$ từ $0.000 \to 0.00179$ (Giai đoạn 1) $\to 0.00238$ (Giai đoạn 2) $\to 0.00202$ (Giai đoạn 3).
- **Seed 2**: $\alpha$ từ $0.000 \to 0.00225$ (Giai đoạn 1) $\to 0.00217$ (Giai đoạn 2) $\to 0.00177$ (Giai đoạn 3).
- **Seed 3**: $\alpha$ từ $0.000 \to 0.00175$ (Giai đoạn 1) $\to 0.00187$ (Giai đoạn 2) $\to 0.00173$ (Giai đoạn 3).

**Ý nghĩa cơ học và giới hạn suy diễn nhân quả**:
1. Khởi tạo $\alpha=0.0$ bảo vệ mạng nơ-ron hoàn toàn khỏi cú sốc phân bố embedding ở bước khởi đầu.
2. Độ dốc gradient đã cập nhật $\alpha$ sang giá trị dương nhỏ $\approx 0.0017 - 0.0024$, xác nhận nhánh điều kiện hóa có nhận tín hiệu lan truyền ngược và không bị khóa cứng ở mức $0.0$. Trạng thái chứng cứ: `CONSISTENT_WITH_SIGNAL_USE`, `NOT_CAUSALLY_ESTABLISHED` (giá trị $\alpha > 0$ là bằng chứng nhất quán với việc sử dụng tín hiệu, nhưng chưa cấu thành bằng chứng nhân quả chứng minh thông tin thời gian cụ thể được mô hình khai thác có lợi nếu chưa qua kiểm định hoán vị đặc trưng / feature-shuffle ablation).

---

## 5. Quyết Định Quản Trị Mô Hình (Scientific Verdict)

1. **BÁC BỎ CƠ CHẾ TIÊM TRỰC TIẾP TOKEN-0 KHÔNG ĐIỀU TIẾT (REJECT UNGATED TOKEN-0 INJECTION)**: Bằng chứng thực nghiệm quy trách nhiệm lớn cho cơ chế tiêm thô vào token-0 ChatML; tuy nhiên mức độ đóng góp nhân quả riêng biệt của vị trí token so với biên độ vector, chuẩn hóa và giao diện ghép nối vẫn chưa được phân lập hoàn toàn.
2. **CHẤP NHẬN PHÂN NHÁNH B1 (ACCEPT B1)**: Phê duyệt đưa cơ chế Textual Temporal Conditioning vào làm giải pháp mặc định cho các tác vụ hỏi đáp suy luận đa thời kỳ (Prompting / RAG) nhờ khả năng giảm thiểu lỗi định dạng xuống mức thấp kỷ lục ($21.30\%$) với $0$ chi phí tham số.
3. **CHẤP NHẬN PHÂN NHÁNH B2 (ACCEPT B2)**: Phê duyệt kiến trúc Tiêm Ẩn trạng Gated Residual Khởi tạo Bằng Không (*Zero-Init Gated Residual Embedding Injection at Designated Position* — lưu ý B2 can thiệp vào tensor ẩn tại vị trí chỉ định trong pha prefill chứ không nối thêm token mới vào chuỗi) làm cấu trúc nền tảng cho các nghiên cứu bộ chuyển đổi biểu diễn liên tục (Continuous Representation Adapters).
4. **ĐÁNH GIÁ ĐỐI CHỨNG VỚI BASELINE $C_0$**: Cả B1 và B2 giải cứu thành công mô hình khỏi sự cố sụp đổ định dạng của $N_0$ và duy trì tính tương đương thống kê với baseline $C_0$ ($p > 0.05$ qua kiểm định McNemar có hiệu chỉnh Holm-Bonferroni); tuy nhiên cả hai chưa xác lập ưu thế vượt trội về độ chính xác phân loại so với baseline $C_0$. Ưu thế đo đạc được của B1 là mức độ tuân thủ định dạng vượt trội (`Invalid Rate` $21.30\%$ so với $40.19\%$).
