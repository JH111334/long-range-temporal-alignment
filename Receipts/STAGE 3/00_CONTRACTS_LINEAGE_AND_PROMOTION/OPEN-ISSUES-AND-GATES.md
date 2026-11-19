# Các Vấn Đề Mở, Cơ Chế Gatekeeper và Phán Quyết Sau Bộ Chẩn Đoán Thực Nghiệm

Tài liệu này tổng hợp tình trạng kiểm soát các rào cản kỹ thuật (Gatekeeper), kết quả giải quyết các nghi vấn pháp y và phán quyết khoa học đóng cổng Go/No-Go cho Stage 03.

---

## 1. Hệ Thống Rào Chắn Gatekeeper P0–P8 và Bộ Bất Biến Khế Ước

Hệ thống rào chắn tự động Gatekeeper P0–P8 đã vượt qua 100% các tiêu chí kiểm định (`tests/test_run052_contract_invariants.py` - 20/20 PASS):
- **P0–P2:** Nhánh Git cô lập, đối tác STRICT READ-ONLY, khế ước FoNE-CanChi 40D và 72 module MaT-LoRA được bảo toàn.
- **P3–P5:** Phân lập objective không double shift, $0\%$ rò rỉ giữa train và dev relation/MCQA, backbone đóng băng tuyệt đối, syntax valid rate canary $= 1.0$, và parity `BOTH_OFF` với base model $= 0.0$.
- **P6–P8:** Checkpoint compact phân kỳ độc lập, manifest SHA256 đầy đủ, phán quyết dựa trên kiểm định thống kê.

---

## 2. Kết Quả Giải Quyết 3 Vấn Đề Khoa Học Mở Qua Bộ Chẩn Đoán Thực Nghiệm

Tập lệnh chẩn đoán `scripts/run_stage03_diagnostics_suite.py` đã thực thi suy luận trên CPU ($0$ optimizer steps) và tạo ra các kết quả xác thực:

### 2.1. Vấn đề 1: Nghi vấn pseudo-immunity $\Delta=0.0$ của can thiệp năm sai (`C_WRONG_YEAR`)
- **Tình trạng:** **ĐÃ GIẢI QUYẾT TRIỆT ĐỂ (`TREATMENT_VERIFIED`)**.
- **Kết quả:** Sau khi sửa lỗi caller tính tọa độ từ năm bị can thiệp $y_{\text{perturbed}} = y_{\text{true}} + 37$, biến thiên logit trung bình trên 36 mẫu date-based eligible đạt $\mathbf{0.118655}$ (vượt xa ngưỡng chẩn đoán $0.01$). Có $4/36$ ca lật đáp án sinh greedy.
- **Kết luận:** Can thiệp thời gian hoạt động thực chất trên không gian tham số, phá vỡ hoàn toàn nghi vấn về tính miễn nhiễm giả tạo của Run 05.2.

### 2.2. Vấn đề 2: Hiện tượng sụp đổ nhãn và khai thác shortcut chữ cái (Label Collapse)
- **Tình trạng:** **ĐÃ ĐƯỢC CHỨNG MINH THỰC NGHIỆM (`LABEL_COLLAPSE_CONFIRMED`)**.
- **Kết quả đo lường trên 624 biến thể hoán vị vòng tròn (Rotated-Option MCQA):**
  - Khi hoán vị các phương án $A, B, C, D$, mô hình Arm B chuyển sang thiên lệch áp đảo về nhãn `b` (**$488 / 624 = 78.2\%$**) và `a` (**$122 / 624 = 19.6\%$**).
  - Độ chính xác gộp trên 624 biến thể là **$24.20\%$** (xác suất đoán ngẫu nhiên $25.0\%$).
  - **Độ nhất quán ngữ nghĩa (Semantic Consistency Rate):** Đúng cả 4 hoán vị vòng tròn đạt đúng **$0 / 156$ ($0.00\%$)**; đúng ít nhất 3 hoán vị cũng đạt **$0 / 156$ ($0.00\%$)**.
- **Kết luận:** Mô hình Arm B hoàn toàn phụ thuộc vào shortcut vị trí chữ cái, chưa hình thành năng lực suy luận ngữ nghĩa độc lập với vị trí phương án. Độ chính xác $24.44\%$ trước đây chủ yếu là do sự may mắn ngẫu nhiên khi vị trí đáp án trùng với phân bố thiên lệch của mô hình.

### 2.3. Vấn đề 3: Phản thực nghiệm ngữ nghĩa thời gian (Semantic Temporal Counterfactuals)
- **Tình trạng:** **XÁC NHẬN GIỚI HẠN MÔ HÌNH NHỎ (`INSUFFICIENT_POWER`)**.
- **Kết quả:** Trên 5 cặp câu hỏi lịch sử đảo chiều mốc năm, tỷ lệ đúng cả hai mốc là $0/5$, tỷ lệ lật đáp án là $0/5$. Mô hình chọn $100\%$ nhãn 'No' trên toàn bộ các câu Yes/No, không phản hồi trước sự biến thiên chân lý thời gian.

---

## 3. Phán Quyết Khoa Học và Cổng Go/No-Go

1. **Khóa Cổng Go/No-Go cho Huấn Luyện Mới:** **`NO-GO (HALT)`**.
   - Nhóm nghiên cứu chính thức đình chỉ mọi kế hoạch huấn luyện GPU lớn tiếp theo.
   - Khi mô hình chưa đạt độ nhất quán ngữ nghĩa ($0.0\%$ Semantic Consistency), việc bổ sung dữ liệu hay kéo dài thời gian huấn luyện sẽ chỉ lãng phí tài nguyên tính toán và tạo ra các số liệu ngụy tạo.
2. **Can thiệp tối thiểu bắt buộc trước khi tái cấp phép huấn luyện:**
   - **Online Choice Permutation:** Trong quá trình huấn luyện MCQA, ngẫu nhiên hoán vị thứ tự các phương án $A, B, C, D$ ở mỗi batch để triệt tiêu triệt để xu hướng thiên vị chữ cái.
   - **Label Smoothing / Balanced Loss:** Bổ sung ràng buộc entropy hoặc điều hòa phân bố nhãn để ép mô hình phân bổ xác suất đồng đều trên 4 phương án thay vì co cụm về một ký tự duy nhất.

---

## 4. Báo Cáo Quyết Định Mở Rộng Quy Mô (Scale-Up Decision) và 22 Tiêu Chí Rào Chắn

Theo văn kiện quyết định chính thức [[Decision-Making/18_STAGE03_MECHANISM_BEFORE_SCALE|DECISION-STAGE03-MECHANISM-BEFORE-SCALE-107]] ban hành ngày 18/10/2026:

### 4.1. Lựa chọn chiến lược
- **Quyết định:** Chọn **Phương án A (Tiếp tục tinh chỉnh và kiểm toán Stage 03)**; **KHÔNG** cấp phép huấn luyện toàn diện trên tập dữ liệu lớn vào thời điểm này.
- **Ranh giới khoa học:** Vấn đề chưa được định danh rõ nằm ở **cơ chế nhân quả thời gian** và **hiện tượng sụp đổ nhãn chữ cái**, không phải do thiếu quy mô dữ liệu hay thiếu số bước tối ưu hóa.

### 4.2. Lộ trình nghiên cứu 8 tuần (Eight-Week Evidence-to-Scale Program)
1. **Tuần 1 (19/10 - 25/10/2026):** Chuẩn hóa mẫu số Run 05.2, phép nối dữ liệu nguồn và khóa mã băm SHA256.
2. **Tuần 2 (26/10 - 01/11/2026):** Tái đánh giá xác định (deterministic) ghép cặp Base / C1 / Arm A / Arm B.
3. **Tuần 3 (02/11 - 08/11/2026):** Đo đạc trực tiếp lõi $F_t$, cổng Seam, LoRA ON/OFF và đường truyền can thiệp.
4. **Tuần 4 (09/11 - 15/11/2026):** Mở rộng tập phản thực nghiệm mốc năm có lật nhãn chân lý & hoàn tất phân tích phân bố nhãn.
5. **Tuần 5 (16/11 - 22/11/2026):** Cấu trúc cụm nguồn, mô phỏng lực thống kê (power simulation) và tiền đăng ký thử nghiệm.
6. **Tuần 6 (23/11 - 29/11/2026):** Thí nghiệm khói can thiệp đúng một biến độc lập (**Online Option Permutation**).
7. **Tuần 7 (30/11 - 06/12/2026):** Tái lập đa hạt giống trên tập holdout sạch nguồn và kiểm tra suy thoái ngôn ngữ (NTP retention).
8. **Tuần 8 (07/12 - 13/12/2026):** Thẩm định 22 tiêu chí Promotion Gate và ra phán quyết GO hoặc NO-GO cho Full Training.

### 4.3. Bộ 22 Tiêu Chí Thẩm Định Cấp Phép Huấn Luyện Toàn Diện (Full-Training Promotion Gate)
Hệ thống kiểm soát fail-closed: chỉ cần 1 tiêu chí không đạt, dự án lập tức duy trì Phương án A.
1. `[ ]` **Checkpoint identity:** Xác thực đầy đủ SHA của parent, adapter, tensor fingerprint, model, tokenizer, dataset, evaluator và git commit.
2. `[ ]` **Frozen backbone:** Trọng số base model không đổi trước và sau smoke optimizer run.
3. `[ ]` **Adapter inventory:** Nạp đúng 72 module MaT-LoRA; ghi nhận trạng thái active/inactive rõ ràng.
4. `[ ]` **Base parity:** `LoRA-OFF + Seam-OFF` tái lập logits của Base model trong ngưỡng sai số định trước.
5. `[ ]` **Adapter sensitivity:** Kiểm chứng thực tế LoRA-ON tạo ra biến thiên logits khác biệt so với LoRA-OFF.
6. `[ ]` **State hygiene:** Vector thời gian được reset độc lập trên từng mẫu; số lượng live hooks sau eval $= 0$.
7. `[ ]` **Treatment proof:** Mọi cặp can thiệp thời gian phải tái tính vector từ đúng năm bị làm sai ($y_{\text{perturbed}}$).
8. `[ ]` **F_t proof:** Ghi nhận trực tiếp sự biến thiên của ma trận lõi động $F_t$ qua các mốc năm can thiệp.
9. `[ ]` **Causal denominator:** Loại trừ 144 câu đa mốc năm khỏi mẫu số tính hiệu ứng can thiệp đơn mốc.
10. `[ ]` **Evaluator determinism:** Tắt hoàn toàn sampling ngẫu nhiên (`do_sample=False`) trong đánh giá so sánh ghép cặp.
11. `[ ]` **Denominator integrity:** Toàn bộ điểm số tổng hợp phải tái lập được từ từng bản ghi mẫu riêng biệt.
12. `[ ]` **Single-gold integrity:** $100\%$ câu hỏi trắc nghiệm chỉ có đúng 1 đáp án chân lý duy nhất.
13. `[ ]` **Primary holdout lineage:** Tập holdout chính thức không có bất kỳ chồng lấn nguồn/sự kiện nào với tập train.
14. `[ ]` **Duplicate audit:** Không có câu hỏi trùng lặp hoặc cận trùng lặp giữa train và holdout.
15. `[ ]` **Option-bias gate:** Không có chữ cái nào chiếm $>50\%$ dự đoán trên tập 4 hoán vị; recall mọi nhãn $>0$.
16. `[ ]` **Semantic permutation gate:** Tính nhất quán ngữ nghĩa sau ánh xạ ngược phải cải thiện vượt bậc so với $0.0\%$.
17. `[ ]` **Grounded temporal gate:** Tập phản thực nghiệm có sự thay đổi chân lý ngữ nghĩa thật theo mốc thời gian.
18. `[ ]` **Positive small-run result:** Can thiệp thời gian đánh bại đối chứng âm tĩnh trong thử nghiệm khói nhỏ.
19. `[ ]` **Power:** Cỡ mẫu ghép cặp độc lập đạt tối thiểu lực thống kê $80\%$ (ước tính $\approx 384$ cặp cho gain $10\text{ pp}$).
20. `[ ]` **Retention:** Tỷ lệ suy thoái Perplexity ngôn ngữ trên tập NTP held-out không vượt quá biên độ $5\%$.
21. `[ ]` **One-variable attribution:** Huấn luyện mở rộng chỉ scale can thiệp đã vượt qua bài test khói (không trộn nhiều thay đổi).
22. `[ ]` **Matched exposure:** Nhánh can thiệp và nhánh đối chứng nhận cùng số bước và cùng số token giám sát.
