# Ma Trận Năng Lực Mô Hình Thực Nghiệm (Run 04.1 Capability Matrix)

Tài liệu này bóc tách năng lực mô hình theo 4 góc phần tư độc lập: **Tri thức sự thật (K)**, **Căn chỉnh thời gian (G)**, **Suy luận quan hệ số học (R)**, và **Định dạng & Duy trì (F)**.

---

## 1. Bảng Tổng Hợp 4 Góc Phần Tư Năng Lực

| Góc Phần Tư | Câu Hỏi Khoa Học Khảo Sát | Thiết Kế Thử Nghiệm & Điều Kiện | Bằng Chứng Thực Nghiệm Đo Đạc | Hạn Chế & Phản Ví Dụ | Phán Quyết Khoa Học |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **K: Tri Thức Sự Thật (Knowledge of Facts)** | Mô hình có tự ghi nhớ được giá trị số học lịch sử của thực thể đóng kín hay không? | Tập DEV Clean MCQA (120 câu closed-book) trên hai thực thể mới `BCXZ` và `GVRO` không có trong tập TRAIN. | Mô hình nền đạt $40{,}00\%$, $R4\_B$ đạt $65{,}83\%$. Tuy nhiên, phân tích logits cho thấy mô hình chủ yếu dựa vào tần suất từ vựng và tiên nghiệm ngôn ngữ thay vì biết chính xác số thực. | Một vector 40D duy nhất không thể mã hóa số liệu chi tiết của hàng nghìn chỉ số độc lập nếu không có văn cảnh. | **`KNOWLEDGE_ABSENT_WITHOUT_CONTEXT`** |
| **G: Căn Chỉnh Thời Gian (Temporal Grounding)** | Vector thời gian 40D có thực sự định hướng câu trả lời phụ thuộc vào mốc năm hay không? | So sánh trên cùng checkpoint $R4\_B$: Điều kiện đúng năm (`ADAPTED_TRUE`) vs Điều kiện đảo năm (`ADAPTED_WRONG`). Đánh giá trên 60 câu Dev Relation. | Trên Scoreboard 2 (Relation): $R4\_B$ đạt **$86{,}67\%$** (dự đoán cân bằng $29$ Yes / $31$ No), vượt trội $+16{,}67\text{ pp}$ so với đối chứng âm $R4\_C$ ($70{,}00\%$, $p < 0{,}05$) và $+38{,}34\text{ pp}$ so với Base model ($48{,}33\%$). | Trên MCQA 4 lựa chọn, đảo năm chỉ làm lệch điểm $+1{,}66\text{ pp}$ ($p = 0{,}625$). Tiên nghiệm ngôn ngữ trên văn bản phương án dài vẫn chi phối quá trình sinh. | **`GROUNDED_ON_BINARY_RELATIONS / INCONCLUSIVE_ON_COMPLEX_MCQA`** |
| **R: Suy Luận Quan Hệ Số Học (Relational Reasoning)** | Mô hình có năng lực so sánh hai số thực khi được cung cấp đầy đủ số liệu hay không? | Scoreboard 3 (Oracle Evidence Diagnostic, 60 câu): Đưa trực tiếp số liệu lịch sử thật vào câu hỏi: *"In 1947, X was 0.118, and in 2022, it was 0.771. Which statement is correct?"* | Độ chính xác của $R4\_B$ đạt **$65{,}00\%$** (nhánh $R4\_A$ đạt $70{,}00\%$), vượt trội hoàn toàn so với mô hình nền ($38{,}33\%$). | Mô hình đôi khi nhầm lẫn giữa toán tử "cao hơn" và "thấp hơn" khi thứ tự phát biểu bị đảo ngược giữa các phương án. | **`VERIFIED_INTACT_UNDER_EVIDENCE`** |
| **F: Định Dạng & Lưu Giữ (Format & Retention)** | Huấn luyện thích ứng có phá vỡ định dạng suy luận hay gây quên thảm khốc benchmark cũ không? | Scoreboard 4 (Retention 60-slice TempUN Small) và kiểm toán tỷ lệ cú pháp không hợp lệ (`Invalid Rate`). | Tỷ lệ `Invalid Rate` duy trì tuyệt đối **$0{,}00\%$** trên toàn bộ các bài đánh giá. Điểm số TempUN phục hồi về **$28{,}33\% \dots 31{,}67\%$** (ngang ngửa Base model $31{,}67\%$). | Khắc phục hiện tượng $0/60$ cũ của Run 03: nguyên nhân là do trôi dạt định dạng lời nhắc và lỗi parser, không phải do suy giảm dung lượng tham số rank 8. | **`RETENTION_RESCUED / SYNTAX_INTACT`** |

---

## 2. Phân Tích Cơ Chế Kích Hoạt và Tỷ Lệ Đóng Góp Tham Số

1. **Lớp Thích Ứng MaT-LoRA MLP ($r=8, \alpha=16{,}0$):**
   - Định mức chuẩn Frobenius của trọng số gốc $W_0$: trung bình $\|W_0\|_F \approx 18{,}5 \dots 22{,}4$.
   - Định mức chuẩn của $\Delta W = \frac{\alpha}{r} B F(\theta, z) A$: $\|\Delta W\|_F \approx 0{,}15 \dots 0{,}32$.
   - Tỷ lệ đóng góp năng lượng $\frac{\|\Delta W\|_F}{\|W_0\|_F} \approx 0{,}8\% \dots 1{,}5\%$.
   - Tỷ lệ này bảo đảm an toàn, không làm biến dạng biểu diễn từ vựng ngôn ngữ của mô hình Instruct.

2. **Cổng Gated Residual Seam (Token Index 14):**
   - Hệ số cổng $\alpha$ sau 30 bước huấn luyện: $\alpha \approx 0{,}0182$.
   - Vector tiêm $v = \alpha \cdot \text{LayerNorm}(W_{\text{proj}} z)$ có định mức $\|v\|_2 \approx 0{,}054$, vừa đủ tác động vào trạng thái ẩn prefill mà không gây bão hòa kích hoạt.
