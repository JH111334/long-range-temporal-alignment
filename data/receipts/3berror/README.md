# Báo Cáo Kiểm Toán: Điểm Yếu và Chế Độ Lỗi Mô Hình Qwen2.5-3B Trên Dữ Liệu TempUN Full

## 1. Bối Cảnh Thực Nghiệm và Khung Đánh Giá
- **Mô hình**: `Qwen/Qwen2.5-3B-Instruct` ($3.09\text{B}$ tham số, $36$ tầng, $d=2048$).
- **Dữ liệu**: `TempUN Full Clean` qua $76$ năm ($1947$–$2022$).
  - Tập huấn luyện: $104{,}130$ mẫu Next Word Prediction (NWP).
  - Tập đánh giá: $5{,}292$ mẫu Multiple Choice Question Answering (MCQA), cân bằng tuyệt đối qua $6$ nhóm nhiệm vụ: Date-Based (`DB`), Min-Max (`MM`), Range-Based (`RB`), Trend-Based (`TB`), Relative (`Rel`), Window-Based (`WB`).
- **Giao thức đánh giá**: Closed-set Choice-Logit Argmax với cơ chế khử thiên vị xoay vòng $4$ phương án (ABCD, BCDA, CDAB, DABC) đối với trắc nghiệm $4$ lựa chọn và $2$ chiều đối với câu hỏi Yes/No.

---

## 2. Bảng Tổng Hợp So Sánh Ghép Cặp

| Cấu hình | Canonical Acc (%) | Debiased Mean Acc (%) | Strict Consistency (%) | Số mẫu đúng kiên định |
| :--- | :---: | :---: | :---: | :---: |
| **Base 3B Zero-shot** | $28.38$ | $29.20$ | **$6.24$** | $330\ /\ 5{,}292$ |
| **Ticktack 3B** | $27.85$ | $28.22$ | **$6.22$** | $329\ /\ 5{,}292$ |
| **Chênh lệch ($\Delta$)** | **$-0.53$** | **$-0.98$** | **$-0.02$** | $-1$ mẫu |

### Kiểm định thống kê McNemar (Edwards continuity correction)
- **Toàn bộ $5{,}292$ mẫu**:
  - $a (\text{cả hai cùng đúng}) = 1{,}222$; $b (\text{Base đúng, Ticktack sai}) = 280$; $c (\text{Base sai, Ticktack đúng}) = 252$; $d (\text{cả hai cùng sai}) = 3{,}538$.
  - Thống kê: $\chi^2 = 1.3703$, $p = 0.2418 > 0.05$.
  - Phán quyết: **Không có sự khác biệt có ý nghĩa thống kê giữa Base 3B và Ticktack 3B**.
- **Tập con trắc nghiệm $4$ lựa chọn ($4{,}440$ mẫu)**:
  - Base đạt $26.55\%$ ($1{,}179/4{,}440$), Ticktack đạt $25.95\%$ ($1{,}152/4{,}440$).
  - $\chi^2 = 1.2731$, $p = 0.2592 > 0.05$ (dao động ngẫu nhiên xung quanh mức cơ sở $25.0\%$).
- **Độ nhất quán kiên định qua $4$ vòng xoay (Strict Consistency)**:
  - $a = 322$, $b = 8$, $c = 7$, $d = 4{,}955$.
  - $\chi^2 = 0.0000$, $p = 1.0000$. Hai mô hình trùng khớp nhau ở $322$ mẫu ($97.6\%$ số mẫu kiên định).

---

## 3. Bản Chất Bốn Điểm Yếu và Chế Độ Lỗi (Deficiency Taxonomy)

### 3.1. Sụp đổ nhãn tuyệt đối (100% No Collapse) trên bài toán Relative
- **Hiện tượng**: Trên toàn bộ $852$ câu hỏi nhị phân của bài toán Relative (`Rel`), mô hình dự đoán nhãn `'No'` cho $100\%$ trường hợp ở Base 3B ($852/852$) và $99.88\%$ ở Ticktack 3B ($851/852$).
- **Hệ quả**: Điểm số chính xác $37.91\%$ đo được trên bài toán Relative thực chất bằng đúng tỷ lệ xuất hiện của nhãn `'No'` trong tập dữ liệu ($323 / 852 = 37.91\%$). Độ chính xác đối với các mẫu có nhãn thật là `'Yes'` bằng $0.0\%$.
- **Nguyên nhân gốc rễ**: Token `'no'` có logit tiên nghiệm không điều kiện (unconditional prior logit) cao hơn token `'yes'` trong không gian từ vựng của Qwen2.5-3B. Khi lấy argmax logit trực tiếp mà không chuẩn hóa xác suất biên, mô hình bị thiên lệch hoàn toàn về phía `'No'`.
- **Đề xuất khắc phục**: Cần áp dụng chuẩn hóa logit biên $\text{logit}_{\text{calibrated}}(y) = \text{logit}(y) - \text{logit}_{\text{prior}}(y)$ hoặc bổ sung loss cân chỉnh nhị phân.

### 3.2. Thiên vị chữ cái cố định (Option-Letter Collapse) trên trắc nghiệm 4 lựa chọn
- **Hiện tượng**: $72.6\%$ số câu trắc nghiệm ($3{,}843 / 5{,}292$) đạt giá trị điểm khử thiên vị đúng bằng $0.25$ qua $4$ vòng quay hoán vị (ABCD, BCDA, CDAB, DABC).
- **Hệ quả**: Mô hình có xu hướng cố định dự đoán vào một vị trí ký tự chữ cái nhất định bất chấp việc các phương án nội dung bị hoán đổi vị trí. Khi xoay vòng $4$ lần, đáp án đúng chỉ rơi vào vị trí ưa thích đó đúng $1$ lần duy nhất.
- **Ý nghĩa**: Mức điểm khoảng $28\%$ của Canonical Accuracy chỉ phản ánh thiên vị chữ cái cộng thêm xác suất ngẫu nhiên $25\%$, trong khi tri thức thời gian thực chất (Strict Consistency) chỉ đạt khoảng $6.2\%$.

### 3.3. Hiệu chỉnh lỗi hiển thị Strict Accuracy trong báo cáo tóm tắt
- Trong tệp `ticktack_benchmark_summary.json` ban đầu, trường `strict_acc` của `Base_3B_Ref` bị gán nhầm thành **$25.1\%$** (do nhầm lẫn với xác suất ngẫu nhiên $1/4$).
- Dữ liệu thực nghiệm thô đo đạc chính xác ghi nhận Base 3B chỉ đạt **$6.24\%$** ($330 / 5{,}292$ mẫu). Lỗi hiển thị này đã được đính chính đồng bộ về đúng $6.24\%$ để phản ánh tính tương đồng giữa Base 3B và Ticktack 3B.

### 3.4. Trùng tiền tố định danh trong tập đánh giá
- Tập đánh giá ghi nhận $264$ chuỗi `example_id` bị lặp lại do sinh nhiều cửa sổ thời gian cho cùng một thực thể (ví dụ `mcq:WB:1949:DVPR-162`).
- Giải pháp định danh duy nhất thông qua chỉ số ghép `#{idx}` của đối tác là hoàn toàn chính xác và cần được duy trì.

---

## 4. Kết Luận Khoa Học và Hướng Đi Tiếp Theo
1. Mô hình quy mô 3B vận hành ổn định trên tài nguyên Tesla T4 nhưng không tự động giải quyết được bài toán biểu diễn thời gian dài hạn nếu chỉ tiếp tục huấn luyện tự quy hồi (NWP) trên dữ liệu thô.
2. Để vượt qua ngưỡng ngẫu nhiên trên MCQA, kiến trúc cần:
   - Cơ chế cân chỉnh logit tiên nghiệm (Prior Logit Calibration) cho nhãn Yes/No.
   - Hàm mục tiêu huấn luyện trắc nghiệm tương phản (Contrastive Multi-choice Objective) để triệt tiêu thiên vị vị trí phương án.
