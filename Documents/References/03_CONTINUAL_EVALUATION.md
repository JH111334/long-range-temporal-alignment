---
loại: "tham-chiếu"
nguồn: "local-file:References/03_CONTINUAL_EVALUATION.md"
độ_tin_cậy: "exploratory"
chủ_đề: "03_CONTINUAL_EVALUATION"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: false
cập_nhật_lần_cuối: "2026-10-12"
---

# Khung Đánh Giá Học Liên Tục và Giao Thức Phân Tích Cú Pháp Nghiêm Ngặt (Continual Evaluation & Strict Parsing)

Tài liệu này đặc tả toàn diện phương pháp luận đánh giá, các công thức toán học đo lường khả năng ghi nhớ/chuyển giao tri thức và khế ước phân tích cú pháp tạo sinh nghiêm ngặt (**`GENERATIVE_STRICT_PARSE_V1`**) trong học liên tục theo thời gian trên tập dữ liệu TempUN Small.

---

## 1. Ý Nghĩa Bản Chất và Động Lực Nghiên Cứu

### Thách thức đánh giá trong Học Liên Tục (Continual Learning)
Khác với bài toán học máy tĩnh (nơi mô hình chỉ được đánh giá một lần trên tập test độc lập), học liên tục theo dòng thời gian đòi hỏi việc theo dõi sự tiến hóa của tri thức qua nhiều chặng tuần tự:
$$\text{Stage 1 (1947)} \longrightarrow \text{Stage 2 (1984)} \longrightarrow \text{Stage 3 (2022)}$$

Một phương pháp thích ứng tốt không chỉ cần đạt kết quả cao trên dữ liệu hiện tại đang học, mà bắt buộc phải:
1. **Duy trì ký ức quá khứ:** Không làm suy giảm độ chính xác trên các sự kiện lịch sử năm 1947 sau khi đã học xong năm 1984 và 2022.
2. **Ngăn chặn trôi dạt cú pháp:** Không để mô hình rơi vào trạng thái sinh văn bản lặp vô nghĩa (degeneration) khi tiếp nhận liên tục các thông số mới.

### Đối chiếu Generative Evaluation và Option Scoring
Có hai trường phái đánh giá mô hình ngôn ngữ trên bài toán trắc nghiệm:
- **Option Scoring (Log-Likelihood Scoring):** Ép buộc mô hình tính xác suất có điều kiện trên 4 ký tự $P(\text{'A'}), P(\text{'B'}), P(\text{'C'}), P(\text{'D'})$ và chọn phương án có xác suất cao nhất.
  - *Nhược điểm:* Che giấu hoàn toàn các khuyết tật về khả năng sinh ngôn ngữ tự nhiên và tính ổn định cú pháp của mô hình.
- **Generative Evaluation (Tự do sinh đáp án):** Cung cấp toàn bộ câu hỏi và để mô hình tự hồi quy sinh ra câu trả lời dưới dạng văn bản tự nhiên, sau đó dùng bộ phân tích cú pháp nghiêm ngặt để trích xuất đáp án.
  - *Ưu điểm:* Phản ánh trung thực năng lực triển khai thực tế của hệ thống.
  - *Yêu cầu bắt buộc:* Phải tách biệt rõ ràng giữa **Lỗi Tri Thức** (chọn sai đáp án) và **Lỗi Cú Pháp** (sinh câu không thể bóc tách).

---

## 2. Công Thức Toán Học và Kích Thước Ma Trận Hiệu Năng

### 2.1. Ma Trận Đánh Giá Liên Tục $R \in \mathbb{R}^{T \times T}$
Với chuỗi $T$ chặng thời gian ($T=3$: 1947, 1984, 2022), sau khi hoàn thành huấn luyện tại chặng thứ $i \in \{1, \dots, T\}$, ta đánh giá mô hình trên toàn bộ tập kiểm tra của các chặng $j \in \{1, \dots, T\}$.

Đặt $R_{i, j}$ là độ chính xác nghiêm ngặt (Strict Accuracy) trên tập kiểm tra của chặng $j$ được đo tại thời điểm mô hình vừa hoàn thành huấn luyện chặng $i$:
$$R = \begin{pmatrix} R_{1, 1} & R_{1, 2} & R_{1, 3} \\ R_{2, 1} & R_{2, 2} & R_{2, 3} \\ R_{3, 1} & R_{3, 2} & R_{3, 3} \end{pmatrix}$$

Trong đó:
- Đường chéo chính $R_{i, i}$: Hiệu năng trực tiếp tại thời điểm vừa học xong chặng $i$.
- Nửa dưới tam giác $R_{i, j}$ ($i > j$): Khả năng bảo tồn tri thức quá khứ của chặng $j$ sau khi đã học thêm các chặng sau.
- Nửa trên tam giác $R_{i, j}$ ($i < j$): Khả năng khái quát hóa về tương lai (Zero-shot Forward Transfer).

### 2.2. Các Chỉ Số Đánh Giá Cốt Lõi

#### 1. Độ chính xác trung bình cuối cùng (Final Average Accuracy - FAA):
Hiệu năng trung bình trên toàn bộ các thời kỳ sau khi kết thúc toàn bộ lộ trình học:
$$\text{FAA} = \frac{1}{T} \sum_{j=1}^T R_{T, j}$$

#### 2. Khả năng chuyển giao ngược (Backward Transfer - BWT):
Đo lường mức độ tác động của việc học các chặng mới lên tri thức của các chặng quá khứ:
$$\text{BWT} = \frac{1}{T - 1} \sum_{j=1}^{T - 1} (R_{T, j} - R_{j, j})$$
- $\text{BWT} < 0$: Hiện tượng quên lãng tai hại (Catastrophic Forgetting). Giá trị âm càng lớn thể hiện sự quên lãng càng nghiêm trọng.
- $\text{BWT} = 0$: Bảo toàn hoàn hảo tri thức quá khứ.
- $\text{BWT} > 0$: Học dữ liệu mới giúp củng cố và nâng cao hiểu biết về dữ liệu cũ (Positive Backward Transfer).

#### 3. Độ sụt giảm quên lãng tối đa (Average Forgetting - F):
Mức suy giảm lớn nhất so với đỉnh cao hiệu năng từng đạt được trong quá khứ:
$$F = \frac{1}{T - 1} \sum_{j=1}^{T - 1} \max_{k \in \{1, \dots, T - 1\}} (R_{k, j} - R_{T, j})$$

---

## 3. Khế Ước Phân Tích Cú Pháp Nghiêm Ngặt (`GENERATIVE_STRICT_PARSE_V1`)

### 3.1. Phân loại Bộ Ba Kết Quả (Outcome Taxonomy)
Đối với mỗi câu hỏi trong 180 mẫu của tập `eval_mcqa.jsonl`, câu trả lời sinh ra bởi mô hình $\hat{y}$ được phân loại thành một trong ba trạng thái duy nhất:
1. **Chính xác (CORRECT):** Mô hình sinh đúng định dạng cú pháp chuẩn VÀ giá trị trích xuất trùng khớp hoàn toàn với đáp án vàng ($y^*$).
2. **Sai lầm (INCORRECT):** Mô hình sinh đúng định dạng cú pháp chuẩn NHƯNG giá trị trích xuất khác đáp án vàng.
3. **Lỗi cú pháp (INVALID / SYNTAX_ERROR):** Mô hình sinh câu trả lời không thể bóc tách thành các phương án hợp lệ (`A, B, C, D, Yes, No`).

### 3.2. Ba Chỉ Số Thống Kê Chuẩn Tắc
Để ngăn chặn hành vi thiên vị hoặc ngụy tạo kết quả, ba chỉ số sau bắt buộc phải được công bố độc lập:

$$\text{Strict Accuracy} = \frac{N_{\text{CORRECT}}}{N_{\text{TOTAL}}} \times 100\% \quad (N_{\text{TOTAL}} = 180)$$
$$\text{Valid Accuracy} = \frac{N_{\text{CORRECT}}}{N_{\text{CORRECT}} + N_{\text{INCORRECT}}} \times 100\%$$
$$\text{Invalid Syntax Rate} = \frac{N_{\text{INVALID}}}{N_{\text{TOTAL}}} \times 100\%$$

**Quy tắc Dừng An Toàn (Quality Gate):** Nếu $\text{Invalid Syntax Rate} > 30{,}0\%$, mô hình bị coi là mất kiểm soát sinh văn bản và toàn bộ run thử nghiệm đó bị gán nhãn `REJECT`.

---

## 4. Ánh Xạ Vào Mã Nguồn Thực Thi (Code Mapping)

### 4.1. Bộ bóc tách trong `tempun_dataset.py`
Hàm `parse_strict_mcqa_answer` áp dụng hệ thống quy tắc làm sạch và biểu thức chính quy chuẩn hóa:
```python
def parse_strict_mcqa_answer(prediction: str) -> Optional[str]:
    cleaned = prediction.strip().strip(".()[]: \n\t")
    # Khớp chính xác phương án trắc nghiệm đơn lẻ
    if cleaned in ("A", "B", "C", "D"):
        return cleaned
    # Khớp câu hỏi Yes/No
    lower = cleaned.lower()
    if lower in ("yes", "y"):
        return "Yes"
    if lower in ("no", "n"):
        return "No"
    # Khớp chữ cái đứng đầu câu có dấu phân cách
    match_letter = re.match(r"^([A-D])[\s\.\:\,\)]", cleaned)
    if match_letter:
        return match_letter.group(1)
    match_yn = re.match(r"^(yes|no)\b", lower)
    if match_yn:
        return "Yes" if match_yn.group(1) == "yes" else "No"
    return None
```

### 4.2. Vòng lặp đo lường trong `run_m0_fc40.py`
Trong hàm `evaluate_stage`:
- Sau mỗi câu hỏi, trạng thái vector tiêm `cond_context.clear()` được dọn dẹp để đảm bảo tính độc lập nhân quả giữa các ca thử nghiệm.
- Kết quả được ghi nhận chi tiết tới từng câu hỏi trong tệp `predictions_all_stages.json` phục vụ kiểm định thống kê.

---

## 5. Phương Pháp và Khế Ước Kiểm Chứng Thực Nghiệm

1. **Kiểm định Ghép Cặp McNemar (Paired McNemar Test):**
   Khi so sánh hai mô hình (ví dụ `M1-TEMP` vs `M0-FC40` hoặc `M1-TEMP` vs `M1-NC`), ta xây dựng bảng tiếp liên $2 \times 2$ trên 180 câu hỏi:
   - $n_{01}$: Số câu mô hình 1 sai nhưng mô hình 2 đúng.
   - $n_{10}$: Số câu mô hình 1 đúng nhưng mô hình 2 sai.
   - Thống kê kiểm định:
     $$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$$
   - Mức ý nghĩa thống kê bắt buộc: $p < 0{,}05$.
2. **Giao thức 3 Hạt Ngẫu Nhiên Cố Định (Fixed 3-Seed Protocol):**
   Mọi thử nghiệm phải chạy trên chính xác ba hạt `SEEDS = [1, 2, 3]`. Kết quả cuối cùng là trung bình cộng và độ lệch chuẩn của cả 3 seed.

---

## 6. Các Cạm Bẫy Kỹ Thuật Thường Gặp (Pitfalls & Failure Modes)

1. **Bóc tách lỏng lẻo (Loose Substring Search):**
   Nếu kiểm tra sự tồn tại của chữ cái bằng lệnh `"A" in prediction`, các câu văn sinh ra chứa từ ngữ thông thường (như *"Australia"*, *"Answer"*) sẽ bị nhận nhầm là chọn phương án A, gây ra hiện tượng dương tính giả (False Positive) rất nghiêm trọng.
2. **Lẫn lộn Mẫu Số (Denominator Confusion):**
   Nếu tính độ chính xác bằng cách lấy `correct / valid`, một mô hình chỉ sinh được 1 câu hợp lệ và đúng câu đó sẽ đạt $100\%$ độ chính xác dù có 179 câu bị lỗi cú pháp. Bắt buộc phải dùng mẫu số $N_{\text{TOTAL}} = 180$.
3. **Rò rỉ Ngữ Cảnh Thời Gian Giữa Các Mẫu Đánh Giá:**
   Nếu không xóa sạch vector thời gian trong `ConditioningContext` sau mỗi mẫu câu, câu hỏi năm 2022 có thể vô tình sử dụng vector của câu hỏi năm 1947 được xử lý ngay trước đó trong vòng lặp.
