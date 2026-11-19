# Báo Cáo Đo Đạc và Kiểm Toán Khoa Học: Qwen2.5-3B Full Baseline & Ticktack

## 1. Thông số Thực Nghiệm Cốt Lõi
- **Mô hình nền**: `Qwen/Qwen2.5-3B-Instruct` (3.09B tham số, 36 tầng, `d=2048`).
- **Phần cứng thực thi**: NVIDIA Tesla T4 (14.56 GB VRAM) trên Kaggle Cloud.
- **Tập dữ liệu**: `TempUN Full Clean` qua 76 năm (1947–2022).
  - Tập huấn luyện NWP: đúng 104,130 mẫu.
  - Tập đánh giá MCQA: đúng 5,292 mẫu (chia đều qua 6 bài toán: DB, MM, RB, TB, Rel, WB).
- **Tiến trình huấn luyện Ticktack 3B**:
  - Thời gian: 29,238.5 giây (~8.12 giờ), 6,508 bước hiệu dụng.
  - Hội tụ: Loss giảm đều từ 4.12 xuống 2.8919; hoàn toàn không có NaN/Inf.
- **Tiến trình đánh giá**:
  - Thời gian: 9,487.9 giây (~2.63 giờ).
  - Độ bao phủ: 5,292/5,292 câu (100.0%). Invalid rate = 0.0%.

---

## 2. Bảng Tổng Hợp Kết Quả Thực Nghiệm

| Cấu hình | Canonical Acc (%) | Debiased Mean Acc (%) | Strict Consistency (%) | Số câu Strict |
| :--- | :---: | :---: | :---: | :---: |
| **Base 3B Zero-shot** | 28.38 | 29.20 | 6.24 | 330 / 5292 |
| **Ticktack 3B** | 27.85 | 28.22 | 6.22 | 329 / 5292 |
| **Chênh lệch (Delta)** | **-0.53** | **-0.98** | **-0.02** | -1 câu |

> **Lưu ý kiểm toán quan trọng**: Giá trị `strict_acc: 25.1%` trong file tóm tắt ban đầu của đối tác là lỗi hiển thị do hardcode/nhầm lẫn tham chiếu. Kết quả đo đạc thô thực tế của Base 3B là **6.24%** (330 câu), tương đương với Ticktack 3B (**6.22%**, 329 câu).

---

## 3. Kiểm Định Thống Kê Ghép Cặp McNemar

### 3.1. Toàn bộ 5,292 mẫu
- Ma trận liên tục: $a (\text{cả hai đúng}) = 1,222$, $b (\text{chỉ Base đúng}) = 280$, $c (\text{chỉ Ticktack đúng}) = 252$, $d (\text{cả hai sai}) = 3,538$.
- Số trường hợp bất đồng ($n_{\text{disc}} = b + c$): 532.
- Kiểm định Edwards: $\chi^2 = 1.3703$, $p = 0.2418 > 0.05$.
- **Kết luận**: Không có sự khác biệt có ý nghĩa thống kê giữa Ticktack 3B và Base 3B.

### 3.2. Tính nhất quán qua 4 vòng quay (Strict Consistency)
- $a = 322$, $b = 8$, $c = 7$, $d = 4,955$.
- $\chi^2 = 0.0000$, $p = 1.0000$. Cả hai mô hình hoàn toàn tương đương về mặt tri thức vững chắc.

### 3.3. Phân rã theo 6 nhiệm vụ
1. **DB (Date-Based, N=912)**: Base 26.10% > TT 24.12% ($b=35, c=17, \chi^2=5.558, p=0.0184$ — Base tốt hơn đáng kể).
2. **MM (Min-Max, N=852)**: Base 25.82% vs TT 23.47% ($b=62, c=42, \chi^2=3.471, p=0.0624$).
3. **RB (Range-Based, N=852)**: TT 27.00% > Base 25.35% ($b=14, c=28, \chi^2=4.024, p=0.0449$ — TT tốt hơn nhẹ).
4. **Rel (Relative, N=852)**: Base 37.91% ≈ TT 37.79% ($b=1, c=0, \chi^2=0.000, p=1.0000$).
5. **TB (Trend-Based, N=912)**: Base 26.32% ≈ TT 26.10% ($b=48, c=46, \chi^2=0.011, p=0.9179$).
6. **WB (Window-Based, N=912)**: Base 29.06% ≈ TT 28.95% ($b=120, c=119, \chi^2=0.000, p=1.0000$).

---

## 4. Kiểm Toán Dữ Liệu và Rò Rỉ (Data Integrity & Leakage)
1. **Rò rỉ dữ liệu (Data Leakage)**: $0$ trùng lặp `example_id`, $0$ trùng lặp văn bản đầu vào giữa Train ($104,130$) và Eval ($5,292$).
2. **Phân chia tập dữ liệu (Splitting)**: Đảm bảo phủ kín 76 năm (1947–2022). Số mẫu mỗi năm hoàn toàn tuân thủ ràng buộc cửa sổ thời gian (1947–1951: 48 mẫu/năm, 1952–2017: 72 mẫu/năm, 2018–2022: 60 mẫu/năm).
3. **Định danh trong tập Eval**: 264 ID có tiền tố trùng lặp do phát sinh nhiều cửa sổ thời gian cho cùng thực thể. Cơ chế nối chỉ số `#{idx}` của đối tác giải quyết triệt để xung đột này.

---

## 5. Nút Nghẽn Bản Chất Phát Hiện Qua Kiểm Toán
1. **Sụp đổ nhãn Yes/No trên bài toán Relative**: Mô hình dự đoán "No" cho toàn bộ 852 câu của bài toán Relative do logit prior của token "No" chiếm ưu thế tuyệt đối. Điểm số 37.91% hoàn toàn do phân bố nhãn "No" trong tập đánh giá quy định.
2. **Thiên vị chữ cái cố định (Option Bias) trên trắc nghiệm 4 lựa chọn**: 72.6% số câu hỏi đạt đúng 0.25 điểm qua 4 hoán vị vòng tròn. Cả hai mô hình đều bị neo vào một vị trí chữ cái cố định, khiến năng lực biểu diễn thời gian không chuyển hóa thành câu trả lời đúng.
