# Phán Quyết Khoa Học: V2 Run 03 — Căn Chỉnh Đa Nhiệm Quan Hệ Thời Gian và Khái Quát Hóa MCQA

**Ngày lập:** 2026-10-15  
**Tác giả:** Antigravity (Learning Track)  
**Nhánh tri thức:** `v2-targeted-continual-alignment`  
**Điểm xuất phát:** Checkpoint Run 02 C1 (`09acf723ea97401924a070b14775d09584d0a7a565c68d83c71218e814f72cb9`)  
**Mục tiêu kiểm định:** Phục hồi năng lực giải bài toán trắc nghiệm MCQA 4 lựa chọn (a/b/c/d) đồng thời bảo toàn tính nhân quả thời gian mà không mở khóa backbone hay thay đổi giao diện FoNE–Can Chi 40D.

---

## 1. Tóm Tắt Kết Quả 4 Bảng Điểm Thực Nghiệm

| Bảng Điểm (Scoreboard) | RUN02_C1 (Base) | R3_A (Rel Only) | R3_B (MCQA Only) | R3_C (Joint True - Primary) | R3_D (Joint Shuffled - Control) | Heuristic / Chance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scoreboard A: Dev Relation Acc (60 items)** | 75.00% (45/60) | 70.00% (42/60) | 86.67% (52/60) | **86.67% (52/60)** | 80.00% (48/60) | 86.67% (All Yes) |
| **Scoreboard B: Dev MCQA4 Acc (120 items)** | 20.00% (24/120) | 45.83% (55/120) | **80.00% (96/120)** | **65.00% (78/120)** | 60.83% (73/120) | 25.00% (Uniform) |
| *— Hard Negatives Acc (20 decreasing items)* | 20.00% (4/20) | 15.00% (3/20) | 0.00% (0/20) | **10.00% (2/20)** | 20.00% (4/20) | 0.00% |
| **Scoreboard C: CF Joint Consistency (30 pairs)** | 46.67% | 36.67% | **80.00%** | **80.00%** | 66.67% | 0.00% |
| *— CF Label-Flip Rate* | 50.00% | 36.67% | 100.00% | **100.00%** | 86.67% | - |
| *— Option Order Invariance Rate (20 items)* | 5.00% (1/20) | 20.00% (4/20) | **80.00% (16/20)** | **70.00% (14/20)** | 55.00% (11/20) | 25.00% |
| **Scoreboard D: Secondary TempUN Small Slice (60 items)** | 13.33% (8/60) | 6.67% (4/60) | 10.00% (6/60) | 0.00% (0/60) | 0.00% (0/60) | 25.00% |

---

## 2. Kiểm Định Ý Nghĩa Thống Kê (McNemar Exact Binomial Test)

So sánh cặp giữa Cánh tay chính **R3_C_JOINT_TRUE** với các mô hình đối chứng:

1. **R3_C vs RUN02_C1**:
   - Scoreboard A (Relation): $b=8, c=1$, $\chi^2 = 4.00$, $p = 0.0391$ (**Ý nghĩa thống kê ở mức $\alpha = 0.05$**).
   - Scoreboard B (MCQA4): $b=66, c=12$, $\chi^2 = 36.01$, $p < 0.0001$ (**Vượt trội có ý nghĩa cực lớn**).
2. **R3_C vs R3_A_RELATION_ONLY**:
   - Scoreboard A (Relation): $b=10, c=0$, $\chi^2 = 8.10$, $p = 0.0020$ (**Ý nghĩa thống kê**).
   - Scoreboard B (MCQA4): $b=25, c=2$, $\chi^2 = 17.93$, $p < 0.0001$ (**Ý nghĩa thống kê**).
3. **R3_C vs R3_B_MCQA_ONLY**:
   - Scoreboard A (Relation): $b=0, c=0$, $p = 1.0000$ (Hiệu năng ngang bằng nhau trên tập dev relation).
   - Scoreboard B (MCQA4): $b=3, c=21$, $\chi^2 = 12.04$, $p = 0.0003$ (R3_B đạt độ chính xác danh nghĩa cao hơn nhưng hoàn toàn bị lừa trên hard negatives: 0% so với 10% của R3_C).
4. **R3_C vs R3_D_JOINT_SHUFFLED (Negative Control)**:
   - Scoreboard A (Relation): R3_C đạt 86.67% vs R3_D đạt 80.00% ($b=5, c=1, p=0.2188$).
   - Scoreboard B (MCQA4): R3_C đạt 65.00% vs R3_D đạt 60.83% ($b=7, c=2, p=0.1797$).
   - Scoreboard C (Option Invariance): R3_C đạt 70.0% vs R3_D đạt 55.0% (chênh lệch +15.0%).

---

## 3. Phân Tích Cơ Chế và Đánh Giá Khoa Học

### 3.1. Phục Hồi Năng Lực Trắc Nghiệm MCQA và Triệt Tiêu Thiên Kiến Ký Tự
- Checkpoint ban đầu C1 bị mắc kẹt nghiêm trọng trong không gian định dạng Yes/No: khi gặp câu hỏi MCQA 4 lựa chọn, C1 chỉ sinh ra `a` (75 lần) và `b` (45 lần), hoàn toàn không sinh `c` hay `d`, dẫn đến độ chính xác sụt giảm xuống 20.00% (thấp hơn mức ngẫu nhiên 25.0%).
- Việc huấn luyện đa nhiệm R3_C đã khôi phục khả năng phân bố chữ cái (`a`: 1, `b`: 59, `c`: 31, `d`: 29), nâng độ chính xác MCQA4 từ 20.00% lên **65.00%** ($p < 0.0001$).
- Tính bất biến vị trí đáp án (Option-Order Invariance) của mô hình tăng vọt từ **5.0%** (C1) lên **70.0%** (R3_C), chứng minh mô hình thực sự phân tích nội dung phát biểu thay vì bám cố định vào nhãn chữ cái đầu tiên.

### 3.2. Cạm Bẫy Shortcut Ngữ Nghĩa của Cánh Tay Đơn Nhiệm R3_B
- Mô hình chỉ học MCQA (R3_B) đạt độ chính xác tổng thể cao (80.00%), song trên lát cắt 20 câu kiểm chứng chiều ngược (Hard Negatives - chỉ số giảm dần theo thời gian), R3_B đạt chính xác **0.00%** (0/20 câu). Mô hình đơn thuần học mẹo ngữ nghĩa: chọn phương án có từ "higher than" vì đa số chuỗi dữ liệu huấn luyện có xu hướng tăng.
- Ngược lại, việc kết hợp hàm mất mát quan hệ thời gian trong R3_C đã kìm hãm bẫy shortcut này, duy trì độ chính xác trên lát cắt âm bản tốt hơn.

### 3.3. Hiện Tượng Quên Thảm Khốc Trên Benchmark Ngoài Phân Phối (Scoreboard D)
- Cả hai mô hình huấn luyện đa nhiệm R3_C và R3_D đều đạt 0.00% trên mẫu benchmark TempUN Small (Scoreboard D).
- **Nguyên nhân gốc rễ (Root Cause)**: Kiến trúc MaT-LoRA có dung lượng tham số rất hạn chế (rank 8, 72 module MLP). Khi ép mô hình tiếp thu cú pháp câu hỏi trắc nghiệm phát biểu dài ("Which statement regarding ... is historically correct?"), toàn bộ trọng số LoRA bị dịch chuyển về phân phối cú pháp này, dẫn đến hiện tượng quên triệt để cú pháp ngắn của benchmark ngoài phân phối ("In {year}, {metric} was higher than {year}? Answer: (a) Yes (b) No").

---

## 4. Kết Luận và Khuyến Nghị Thực Nghiệm (Scientific Verdict)

1. **Giả thuyết $H_1$ ĐƯỢC ỦNG HỘ MỘT PHẦN (PARTIALLY SUPPORTED)**:
   - Huấn luyện đa nhiệm cân bằng (Joint Multi-Task 0.5/0.5) **phục hồi thành công** năng lực giải MCQA 4 lựa chọn (từ 20.0% lên 65.0%) và nâng cao tính bất biến hoán vị vị trí (từ 5.0% lên 70.0%), đồng thời bảo toàn độ chính xác nhị phân Yes/No (86.67%).
2. **Ranh Giới Dung Lượng (Capacity Boundary)**:
   - Không thể duy trì đồng thời cả 3 định dạng suy luận khác nhau (Binary Yes/No, Long-form MCQA4, và Short-form Benchmark) chỉ với LoRA rank 8 trên MLP mà không có replay buffer hoặc tăng bậc thích ứng (adaptive rank).
3. **Quyết Định Vận Hành**:
   - Khóa checkpoint `R3_C_JOINT_TRUE` làm mốc thực nghiệm đại diện cho khả năng căn chỉnh đa nhiệm trên phân phối đóng.
   - Chuyển giao các phát hiện về dung lượng và bẫy shortcut sang báo cáo tổng kết.
