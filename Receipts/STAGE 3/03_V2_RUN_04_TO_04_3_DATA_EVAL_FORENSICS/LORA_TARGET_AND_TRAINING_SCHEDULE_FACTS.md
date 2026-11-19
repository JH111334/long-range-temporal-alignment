# Sự Thật Kỹ Thuật: Vị Trí Module LoRA và Cơ Chế Huấn Luyện (Joint vs Continual)

**Mã tài liệu:** `LORA_TARGET_AND_TRAINING_SCHEDULE_FACTS`  
**Ngày lập:** 2026-10-15  
**Tác giả:** Antigravity (Principal Research Engineer & Scientific Auditor)  
**Nhánh Git:** `experiment/v2-run04-2-partner-paper-fidelity-audit`

---

## 1. Vị Trí Module Thích Ứng LoRA: Văn Bản Bài Báo vs Triển Khai Thực Nghiệm

### 1.1. Văn bản bài báo gốc TickTack công bố những gì?
Trích dẫn nguyên văn từ Mục §4.2 (Training Settings, trang 5) của bài báo *Xue Han et al. (arXiv:2503.04150v3)*:
> *"We incorporate trainable rank-decomposition matrices into each layer of the Transformer architecture, keeping the pre-trained model parameters frozen while only optimizing the LoRA parameters (Hu et al., 2021)."*

**Sự thật kiểm chứng:**
1. Bài báo **hoàn toàn không nêu tên cụ thể các module tuyến tính** được can thiệp (không hề có từ khóa `q_proj`, `v_proj`, `k_proj`, `o_proj`, hay `mlp` trong phần cấu hình huấn luyện).
2. Bài báo **không công bố hạng LoRA ($r$) hay hệ số co tỉ lệ ($\alpha$)**.
3. Cấu hình `target_modules = ['q_proj', 'v_proj']` là một **quy ước mặc định phổ biến của thư viện Hugging Face PEFT**, hoàn toàn là giả định vận hành của người triển khai (`IMPLEMENTATION_ASSUMPTION`), không phải sự thật hiển nhiên từ bài báo gốc.

### 1.2. So sánh vị trí module giữa hai nhánh nghiên cứu
- **Nhánh đối tác (TickTack-inspired Rebuild)**:
  - Vị trí can thiệp: `q_proj` và `v_proj` trên 24 tầng Transformer ($2 \times 24 = 48$ module).
  - Hạng: $r = 8$, hệ số: $\alpha = 16.0$.
  - Bản chất: Thích ứng cơ chế chú ý (Self-Attention mechanism).
- **Nhánh Learning Track (M0-FC40 / V2)**:
  - Vị trí can thiệp: 72 module MLP (`gate_proj`, `up_proj`, `down_proj` trên 24 tầng Transformer).
  - Kiến trúc: Periodic Dual-Cycle MaT-LoRA tích hợp chu kỳ Can-Chi động vào trọng số $\Delta W(\theta, z)$.
  - Bản chất: Can thiệp vào kho lưu trữ tri thức tham số (Parametric memory / FFN layers).

**Kết luận khoa học:** Cả hai cấu hình đều là những lựa chọn kiến trúc hợp lệ xuất phát từ văn bản mở của bài báo gốc. Không bên nào có quyền tuyên bố cấu hình của mình là "chuẩn tắc duy nhất của bài báo gốc".

---

## 2. Cơ Chế Huấn Luyện: Đa Nhiệm Gộp (Joint NTP) vs Chặng Thời Gian Tiếp Nối (Continual Learning)

### 2.1. Bản chất sự khác biệt giữa hai chế độ học
- **Chế độ Joint Post-training của Đối tác**:
  - Toàn bộ $2.461$ mẫu huấn luyện của cả 3 mốc thời gian ($1947, 1984, 2022$) được đưa vào một bể dữ liệu duy nhất và xáo trộn ngẫu nhiên (single i.i.d. shuffle pool).
  - Huấn luyện 2 epochs với hàm mất mát liên hợp $\mathcal{L}_{\text{NTP}} + 0.1 \mathcal{L}_{\text{contrastive}}$.
  - Mục tiêu: Tối đa hóa điểm số trên tập dữ liệu đóng mà không quan tâm đến thứ tự xuất hiện của dòng thời gian.
- **Chế độ Continual Pre-training của Learning Track (M0-FC40)**:
  - Dữ liệu được đưa vào tuần tự theo trật tự lịch sử: Chặng 1 ($1947$) $\to$ Chặng 2 ($1984$) $\to$ Chặng 3 ($2022$).
  - Mục tiêu khoa học: Kiểm tra khả năng **học tăng dần không quên tri thức cũ** (Backward Transfer - BWT) và chống suy thoái khả năng tổng quát (Anti-Alignment Tax).

### 2.2. Phán quyết so sánh phương pháp (Methodological Comparability)
- Việc đối tác huấn luyện Joint đạt $29.07\%$ (trên giao thức Choice Logit) **không phải là bằng chứng cho thấy mô hình Continual M0-FC40 bị lỗi**.
- Hai chế độ này phục vụ hai câu hỏi nghiên cứu hoàn toàn khác nhau:
  1. *Joint Offline Training*: Nghiên cứu biểu diễn thời gian khi có sẵn toàn bộ dữ liệu lịch sử cùng một lúc.
  2. *Continual Temporal Alignment*: Nghiên cứu khả năng LLM tiếp nhận mốc thời gian mới mà không làm hỏng tri thức các thời kỳ trước đó.
- Việc so sánh điểm số trực tiếp giữa một mô hình học Joint gộp toàn bộ dữ liệu với một mô hình học Continual tuần tự là một **phép so sánh lệch biến số cơ bản (confounded comparison)** nếu không kiểm soát chặt chẽ ngân sách tiếp xúc token và mục tiêu đánh giá.
