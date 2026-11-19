# Đối Soát Độ Chuẩn Xác Kỹ Thuật: Bài Báo TickTack Gốc (arXiv:2503.04150v3) vs Triển Khai Thực Nghiệm

**Mã đối soát:** `PAPER_EQ_VS_IMPLEMENTATION_FIDELITY`  
**Ngày lập:** 2026-10-15  
**Tác giả:** Antigravity (Principal Research Engineer & Scientific Auditor)  
**Bài báo quy chiếu:** Xue Han et al., *"Temporal Alignment of LLMs through Cycle Encoding for Long-Range Time Representations"*, arXiv:2503.04150v3 (v3 cố định).

---

## 1. Bảng Đối Chiếu Chuẩn Tắc Từng Phương Trình (Equation-Level Truth Table)

| Mục trong bài báo | Công thức / Nội dung bài báo công bố (`paper_says`) | Triển khai của Đối tác (`partner_implementation`) | Triển khai của Learning Track (`project_design`) | Phân loại trạng thái |
| :--- | :--- | :--- | :--- | :--- |
| **Eq. 1 (Chu kỳ Can Chi)** | Năm CN $\ge 4$: $t_{\text{cycle}} = (y - 3) \pmod{60}$. 1984 $\to$ 1 (Giáp Tý). 1983 $\to$ 0 (Quý Hợi). Đánh chỉ số 1-based. | `if year >= 4: return (year - 3) % 60` (sao chép công thức Eq. 1 của bài báo). | `offset = year - 4; cycle_idx = offset % 60`. 1984 $\to$ 0 (Giáp Tý). Đánh chỉ số 0-based chuẩn lập trình. | `PAPER_CONVENTION_DIFFERENCE` (Hai quy ước đánh số 1-based vs 0-based lệch pha đúng 1 đơn vị $\equiv 6^\circ$, hoàn toàn không phải bug toán học). |
| **Eq. 2 (Bán kính cực)** | $r_{\text{AD}} = \alpha_{\text{AD}} + \beta_{\text{AD}} \theta_{\text{cycle}}$ với $\alpha, \beta \in [0.5, 1.0]$. | Lấy $\alpha = 0.5, \beta = 0.5 \implies r = 0.5 + 0.5 \theta$. Bị trùng lặp bán kính giữa các năm cách nhau 60 năm ($Y$ và $Y+60k$). | Biến thiên đơn điệu theo đại kỷ nguyên: $r = 1.0 + (y - t_0)/60.0$ ($t_0=1900$), phân tách tuyệt đối các chu kỳ lịch sử. | `PAPER_AMBIGUOUS` (Công thức Eq. 2 của bài báo làm trùng bán kính hai năm cùng pha; Learning Track thiết kế đơn điệu tránh alias). |
| **Eq. 3 (Tọa độ Descartes)** | $x = r \cos(\theta), y = r \sin(\theta)$ trên mặt phẳng cực 2D. | Tính $x = r \cos(\theta), y = r \sin(\theta)$ theo Eq. 2. | Tính $x, y$ trên bán kính đơn điệu đại kỷ nguyên kết hợp 8D Can-Chi. | `METHODOLOGICAL_DEVIATION` (Learning Track mở rộng thành 40D FoNE-CanChi). |
| **Eq. 4 (Mã hóa Sinusoidal TE)** | $\mathrm{TE}(t) = \mathrm{TE}(x) + \mathrm{TE}(y)$ qua các hàm $\sin, \cos$ đa tần số logarit. | Tính $\mathrm{TE}(x) + \mathrm{TE}(y) \in \mathbb{R}^{d}$ ($d=896$). Chuẩn Euclidean $\|\mathrm{TE}\|_2 = \sqrt{448} \approx 21.166$. | Biểu diễn 40D bất biến `FIXED_FONE_CANCHI_V1` (FoNE 32D + CanChi 8D), chuẩn hóa biên độ có kiểm soát. | `METHODOLOGICAL_DEVIATION` |
| **Eq. 5 (§3.2 Tiêm thời gian)** | *"add the temporal embedding $\mathrm{TE}(t)$ to the input word embedding at the token corresponding to the year $t$"*. | `embed_tokens(input_ids) + te` (Phép cộng broadcasting cộng vector 21.166 vào **TOÀN BỘ CÁC TOKEN** trong câu). | Tiêm tại **token index 14** cố định qua cổng số dư liên tục `GatedResidualProjection(40 -> 896)` với $\alpha_0 = 0.0$. | `REPRODUCED_SOURCE_BUG` ở bản đối tác (broadcasting tràn toàn câu gây Cosine Collapse) vs `PROJECT_SEAM_CONTRACT` ở Learning Track. |
| **§4.2 (LoRA Backbone)** | *"incorporating trainable low-rank adaptation (LoRA) matrices into frozen LLMs"*. Không hề nêu tên module `q_proj, v_proj` hay $r, \alpha$. | Chọn `target_modules=['q_proj', 'v_proj']`, rank $r=8$, alpha $\alpha=16.0$. | Chọn Dual-Cycle MaT-LoRA trên 72 module MLP (`gate_proj`, `up_proj`, `down_proj`, $r=8, \alpha=16, K=2$). | `IMPLEMENTATION_ASSUMPTION` (Bài báo không chỉ định cụ thể module; cả hai cấu hình đều là giả định vận hành hợp lý). |
| **Table 1 (Mô hình đánh giá)** | Thử nghiệm trên **Qwen2.5-3B, Qwen2.5-7B, LLaMA2-7B, LLaMA2-13B**. | Thử nghiệm trên **Qwen2.5-0.5B-Instruct**. Báo cáo $26.67\%, 29.07\%, 22.41\%, 28.89\%$. | Thử nghiệm trên **Qwen2.5-0.5B-Instruct**. Khóa bất biến backbone $100\%$ đóng băng. | `EXTERNAL_PARTNER_REPORTED` (Bài báo gốc **hoàn toàn không có số liệu cho Qwen2.5-0.5B**). |
| **Table 1 (§4.2 Giao thức eval)** | Đánh giá Zero-shot và 5-shot trên TempLAMA, TempUN. Không công bố prompt template chi tiết hay regex parser. | Sử dụng **Academic Choice Logit Scoring** (so sánh logit trực tiếp các token phương án ngay sau `Answer:`). Invalid $= 0.0\%$. | Sử dụng **P12_TASK_AWARE_CHATML** với **RAW_GREEDY_V1** (sinh tự do toàn từ vựng 151.936 tokens + parser nghiêm ngặt). | `EVALUATION_CONTRACT_SHIFT` (Hai giao thức đo lường trên hai không gian toán học khác biệt hoàn toàn). |
| **§4.2 (Chế độ huấn luyện)** | Huấn luyện chung toàn bộ dữ liệu (Joint Post-training), batch 8, grad accum 2, 10 epochs. | Huấn luyện chung (Joint) 2 epochs trên 2.461 mẫu gộp. | Huấn luyện chặng tiếp nối (Continual) 1947 $\to$ 1984 $\to$ 2022 để kiểm tra suy giảm tri thức (BWT). | `RESEARCH_OBJECTIVE_DIFFERENCE` (Mục tiêu của Learning Track là nghiên cứu Continual Alignment, không phải tái lập Joint NTP). |

---

## 2. Bóc Tách Bản Chất Kỹ Thuật Của Các Vấn Đề Tranh Chấp

### 2.1. Quy ước `year - 3` vs `year - 4`: Lệch pha góc quay 1-based vs 0-based
- Bài báo sử dụng công thức văn hóa truyền thống phương Đông coi năm 4 CN là năm thứ nhất (1-based index) của chu kỳ:
  $$(1984 - 3) \pmod{60} = 1981 \pmod{60} = 1 \quad (\text{Giáp Tý là lớp 1})$$
  $$(1983 - 3) \pmod{60} = 1980 \pmod{60} = 0 \quad (\text{Quý Hợi là lớp 60, modulo 60 trả về 0})$$
- Dự án `long-range-temporal-alignment` sử dụng quy ước mảng số học trong khoa học máy tính (0-based index):
  $$(1984 - 4) \pmod{60} = 1980 \pmod{60} = 0 \quad (\text{Giáp Tý là index 0})$$
  $$(1985 - 4) \pmod{60} = 1981 \pmod{60} = 1 \quad (\text{Ất Sửu là index 1})$$
  $$(1983 - 4) \pmod{60} = 1979 \pmod{60} = 59 \quad (\text{Quý Hợi là index 59})$$
- **Phán quyết:** Hai quy ước đồng dư hoàn hảo: $\text{cycle}_{\text{1-based}} \equiv \text{cycle}_{\text{0-based}} + 1 \pmod{60}$. Phép toán `(year - 4) % 60` áp dụng trên số nguyên **không bao giờ trả về `None`**. Việc quy kết `year - 4` gây lỗi `None` là nhận định ngụy biện kỹ thuật do không phân biệt được lỗi thiếu dữ liệu đầu vào với phép toán đồng dư.

### 2.2. Nguồn gốc thật sự của giá trị thiếu `None`
- Trong tệp gốc `Data/tempun_small/canchi_eval_mcqa.jsonl.gz`, trường `canonical_year` bị rỗng (`null`) tại **$144/180$ mẫu ($80.0\%$)** (chỉ có 36 câu Date-Based có năm; 72 câu Min-Max và 72 câu Relative-Based không có trường năm).
- Trước đây, executor cũ áp dụng fallback nguy hại `cond.get('effective_year') or item.get('canonical_year') or 1984`, dẫn đến việc tự động gán năm 1984 cho các câu không có năm.
- **Phán quyết:** Lỗi thuộc về khâu trích xuất siêu dữ liệu thời gian (Metadata Extraction Defect), hoàn toàn không liên quan đến phép trừ 3 hay trừ 4.

### 2.3. Hiện tượng "Cosine Collapse" và sai lầm tiêm Broadcast
- Trong script của đối tác (`build_kaggle_pure_ticktack_baseline.py`, dòng 323):
  `inputs_embeds = model.model.embed_tokens(inputs['input_ids']) + te`
- Vector $\mathrm{TE}(t)$ có chuẩn $\|\mathrm{TE}\|_2 = \sqrt{448} \approx 21.166$, trong khi vector embedding từ vựng có chuẩn $\approx 1.0$.
- Phép cộng trên đã broadcast vector 21.166 vào **tất cả mọi token trong câu hỏi**, khiến mọi từ ngữ đều bị chi phối bởi vector thời gian, dẫn đến độ tương đồng Cosine giữa mọi token tiến tới $0.98 \sim 0.99$.
- Bài báo gốc tại §3.2 nói rõ: chỉ cộng $\mathrm{TE}(t)$ tại token biểu diễn năm $t$. Đối tác đã cài đặt sai phép tiêm thành tiêm toàn câu, dẫn đến hiện tượng suy biến góc mà đối tác tự gọi là "Cosine Collapse".
