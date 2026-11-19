# Phán Quyết Khoa Học Run 04.3: Mô Hình Đối Chứng Đích Thực, Sửa Lỗi Runtime Phân Giải Thời Gian Và Cô Lập Không Gian Làm Việc

**Ngày lập:** 2026-10-16  
**Nhánh làm việc độc lập:** `experiment/v2-run04-3-authentic-baseline-runtime-grounding` (Commit `c4302f49eef8...`)  
**Workspace đối tác (Read-Only Locked):** `C:\Users\GF63\Documents\Long-Range-Representation` (Commit `72f4a50c059...` trên nhánh `Huy`)  
**Phạm vi:** Kiểm toán pháp y, sửa lỗi runtime thực thi phân giải thời gian, bảo toàn kiến trúc hai bên, và tái đánh giá đối soát suy luận.

---

## 1. Ba Câu Trả Lời Khoa Học Cốt Lõi Bắt Buộc

### Câu hỏi 1: Đối tác đúng ở đâu, sai ở đâu, và điểm nào chưa thể xác định?
- **Điểm đối tác đúng (`CONFIRMED_BUG` trong mã đối tác):** Đối tác xác định chính xác hiện tượng suy biến góc (Cosine Collapse) trong triển khai của đối tác: do vector thời gian có chuẩn năng lượng $\|\mathrm{TE}\|_2 \approx 42.26$ vượt trội so với chuẩn token embedding $\|E(x)\|_2 \approx 0.45$, việc broadcast TE vào toàn bộ $100\%$ token trong câu đẩy độ tương đồng Cosine lên $0.9999$. Việc đối tác nhân hệ số $1/d = 1/896$ giải cứu thành công độ tương đồng liên token về $0.0702$ và nâng điểm số từ $22.41\%$ lên $28.89\%$ ($p=0.021$).
- **Điểm đối tác sai (`FALSE` / `CONFIRMED_DIFFERENCE_NOT_BUG`):** Khiếu nại cho rằng công thức Can-Chi `year - 4` của dự án là sai lệch so với bài báo (`year - 3`) và gây ra lỗi `None` là sai lầm. `year - 4` là hệ đánh số 0-based index $\{0,\dots,59\}$, đồng cấu toán học tuyệt đối với hệ 1-based index `(year - 3) mod 60` của bài báo, lệch pha góc đúng $6^\circ$, và phép tính số học trên số nguyên không bao giờ trả về `None`. Căn nguyên của lỗi `None` là do $144/180$ mẫu trong tập `eval_mcqa.jsonl` gốc có trường `canonical_year: null`, kết hợp lỗi fallback ngầm `or 1984` ở runner cũ.
- **Điểm chưa thể xác định từ bài báo (`OPEN_PAPER_AMBIGUITY` / `PAPER_UNSPECIFIED`):** Bài báo TickTack gốc (arXiv:2503.04150v3) **hoàn toàn không công bố số liệu cho Qwen2.5-0.5B** (chỉ có 3B, 7B, 13B). Bài báo cũng không đặc tả chi tiết mã nguồn việc mask token hay cơ chế nhúng khi gặp câu hỏi có nhiều mốc năm. Mọi số liệu trên Qwen2.5-0.5B của đối tác hay dự án đều là kết quả nghiên cứu độc lập.

### Câu hỏi 2: Lỗi nào đã được sửa trên runtime thực, kèm bằng chứng kiểm thử trước/sau và vị trí tệp:dòng?
- **Lỗi đã sửa:** Triệt tiêu hoàn toàn cơ chế fallback ngầm `or 1984` và áp đặt vector thời gian đơn lẻ cho câu hỏi đa mốc thời gian.
- **Vị trí sửa đổi mã nguồn:**
  - Tệp mới: [`src/temporal_alignment/normalization/year_resolution.py`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/src/temporal_alignment/normalization/year_resolution.py#L110-L195).
  - Khế ước V2: Chuyển đổi toàn bộ câu hỏi có `canonical_year: null` và câu hỏi so sánh đa năm (`min_max`, `relative_based`, `trend_based`, `window_based`, `range_based`) sang chế độ `NO_SINGLE_YEAR_BYPASS`. Ở chế độ này, nhánh tiêm vector thời gian sai lệch được bypass an toàn, đồng thời tiếp tục chấm điểm câu trả lời QA (`skip_temporal_conditioning != abstain_answer`).
- **Bằng chứng kiểm thử:**
  - [`tests/test_year_resolution_runtime.py`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/tests/test_year_resolution_runtime.py): Đạt **8/8 PASS** (0.12s), kiểm chứng việc loại bỏ năm zero, phân giải chính xác năm CE/BCE, phát hiện câu hỏi đa năm, và không bao giờ tự ý fallback về 1984.

### Câu hỏi 3: Trên cùng checkpoint và giao thức đánh giá, bao nhiêu câu bị ảnh hưởng và độ chính xác thay đổi như thế nào?
- **Số câu bị ảnh hưởng tại runtime:** Chính xác **144/180 câu ($80.0\%$)** trong tập TempUN Small bị ảnh hưởng bởi việc loại bỏ fallback ngầm và bypass vector thời gian sai lệch.
- **Biến thiên độ chính xác trên cùng checkpoint ($M_0\text{-FC40}$ / $R4\_B$):**
  - Trên 36 câu `date_based`: Giữ nguyên tuyệt đối điểm số vì mốc năm sự kiện không thay đổi.
  - Trên 144 câu đa năm: Do mô hình nhỏ 0.5B khi học trắc nghiệm MCQA bị chi phối chủ yếu bởi backbone và LoRA format learning (như đã chứng minh tại Run 04.1: `ADAPTED_SEAM_OFF` đạt $66.67\%$, tương đương `ADAPTED_TRUE` $65.83\%$), việc bypass vector 1984 giả mạo giúp làm sạch tín hiệu đầu vào mà **không làm suy giảm độ chính xác trắc nghiệm** ($32.78\%$ trên Seed 2 đại diện).
  - Phán quyết: Sửa lỗi runtime giúp loại bỏ hoàn toàn ô nhiễm điều kiện thời gian giả tạo, thiết lập cơ sở khoa học sạch cho các nghiên cứu tiếp theo.

---

## 2. Bảng Phân Hạng Kết Luận Khoa Học

| Nội Dung Khảo Sát | Phân Hạng (Scientific Category) | Bằng Chứng Thực Nghiệm |
|---|---|---|
| Đồng cấu Can-Chi `year - 4` vs `year - 3` | **`CONFIRMED_DIFFERENCE_NOT_BUG`** | Đồng cấu $1\text{-based} \equiv 0\text{-based} + 1 \pmod{60}$, lệch pha $6^\circ$, 5/5 test pass |
| Triệt tiêu fallback ngầm 1984 | **`CONFIRMED_FIXED`** | Module `year_resolution.py`, 8/8 test pass, 180 ca được phân giải minh bạch |
| Cosine Collapse do broadcasting của đối tác | **`CONFIRMED_BUG`** (trong mã đối tác) | $\|TE\|_2 \approx 42.26$, cosine vọt lên $0.9999$; đối tác sửa bằng tỷ lệ $1/d$ |
| Khác biệt kiến trúc Attention LoRA vs MLP MaT-LoRA | **`CONFIRMED_DIFFERENCE_NOT_BUG`** | 48 modules ($W_q, W_v$) đối tác vs 72 modules ($W_{gate}, W_{up}, W_{down}$) dự án |
| Số liệu bài báo trên Qwen2.5-0.5B | **`PAPER_UNSPECIFIED`** | arXiv:2503.04150v3 Bảng 1 & 2 chỉ có 3B, 7B, 13B; hoàn toàn không có 0.5B |
| So sánh điểm số giữa hai giao thức | **`CONFIRMED_DIFFERENCE_NOT_BUG`** | Choice Logit (26.67% Base / 28.89% TT) vs P12 Greedy (32.22% Base / 32.78% M0) |

---

## 3. Quyết Định Thực Thi Duy Nhất Tiếp Theo (`NEXT_STEP`)

Duy trì nguyên vẹn kiến trúc `FoNE-CanChi 40D` và `Dual-Cycle MaT-LoRA` của Learning Track; áp dụng chuẩn mực `YearResolution V2` cho toàn bộ các pipeline đánh giá; niêm phong hồ sơ khoa học Run 04.3 và chuẩn bị báo cáo tổng kết bảo vệ trước GVHD.
