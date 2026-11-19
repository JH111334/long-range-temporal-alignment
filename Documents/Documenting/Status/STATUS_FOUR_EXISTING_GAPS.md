---
document_class: "OFFICIAL_DERIVED"
authority: "NON_CANONICAL"
retention: "LONG_TERM"
status: "VALIDATED"
---

# Hiện Trạng: 4 Yếu Tố Khoảng Cách Kỹ Thuật Giữa Thực Nghiệm Hiện Tại và Bài Báo Gốc Ticktack

**Vị trí lưu trữ**: `Documents/Documenting/Status/STATUS_FOUR_EXISTING_GAPS.md`  
**Dự án**: `long-range-temporal-alignment`  
**Thời điểm ghi nhận**: 2026-10-23  
**Phân loại tài liệu**: Trạng thái hiện hữu hệ thống (System Status & Gap Ledger), phân biệt rạch ròi với lỗi cục bộ (Errors) hay bài học thiết kế (Insights).  
**Tài liệu tham chiếu**: Bài báo Ticktack (Han et al., [arXiv:2503.04150v3](https://arxiv.org/html/2503.04150v3)), Mã nguồn thực nghiệm đối tác (`scratch/partner_repo`), Sổ cái sai lệch [`STAGE03_3B_DISCREPANCY_REGISTER.jsonl`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Receipts/STAGE%203/10_QWEN25_3B_FORENSIC_AUDIT/STAGE03_3B_DISCREPANCY_REGISTER.jsonl).

---

## 1. Bối Cảnh và Mục Đích

Tài liệu này ghi nhận tường minh 4 khoảng cách kỹ thuật mang tính cấu trúc đang hiện hữu giữa thiết lập thực nghiệm của dự án (và nhánh đối tác) so với các điều kiện chuẩn tắc đã được công bố trong bài báo gốc Ticktack. Việc không đáp ứng đầy đủ 4 yếu tố này lý giải tại sao mô hình đạt độ chính xác thấp (~28% so với 64.96% trong bài báo), đồng thời xác lập ranh giới các vấn đề cần giải quyết trước khi tiến hành bất kỳ can thiệp huấn luyện quy mô lớn nào.

---

## 2. Bốn Vấn Đề Kỹ Thuật Đang Hiện Hữu

### Vấn đề 1: Phân kỳ lựa chọn mô hình nền (Model Choice Divergence: Base vs Instruct)
- **Thiết lập bài báo gốc**: Sử dụng mô hình nền thuần túy `Qwen/Qwen2.5-3B` (Base Model). Mô hình này được huấn luyện tối ưu cho hàm mất mát dự đoán từ tiếp theo (Next Token Prediction), khớp hoàn toàn với mẫu lời nhắc tiếp nối chuỗi dạng `Question: ...\nAnswer: [Token]`.
- **Thiết lập hiện tại**: Sử dụng `Qwen/Qwen2.5-3B-Instruct`. Đây là mô hình đã qua tinh chỉnh chỉ dẫn (SFT) và căn chỉnh hành vi (RLHF/DPO) để hoạt động như một trợ lý hội thoại.
- **Tác động hiện hữu**: Khi mô hình Instruct tiếp nhận văn bản trần không có cấu trúc ChatML (`<|im_start|>user...`), nó rơi vào trạng thái lệch pha phân phối nghiêm trọng. Trọng số căn chỉnh hành vi khuếch đại thiên lệch logit tiên nghiệm tự nhiên (token `'no'` chiếm ưu thế tuyệt đối trước `'yes'`), dẫn đến sụp đổ toàn bộ bài toán quan hệ thời gian Rel ($852/852$ câu đoán `'No'`).

### Vấn đề 2: Thiếu hụt nghiêm trọng ngân sách tính toán (Compute & Training Epoch Deficit)
- **Thiết lập bài báo gốc**: Huấn luyện đầy đủ **10 epochs** trên cụm phân tán **$8 \times \text{NVIDIA A100}$** (80GB VRAM), với bộ tối ưu hóa Cosine Annealing, tốc độ học $\eta = 10^{-4}$ và giai đoạn khởi động (warmup) ổn định.
- **Thiết lập hiện tại**: Huấn luyện **1 epoch duy nhất** trên **1 GPU NVIDIA Tesla T4** (14.56GB VRAM), tốc độ học lớn $\eta = 2 \times 10^{-4}$ không qua warmup đầy đủ.
- **Tác động hiện hữu**: Cắt giảm hơn $90\%$ số bước tối ưu hóa khiến các ma trận thích ứng LoRA ($q\_proj, v\_proj$) hoàn toàn chưa hội tụ. Mô hình chưa kịp học cách dịch chuyển phân phối logit để vượt qua các thiên lệch tiên nghiệm của mô hình nền, khiến hiệu năng đo đạc không khác biệt so với Base model (McNemar $p = 0.2418$).

### Vấn đề 3: Sự thưa thớt cặp dương trong hàm mất mát tương phản nội chu kỳ (Intra-loss Pair Sparsity)
- **Thiết lập bài báo gốc**: Triển khai trên cụm máy lớn với kích thước batch hiệu dụng cao ($B \ge 64$), bảo đảm trong mỗi mẻ dữ liệu luôn tồn tại nhiều cặp mẫu cùng năm hoặc cùng chu kỳ Can Chi ($c_i = c_j$).
- **Thiết lập hiện tại**: Do giới hạn 16GB VRAM của Tesla T4, kích thước micro-batch bị giới hạn ở $B = 8$.
- **Tác động hiện hữu**: Với 60 chu kỳ Can Chi phân bố đều trên tập dữ liệu, xác suất toán học để một micro-batch $B = 8$ không chứa bất kỳ cặp mẫu nào trùng chu kỳ là:
  $$P(\text{không trùng}) = \prod_{k=0}^{7} \left(1 - \frac{k}{60}\right) \approx 60.52\%$$
  Hệ quả là hơn **$60.5\%$ số bước cập nhật tham số hoàn toàn không có gradient tương phản thời gian** ($\mathcal{L}_{\text{intra}} = 0.0$), làm vô hiệu hóa phần lớn động lực học của cơ chế biểu diễn thời gian.

### Vấn đề 4: Phân kỳ giao thức đánh giá khử thiên lệch (Evaluation Protocol Divergence)
- **Thiết lập bài báo gốc**: Đánh giá đơn lượt trên thứ tự phương án ban đầu (Single-pass canonical order), không áp dụng kỹ thuật xoay vòng phương án trắc nghiệm. Phương thức xếp hạng logit đóng (Choice Logit Ranking) mặc nhiên triệt tiêu lỗi cú pháp (`invalid_rate = 0.0%`).
- **Thiết lập hiện tại**: Áp dụng kiểm tra nghiêm ngặt bằng **4 vòng xoay hoán vị phương án (4-fold cyclic rotation debiasing)**.
- **Tác động hiện hữu**: Giao thức xoay vòng trừng phạt triệt để hành vi thiên lệch vị trí chữ cái. Kết quả kiểm toán bóc tách rằng $86.55\%$ số câu hỏi trắc nghiệm ($3{,}843 / 4{,}440$) chỉ chọn cố định một vị trí chữ cái (điểm debiased $= 0.25$), và độ nhất quán ngữ nghĩa thực tế chỉ đạt $0.16\%$. Điểm số thấp ở local không phải do mô hình bị lỗi cú pháp, mà do giao thức đánh giá nghiêm ngặt đã bóc trần sự phụ thuộc vào phím tắt vị trí của mô hình chưa hội tụ.

---

## 3. Ranh Giới Quản Trị Hệ Thống

Bốn vấn đề trên được phân loại là **Khoảng cách thiết lập hệ thống hiện hữu (Systemic Setup Gaps)**. Chúng không cấu thành lỗi thực thi (Execution Bug) trong mã chạy, mà phản ánh sự thiếu hụt tài nguyên tính toán và sai lệch mô hình nền khi chuyển đổi thiết kế bài báo sang môi trường thử nghiệm hạn chế. Mọi đề xuất cải tiến trong Stage 03 phải trực tiếp giải quyết 4 khoảng cách này trước khi xem xét các kỹ thuật nâng cao.
