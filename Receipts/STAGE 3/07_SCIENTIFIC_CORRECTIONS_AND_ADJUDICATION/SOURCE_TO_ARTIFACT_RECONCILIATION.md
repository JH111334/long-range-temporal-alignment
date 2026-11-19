# Báo Cáo Đối Soát Pháp Y Nguồn Dữ Liệu và Tạo Tác Thực Nghiệm (Run 04 -> Run 04.1)

Tài liệu này ghi nhận quá trình kiểm toán độc lập, truy nguyên nguồn gốc nhị phân và đối soát từng dòng kết quả giữa mã nguồn thực thi, tệp tạo tác đầu ra và các khẳng định khoa học giữa đợt thử nghiệm Run 04 và bản hoàn thiện Run 04.1.

---

## 1. Bản Đồ Truy Nguyên Nguồn Gốc và Tính Bất Biến Thực Thi

| Thành Phần | Danh Tính Chuẩn Tắc (Run 04.1) | Mã Băm SHA256 / Phiên Bản | Ghi Chú Đối Soát Pháp Y |
| :--- | :--- | :--- | :--- |
| **Mã Nguồn Kernel Thực Thi** | `scripts/kaggle_v2_run04_executor.py` | `94d69a3c38eca6c66fcd22939b73c1bd0e563388db30b487aefe0574c0071fee` | Trùng khớp từng byte với `v2_run04_app/executor.py` đóng gói trong payload gửi lên Kaggle. |
| **Mã Nguồn Đánh Giá Cục Bộ** | `scripts/evaluate_v2_run04_clean.py` | `1101d961a1a9c018ede5a4af1a6fdb405f3d526a457ebd86074b7b5c71833f77` | Script tạo ra các tệp kết quả JSON cục bộ trong thư mục `run_04/`. |
| **Mô Hình Nền (Backbone)** | `Qwen/Qwen2.5-0.5B-Instruct` | Commit: `7ae557604adf67be50417f59c2c2f167def9a775` | Khóa cứng 100% tham số (`requires_grad = False`). Đã kiểm tra logit parity. |
| **Biểu Diễn Thời Gian** | `FIXED_FONE_CANCHI_V1` (40D) | Golden Manifest SHA: `6c35fc55cbdfd571...` | 32D FoNE số học + 8D Can-Chi vi chu kỳ. Tiêm tại prefill index 14. |
| **Checkpoint Xuất Phát** | `C1_COMPACT_ADAPTER_ONLY.pt` | File SHA: `30cadb35504e7f15...` / Tensor SHA: `322682a8b435...` | Đã đính chính mã băm tệp trên đĩa thay cho chuỗi gõ nhầm `7737229...` trong receipt cũ. |
| **Tập Dữ Liệu Train Sạch** | `MCQA4_TRAIN_CLEAN.jsonl` (400) | `dbbc9a9bbcf1567ba05342a34fb4697c11f714eb617672b1d3d6ba2697858d4a` | 100% `oracle_truth_count == 1`. 200 tăng / 200 giảm. 100 mỗi chữ cái `a,b,c,d`. |
| **Tập Dữ Liệu Dev Sạch** | `MCQA4_DEV_CLEAN.jsonl` (120) | `f041ff9cf80ba9866ff3c1143c121404c0528489437b7ca3b06323cf106b51c1` | 100% `oracle_truth_count == 1`. 60 tăng / 60 giảm. 30 mỗi chữ cái `a,b,c,d`. |
| **Tập Quan Hệ Dev (S2)** | `dev_relation_pairs.jsonl` (60) | `85f09623fa62fc86fa323fca8ebef53d4ebda9fc2c4bbfbc347209772594a11b` | 60 câu Yes/No phân bổ cân bằng 29 Yes / 31 No. |
| **Tập Benchmark TempUN (S4)** | `eval_mcqa.jsonl` (180) | `f0ca8db323d4677ee5db0c65c52c93046747df3ca3f00f074a6fc403666b6c16` | Benchmark canonical 180 câu (156 MCQA + 24 Yes/No). |

---

## 2. Kết Quả Thẩm Định 12 Giả Thuyết Pháp Y (F01 – F12)

### F01. Nhiễm Trọng Số C1 Vào Đánh Giá `W0_BASE`
- **Trạng thái:** `CONFIRMED_ON_EXECUTED_CODE`
- **Phân tích:** Trong `scripts/kaggle_v2_run04_executor.py`, hàm `generate_single_prediction(..., use_adapter=False)` chỉ bỏ qua việc gán vector điều kiện và không tiêm seam. Lớp `DualCycleMaTLoRALinear.forward()` vẫn tính toán `base_out + lora_out` với tọa độ $(0, 0)$. Vì checkpoint C1 đã được nạp sẵn vào bộ nhớ trước khi chạy vòng lặp đánh giá `W0_BASE`, các ma trận `lora_B` mang trọng số khác không của C1 đã làm sai lệch kết quả của Base model.
- **Biện pháp khắc phục:** Bổ sung thuộc tính `self.enabled = False` vào `DualCycleMaTLoRALinear`. Khi `enabled = False`, mô hình trả về trực tiếp `base_layer(x)` với sai khác tuyệt đối bằng đúng $0{,}0000000000$ so với mô hình Qwen2.5-0.5B nguyên bản (Gate G1 PASS).

### F02. Rò Rỉ Trạng Thái Tọa Độ Thời Gian Qua Các Lần Gọi Suy Luận
- **Trạng thái:** `CONFIRMED_ON_EXECUTED_CODE`
- **Phân tích:** Các biến `current_theta` và `current_z` được lưu trực tiếp trên thuộc tính module và không được hoàn nguyên sau khi sinh xong. Hook nhúng không có cơ chế thu hồi handle khi phát sinh exception.
- **Biện pháp khắc phục:** Bổ sung phương thức `clear_temporal_coordinates()` và xây dựng context manager `scoped_model_intervention` bảo đảm xóa sạch hook và hoàn nguyên tọa độ trong khối `finally`.

### F03. Nhánh Replay $R4\_D$ Chưa Được Cài Đặt Thực Tế
- **Trạng thái:** `CONFIRMED_ON_EXECUTED_CODE`
- **Phân tích:** Dòng 331–336 của executor cho thấy khối lệnh của $R4\_D$ lấy chính xác các phần tử từ `train_rel_pairs` và `train_mcqa_clean` giống hệt nhánh $R4\_B$. Không có mẫu replay nào từ dữ liệu cũ được nạp hay đưa vào tính gradient.
- **Biện pháp khắc phục:** Đình chỉ kết luận về replay; ghi nhận trạng thái khoa học là **`REPLAY_NOT_TESTED`**. Xác nhận 508/508 tensors của $R4\_D$ và $R4\_B$ trùng khớp $100\%$ (`max_abs_diff = 0.0`).

### F04. Tính Toán Hàm Mất Mát Đa Nhiệm Bằng Phép Bình Quân Tuần Tự
- **Trạng thái:** `CONFIRMED_ON_EXECUTED_CODE`
- **Phân tích:** Executor thực thi `loss_i = out.loss / batch_size` với `batch_size = 2` trên từng mẫu (1 câu relation và 1 câu MCQA).
- **Đánh giá:** Đại số tích lũy gradient $0{,}5 \nabla \mathcal{L}_{\text{rel}} + 0{,}5 \nabla \mathcal{L}_{\text{mcqa}}$ hoàn toàn tương đương với hàm mục tiêu $\mathcal{L} = 0{,}5 \mathcal{L}_{\text{rel}} + 0{,}5 \mathcal{L}_{\text{mcqa}}$. Mục tiêu toán học là chính xác.

### F05. Sai Lệch Mẫu Số Đánh Giá Giữa Executor Kaggle và Script Cục Bộ
- **Trạng thái:** `CONFIRMED_ON_EXECUTED_CODE`
- **Phân tích:** Executor chạy toàn bộ 120 câu Oracle Diagnostic và 180 câu Retention benchmark. Script cục bộ `evaluate_v2_run04_clean.py` lấy mẫu 60 câu Oracle và 60 câu Retention để tạo tệp `KNOWLEDGE_VS_RELATION_DIAGNOSTIC.json`.
- **Biện pháp khắc phục:** Phân định rõ ràng trong tài liệu nghiệm thu giữa lát cắt 60 câu (`sample_60_slice`) và tập chuẩn 180 câu (`canonical_180_benchmark`), lưu trữ đầy đủ `sample_id` tương ứng.

### F06. Bản Chất Đối Chứng của $R4\_C$ Là Đối Chứng Huấn Luyện, Không Phải Can Thiệp Suy Luận
- **Trạng thái:** `CONFIRMED_ON_EXECUTED_CODE`
- **Phân tích:** $R4\_C$ được huấn luyện với năm bị xáo trộn, nhưng khi suy luận vẫn nhận năm đúng `year_eval`. Phép so sánh giữa $R4\_B$ và $R4\_C$ phản ánh ảnh hưởng của quy trình huấn luyện, không phải là phép đo nhân quả cô lập điều kiện trên cùng một mô hình.
- **Biện pháp khắc phục:** Thực hiện bộ 6 can thiệp điều kiện suy luận trên **cùng một checkpoint** $R4\_B$ (TRUE, WRONG, ZERO, SEAM_OFF, LORA_OFF, BOTH_OFF) tại Phase P4.

### F07. Rủi Ro Che Giấu Lỗi Nạp Trọng Số Do `strict=False`
- **Trạng thái:** `CONFIRMED_ON_EXECUTED_CODE`
- **Phân tích:** Việc sử dụng `strict=False` khi nạp từ điển `adapters_state` có nguy cơ bỏ qua các tham số bị thiếu.
- **Biện pháp khắc phục:** Triển khai hàm kiểm tra nghiêm ngặt `strict_verify_adapter_checkpoint`, xác thực đầy đủ 72 module, kiểm tra định dạng tên tensor, kiểu dữ liệu và tính hữu hạn (`torch.isfinite`).

### F08. Mã Băm Tệp Checkpoint Khác Nhau Dù Trọng Số Đồng Nhất
- **Trạng thái:** `CONFIRMED_ON_EXECUTED_CODE`
- **Phân tích:** Tệp $R4\_B$ và $R4\_D$ có SHA256 khác nhau trên đĩa do chứa khóa metadata `"arm_name"` mang giá trị khác nhau, trong khi toàn bộ 508 tensor khả huấn bên trong giống nhau từng bit.
- **Biện pháp khắc phục:** Xây dựng cơ chế băm tensor thuần túy (`tensor_fingerprint`), tách biệt hoàn toàn trọng số mô hình khỏi siêu dữ liệu lưu trữ.

### F09. Giới Hạn Cụm và Tính Toán P-Value của McNemar
- **Trạng thái:** `CONFIRMED_ON_EXECUTED_CODE`
- **Phân tích:** `GROUP_AWARE_STATS.json` thực chất tính McNemar ghép cặp trên từng ca riêng lẻ và làm tròn p-value về `0.0`. Tập DEV chỉ chứa 2 thực thể nguồn (`BCXZ` 107 câu, `GVRO` 13 câu), không thể suy diễn tổng quát hóa cấp quần thể.
- **Biện pháp khắc phục:** Đổi tên hạng mục thống kê thành `paired_case_level`, xuất p-value dạng số học khoa học ($p = 2{,}47 \times 10^{-5}$), và bổ sung cảnh báo rõ ràng về giới hạn ngoại suy cụm.

### F10. Giá Trị Mặc Định Ẩn Khi Thiếu Siêu Dữ Liệu Năm
- **Trạng thái:** `CONFIRMED_ON_EXECUTED_CODE`
- **Phân tích:** Mã nguồn cũ sử dụng cú pháp `or 1984` che giấu các trường hợp dữ liệu đầu vào thiếu thông tin năm.
- **Biện pháp khắc phục:** Loại bỏ hoàn toàn giá trị fallback ngầm; dữ liệu thiếu mốc thời gian bắt buộc phải báo lỗi `ValueError` đóng kín.

### F11. Giới Hạn của Thử Nghiệm 30 Bước trên Một Hạt Giống
- **Trạng thái:** `CONFIRMED_ON_EXECUTED_CODE`
- **Phân tích:** Thử nghiệm Run 04 thực thi 30 bước tối ưu hóa trên một hạt giống duy nhất (`seed=42`).
- **Biện pháp khắc phục:** Không tuyên bố độ lợi mang tính tổng quát hóa đa hạt giống; ghi nhận chính xác ngân sách phơi nhiễm dữ liệu thực tế (60 lượt tiếp xúc).

### F12. Bản Chất của Bảng Điểm Chẩn Đoán Oracle Evidence
- **Trạng thái:** `CONFIRMED_ON_EXECUTED_CODE`
- **Phân tích:** Trong bảng điểm S3, số liệu thực tế được cung cấp trực tiếp trong câu hỏi.
- **Biện pháp khắc phục:** Tách biệt rõ ràng năng lực ghi nhớ sự thật lịch sử (K) khỏi năng lực suy luận quan hệ số học (R). Khi có số liệu, mô hình đạt $65{,}00\%$, chứng minh năng lực so sánh số học của mô hình hoàn toàn bình thường.
