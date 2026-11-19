# BIÊN BẢN ĐÍNH CHÍNH CÁC SAI SỐ KỸ THUẬT VÀ BÁO CÁO TIỀN NHIỆM (ERRATA)

Thời điểm lập biên bản: 2026-11-04
Phạm vi kiểm toán: Các tệp kết quả và tổng hợp trong `Reports/6-11Report/`

Biên bản này ghi nhận khách quan các sai khác số học, cấu trúc và quy trình được phát hiện trong đợt kiểm toán độc lập mà không thay đổi lịch sử các tệp gốc.

---

## 1. Erratum 1: Lỗi vòng lặp tự băm (Self-Hash Circular Dependency) và trôi dạt byte trong EVIDENCE-INDEX.json
- **Hiện tượng**: Tệp `EVIDENCE-INDEX.json` trong `Reports/6-11Report` tự lập danh mục chính nó trước khi ghi nội dung mới, dẫn đến sai lệch mã băm SHA-256 và kích thước byte. Đồng thời, 3 tệp (`ARMS_1_TO_5_CONFIGS_AND_OBJECTIVES.json`, `DATASET_PATHS_AND_PROVENANCE_COUNTERFACTUAL_STRICT.json`, `EVIDENCE-INDEX.json`) có kích thước thực tế khác với chỉ mục.
- **Biện pháp xử lý**: Chuyển sang sử dụng bản kê khai băm tách rời (`SOURCE_AND_SHA256_MANIFEST.json`) đặt trong thư mục kiểm toán độc lập `_FORENSIC_AUDIT_20261104/`, loại bỏ hoàn toàn tự tham chiếu.

## 2. Erratum 2: Khai báo vượt số lượt suy luận kiểm chuẩn bền vững (Rotation Inference Inflation)
- **Khai báo tiền nhiệm**: Báo cáo tổng hợp ghi nhận Strict Accuracy thực hiện $5.292 \times 4 = 21.168$ lượt suy luận, và tổng thể kiểm định 3 tầng là $22.980$ lượt suy luận.
- **Thực tế kiểm toán**: Tập kiểm chuẩn `eval_5292_clean.jsonl.gz` gồm 4.440 câu hỏi trắc nghiệm 4 phương án (thực hiện 4 vòng hoán vị = $17.760$ lượt) và 852 câu hỏi nhị phân `Rel` (chỉ có 2 phương án Yes/No, thực hiện 2 vòng hoán vị = $1.704$ lượt).
- **Số lượt suy luận chính xác**:
  - Strict Accuracy thực tế: $17.760 + 1.704 = 19.464$ lượt suy luận (chênh lệch giảm $1.704$ lượt).
  - Tổng thể độ bền vững: $19.464 \text{ (Strict)} + 108 \text{ (CF V1)} + 1.704 \text{ (CF V2)} = 21.276$ lượt suy luận (chênh lệch giảm $1.704$ lượt so với con số $22.980$).

## 3. Erratum 3: Thiếu hụt 342 khóa định danh trong tệp ARM_1_TICKTACK_BASE_CANONICAL_RESULTS.json
- **Hiện tượng**: Tệp kết quả Arm 1 khai báo tổng mẫu kiểm định là $5.292$, nhưng từ điển `case_results` chỉ chứa $4.950$ khóa duy nhất.
- **Nguyên nhân gốc rễ (Root Cause)**: Tệp dữ liệu `eval_5292_clean.jsonl.gz` chứa $4.950$ mã `example_id` duy nhất và $342$ mã trùng lặp. Script đánh giá của Arm 1 sử dụng trực tiếp `case_results[example_id] = ...`, khiến 342 câu hỏi trùng ID bị ghi đè trong từ điển. Các nhánh từ Arm 2 đến Arm 5 đã khắc phục bằng khóa `f"{idx}_{example_id}"`, bảo toàn đủ $5.292$ trường hợp.
- **Hệ quả thống kê**: Không thể áp dụng kiểm định ghép cặp McNemar quy mô $N=5.292$ giữa Arm 1 và các nhánh khác; bắt buộc phải giới hạn trên tập giao hoán $N=4.950$ trường hợp.

## 4. Erratum 4: Tuyên bố sớm trạng thái 5EPOCHS_EXECUTED_AND_VERIFIED khi chưa đối soát chi phí
- **Hiện tượng**: Báo cáo tổng hợp trước đó đã gắn nhãn `5EPOCHS_EXECUTED_AND_VERIFIED` ngay sau khi tải tệp kết quả.
- **Điều chỉnh**: Trạng thái được hạ về `QUARANTINED_PENDING_PROVENANCE` cho đến khi toàn bộ 6 cổng kiểm toán (`G_DATA`, `G_METRICS`, `G_RUN`, `G_CHECKPOINT`, `G_BILLING`, `G_REPORT`) được đối soát độc lập với hệ thống Modal và Google Drive.

## 5. Erratum 5: Lối tắt cấu trúc trong tập kiểm nghiệm CF V1 (Historical Milestones)
- **Hiện tượng**: Toàn bộ 36 cặp đảo chiều (flip pairs) trong 54 cặp của CF V1 đều có hướng cố định: Câu 1 là `Yes`, Câu 2 là `No`.
- **Hệ quả**: Một bộ phân lớp ngây thơ (heuristic) luôn dự đoán `Yes` cho câu 1 và `No` cho câu 2 sẽ đạt $36/54 = 66.67\%$ độ chính xác trên CF V1 mà không cần hiểu biết lịch sử. Cần lưu ý khi diễn giải kết quả CF V1.
