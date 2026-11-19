# Kế hoạch replay Stage03 trên 128 mẫu

Tiền đề là forensic ngày 04/11, [contract metric](METRIC_CONTRACT_V2.json) và [128 dòng đã khóa](REPLAY_128_CASES.json). Chọn mẫu round-robin theo task và pattern đúng/sai của hai arm, sắp bằng SHA của revision dữ liệu, chỉ số dòng và ID. Giữ eval_5292 ở vai trò development benchmark.

- `RUN_REPLAY_GPU=false`. Provider dự kiến Modal A10G. Trước khi launch, chốt maximum spend do người dùng phê duyệt, deadline, image/container identity, revision Base/tokenizer đúng lần deploy, dữ liệu/adapter/evaluator SHA và command evaluation-only. Không gọi runner 5 epochs hiện có vì đường đó có nhánh training. Kế hoạch hiện thiếu deploy binding nên chưa có command GPU hợp lệ.
- Dùng chính template, padding, truncation, precision, vị trí Seam, thứ tự lô và choice-token scorer của run gốc. Tái tạo độ dài pad của từng lô gốc khi chọn 128 dòng. Kiểm exact adapter key/shape/dtype và tensor đọc lại. Audit padding/last-token riêng, với kết quả version mới nếu thay evaluator.
- So Arm 4/5 final với epoch 1 và epoch 3/5 cùng mẫu. Giữ thêm adapter OFF, Seam OFF, wrong year, zero-time và dynamic core OFF ở Arm 5. Ghi identity, predicted label, gold chỉ ở scorer, vector log-prob/logit, top-2 margin, permutation và treatment witness.
- Dừng nếu lệch binding, nonfinite, hết ngân sách hoặc có argmax mismatch trên dữ liệu đối chứng. Gate định danh cần 128/128 dòng, còn gate argmax cần khớp tất cả dự đoán lịch sử đã xác minh theo từng phép xoay. JSON lịch sử chỉ giữ correctness/debiased/strict, nên exact argmax agreement hiện là `INCONCLUSIVE`. Chỉ xét full replay sau phê duyệt riêng.

Chạy tiền kiểm CPU từ ProjectSetup bằng `py -3.12 -B scripts/preflight_stage03_replay.py --report <thư_mục_này> --smoke`. Exit 3 thể hiện HOLD. Chi phí GPU và thời gian thực vẫn `HOLD_UNTIL_BOUND_REPLAY_AND_APPROVAL`.
