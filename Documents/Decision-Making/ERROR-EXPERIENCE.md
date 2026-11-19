# Error Experience

Đây là sổ lỗi và bài học vận hành, không phải danh sách kết quả nghiên cứu. Mỗi mục mới cần ghi ngày, triệu chứng, bằng chứng, nguyên nhân đã xác minh hoặc giả thuyết, cách khắc phục và regression gate.

| Giai đoạn | Lỗi từng gặp/rủi ro đã thấy | Bài học áp dụng tại đây |
|---|---|---|
| Chọn bài toán | Mở scope thành hệ thống tổng quát trước khi có dữ liệu/evaluator | Giữ một bài toán chính: temporal retrieval/QA tiếng Việt có bằng chứng và khả năng từ chối |
| Khảo sát nguồn | Đồng nhất mô tả thứ cấp với thông tin chính thức | Nguồn khám phá chỉ dẫn đường; claim kỹ thuật phải về paper, model card, dataset card hoặc docs chính thức |
| Thu thập dữ liệu | Tải asset khi thiếu license, revision hoặc vai trò split | Adapter fail-closed; registry là điều kiện trước download, không phải ghi bổ sung sau |
| Split | Random split làm near-duplicate/sự kiện cùng nguồn rơi qua các tập | Chia theo source/entity/event/time trước mọi bước học từ dữ liệu; audit leakage riêng |
| Tiền xử lý | Dùng số 0 mơ hồ cho BCE/CE; fit preprocessing trên toàn corpus | Dùng `HistoricalYear`; fit trên train và lưu provenance/hash |
| Training | Suy luận “model nhỏ sẽ vừa GPU”; áp EWC lên tham số frozen | Đo peak RAM/VRAM trước; audit trainable parameters và Fisher/reference checkpoint |
| Evaluation | Gọi fixture/smoke là benchmark; chỉ báo mean một seed | Tách evidence class; báo invalid-output, slice, seed và khoảng tin cậy |
| Dịch vụ ngoài | Kết nối dịch vụ bị hiểu nhầm là dữ liệu/model đã sẵn sàng | Tách connectivity, authorization, manifest, checksum và semantic validation |
| Dịch vụ inference | Model có trang Hub nhưng không có provider; provider tự chọn làm route thay đổi | Preflight provider; pin model/provider/revision; cấm silent fallback và ghi route vào artifact |
| Embedding | HTTP lỗi hoặc output sai shape bị đổi thành zero vector | Validate status/schema/dimension/finite/norm; lỗi phải dừng đóng, không tạo ranking giả |
| Dữ liệu lịch sử | Cùng sự kiện/đoạn sách xuất hiện ở nhiều wording hoặc nguồn chép lẫn nhau | Cluster theo event/source/near-duplicate trước split; kiểm thủ công vùng ngưỡng |
| Annotation | Agreement cao giả do phần lớn evidence là không liên quan | Báo agreement theo lớp, answer/evidence riêng và adjudication trên mẫu khó |
| Qrels | So hai run dùng candidate pool hoặc qrels revision khác nhau | Lưu pool/qrels hash; paired comparison chỉ trên cùng query và cùng evaluator |
| Phần cứng | Gọi thiết bị là RTX 2050 Ti dù hệ thống báo RTX 2050 4.096 MiB | Artifact dùng tên/VRAM do công cụ hệ thống ghi nhận; mô tả người dùng chỉ là ngữ cảnh |
| Python | Lệnh `python` có thể trỏ tới 3.10 trong khi project yêu cầu ≥3.11 | Dùng `py -3.12`, preflight version và ghi interpreter path; xem error log trong `General` |
| Quản lý model | Dùng nhánh `main` thay revision bất biến | Pin commit/revision, checksum, config/data/code hash và điều kiện rollback |
| Streamlit | UI che mất nguồn artifact hoặc kích hoạt training/download | UI chỉ đọc artifact đã duyệt; hiển thị lineage, warning và evidence class |
| DOCX | Hậu xử lý DOCX có thể làm mất media đã nhúng | Giữ nguồn Markdown, render LibreOffice và kiểm tra PDF bằng PyMuPDF |
| MCP | Tắt tool ở cấu hình toàn cục làm ảnh hưởng repo khác | Chỉ dùng `.codex/config.toml` theo project; ghi cách hoàn tác |
| Tối ưu cục bộ (DSC A4.2) | Tăng k=100 cải thiện retrieval nhưng packing P2 làm giảm complete support (153 vs 156) | Đo end-to-end evidence availability, không dừng ở intermediate metric; local optimization ≠ global optimization |
| Điều kiện so sánh (DSC) | So sánh hai mô hình khi context không đồng nhất (0/700 context giống nhau) | Khóa chặt measurement contract; cô lập duy nhất một biến thực nghiệm |
| Tích hợp hệ thống (AIC2026) | 55/55 unit test pass nhưng live integration bị chặn do Docker runtime | Unit PASS không tương đương Integration PASS; phải kiểm thử E2E vertical slice |
| Phục hồi huấn luyện (CPT) | Chạy bị mất app, checkpoint thiếu optimizer/scheduler/RNG state | Kiểm tra checkpoint durability và tính khôi phục toàn vẹn trước khi resume |
| Bùng nổ số học (M0-FC40) | GradScaler khuếch đại 65.536x làm tràn FP16 intermediate VJP, clip_grad_norm đổi Inf thành NaN | Tham số khả huấn lưu FP32, upcast loss FP32, loại bỏ GradScaler khi gradients tự nhiên nằm trong ngưỡng an toàn |
| Đánh đổi căn chỉnh (M0-FC40) | Causal LM loss giảm mạnh nhưng trôi dạt cú pháp làm Invalid Rate tăng lên 93.33% | Nhận diện Alignment Tax; phải bổ sung cơ chế Instruction-Preserving Regularization trong Nhánh V2 |
| Quản lý đĩa/Transfer | Pipeline transfer nạp 4x duplicates 25 file zip làm cạn 59.5 GB đĩa C: | Sau khi upload thành công lên remote LFS, bắt buộc auto-purge temporary transfer/staging bundles và đặt cảnh báo quota C: <25 GiB |

Các lỗi thực tế, command và cấu hình không được ghi thêm vào file tổng hợp này; chúng thuộc `Documents/General/<Project>_Error_DD-MM-YY_Vn.md` theo `RULE.md`.

## Mẫu log

```text
Ngày / commit:
Stage và input revision:
Triệu chứng:
Bằng chứng quan sát:
Nguyên nhân: confirmed | hypothesis
Khắc phục/rollback:
Test ngăn tái phát:
Chủ sở hữu và trạng thái:
```

<!-- BEGIN delivery-must-do-prevent:v1 -->

## MUST DO và MUST PREVENT bổ sung

| MUST DO | MUST PREVENT |
|---|---|
| Chạy lexical/interval anchor trước learned representation | Xây abstraction tổng quát khi chưa có pilot/qrels |
| Một hypothesis, một biến chính, một stop gate | Cue nhiều model rồi chỉ so ví dụ đẹp |
| Ghi negative result `REJECT|REPEAT` như tri thức | Coi không tăng metric là không có tiến độ rồi né evaluator |
| Tạo proof packet định kỳ | Dùng paper map, UI hoặc diagram thay experiment evidence |
| Local/free trước; external chỉ để trả lời câu hỏi đã khóa | Dùng GPU/cloud để che data, split hoặc parser defect |
<!-- END delivery-must-do-prevent:v1 -->

## 5 Phân loại Thất bại Chuẩn tắc

1. **Problem failure**: Giải quyết sai bài toán hoặc bài toán không đủ giá trị (ví dụ: cải thiện generator trong khi retrieval thiếu bằng chứng).
2. **Scientific failure**: Giả thuyết không được thực nghiệm ủng hộ (ví dụ: FoNE thô không vượt qua baseline không thời gian).
3. **Experimental failure**: Phép đo không cho phép kết luận đáng tin cậy do confounding variables hoặc đo lệch điều kiện.
4. **Engineering failure**: Ý tưởng đúng nhưng triển khai kỹ thuật bị lỗi (ví dụ: overflow số học, checkpoint không thể phục hồi).
5. **Delivery failure**: Có kết quả kỹ thuật nhưng không bàn giao được sản phẩm hoặc không thể tái lập độc lập.

## Bộ Cổng Kiểm Định Dự Án (Project Failure Gates G0 -> G7)

$$\boxed{\text{Problem} \longrightarrow \text{Baseline} \longrightarrow \text{Controlled Experiment} \longrightarrow \text{Evidence} \longrightarrow \text{Decision} \longrightarrow \text{Verified Delivery}}$$

| Cổng | Tên cổng | Trọng tâm kiểm tra | Minh chứng bắt buộc |
| :--- | :--- | :--- | :--- |
| **G0** | Problem & Success Contract | Bài toán, KPI, đầu ra mong muốn và giới hạn phạm vi | Bản mô tả bài toán và success criteria được khóa chặt |
| **G1** | Data & Evaluation | Split độc lập, nhãn/qrels, audit leakage, evaluator mẫu số | Manifest dữ liệu và kết quả audit rò rỉ |
| **G2** | Baseline Reproducibility | Baseline đơn giản, reproducible trên cùng điều kiện | Mã nguồn run baseline + config hash + metrics JSON |
| **G3** | Experimental Validity | Cô lập duy nhất một biến thử nghiệm; có ablation, CI/p-value | Experiment card + paired comparison cùng substrate |
| **G4** | Integration Contract | Ranh giới giữa các component; E2E vertical slice test | Interface invariant test PASS; schema tương thích |
| **G5** | Reliability & Observability | Checkpoint bền vững, số học hữu hạn fail-fast, telemetry | Logs vi mô không lỗi; kịch bản retry/recovery xác thực |
| **G6** | Collaboration & Governance | Ranh giới sở hữu phân công rõ; audit nguồn kiểm chứng | Handoff manifest rõ ràng; không can thiệp code chéo |
| **G7** | Delivery & Claims | Tuyên bố dựa trên bằng chứng đo đạc; bàn giao tái lập | Final smoke test bundle, reproducibility receipt |

