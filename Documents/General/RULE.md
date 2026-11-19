# Quy tắc nhật ký Documents/General

`Documents/General` chỉ chứa file quy tắc này và hai họ nhật ký dưới đây. Tài liệu khảo sát, quyết định, hướng dẫn, lịch môn, cấu hình MCP, báo cáo DOCX/HTML và nội dung tinh chỉnh hệ thống phải đặt tại `Documents/References` hoặc `Documents/Decision-Making`.

## Tên file

- Checkpoint: `long-range-temporal-alignment_Checkpoints_DD-MM-YY_Vn.md`.
- Run không hợp lệ hoặc quality gate thất bại: `long-range-temporal-alignment_Error_DD-MM-YY_Vn.md`.
- `DD-MM-YY` thay cho `DD/MM/YY` vì Windows không cho phép dấu `/` trong tên file.
- `Vn` bắt đầu từ `V1`, tăng tuần tự cho cùng project, loại và ngày; không ghi đè hoặc tái sử dụng phiên bản. Ngày mới bắt đầu lại từ `V1`.
- Ngày là ngày chạy/checkpoint theo `Asia/Bangkok`, không phải ngày sửa văn bản.

## Checkpoint phải ghi

1. Mục tiêu/stage, thời điểm, owner, Git revision và trạng thái worktree.
2. System profile liên quan: OS, Python, CPU/RAM, GPU/VRAM, package/service version; ghi `không thu thập` nếu thiếu, không suy đoán.
3. Lệnh đã chạy; config, seed, input/data/model revision và hash khi có.
4. Artifact/output, kết quả `PASS|FAIL|UNAVAILABLE|BLOCKED` và mức bằng chứng.
5. Giới hạn, điều kiện tiếp theo và rollback/replay.

Với run `measured` hoặc `benchmark`, phải ghi thêm:

1. `anchor_checkpoint` của baseline đóng băng; `parent_checkpoint` của run trực tiếp trước đó. Dùng `NONE` và nêu lý do khi chưa có.

2. `hypothesis_id`, `evaluation_role=development|error_analysis|sealed_test`, `gold_status=complete|partial|absent` và `evaluation_contract_hash`.

3. Evaluation contract gồm data manifest, split, gold/qrels, evaluator, metric, parser và slice registry. Nếu khác run đối chứng hoặc có nhiều thay đổi không tách được, ghi `comparison=confounded`; không kết luận gain hay nguyên nhân.

4. Paired delta theo `case_id|query_id|group_id` so với cả anchor và parent; báo overall, lát cắt, denominator, seed/CI, `wins|ties|losses` và đường dẫn/hash artifact theo mẫu.

5. `case_budget` và quy tắc chọn trước run. Tối thiểu xét ca cải thiện, regression, lỗi dai dẳng, low-margin/high-variability, counterexample và mẫu random có seed trên lát cắt trọng yếu. Nếu một nhóm không có ca, ghi `0`, phạm vi tìm và quy tắc chọn.

6. `observation` tách khỏi `cause`; trạng thái hypothesis là `supported|refuted|inconclusive`, kèm dự đoán có thể bác bỏ, minimal next experiment và stop/rollback gate.

Khi `gold_status=partial`, ghi judged coverage và cách xử lý unjudged. Khi `gold_status=absent`, chỉ báo chất lượng dữ liệu, stability, behavioral test hoặc human-audit protocol. Pseudo-label, synthetic label và LLM judge là `proxy`, không phải gold và không đủ để tuyên bố effectiveness tăng.

Không dùng lại `sealed_test` để sinh giả thuyết. Khi đã phân tích ca lỗi trên test, đánh dấu tập đó là `error_analysis` trong lần dùng tiếp theo.

## Error phải ghi

Ngoài metadata checkpoint, ghi checkpoint liên quan, `first_bad`, `last_known_good`, invariant hoặc quality gate bị vi phạm, lát cắt/ca bị ảnh hưởng, expected/actual, cách tái hiện, log đã loại secret, bằng chứng ủng hộ và phản bác từng nguyên nhân, khắc phục/rollback, regression test và rủi ro còn lại.

Run thực thi hợp lệ nhưng metric kém hơn phải nằm trong Checkpoint với `decision=REJECT|REPEAT`; không tạo Error chỉ vì giả thuyết bị bác bỏ. Chỉ tạo Error khi run hoặc kết quả không hợp lệ, hoặc quality gate đã định trước chuyển sang `FAIL`. Rủi ro dự kiến và bài học tổng hợp vẫn thuộc `Documents/Decision-Making/ERROR-EXPERIENCE.md`.

Không tạo log chỉ để mô tả hoạt động nghiên cứu hay làm đầy thư mục. Khi nghiên cứu/bổ sung/loại bỏ/tinh chỉnh hệ thống, nội dung chuyên môn nằm ở `Decision-Making`/`References`; checkpoint chỉ ghi cấu hình, lệnh và bằng chứng thực sự đã chạy.

Nội dung viết bằng tiếng Việt có dấu, viết hoa đầu câu và dùng câu ngắn. Chỉ giữ tiếng Anh cho identifier, enum, API và thuật ngữ cần thiết; không chép tiến trình debug dài vào log, chỉ liên kết artifact đã lọc và có hash.
