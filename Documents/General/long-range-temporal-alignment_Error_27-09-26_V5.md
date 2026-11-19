# ERROR — Khắc phục lỗi cú pháp YAML frontmatter trong SKILL.md phục vụ Codex native discovery (V5)

## Định danh và trạng thái

- Project: long-range-temporal-alignment
- Thời điểm: 2026-09-27, Asia/Bangkok
- Phiên bản: V5
- Nhánh Git: main, HEAD c4302f49eef825c5ce9f84a076e2be3707b0a0da
- Phân loại: Sửa chữa lỗi cú pháp YAML frontmatter trong SKILL.md phục vụ Codex native discovery
- Trạng thái: RESOLVED_BY_HOTFIX_V5
- Expected: Khi khởi động Codex CLI native (0.157.1 trên Windows), hệ thống nạp toàn bộ danh mục kỹ năng hợp lệ mà không phát sinh bất kỳ cảnh báo "Skipped loading ... skill(s) due to invalid SKILL.md files" hay lỗi cú pháp YAML nào.
- Actual: Khi khởi động Codex, xuất hiện cảnh báo "Warnings · 1 of 13: Skipped loading 12 skill(s) due to invalid SKILL.md files", kèm 12 cảnh báo chi tiết chỉ ra lỗi cú pháp YAML tại các tệp kỹ năng caveman/cavecrew ("invalid YAML: description: invalid type: sequence, expected a string at line 2 column 14").
- Nguyên nhân:
  1. Sai lệch cú pháp YAML frontmatter (MALFORMED_YAML / DESCRIPTION_PARSED_AS_LIST_OR_MAP): Tiền tố `[MANUAL CALL ONLY - No auto route] >` được chèn trực tiếp sau khóa `description:` mà không có dấu ngoặc kép. Dấu ngoặc vuông `[` khiến bộ phân tích cú pháp serde_yaml trong Codex diễn giải giá trị của `description` là một sequence (danh sách) thay vì chuỗi ký tự (string), và ký tự `>` đứng ngay sau danh sách gây lỗi cú pháp nghiêm trọng.
  2. Bỏ sót trong đợt kiểm toán trước: Đợt kiểm toán S05 xem các công cụ caveman này là trạng thái DORMANT (không kích hoạt tự động) trong tài liệu báo cáo nhưng vẫn giữ các tệp vật lý trong thư mục `C:\Users\GF63\.agents\skills\`. Do Codex CLI tự động quét toàn bộ thư mục này khi khởi chạy tiến trình, lỗi cú pháp đã làm tê liệt quá trình nạp kỹ năng native.
  3. Ký tự so sánh góc nhọn vi phạm quy chuẩn Codex: Trường description trong `cavecrew` và `caveman-commit` chứa các ký tự góc nhọn (`>`, `<=`) bị cấm bởi quy chuẩn Codex skill validator.
- Khắc phục:
  1. Khôi phục và chuẩn hóa frontmatter của toàn bộ 12 kỹ năng mục tiêu:
     - `cavecrew`, `caveman`, `caveman-commit`, `caveman-compress`, `caveman-discover`, `caveman-evidence-review`, `caveman-help`, `caveman-manage`, `caveman-optimize`, `caveman-review`, `caveman-setup`, `caveman-stats`.
     - Chuyển `description` thành chuỗi được bao bọc chuẩn xác bởi dấu ngoặc kép, loại bỏ hoàn toàn các ký tự góc nhọn vi phạm (`<`, `>`).
     - Bảo toàn nguyên vẹn 100% phần thân Markdown body của toàn bộ các kỹ năng (zero body mutation).
  2. Chuẩn hóa bổ trợ cho các kỹ năng phụ trợ:
     - `caveman-learn`: Chuẩn hóa chuỗi description.
     - `caveman-explore`: Đưa danh sách công cụ vào `allowed-tools` chuẩn hóa và chuyển thông số mô hình vào `metadata`.
     - `caveman-default`: Chuẩn hóa chuỗi description trong `C:\Users\GF63\.codex\skills\caveman-default\SKILL.md`.
     - Chuẩn hóa 8 kỹ năng thuộc plugin `understand-anything` để đảm bảo tương thích tuyệt đối.
  3. Xây dựng công cụ kiểm định độc lập `scripts/validate_codex_skills.py` áp dụng đầy đủ bộ kiểm tra của OpenAI/Codex reference validator (`quick_validate.py`).
  4. Lập hồ sơ kiểm toán [INVALID_SKILL_AUDIT_V5.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/INVALID_SKILL_AUDIT_V5.json) và báo cáo nghiệm thu [SKILL_YAML_REPAIR_V5.json](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/SKILL_YAML_REPAIR_V5.json).
- Regression:
  1. Thực thi `scripts/validate_codex_skills.py`: 59/59 active skills đạt kết quả VALID (0 invalid, 0 shadowing, 0 directory mismatch).
  2. Toàn bộ test suite của dự án đạt 73/73 PASS.
  3. Các cổng kiểm tra chất lượng `validate_project.py` và `validate_knowledge_vault.py` đạt kết quả PASS tuyệt đối.

## 1. Bản chất sự cố và bằng chứng phân tích

Trong nhật ký vận hành Codex CLI lưu trữ tại cơ sở dữ liệu `C:\Users\GF63\.codex\logs_2.sqlite`, sự kiện `session_init` khi khởi tạo thread liên tục ghi nhận 12 thông báo lỗi nạp kỹ năng:
`failed to load skill C:\Users\GF63\.agents\skills\<skill_name>\SKILL.md: invalid YAML: description: invalid type: sequence, expected a string at line 2 column 14`.

Hiện tượng này đối chiếu chính xác 100% với cảnh báo 13 mục trên giao diện TUI của Codex CLI:
- Cảnh báo tổng quát: `Skipped loading 12 skill(s) due to invalid SKILL.md files.`
- 12 cảnh báo thành phần: Tương ứng với từng tệp kỹ năng trong thư mục `.agents\skills\`.

## 2. Quy chuẩn kiểm định Codex Native Skill

Tham chiếu theo bộ mã nguồn kiểm định chính thức của Codex tại `C:\Users\GF63\.codex\skills\.system\skill-creator\scripts\quick_validate.py`, một tệp `SKILL.md` hợp lệ bắt buộc thỏa mãn:
1. Bắt đầu và kết thúc phần frontmatter bằng cặp dấu phân cách `---`.
2. Khối frontmatter phải là một mapping/dictionary hợp lệ trong YAML.
3. Chỉ chứa các trường thuộc tập hợp cho phép: `name`, `description`, `license`, `allowed-tools`, `metadata`.
4. Trường `name`: Là chuỗi định dạng hyphen-case (chữ thường, số, dấu gạch nối đơn), độ dài tối đa 64 ký tự.
5. Trường `description`: Là chuỗi mô tả rõ ràng, độ dài tối đa 1024 ký tự, tuyệt đối không chứa ký tự góc nhọn (`<`, `>`).
6. Phần thân tài liệu (body): Không chứa khối mã lệnh chưa đóng hoặc thẻ giữ chỗ `[TODO:...]`.

## 3. Kết quả kiểm tra và xác nhận không có xung đột (Shadowing Audit)

Kiểm toán tự động trên 59 kỹ năng hoạt động ghi nhận:
- Sự sai lệch giữa tên thư mục và tên kỹ năng: 0.
- Số lượng kỹ năng bị trùng lặp tên gây đè lấp (accidental shadowing): 0.
- Các tệp gốc của Matt Pocock trong `vendor/matt` được bảo toàn nguyên trạng, các wrapper hoạt động độc lập và minh bạch.
- Tất cả các kỹ năng quản trị toàn cục (GLOBAL) và kỹ năng riêng biệt của dự án (LOCAL) đều được phân định quyền sở hữu rõ ràng.
