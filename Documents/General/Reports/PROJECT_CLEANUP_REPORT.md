# BÁO CÁO DỌN DẸP DỰ ÁN (PROJECT CLEANUP REPORT)

**Thời gian thực hiện:** 2026-09-27 20:31:20  
**Repository:** `long-range-temporal-alignment`  
**Executor:** Antigravity (Evidence-First Maintenance Mode)  

---

## 1. Tổng quan Dọn dẹp & Dung lượng Thu hồi

- **Tổng số tệp đã xóa an toàn (`DELETE_SAFE`)**: 424 tệp
- **Tổng dung lượng thu hồi**: 206,225,383 bytes (~196.67 MB)
- **Thư mục rác/tạm đã xóa triệt để**:
  - Toàn bộ cây thư mục `.tmp/` (~193.98 MB) bao gồm các kết quả trích xuất tạm và mock directories sinh ra trong các đợt chạy `pytest` trước đó.
  - Các tệp bytecode Python `.pyc` trong `__pycache__/` tại `src/`, `tests/`, `scripts/`, `SourceAPI/`.
  - Bộ nhớ đệm `.pytest_cache/` và `.ruff_cache/`.
  - 5 tệp script/kết quả trung gian dùng một lần trong `scratch/` đã hoàn thành sứ mệnh (`audit_tool_mcp.py`, `generate_hook_source_inventory.py`, `generate_optimization_report.py`, `run_regression_canaries.py`, `test_skills_audit.json`).

---

## 2. Các Tệp Được Bảo Toàn (Preserved Evidence & Reproducibility)

Tuân thủ nghiêm ngặt Quy tắc A4 (*Preserve official evidence*), các tệp có nguồn gốc từ quá trình chẩn đoán/gỡ lỗi nhưng hiện là bằng chứng chính thức hoặc được trích dẫn trong `Documents/General` đều được giữ nguyên 100%:

| Tệp / Thư mục | Kích thước | Phân loại | Căn cứ Giữ lại |
| :--- | :--- | :--- | :--- |
| `FINAL_SYSTEM_INTEGRITY_V6.json` | 10,030 B | `KEEP_EVIDENCE` | Biên lai nghiệm thu hệ thống toàn diện, nằm trong sync manifest |
| `CROSS_PROJECT_FIX_MATRIX.json` | 7,123 B | `KEEP_EVIDENCE` | Ma trận nghiệm thu 7 dự án thành phần |
| `CROSS_PROJECT_AGENT_SYSTEM_FIX.json` | 11,276 B | `KEEP_EVIDENCE` | Biên bản kiểm toán kỹ thuật khế ước Wire Protocol V4 |
| `CROSS_PROJECT_AGENT_SYSTEM_FIX.md` | 7,858 B | `KEEP_EVIDENCE` | Báo cáo nghiệm thu liên dự án |
| `TOOL_MCP_OPTIMIZATION_AUDIT.json` | 79,088 B | `KEEP_EVIDENCE` | Hồ sơ kiểm toán năng lực Tool & MCP |
| `TOOL_MCP_OPTIMIZATION_REPORT.md` | 31,171 B | `KEEP_EVIDENCE` | Báo cáo tối ưu hóa Tool & MCP |
| `CODEX_SKILL_VALIDATION_V5.json` | 18,008 B | `KEEP_EVIDENCE` | Kết quả kiểm định tính hợp lệ native của Codex skills |
| `SKILL_YAML_REPAIR_V5.json` | 5,001 B | `KEEP_EVIDENCE` | Lịch sử sửa lỗi cú pháp YAML frontmatter |
| `HOOK_SOURCE_INVENTORY_V4.json` | 9,449 B | `KEEP_EVIDENCE` | Kiểm kê toàn bộ hook sources theo Wire Protocol V4 |
| `HOOK_PERFORMANCE_AUDIT.json` | 669 B | `KEEP_EVIDENCE` | Đo đạc độ trễ và số chu kỳ CPU của hook adapters |
| `WINDOW_POPUP_DIAGNOSIS.json` | 2,329 B | `KEEP_EVIDENCE` | Chẩn đoán nguyên nhân gốc hiện tượng chớp console Windows |
| `HOOK_FAILURE_TIMELINE.json` | 34,646 B | `KEEP_EVIDENCE` | Được trích dẫn trong `Documents/General/..._Error_27-09-26_V3.md` |
| `INVALID_SKILL_AUDIT_V5.json` | 17,464 B | `KEEP_EVIDENCE` | Được trích dẫn trong `Documents/General/..._Error_27-09-26_V5.md` |
| `scratch/final_audit_v6.py` | 17,089 B | `KEEP_EVIDENCE` | Được trích dẫn trong `Documents/General/..._Checkpoints_27-09-26_V2.md` |
| `scratch/regression_canaries_report.json` | 3,001 B | `KEEP_EVIDENCE` | Được trích dẫn trong `Documents/General/..._Error_27-09-26_V2.md` |
| `scratch/apply_cross_project_rollout.py` | 20,880 B | `KEEP_REPRODUCIBILITY` | Mã nguồn tái lập quy trình rollout và kiểm định 7 dự án |

---

## 3. Các Mục Đề Xuất Xem Xét Thủ Công (Manual Review)

- **`lab01_completed/` (10 tệp, ~10.81 MB)**:
  - Bản chất: Bài tập thực hành xử lý ảnh (PIL, OpenCV) cùng các hình ảnh mẫu (`baboon.png`, `cat.png`, `lenna.png`).
  - Đánh giá: Không có tham chiếu trong mã nguồn, kiểm thử hoặc tài liệu của dự án `long-range-temporal-alignment`. Tuy nhiên, đây là bài tập học tập đã hoàn thành của người dùng, việc xóa tự động sẽ vi phạm nguyên tắc bảo vệ dữ liệu người dùng.
  - Khuyến nghị: Người dùng có thể di chuyển thư mục này về đúng repository môn học (ví dụ: `DS335-VLM-Research` hoặc lưu trữ cá nhân) thay vì xóa bỏ.

---

## 4. Kết quả Kiểm định Tính Toàn vẹn (Post-Cleanup Verification)

- **`validate_project.py`**: `VALIDATION PASSED` (66 required files, 10 DOCX/source pairs, progress HTML, smoke pipeline).
- **`validate_knowledge_vault.py`**: `Living Knowledge Vault: PASS` (56 Markdown, 20 managed, 3 decisions, 2 references).
- **Pytest**: `73 passed in 0.97s` (100% test suites hoàn thành không lỗi).
- **Google Drive NCKH Sync**: `copy=0`, `update=0`, `delete_stale=0` (Kho Drive và Local đồng bộ tuyệt đối).
