# Checkpoint: Hoàn Tất Tích Hợp Bộ Kỹ Năng Nature-Skills Cho Cả ANTI Và CODEX Trên 8 Dự Án

## 1. Định danh

- Project: long-range-temporal-alignment
- Canonical successor: Long-Range-Representation
- Thời điểm: 2026-10-02, Asia/Bangkok.
- Phiên bản: V1
- Owner: Antigravity Agent & Codex Pair
- Git revision: git_tracked (controlled, push_performed = false)
- Stage: Hoàn tất production integration của bộ kỹ năng Nature-Skills (pinned upstream) trên 8 dự án với tính tương đồng (parity) tuyệt đối giữa ANTI và CODEX.
- System profile: Python 3.12, Windows 11.
- Mức bằng chứng: verified_production_integration_audit|reproducible_canary_suite.

---

## 2. Thông Tin Thượng Nguồn Định Danh (Authoritative Upstream Identity)

- Repository: `https://github.com/Yuan1z0825/nature-skills`
- Commit SHA cố định: `84880815fb37317b3766bff2c2abba395b8993c3`
- Giấy phép: `Apache-2.0`
- Cơ chế cập nhật: `auto_update = false` (nghiêm cấm auto-update; nâng cấp theo `UPGRADE_RUNBOOK.md`).
- Tệp khóa quản trị: `NATURE_SKILLS_LOCK.json` (được lưu tại `_vendor/nature-skills/`, `long-range-temporal-alignment/`, và `Long-Range-Representation/`).

---

## 3. Phân Bổ Không Gian Tên Và Năng Lực (Namespace & Capabilities)

### AUDIT Namespace (Adopted Local Adapters)
1. **`audit:nature-statistics`**:
   - Thư mục chuyển đổi: `C:\Users\GF63\.skills\global\shared\audit\nature-statistics\`
   - Khám phá toàn cục: `C:\Users\GF63\.agents\skills\nature-statistics` và `C:\Users\GF63\.codex\skills\nature-statistics`
   - Chức năng: Kiểm toán báo cáo thống kê, đơn vị thực nghiệm, độ bất định, khoảng tin cậy CI, kích thước hiệu ứng và tương tác nhân quả giai thừa 2×2.
   - Rào chắn: Không bịa đặt p-value/CI; bảo toàn mẫu số hợp lệ/nghiêm ngặt của khế ước đánh giá dự án; không nâng cấp bằng chứng mô tả thành ý nghĩa thống kê.

2. **`audit:nature-reviewer`**:
   - Thư mục chuyển đổi: `C:\Users\GF63\.skills\global\shared\audit\nature-reviewer\`
   - Khám phá toàn cục: `C:\Users\GF63\.agents\skills\nature-reviewer` và `C:\Users\GF63\.codex\skills\nature-reviewer`
   - Chức năng: Mô phỏng phản biện học thuật độc lập, kiểm toán nhất quán giữa tuyên bố khoa học và bằng chứng thực nghiệm, phân loại quan ngại Major/Minor.
   - Rào chắn: Dựa 100% trên bằng chứng (evidence-grounded); yêu cầu snapshot báo cáo đóng băng; không bịa đặt số trang/dòng/hình vẽ; khế ước thực nghiệm và receipts thắng lời văn diễn giải.

### RESEARCH Namespace (Adapted Patterns Only)
3. **`research:data-evidence-gate`**:
   - Thư mục chuyển đổi: `C:\Users\GF63\.skills\global\shared\research\data-evidence-gate\`
   - Khám phá toàn cục: `C:\Users\GF63\.agents\skills\data-evidence-gate` và `C:\Users\GF63\.codex\skills\data-evidence-gate`
   - Chức năng: Kiểm soát nguồn gốc dữ liệu, quyền truy cập, giấy phép, mã băm SHA-256, điều kiện tái phân phối và kiểm toán siêu dữ liệu FAIR.
   - Rào chắn: Tuyệt đối không tự động tải dữ liệu từ internet khi chưa vượt qua cổng G0 Data Rights Gate; cấm commit dữ liệu thô vào Git.

4. **`research:academic-evidence-search`**:
   - Thư mục chuyển đổi: `C:\Users\GF63\.skills\global\shared\research\academic-evidence-search\`
   - Khám phá toàn cục: `C:\Users\GF63\.agents\skills\academic-evidence-search` và `C:\Users\GF63\.codex\skills\academic-evidence-search`
   - Chức năng: Khảo sát tài liệu có hệ thống, phân tầng nguồn (Tier 1/2/3), kiểm minh trích dẫn và DOI, rà soát khoảng trống công trình liên quan.
   - Rào chắn: Ưu tiên tuyệt đối nguồn bài báo chính thức; tách biệt bằng chứng trực tiếp và lân cận; không bịa đặt trích dẫn/DOI; ghi vết truy vấn hệ thống.

---

## 4. Danh Mục 8 Dự Án Và Cấu Hình Cục Bộ (8-Project Inventory & Configuration)

Đã khởi tạo và xác thực thành công tệp kiểm kê `NATURE_SKILLS_8PROJECT_INVENTORY.json` trên đúng 8 dự án authoritative:
1. `long-range-temporal-alignment` (canonical: `Long-Range-Representation`)
2. `DSC-TinDipLaPo`
3. `ds314-evidence-grounded-retrieval`
4. `ds201-temporal-retrieval-learning`
5. `ds111-retrieval-data-analysis`
6. `DS335-VLM-Research`
7. `evidence-grounded-text-agent`
8. `general-reasoning-fact-checking` (Source: `QA-TEXT/FactChecking`)

Toàn bộ 8 dự án đã được cập nhật ma trận kích hoạt `nature_capabilities` cùng siêu dữ liệu miền (`domain_metadata`) tại `.agents/agent-system-v2.json` và đăng ký vào `.agents/capabilities.json`.

---

## 5. Kết Quả Kiểm Thử Canary Tương Đồng (Canary Suite & Parity Acceptance)

- **Quy mô kiểm thử**: 8 dự án × 2 tác tử (ANTI & CODEX) × 5 ca canary = **80 ca định tuyến**.
  - C1 (Thống kê hạt giống) $\rightarrow$ `AUDIT -> audit:nature-statistics` (PASS 16/16)
  - C2 (Phản biện báo cáo) $\rightarrow$ `AUDIT -> audit:nature-reviewer` (PASS 16/16)
  - C3 (Bản quyền dữ liệu) $\rightarrow$ `RESEARCH -> research:data-evidence-gate` (PASS 16/16)
  - C4 (Tính mới / tài liệu) $\rightarrow$ `RESEARCH -> research:academic-evidence-search` (PASS 16/16)
  - C5 (Điều khiển âm - sửa lỗi code) $\rightarrow$ `IMPLEMENT -> Không kích hoạt Nature` (PASS 16/16, Zero False Positives)
- **Tỷ lệ vượt qua**: **80/80 (100.0%)**.
- **Sai số phân loại**: `False Positives = 0`, `False Negatives = 0`.
- **Kiểm thử vòng đời (Lifecycle Canaries)**: Vượt qua 4/4 kịch bản, trong đó kịch bản `PRE_REPORT_FREEZE` thực hiện chuỗi nối tiếp có kiểm soát: `statistics -> reviewer` (chiều sâu chuỗi tối đa = 2).
- **Tính tương đồng ANTI & CODEX**: Khớp 100% về ngữ nghĩa định tuyến, phân bổ pha, và rào chắn thực thi.

---

## 6. Hiệu Quả Ngữ Cảnh (Context Efficiency)

- Cơ chế tải lười 2 giai đoạn (two-stage lazy loading): Bộ định tuyến chạy bằng Python cục bộ (< 2ms) trước khi nạp tệp `SKILL.md`.
- Độ hồi quy ngữ cảnh khởi tạo (Startup Context Regression): **0,0%** (không có token thừa nào bị nạp vào context khi bắt đầu phiên).
- Tài liệu tham chiếu chi tiết (vendor references) nằm cách ly tại `_vendor/nature-skills/` và chỉ được truy xuất khi có yêu cầu tường minh.

---

## 7. Trạng Thái Các Cổng Kiểm Định (Quality Gates)

- `scripts/validate_project.py` $\rightarrow$ **PASS** (66 files, 10 DOCX/source pairs, smoke pipeline).
- `scripts/validate_knowledge_vault.py` $\rightarrow$ **PASS** (79 Markdown files, 23 managed, 5 decisions, 2 references).
- `pytest tests/` $\rightarrow$ **80 passed in 1.32s**.
- `agent-system-v2 unit tests` $\rightarrow$ **13 passed in 0.180s**.
- `DS314, DS201, DS111, DS335, FactChecking validators` $\rightarrow$ **ALL PASS**.
