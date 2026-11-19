# Bộ Chuyển Đổi Kỹ Năng Nghiên Cứu (Nature-Skills Local Adapters)

## 1. Ranh Giới Quản Trị Tuyệt Đối (Authoritative Boundaries)

Mọi kỹ năng chuyển giao từ `Yuan1z0825/nature-skills` (Revision `84880815fb37317b3766bff2c2abba395b8993c3`, Giấy phép Apache-2.0) khi được tham chiếu trong dự án này bắt buộc phải tuân thủ nghiêm ngặt 7 nguyên tắc bất biến:

1. **Thẩm quyền tối cao**: `AGENTS.md`, quy tắc tri thức `Documents/HOME/Quy tắc tri thức.md` và `Documents/General/RULE.md` giữ thẩm quyền cao nhất. Không bất kỳ kỹ năng bên ngoài nào được phép ghi đè các khế ước thực nghiệm đóng băng.
2. **Không bịa đặt (Zero Hallucination)**: Tuyệt đối không tự suy diễn, ngụy tạo trích dẫn, DOI, số liệu mẫu ($n$), giấy phép hay chỉ số đánh giá.
3. **Phân rã mức độ bằng chứng**: Không bao giờ nâng diễn ngôn phái sinh thành bằng chứng canonical.
4. **Phân tầng dữ liệu**: Runtime artifacts $\rightarrow$ receipts/audits $\rightarrow$ decisions $\rightarrow$ documents.
5. **Fail-closed**: Mọi hành vi ghi tệp ra ngoài hoặc tải dữ liệu từ internet mà không có phê duyệt quality gate trước đều bị cấm.
6. **Vô hiệu hóa Auto-Update**: Cấm tuyệt đối tính năng tự động cập nhật trong SessionStart để bảo toàn 100% tính đơn định và tái lập của nghiên cứu.
7. **Ghim cố định mã phiên bản**: Ghim cứng commit SHA `84880815fb37317b3766bff2c2abba395b8993c3`.

---

## 2. Danh Mục Kỹ Năng Đã Thẩm Định

- **nature-statistics** (`ADAPT_PATTERN_ONLY`): Áp dụng chuẩn kiểm toán thống kê Nature (effect sizes, bootstrap CI, phân định rõ independent experimental unit vs technical replicates/seeds, kiểm tra tương tác 2×2). Không thay đổi công thức canonical hiện hành của dự án.
- **nature-reviewer** (`ADOPT_LOCAL`): Cung cấp góc nhìn phản biện độc lập (3-blind reviewers) trên bản thảo hoặc báo cáo đã đóng băng trước khi nộp hoặc công bố tuần.
- **nature-data** (`ADAPT_PATTERN_ONLY`): Vận dụng khung kiểm toán tính sẵn sàng của mã nguồn và dữ liệu (FAIR), tuyệt đối không để hướng dẫn chung của tạp chí vượt qua các hợp đồng quyền dữ liệu cục bộ (`rights_override.json`).
- **nature-academic-search** (`ADAPT_PATTERN_ONLY`): Vận dụng nguyên tắc phân tầng nguồn tài liệu (Tier-1 Peer-Reviewed vs Tier-2 Preprints vs Tier-3 Gray Literature).
