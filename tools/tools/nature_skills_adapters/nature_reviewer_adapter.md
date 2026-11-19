---
name: nature-reviewer-adapter
version: 1.0.0
upstream_repo: https://github.com/Yuan1z0825/nature-skills
upstream_commit: 84880815fb37317b3766bff2c2abba395b8993c3
license: Apache-2.0
mode: ADOPT_LOCAL
auto_update: false
---

# Nature Reviewer Local Adapter Wrapper

## 1. Ranh Giới Bắt Buộc Của Dự Án
- Chỉ được áp dụng trên snapshot tài liệu/báo cáo đã đóng băng (ví dụ: `WEEK1TARGET/report/`).
- Tuyệt đối không được can thiệp vào quá trình sinh kết quả thực nghiệm thô hay ghi đè mã băm canonical.
- Mọi nhận xét phản biện (Concerns) phải đi kèm:
  + `claim_pointer`: Trích dẫn chính xác dòng/đoạn trong báo cáo.
  + `evidence_pointer`: Liên kết đến biên nhận JSON hoặc mã băm SHA-256 cụ thể.
  + Phân loại rõ ràng `Major Concerns` (Blocking Yes/No) và `Minor Comments`.
- Đánh giá khách quan trên 5 trục:
  1. `originality` (Tính nguyên bản, phân định rõ prior art)
  2. `scientific importance` (Tầm quan trọng khoa học)
  3. `technical soundness` (Tính chuẩn xác kỹ thuật và đối chứng)
  4. `readability for nonspecialists` (Tính mạch lạc, ngôn ngữ học thuật)
  5. `replication feasibility` (Khả năng tái lập 100%)
