---
name: project-context-router
description: Dùng khi bắt đầu task hoặc đổi phạm vi trong repo temporal alignment để định tuyến ngữ cảnh Documents vừa đủ; dừng đọc khi đã rõ mục tiêu, quyền, hành động tiếp theo và cách kiểm chứng. Không thay workflow nghiên cứu domain.
---

# Project Context Router

## Preconditions

Xác nhận repo root và đọc `AGENTS.md`. Giữ nguyên thay đổi chưa commit.

## Context Router và Ranh Giới Mục Tiêu (Goal Ladder)

1. Đọc `Documents/HOME/Trang chủ.md`, rồi `Documents/Decision-Making/12_PROJECT-CONTEXT.md`.
2. Kiểm tra mục tiêu chi phối tại `Documents/Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105.md` và `Documents/Global/GOAL_BRANCH_STAGE_CROSSWALK.md` để xác định rõ `goal_id` (`G1`, `G2`, hoặc `G3`) của phiên làm việc.
3. Chỉ đọc tiếp các ADR, Reference, mã nguồn hoặc cấu hình gắn trực tiếp với `goal_id` đã xác định. Chỉ mở `Documents/General` khi cần bằng chứng lịch sử và đã đọc `General/RULE.md`.

## Procedure và Tool Requirements

Đọc trực tiếp file khi biết path. Dùng capability registry tại `.agents/capabilities.json` cho tìm kiếm lớn hoặc công cụ ngoài; `unverified` không được gọi. Không quét toàn repo hoặc nạp mọi skill theo mặc định nhằm tiết kiệm token budget và tránh tràn context.

## Hard Guards

- Dừng đọc ngay khi đã biết objective (`goal_id`), current state, authority, next valid action và acceptance/verification. Giữ `UNKNOWN` nếu nguồn thiếu hoặc xung đột; không suy trạng thái từ filename.
- Không nạp `Documents/Documenting/` khi lập trình thông thường. Khi được giao trình bày/paper/portfolio/hồi cứu, đọc `Documents/Documenting/RULE.md`, rồi đúng file tổng hợp/thành phần và nguồn canonical cần đối chiếu; giữ significance gate, provenance, authority và nhãn bắt buộc `GOAL1_YEAR_CLOSED_ANSWER` cho các tài liệu Stage 03.

## Verification, Writeback và Stop

Ghi ngắn các nguồn đã đọc và phần chưa xác minh. Skill này chỉ định tuyến, không tự ghi Documents. Chuyển sang `project-change-verification` khi bắt đầu sửa.

## Tài liệu có tiền đề

Theo mục “Tài liệu gọn và có ngữ cảnh” trong `AGENTS.md`: mở đúng file tiền đề và đề tài trước khi sửa; dẫn nguồn thay chép lại. Chỉ đọc phụ thuộc cần cho hành động/kiểm chứng; giữ gate, bằng chứng và lịch sử.
