# Checkpoint — Kiểm tra seed chia nhóm

## Định danh

- Project: long-range-temporal-alignment
- Thời điểm: 2026-09-23, Asia/Bangkok.
- Phiên bản: V1
- Owner: phiên Codex theo yêu cầu người dùng.
- Git revision: c4302f49eef825c5ce9f84a076e2be3707b0a0da; worktree dirty trước khi sửa, chưa commit.
- Stage: kiểm tra contract chia nhóm trên CPU.
- System profile: Windows/PowerShell; Python 3.12 qua `py -3.12`; CPU, RAM, GPU, package và service version không thu thập.
- Mức bằng chứng: `static_check|smoke_test`.

## Cấu hình và kiểm chứng

- Data/model revision, manifest, evaluation contract và seed thực nghiệm: `NOT_APPLICABLE`.
- Trước sửa: `py -3.12 scripts/validate_project.py` PASS; `py -3.12 -m pytest -q -p no:cacheprovider --basetemp .tmp/pytest-agent-refactor` 50 PASS; `py -3.12 scripts/validate_knowledge_vault.py` PASS.
- Sau sửa: cùng ba lệnh PASS; 51 test PASS; vault có 40 Markdown, 12 managed node, 3 decision và 2 reference trước khi thêm log này.
- `assign_group_split` bác bỏ seed rỗng hoặc chỉ có khoảng trắng bằng `ContractError`; `tests/test_splits.py` kiểm tra trường hợp này.
- Artifact: `src/temporal_alignment/splits.py` SHA-256 `b8ab22eb3c969a42d7a49d6a75797e37dc757faa0bb1da4506d0dbba85c2eb3`; `tests/test_splits.py` SHA-256 `9256041d93f844bb75c599cc9e45a8494f7f9e1dabd5bad10c2b697afd5a2326`.
- Trạng thái tri thức: contract hiện hành giữ nguyên; checkpoint ghi một validation cục bộ, không nâng trạng thái nghiên cứu.

## Giới hạn và hoàn tác

- Không tải dữ liệu/model, không chạy GPU hoặc benchmark; kết quả test không đo chất lượng mô hình.
- Các thay đổi khác trong worktree có trước hoặc nằm ngoài phạm vi checkpoint này.
- Replay bằng ba lệnh trên. Hoàn tác bằng cách bỏ riêng điều kiện `seed.strip()` và test tương ứng, rồi chạy lại ba gate; không reset worktree.
