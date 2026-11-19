# Checkpoint — Living Knowledge Vault v1

- Project: long-range-temporal-alignment
- Thời điểm: 2026-09-06, Asia/Bangkok.
- Phiên bản: V1
- Loại: checkpoint cấu trúc và knowledge-vault.
- Git/worktree: dirty trước phiên; không commit hoặc reset.
- Mức bằng chứng: static_check|smoke_test; không tuyên bố benchmark.

## Định hướng kỹ thuật

1. Documents là vault root; .obsidian được chuyển lên Documents/.obsidian.
2. Hai track trong 12_PROJECT-CONTEXT vẫn giữ nguyên.
3. Metadata chỉ thêm cho context, design, experiment plan và source registry.

## Lệnh và kết quả

- py -3.12 scripts/validate_knowledge_vault.py: PASS.
- py -3.12 scripts/validate_project.py: PASS.
- py -3.12 -m pytest -q -p no:cacheprovider --basetemp .tmp/pytest-vault: 48 passed.
- uv lock --check: PASS.

## Giới hạn và lineage

- Backup và before/after SHA-256: .migration-backups/living-knowledge-vault-20260906-v1/manifest.json.
- Dataset, model, qrels và measured comparison: NOT_RUN.
- Không promote delta 0,5, Dual-Cycle hoặc bất kỳ threshold nào.

## Việc tiếp theo

- Khóa data-source dossier, evaluator và lexical interval baseline trước R0-R3.
