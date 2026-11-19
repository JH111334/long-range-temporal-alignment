# Checkpoint — Capability compute và SourceAPI

## Định danh

- Project: long-range-temporal-alignment
- Thời điểm: 2026-09-06, Asia/Bangkok.
- Phiên bản: V2
- Owner: Hoàng Nguyễn Minh Khang.
- Git revision: c4302f4; worktree dirty trước phiên, không commit hoặc reset.
- Stage: capability-based compute và temporal SourceAPI boundary.
- System profile: Windows 11 build 26200; Python 3.13 qua uv, Python 3.12 cho
  vault CLI; Intel i5-12450H, RAM 15,7 GiB, RTX 2050 4 GiB.
- Mức bằng chứng: static_check|smoke_test
- Quyết định: ACCEPT hai contract; không phải model/corpus benchmark.

## Cấu hình, đầu vào và lệnh

- Data/model/config revision, seed, qrels và evaluation contract: NOT_APPLICABLE.
- Network, provider credential, Kaggle upload, GPU training và remote inference: NOT_RUN.
- `uv lock --check`: PASS.
- `uv run pytest -q -p no:cacheprovider --basetemp .tmp/pytest-final`: 50 PASS.
- `uv run ruff check SourceAPI tests/test_source_api.py`: PASS.
- `uv run python scripts/validate_project.py`: PASS, 66 required file và 10
  DOCX/source pair.
- `py -3.12 scripts/validate_knowledge_vault.py`: PASS trước promotion.
- `py -3.12 SourceAPI/pipeline.py contract`: exit 0, `NO_PROVIDER_ENABLED`.
- Full-repository Ruff: FAIL do 458 lỗi legacy/generated ngoài SourceAPI; không dùng
  kết quả targeted để tuyên bố toàn repository lint sạch.

## Artifact và kết quả

- `SourceAPI/contracts.py`: SHA-256
  `de9c5c573b94df29e21c1c4e6c247d73faeb288b1b65d6f8e69d3900897bfe70`.
- `SourceAPI/pipeline.py`: SHA-256
  `10bc480558bd26d07c5db0eb6351dcba1001cd6bd959ea3e8f284638ab30eaf5`.
- Fake client giữ raw temporal expression, canonical interval tùy chọn, provenance
  và partial failure. Không có raw provider payload trong internal record.

## Giới hạn, replay và rollback

- Chưa có source thật, corpus ingest, qrels, latency/VRAM, paired delta hoặc
  production claim.
- Replay bằng các lệnh targeted ở trên. Rollback SourceAPI không ảnh hưởng offline
  baseline; external artifact sai contract bị đánh invalid và không đăng ký.
- Next gate: provider-specific fixture + terms + explicit integration request; external
  compute chỉ mở sau hash-bound manifest và cùng evaluator local.
