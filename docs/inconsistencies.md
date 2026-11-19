# Inconsistencies

Rà soát ngày 2026-08-23.

| ID | Xung đột trước chỉnh sửa | Trạng thái | Cách xử lý |
|---|---|---|---|
| I01 | README gọi `python` nhưng máy mặc định là Python 3.10, trong khi `pyproject.toml` yêu cầu Python ≥3.11 | Đã xử lý | Hướng dẫn dùng `py -3.12` và `PYTHONPATH=src` khi chưa cài package |
| I02 | Tài liệu đặt Qwen2.5-0.5B làm probe đầu tiên nhưng Qwen3-0.6B đã có model card Apache-2.0 và hỗ trợ Transformers ≥4.51 | Đã xử lý ở mức candidate | Qwen3-0.6B là candidate chính; Qwen2.5 giữ làm control, cả hai chưa qua resource probe |
| I03 | Hướng nghiên cứu chỉ xoay quanh Ticktack/hybrid encoder, thiếu use case Việt Nam và application I/O | Đã xử lý ở mức hoạch định | ADR 0004–0005 và scaffold request/response/lifecycle |
| I04 | 10 DOCX không có nguồn Markdown duy trì riêng | Đã xử lý | `docs/converted/` là nguồn tái dựng; 10/10 DOCX đã qua structural, round-trip và rendered-layout QA |
| I05 | Dataset VLSP, sách Lịch sử 12 và đề thi có giá trị nhưng registry chưa có terms/checksum | Chưa giải quyết | Giữ `hold`; không ingest trong phiên này |
| I06 | Ruff nằm trong extra `dev` nhưng chưa cài trên interpreter hiện tại | Chưa giải quyết, không chặn scaffold | Compile, unittest và validator là gate hiện tại; cài môi trường dev ở bước setup riêng |
