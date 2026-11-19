# Skills, Tools Và Workflow Đã Chọn

## Luồng nghiên cứu tài liệu

1. `document-project-research-orchestration`: điều phối source–report–code consistency.
2. `research-source-routing`: HyperAI/Papers with Code để khám phá; paper/trang tác giả/trang chính thức để xác minh.
3. `document-to-markdown-with-markitdown`: tạo working text từ PDF.
4. `pymupdf-pdf-analyzer`: trích xuất theo trang và render khi PDF hai cột/công thức làm Markdown sai thứ tự.

Quyết định: MarkItDown không đủ tin cậy cho bảng/công thức Ticktack; PDF gốc + ảnh trang PyMuPDF là source of truth.

## Luồng tài liệu giao

1. Viết nguồn Markdown ngắn theo từng vấn đề.
2. `author-validate-docx`: build DOCX, round-trip, LibreOffice render, PyMuPDF layout check.
3. `weekly-research-html-report`: brief tĩnh tự chứa với source/license/decision matrix.

Kết quả: 10 DOCX qua structural/rendered QA và 1 HTML qua structural gate.

## Luồng code/repository

- `python-code-standard`: contracts nhỏ, type hints, docstring, fail-closed validation.
- Understand Anything được xem xét; do repository tham chiếu nhỏ và workflow graph yêu cầu ghi vào repo/sub-agent, dùng fallback targeted scan trên README, workflow, contracts, schemas, pipeline và tests.
- `github`/`yeet`: kiểm tra remote, commit, push, public/MIT/SHA/CI. GitHub App được dùng để thêm workflow khi OAuth token thiếu scope `workflow`.
- Local skill `temporal-alignment-research`: mã hóa data, representation, objective, model/resource và evaluation gates cho các phiên sau.
- Local skill `temporal-research-planning`: chọn use case Việt Nam-first, problem contract,
  lifecycle, compute capability profile, risk register và stop condition.

## Công cụ chỉ thêm khi có nhu cầu

| Công cụ | Baseline hiện tại | Điều kiện thêm |
|---|---|---|
| Ruff/Pytest | compile + unittest + validator | Khi dependencies dev được cài/lock |
| MLflow | JSON registry + hash | Nhiều người/máy và nhiều run |
| DVC | manifest + checksum | Dataset lớn, có remote storage ổn định |
| Hydra | JSON config | Ma trận cấu hình tăng mạnh |
| Torch/Transformers/PEFT | Protocol adapters | Data/resource gate đạt |

Không cài framework nặng chỉ để scaffold. Phiên bản môi trường phải được khóa tại thí nghiệm model đầu tiên.

Checkpoint ngày 2026-08-23: `py -3.12` + `PYTHONPATH=src` chạy 27 unit test đạt; Ruff chưa cài nên không được tuyên bố đã lint.
