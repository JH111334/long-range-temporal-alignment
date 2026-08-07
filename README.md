# Long-Range Temporal Alignment

Scaffold nghiên cứu để tái lập có kiểm soát và kiểm tra các biểu diễn thời gian dài hạn cho mô hình ngôn ngữ. Trọng tâm ban đầu là Ticktack (arXiv:2503.04150), sau đó so sánh với các baseline trung lập về lịch và một biểu diễn lai đơn điệu–chu kỳ.

## Trạng thái

- Đây là scaffold và smoke baseline, chưa phải kết quả benchmark hay tuyên bố SOTA.
- Dữ liệu trong `data/synthetic/` là dữ liệu tổng hợp dùng kiểm thử hợp đồng.
- TempLS chưa có bản phát hành chính thức được xác minh; không dùng số liệu mô phỏng để thay thế benchmark này.
- Mọi dataset thật phải qua manifest nguồn, ngày truy cập, điều khoản, checksum, người rà soát và vai trò split.

## Câu hỏi nghiên cứu

1. Có thể tái hiện phần nào của Ticktack khi paper thiếu mã nguồn, dữ liệu TempLS và một số siêu tham số?
2. Biểu diễn lai giữ trục năm đơn điệu và nhiều chu kỳ có giảm va chạm cách nhau 60 năm so với công thức đã công bố không?
3. Loss được chuẩn hóa theo số cặp có ổn định hơn khi đổi batch size không?
4. Cải thiện có giữ được khi chia theo thời gian, nguồn, thực thể, ngôn ngữ và độ hạt không?
5. EWC tác động lên tham số nào khi base model bị đóng băng và chỉ LoRA được huấn luyện?

## Luồng hệ thống

`Nguồn + manifest → Chuẩn hóa thời gian → Chia nhóm chống leakage → Encoder → Objective → Model adapter → Evaluator theo lát cắt → Artifact + provenance`

Các hợp đồng dữ liệu nằm trong `src/temporal_alignment/contracts.py`; schema trao đổi nằm trong `schemas/`. Pipeline smoke không tải model hoặc dataset từ mạng.

## Baseline theo tầng phần cứng

| Tầng | Mục tiêu | Phần cứng |
|---|---|---|
| 0 | Kiểm thử lịch, encoder, loss, split và metric trên dữ liệu tổng hợp | CPU, Python chuẩn |
| 1 | Tokenizer/frozen-embedding probe với model nhỏ, sau khi đo RAM/VRAM | 16 GB RAM, RTX 2050 4 GB |
| 2 | LoRA model nhỏ, micro-batch 1, sequence ngắn; chỉ chạy nếu resource probe đạt | Máy cục bộ, kết quả phải ghi peak RAM/VRAM |
| 3 | Tái lập 3B–13B và multi-seed | GPU phòng lab hoặc cloud; không mặc định phù hợp máy cục bộ |

Model ứng viên tầng 1 là Qwen2.5-0.5B-Instruct vì giấy phép Apache-2.0 và kích thước nhỏ hơn các model trong paper. Đây là lựa chọn thử nghiệm, chưa phải xác nhận vừa 4 GB VRAM.

## Chạy nhanh

```powershell
python -m temporal_alignment --config configs/smoke.json --output artifacts/smoke_run.json
python -m unittest discover -s tests -v
python scripts/validate_project.py
```

Khi chưa cài package, chạy từ root repository với:

```powershell
$env:PYTHONPATH = "src"
python -m temporal_alignment --config configs/smoke.json --output artifacts/smoke_run.json
```

## Bằng chứng và tài liệu

- `Documents/`: tài liệu tiếng Việt và brief nghiên cứu.
- `registries/`: trạng thái nguồn dữ liệu, model và thí nghiệm.
- `docs/adr/`: quyết định kiến trúc và ranh giới tái lập.
- `AGENTS.md`, `Workflows/flowinfo.md`: quy tắc cho các phiên làm việc sau.

## Giấy phép

Mã nguồn được phát hành theo MIT. Dataset, model và paper giữ giấy phép/điều khoản của nguồn tương ứng; MIT của repository không cấp lại quyền cho các tài sản đó.
