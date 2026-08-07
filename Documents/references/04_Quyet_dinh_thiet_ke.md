# Quyết Định Thiết Kế

## D01. Internal time

Dùng `HistoricalYear(era, year)` và quy đổi nội bộ sang astronomical numbering: 1 BCE = 0, 2 BCE = -1, 1 CE = 1. API không nhận display year 0.

## D02. Encoder taxonomy

- Scalar và Fourier: baseline.
- Published Ticktack: đúng phương trình đã in, giữ va chạm để phân tích.
- Operational radial: giả định bổ sung để phân biệt chu kỳ.
- Hybrid cycle–monotonic: phương pháp đề xuất.

## D03. Objective

Giữ Eq. 8 như baseline unnormalized. Phương án nghiên cứu dùng mean trên unique pairs, không self-pair, inter-margin và ghi rõ class weighting.

## D04. EWC

EWC chỉ hợp lệ khi artifact ghi current/reference/Fisher và danh sách trainable parameters. Phải có LoRA-only, EWC-only hoặc retention baseline trước khi giữ thành phần này.

## D05. Data

Group/time/source split là mặc định. Candidate dataset không được tải trước evidence gate. Synthetic fixture chỉ kiểm thử hợp đồng.

## D06. Phần cứng

Máy đo ngày 2026-08-07: Python 3.12.6, NVIDIA GeForce RTX 2050 4096 MiB; ngân sách người dùng 16 GB RAM. Model fit là giả thuyết cho tới khi resource probe ghi peak.

## D07. Framework

Core thuần Python. Torch/Transformers/PEFT/Datasets là optional adapter. JSON + hash là tracking baseline; chỉ thêm MLflow/DVC khi số run/data artifact đủ lớn để bù chi phí vận hành.
