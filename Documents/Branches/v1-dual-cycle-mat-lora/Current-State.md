---
loại: "trạng-thái"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
cập_nhật_lần_cuối: "2026-10-13"
---

# Trạng Thái Hiện Tại: Nhánh V1 (Starting Anchor M0-FC40)

- **Trạng thái nhánh**: `COMPLETED_STARTING_ANCHOR` (Đã hoàn tất đo đạc, đóng băng làm mốc cơ sở).
- **Mã thực nghiệm**: `M0-FC40`.
- **Hạ tầng thực thi**: Kaggle Tesla T4 GPU (Kernel `qh0205/lrr-fc105-m0-fc40-3seed`).
- **Thời gian hoàn thành**: 13/10/2026 01:10 UTC (08:10 Asia/Ho_Chi_Minh).

## Bảng tổng hợp số liệu thực nghiệm V1 (3 Seed)

| Seed | Chặng 1 (`1947`) Loss | Chặng 2 (`1984`) Loss | Chặng 3 (`2022`) Loss | Strict Accuracy | Valid Accuracy | Invalid Syntax Rate | BWT (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 1** | $1.8663$ | $1.7233$ | $1.7417$ | $2.22\%$ | $33.33\%$ | $93.33\%$ | $-3.125\%$ |
| **Seed 2** | $1.8736$ | $1.7121$ | $1.7333$ | $2.22\%$ | $33.33\%$ | $93.33\%$ | $-3.125\%$ |
| **Seed 3** | $1.8577$ | $1.7258$ | $1.7406$ | $2.22\%$ | $33.33\%$ | $93.33\%$ | $-3.125\%$ |
| **Trung bình** | $\mathbf{1.8659 \pm 0.008}$ | $\mathbf{1.7204 \pm 0.007}$ | $\mathbf{1.7385 \pm 0.004}$ | $\mathbf{2.22\% \pm 0.00\%}$ | $\mathbf{33.33\% \pm 0.00\%}$ | $\mathbf{93.33\% \pm 0.00\%}$ | $\mathbf{-3.125\%}$ |

## Phán quyết nghiệm thu V1

- **Tính toàn vẹn số học**: ĐẠT (`PASS 100%`). Toàn bộ 7.383 bước huấn luyện không có NaN/Inf.
- **Tính toàn vẹn mô hình nền**: ĐẠT (`PASS 100%`). Không rò rỉ gradient sang trọng số $W_0$.
- **Tuân thủ khế ước đánh giá**: ĐẠT (`PASS 100%`). Đo trên đúng 180 ca MCQA đóng băng.
- **Đóng băng**: Nhánh V1 không nhận thêm thay đổi mã nguồn. Mọi nỗ lực tối ưu tiếp theo chuyển giao sang [[Documents/Branches/v2-targeted-continual-alignment/HOME|Nhánh V2]].
