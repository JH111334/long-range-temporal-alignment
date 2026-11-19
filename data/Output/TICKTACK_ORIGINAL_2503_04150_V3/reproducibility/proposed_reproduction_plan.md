# Kế Hoạch Đề Xuất Tái Lập Thích Ứng (Adapted Re-Implementation Plan)

> **CẢNH BÁO QUẢN TRỊ:** Đây là tài liệu quy cách kỹ thuật để chuẩn bị và xin phê duyệt. **Tuyệt đối KHÔNG tự động phát động huấn luyện GPU hay đẩy Kaggle kernel khi chưa có sự phê duyệt rõ ràng từ Người Nghiên Cứu và GVHD.**

## 1. Mục Tiêu Khoa Học
Tái lập trung thực **phương pháp gốc của Xue Han et al. (arXiv:2503.04150v3)** trên cơ sở dữ liệu `TempUN_Small` và mô hình `Qwen2.5-0.5B-Instruct` nhằm cung cấp mốc đối chứng phương pháp sạch (`ADAPTED_TICKTACK_ORIGINAL`), phân lập với kiến trúc nội bộ `Dual-Cycle MaT-LoRA`.

## 2. Ranh Giới Kỹ Thuật (Specification)
- **Mô hình nền:** `Qwen/Qwen2.5-0.5B-Instruct` (revision: `7ae557604adf67be50417f59c2c2f167def9a775`), frozen $W_0$.
- **Biểu diễn thời gian:** Tọa độ cực Lục Thập Hoa Giáp $	o$ Descartes $(x, y) 	o TE(x), TE(y)$ theo đúng Phương trình (1) - (4) của bài báo gốc.
- **Vị trí tiêm:** Cộng trực tiếp vào Input Embedding $h'_i = h_i + TE(x_i) + TE(y_i)$ (Eq 5) tại token thời gian.
- **Cấu hình Adapter:** Chuẩn LoRA ($r=8, lpha=16$) trên attention / MLP projection.
- **Hàm mất mát:** $\mathcal{L} = \mathcal{L}_{NTP} + \sigma \mathcal{L}_{temporal}$ với $\sigma = 0.5$ (Eq 8, 10).
- **Phần cứng dự kiến:** 1 x Tesla T4 GPU (Kaggle), thời lượng ước tính ~30-45 phút / seed.

## 3. Cổng Phê Duyệt Bắt Buộc Trước Khi Thực Thi
1. **Gate C1 (Approval Gate):** Người nghiên cứu ký duyệt kế hoạch và phân bổ tài nguyên.
2. **Gate C2 (Deterministic CPU Dry-Run):** Chạy kiểm thử forward pass và hàm mất mát $TE(x), TE(y)$ trên CPU cục bộ.
3. **Gate C3 (Canary 1-seed 45 steps):** Xác minh số học hữu hạn ($0$ NaN/Inf) trên GPU trước khi mở 3 seed đầy đủ.
