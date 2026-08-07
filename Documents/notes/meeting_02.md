# Ghi Chú Buổi Trao Đổi 02

## Quyết định

- Legal Text QA không còn là hướng chính vì dữ liệu Việt Nam thiếu, VLSP có agreement, input/task chưa chốt và phải đầu tư gold dataset lớn.
- NCKH/KLTN cần đóng góp kỹ thuật rõ hơn.
- Hướng ưu tiên mới: temporal alignment for long-range time representations, bắt đầu từ error analysis và limitations của arXiv:2503.04150.

## Chuyển thành công việc

1. Audit khả năng tái lập Ticktack.
2. Xác định lỗi công thức, objective, EWC-LoRA, data split và evaluator.
3. Dựng baseline nhẹ phù hợp 16 GB RAM/4 GB VRAM.
4. Chọn dataset sau khảo sát license/availability; không ingest vội.
5. Phát triển một cải tiến có giả thuyết và ablation rõ.

## Điều cần hỏi GVHD tiếp theo

- Mức đóng góp ưu tiên: representation, objective hay evaluation protocol?
- Có quyền truy cập GPU lab/dataset nội bộ hay không?
- Chấp nhận benchmark public nào làm task chính?
- Mốc kỳ vọng cho reproduction audit và proposal thí nghiệm?
