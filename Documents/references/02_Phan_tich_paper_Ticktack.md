# Phân Tích Paper Ticktack

Nguồn: [arXiv:2503.04150v3](https://arxiv.org/abs/2503.04150), bản ngày 2025-10-21. Phân loại: preprint.

## Đóng góp paper tuyên bố

1. Chuyển năm Gregorian/BCE sang 60 lớp can-chi.
2. Biểu diễn lớp chu kỳ bằng góc trong tọa độ cực, dùng bán kính để phân biệt các năm cùng lớp.
3. Chuyển sang tọa độ Descartes, áp dụng sin/cos rồi cộng vào embedding đầu vào.
4. Post-train bằng next-token loss, temporal alignment loss và EWC; dùng LoRA.
5. Tạo TempLS: 137.090 cặp QA tiếng Trung từ Baidu Baike, khoảng 75.000 BCE–2025 CE; 132.830 train và 4.260 test.

## Kết quả paper báo cáo

Trên TempLS, Ticktack báo cáo Qwen2.5-7B đạt 82,82% zero-shot và 83,29% few-shot; LLaMA2-13B đạt 65,28% và 70,18%, so với base 32,63% và 33,97%. Cụm “tăng trung bình 34,43%” phù hợp với trung bình chênh lệch **điểm phần trăm**, không phải mức tăng tương đối.

Đây là số liệu của paper, chưa được nhóm tái lập.

## Giới hạn tác giả tự nêu

- Chỉ xử lý độ hạt năm; tháng/ngày để lại cho tương lai.
- TempLS còn nhỏ và chỉ lấy từ một nguồn; cần benchmark đa nguồn để tăng khả năng khái quát và độ tin cậy.

## Error analysis độc lập

| Vấn đề | Bằng chứng trong paper | Hệ quả tái lập | Kiểm thử dự án |
|---|---|---|---|
| Bán kính chưa đặc tả | `r = alpha + beta × theta`, alpha/beta chỉ cho khoảng 0,5–1,0 | Nếu alpha/beta là hằng số, 1965 và 2025 cùng góc và cùng bán kính, trái mô tả | Alias-collision test 60 năm |
| Loss phụ thuộc batch | Intra/inter dùng tổng cặp, không chuẩn hóa | Thay batch/class count làm đổi độ lớn loss | Nhân đôi batch và so loss |
| Delta bằng 1 | Paper nói delta bị ảnh hưởng bởi batch và đặt 1 | Hệ số `(1-delta)` triệt tiêu inter-class term | Kiểm tra gradient/term contribution |
| EWC + LoRA mơ hồ | Base weights đóng băng; paper không nêu Fisher áp lên tham số nào | EWC trên base frozen không tạo cập nhật; adapter không có reference rõ | Contract current/reference/Fisher/trainable names |
| Thiếu cấu hình | Không nêu lambda EWC, LoRA rank/alpha/dropout/target modules, seed, Fisher estimator | Không thể tái lập chính xác | Gắn nhãn operational assumption |
| Data chưa phát hành | Không thấy URL/license TempLS trong paper | Không thể audit source, duplicate, contamination | Status hold; không ingest |
| Split yếu | Chỉ nêu train/test, không nêu group/time/source split | Nguy cơ leakage theo thực thể/sự kiện/nguồn | Group/time/source-held-out |
| Độ tin cậy thống kê | Không báo multi-seed, CI hoặc significance | Không biết variance | Multi-seed và bootstrap CI |
| Ablation thiếu | Chủ yếu xét encoding; thiếu loss-only/EWC-only/simple baseline | Không quy trách nhiệm được hiệu ứng | Ma trận ablation đầy đủ |
| Evaluator format | TimeLLaMA bằng 0 trên TempLAMA do không hợp format | Trộn năng lực với lỗi parser | Báo invalid-output rate riêng |
| T-SNE | Hình cụm embedding mang tính minh họa | Không chứng minh cải thiện nhân quả | Metric định lượng trước visualization |
| Bias lịch | 60-year sexagenary cycle là lựa chọn văn hóa | Có thể không chuyển tốt sang ngôn ngữ/lịch khác | Fourier/hybrid culture-neutral baseline |

## Kết luận kỹ thuật

Mục tiêu phù hợp không phải “chạy lại nguyên paper”, vì reproduction package chưa đủ. Dự án nên đóng góp bằng một reproduction audit có kiểm thử, một baseline trung lập mạnh, objective chuẩn hóa và đánh giá extrapolation/leakage có kiểm soát.
