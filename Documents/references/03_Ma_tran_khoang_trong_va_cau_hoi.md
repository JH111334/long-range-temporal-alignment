# Ma Trận Khoảng Trống Và Câu Hỏi Nghiên Cứu

| Câu hỏi | Baseline tối thiểu | Can thiệp | Tiêu chí hỗ trợ giả thuyết | Tiêu chí bác bỏ |
|---|---|---|---|---|
| RQ1: Phần nào của Ticktack tái lập được? | Published equations + next-token baseline | Operational radial encoder | Kết quả ổn định qua seed/split, config đầy đủ | Hiệu ứng biến mất khi chống leakage hoặc không thể định nghĩa encoder nhất quán |
| RQ2: Hybrid có giảm alias 60 năm? | Scalar, Fourier, published Ticktack | Monotonic axis + multi-period features | Collision giảm, chronological correlation tăng và QA không giảm | Chỉ metric hình học tăng nhưng QA/generalization giảm |
| RQ3: Loss chuẩn hóa ổn định hơn? | Unnormalized Eq. 8 | Mean unique pairs + inter margin | Ít nhạy batch/class count, gradient hữu ích | Không cải thiện stability hoặc làm giảm downstream |
| RQ4: Có khái quát dài hạn? | Random split để đối chiếu | Time/source/entity-held-out | Hiệu ứng giữ trên BCE/CE, xa train years, đa nguồn | Chỉ tăng trên random split/near-duplicate |
| RQ5: EWC có cần thiết với LoRA? | LoRA không EWC | EWC adapter-aware và retention eval | Tăng retention với temporal gain tương đương | Penalty gần zero hoặc không hơn regularization đơn giản |

## Đóng góp khả thi

1. Reproduction audit định lượng các điểm thiếu đặc tả của Ticktack.
2. Bộ contracts và evaluator tách lỗi representation, objective, parser và leakage.
3. Hybrid cycle–monotonic encoding; chỉ gọi là đề xuất cho đến khi benchmark xác nhận.
4. Protocol đánh giá long-range temporal alignment dưới giới hạn phần cứng nhỏ.
