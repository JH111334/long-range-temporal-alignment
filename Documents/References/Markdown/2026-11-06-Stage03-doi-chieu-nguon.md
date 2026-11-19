---
loại: "tham-chiếu"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: false
độ_tin_cậy: "exploratory"
chủ_đề: "Đối chiếu Stage 03"
cập_nhật_lần_cuối: "2026-11-06"
nguồn: "https://chatgpt.com/share/6aedcbe7-d488-83ec-8f17-2a26a1352765"
---

# Đối chiếu nguồn và Stage 03

Ghi chú tra cứu ngày 06/11/2026 cho G1 YEAR-only Closed QA. Tiền đề là [ADR 23](../../Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105.md), [trạng thái V2](../../Branches/v2-targeted-continual-alignment/Current-State.md) và [chat nghiên cứu](https://chatgpt.com/share/6aedcbe7-d488-83ec-8f17-2a26a1352765). Mở [mục lục PDF](../INDEX.md) theo câu hỏi đang làm.

## Kiểm từ hiện vật lên kết luận

| Quan sát đã đối chiếu | Ý nghĩa với nghiên cứu |
|---|---|
| Launcher local trùng SHA trong receipt deploy. Đường train đặt padding phải, đường eval đặt padding trái ở dòng 593. Eval chèn ở chỉ số tensor 14 trong batch đã pad/truncate. | Mô tả padding phải trong audit cũ lệch với launcher đã pin. Vị trí 14 có thể rơi vào PAD, tiền tố hoặc phần nội dung khác tùy độ dài. Cần đo span năm và mask từng hàng trước khi kết luận tỷ lệ lỗi. Với padding trái, hidden state cuối vẫn là vị trí sinh tiếp khi chuỗi hợp lệ. |
| Scorer đọc logit nhóm Yes/No hoặc A–D. Nhãn gold nằm ngoài chuỗi gửi model nhưng hiện được dùng để chọn nhóm ứng viên, rồi chấm đúng/sai. | Đây là phép đo closed-choice có routing theo loại nhãn. Khi đối chiếu parser mới, lấy loại câu từ contract hoặc metadata và giữ gold cho phép chấm. Chia riêng kết quả restricted logits, free generation với parser nhãn, và ánh xạ đáp án số duy nhất nếu đánh giá thêm. |
| Arm 5 còn truyền theta/z vào dynamic core. CPU audit xác nhận nội dung 12 checkpoint và final khớp epoch 5. | Thử nghiệm bỏ Seam chỉ kiểm đường chèn tại input. Khi muốn đo toàn bộ temporal conditioning, tách Seam và core theo đối chứng đã khóa. Tensor khác nhau là chứng cứ nạp/huấn luyện, còn gain cần replay ghép cặp. |
| Receipt ghi train 104.130 NWP và eval 5.292 gồm 4.440 MCQA, 852 Rel. Prompt trùng nguyên văn bằng 0 nhưng fact-level chưa sạch và DEV đã dùng chọn cấu hình. | Có thể phát biểu chuyển định dạng khi không fine-tune trên MCQA/Rel, với đánh giá hiện tại trên DEV. Kết luận khái quát cần holdout độc lập theo thực thể–chỉ báo–thời gian và kiểm rò dữ kiện gần nhau. |

Nguồn local. [Launcher dòng 593, 608, 680–703](../../../scripts/run_stage03_targeted_arms4_5_5epochs_modal.py) · [Binding](../../../Reports/6-11Report/_POST_FORENSIC_OPEN_GATES_20261105/DEPLOYED_SOURCE_BINDING.json) · [CPU verdict](../../../Reports/6-11Report/_POST_FORENSIC_OPEN_GATES_20261105/CPU_GATE_VERDICT.json) · [Audit fact-level](../../../Reports/6-11Report/_POST_FORENSIC_OPEN_GATES_20261105/FACT_LEVEL_LEAKAGE_AUDIT.json).

## Đọc nguồn theo câu hỏi

- **Hướng chuyển tác vụ có gì mới?** TempUN §3 đã dùng NWP để fine-tune rồi dùng MCQA để đánh giá. Phần nghiên cứu cần kiểm ở Stage 03 là ảnh hưởng của cách đưa tín hiệu thời gian vào Qwen dưới cùng dữ liệu, ngân sách và scorer, kèm độ bền khi chuyển định dạng. [TempUN, trang 4](../Papers/doi-10.18653-v1-2024.findings-emnlp.953/PAPER.md).
- **So với Ticktack đến đâu?** Ticktack Eq. (5) cho công thức cộng vào input embedding, còn mapping token/padding và scorer cần nguồn triển khai cụ thể để xác nhận. §4.2 dùng TempLS để post-train và giữ dữ liệu giống nhau giữa w/ PT với Ticktack. Table 1 công bố TempUN zero-shot cho Qwen2.5-3B Base 59,25%, w/ PT 44,82%, Ticktack 64,96%. Giữ các điểm ở mức paper-reported. So sánh nội bộ dùng Base, NWP no-temporal và các Arm cùng evaluator. Gọi đường hiện tại là Ticktack-inspired khi nói về lựa chọn mã hóa. [Ticktack v3, trang 4 và 6](../Papers/arxiv-2503.04150/PAPER.md).
- **Dùng nguồn nào để kiểm injection?** FoNE v2 mã hóa số một token bằng Fourier. Stage 03 có đặc trưng FoNE/Can-Chi 40D và projection riêng. Khi hữu ích, đối chiếu 14 cố định, span năm đã map và sequence-wide trên token hợp lệ như ba phương án để kiểm. Khóa quy tắc khi có nhiều năm, trùng năm hoặc truncation. Đo câu đúng, năm đúng và thay đổi đầu ra khi đổi năm trong cùng ca. [FoNE v2](../Papers/arxiv-2502.09741/PAPER.md).
- **Đọc đáp án bằng cách nào?** My Answer is C và Look at the Text cho thấy kênh first-token và text answer có thể khác nhau ở model instruction-tuned. Trên Base Qwen, lưu raw response và trạng thái invalid/ambiguous cùng kết quả chấm. So trên cùng ID và cùng hoán vị. MCQFormatBench giúp tách nhạy format khỏi nhạy thời gian. Ghi số biến thể thực tế riêng cho Rel và MCQA. [My Answer is C](../Papers/doi-10.18653-v1-2024.findings-acl.441/PAPER.md) · [Look at the Text](../Papers/arxiv-2404.08382/PAPER.md) · [MCQFormatBench](../Papers/acl-2025.gem-1.69/PAPER.md).
- **Xây đối chứng và holdout thế nào?** Tách kiểm sinh giá trị số NWP với metric QA, rồi đo ảnh hưởng temporal trên cùng bộ ghép cặp. TDBench gợi ý gold từ temporal SQL và kiểm thời điểm độc lập với câu trả lời. METSQA gợi ý lát cắt trật tự timestamp và độ lớn số, trong thiết lập có bảng. Nguồn về knowledge acquisition giúp tách học dữ kiện và hiểu định dạng câu hỏi. [TDBench](../Papers/arxiv-2508.02045/PAPER.md) · [METSQA](../Papers/doi-10.5715-jnlp.33.1555/PAPER.md) · [Knowledge Learners](../Papers/doi-10.18653-v1-2024.acl-long.296/PAPER.md).

## Trạng thái bằng chứng

Đợt này xác minh nguồn, passage, code path và receipt. Gate Stage 03 giữ theo [trạng thái V2](../../Branches/v2-targeted-continual-alignment/Current-State.md). Mẫu 128 theo manifest là kế hoạch replay. Chất lượng sau sửa injection hoặc đổi parser cần kết quả ghép cặp. [Manifest 128](../../../Reports/6-11Report/_POST_FORENSIC_OPEN_GATES_20261105/REPLAY_SAMPLE_MANIFEST.json) · [Thiết kế test độc lập](../../../Reports/6-11Report/_POST_FORENSIC_OPEN_GATES_20261105/INDEPENDENT_TEST_DESIGN.md).
