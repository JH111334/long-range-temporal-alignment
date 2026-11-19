# Quyết Định Khoa Học và Hạn Chế Cấu Trúc: Thực Nghiệm V2 Run 02

## 1. Mục Tiêu Nghiên Cứu và Bản Chất Can Thiệp

Nghiên cứu trả lời câu hỏi cốt lõi: *Liệu việc huấn luyện thực sự mô hình Qwen2.5-0.5B-Instruct với hàm mất mát quan hệ thời gian (Temporal-Relation Objective) có cải thiện năng lực suy luận dựa trên bằng chứng thời gian so với (a) tiếp tục học tiếp nối số thực (Continued NWP), (b) điều kiện hóa thời gian bị xáo trộn (Corrupted Conditioning), và (c) chỉ học định dạng câu trả lời (Format-Only)?*

Thực nghiệm Run 02 thiết lập sự phân định rạch ròi so với Run 01:
- **Run 01 (Đã kiểm toán sửa sai tại `RUN01_FORENSIC_CORRECTION.json`)**: Là can thiệp suy luận thuần túy (`INFERENCE_ONLY_UNTRAINED_INTERVENTIONS`) trên cùng một checkpoint M0 tĩnh, không có bước tối ưu hóa nào.
- **Run 02**: Huấn luyện thực sự bằng PyTorch với 30 bước optimizer AdamW cho mỗi nhánh, tạo ra 4 checkpoint độc lập với các tensor trọng số và mã SHA256 hoàn toàn khác biệt.

## 2. Thiết Kế Bốn Nhánh Đối Chứng Cân Bằng Ngân Sách (Equal-Budget Factorial Design)

| Nhánh | Mục tiêu tối ưu | Vector điều kiện $e(t)$ | Vai trò khoa học |
| :--- | :--- | :--- | :--- |
| **`C0_CONTINUED_MASKED`** | $\mathcal{L}_{\text{masked}}$ (giá trị số thực) | True Year | Đối chứng tiếp tục CPT chuẩn |
| **`C1_TRUE_TEMPORAL_RELATION`** | $\mathcal{L}_{\text{masked}} + 0.5 \mathcal{L}_{\text{rel}}$ | True Year | Can thiệp đề xuất (Targeted Relation) |
| **`C2_SHUFFLED_CONDITIONING`** | $\mathcal{L}_{\text{masked}} + 0.5 \mathcal{L}_{\text{rel}}$ | Shuffled Random Year | Đối chứng âm (Negative Control) |
| **`C3_FORMAT_ONLY`** | $\mathcal{L}_{\text{masked}} + 0.5 \mathcal{L}_{\text{format}}$ | True Year | Kiểm soát độ quen thuộc định dạng |

Tất cả 4 nhánh cùng xuất phát từ checkpoint M0-FC40 gốc (`m0_fc40_seed_1_after_02.pt`), đóng băng 100% mô hình nền Qwen2.5-0.5B-Instruct, cùng learning rate $2 \times 10^{-4}$, cùng số bước cập nhật và token exposure.

## 3. Các Hạn Chế Cấu Trúc Đã Được Xác Lập (Structural Limitations)

1. **Vị trí Seam tại Token 14 (Seam Token Misplacement)**:
   Vector thời gian 40D được tiêm vào token thứ 14 (`' selecting'`) thuộc phần chỉ dẫn hệ thống (System Prompt) của ChatML. Do đó, gradient từ mục tiêu quan hệ thời gian phải lan truyền qua cơ chế Attention của toàn bộ chuỗi ngữ cảnh để tác động lên thực thể ở câu hỏi người dùng.
2. **Nghẽn Giao Diện Thời Gian Đơn Điểm (Single-Vector Conditioning Interface)**:
   Kiến trúc hiện hành chỉ cho phép tiêm một vector $e(t)$ duy nhất. Đối với câu hỏi so sánh hai mốc thời gian ($t_1, t_2$), mô hình chỉ nhận vector của một mốc thời gian qua seam ngoài, trong khi mốc thời gian còn lại hoàn toàn phụ thuộc vào token văn bản trong prompt.
3. **Tiên Nghiệm Ngôn Ngữ Át Chế (Dominant Language Priors)**:
   Trọng số mô hình nền 0.5B chứa các tiên nghiệm ngôn ngữ mạnh mẽ về cú pháp hội thoại và từ vựng, đòi hỏi tín hiệu thích ứng thời gian phải có cơ chế chú ý trực tiếp vào thực thể (như Cross-Attention hoặc Dynamic Entity Gating) thay vì chỉ cộng vào embedding của chỉ dẫn chung.

## 4. Quyết Định Khoa Học

- **Khóa Cổng Can Thiệp Seam Cố Định**: Giữ nguyên khế ước Decision 105 (`FIXED_FONE_CANCHI_V1`) và ranh giới đóng băng backbone. Không tự ý dời seam hoặc thay đổi số chiều 40D trên mã nguồn chuẩn tắc.
- **Ranh Giới Thống Kê**: Mọi kết luận về tính ưu việt của C1 bắt buộc phải vượt qua kiểm định McNemar chính xác hai phía ($p < 0.05$) so với cả C2 (đối chứng âm) và C3 (kiểm soát định dạng) trên tập dev held-out.
