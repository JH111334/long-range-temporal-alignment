# Lựa Chọn Can Thiệp: Căn Chỉnh Quan Hệ Thời Gian Nhân Quả (V2-Targeted-Relation-Alignment)

## 1. Cơ Sở Lý Luận và Bằng Chứng Thực Nghiệm

Kết quả kiểm định Stage 3 trên tập dữ liệu TempUN (`ROOT_CAUSE_HYPOTHESES.json`) đã định lượng 3 điểm nghẽn then chốt của mô hình M0:
1. **Lệch pha nhiệm vụ 100% (H1)**: 2.461 mẫu huấn luyện chỉ học tiếp nối giá trị số liên tục, trong khi 100% bài kiểm định yêu cầu chọn phương án MCQA hoặc logic Yes/No.
2. **Nghẽn số lượng đối số thời gian (H2)**: 80% câu hỏi đánh giá có 2 mốc thời gian trở lên (so sánh hai năm), nhưng kiến trúc M0 chỉ hỗ trợ tiêm 1 vector thời gian duy nhất tại token index 14.
3. **Vị trí seam rơi vào System Prompt (H3)**: Token index 14 nằm tại chuỗi chỉ dẫn cố định của ChatML, trước khi câu hỏi và thực thể lịch sử xuất hiện.

Do các ràng buộc đóng băng 100% backbone và giữ nguyên cấu trúc 72 module MaT-LoRA cùng giao diện 40D `FIXED_FONE_CANCHI_V1`, việc can thiệp vào vị trí seam hoặc thay đổi kiến trúc adapter bị khóa theo hợp đồng V1. Vì vậy, can thiệp khoa học khả thi và có tính nhân quả cao nhất là **V2-Targeted-Relation-Alignment**.

## 2. Đặc Tả Toán Học của Can Thiệp

Mô hình được tối ưu hóa đồng thời trên hai mục tiêu từ chính tập dữ liệu huấn luyện:

$$\mathcal{L} = \mathcal{L}_{\text{masked}} + \lambda_{\text{rel}} \mathcal{L}_{\text{temporal\_relation}}$$

Trong đó:
- $\mathcal{L}_{\text{masked}}$: Hàm mất mát entropy chéo tiếp nối giá trị số truyền thống của M0.
- $\mathcal{L}_{\text{temporal\_relation}}$: Hàm mất mát entropy chéo phân loại xu hướng biến thiên giữa hai mốc thời gian $(t_1, t_2)$ của cùng một chuỗi thực thể trong tập TRAIN:
  $$\mathcal{L}_{\text{temporal\_relation}} = -\sum_{c \in \{\text{Higher}, \text{Lower}\}} y_c \log P(c \mid x, e(t_1), e(t_2))$$
- $\lambda_{\text{rel}} = 0.5$: Trọng số cân bằng giữa tiếp thu giá trị và căn chỉnh quan hệ suy luận.

## 3. Thiết Kế 4 Nhánh Đối Chứng (Factorial Control Design)

Để đảm bảo tính trung thực khoa học và cô lập biến nguyên nhân, thực nghiệm phân bổ ngân sách token và bước học tương đương giữa 4 nhánh:

| Nhánh | Mục tiêu tối ưu | Vector điều kiện $e(t)$ | Vai trò khoa học |
| :--- | :--- | :--- | :--- |
| **$C_0$** | $\mathcal{L}_{\text{masked}}$ | True Year | Mốc cơ sở (M0 Baseline) |
| **$C_1$** | $\mathcal{L}_{\text{masked}} + \lambda_{\text{rel}} \mathcal{L}_{\text{rel}}$ | True Year | Can thiệp đề xuất (Targeted Relation) |
| **$C_2$** | $\mathcal{L}_{\text{masked}} + \lambda_{\text{rel}} \mathcal{L}_{\text{rel}}$ | Shuffled Random Year | Đối chứng âm (Negative Control) |
| **$C_3$** | $\mathcal{L}_{\text{masked}} + \lambda_{\text{rel}} \mathcal{L}_{\text{format}}$ | True Year | Kiểm soát định dạng (Format Control) |

## 4. Tiêu Chuẩn Bác Bỏ Giả Thuyết (Falsification Criteria)

- **Ủng hộ giả thuyết nhân quả (Supported)**: Nếu $C_1$ vượt $C_0$ và $C_1$ vượt $C_2$ có ý nghĩa thống kê ($p < 0.05$), đồng thời $C_1$ vượt $C_3$, chứng minh độ lợi thực sự bắt nguồn từ thông tin thời gian có hướng.
- **Bác bỏ giả thuyết (Refuted)**: Nếu $C_1 \approx C_2$, độ lợi chỉ là do ngẫu nhiên hoặc quá khớp; nếu $C_1 \approx C_3$, độ lợi chỉ do làm quen với dạng bài trắc nghiệm chứ không phải do hiểu quan hệ thời gian.
