# KHUNG PHƯƠNG PHÁP LUẬN: TARGETED TEMPORAL CONTINUAL PRETRAINING

## 1. Bản Chất Kỹ Thuật Của Huấn Luyện Có Định Hướng (Targeted Training)

Trong các mô hình ngôn ngữ tự hồi quy (Causal Language Models) được tinh chỉnh theo định dạng chỉ thị (Instruction-Following, ví dụ định dạng ChatML của `Qwen2.5-0.5B-Instruct`), chuỗi đầu vào bao gồm hai phần tách biệt:
$$\text{Sequence} = [\text{Prompt / Context Tokens}, \quad \text{Target Response Tokens}]$$

### 1.1. So sánh cơ chế tính hàm mất mát
- **Huấn luyện không che mặt nạ (Unmasked Language Modeling Loss):**
  Hàm mất mát Cross-Entropy được tính trên toàn bộ chuỗi token:
  $$\mathcal{L}_{\text{unmasked}} = -\frac{1}{N} \sum_{t=1}^N \log P(x_t \mid x_{<t})$$
  *Hệ quả tiêu cực:* Mô hình buộc phải tối ưu hóa việc dự đoán các token thuộc prompt, câu hỏi và cấu trúc cú pháp cố định (`<|im_start|>user`, `<|im_end|>`). Điều này làm tiêu hao dung lượng gradient của adapter hạng thấp ($r=2$) vào việc ghi nhớ văn phong câu hỏi thay vì tập trung vào tri thức thực tế của câu trả lời.
- **Huấn luyện che mặt nạ định hướng (Masked-Target Cross-Entropy Loss):**
  Áp dụng mặt nạ nhị phân $M \in \{0, 1\}^N$, trong đó $M_t = 1$ khi và chỉ khi token $x_t$ thuộc về phần câu trả lời mục tiêu (target label / completion), và $M_t = 0$ cho toàn bộ các token tiền đề (instruction, question, options):
  $$\mathcal{L}_{\text{target}} = -\frac{1}{\sum_{t=1}^N M_t} \sum_{t=1}^N M_t \log P(x_t \mid x_{<t})$$

---

## 2. Rút Ra Từ Bài Học Tuần 1 (Week 1 Canonical Decision)

Trong giai đoạn đầu của đề tài (Week 1), thực nghiệm so sánh ghép cặp giữa $B_1$ (Unmasked LM Loss) và $B_3$ (Masked-Target CE) trên cùng một kiến trúc và seed cho thấy:
1. Việc huấn luyện không che mặt nạ khiến mô hình dễ bị trôi dạt phân phối (distribution shift), làm suy giảm khả năng tuân thủ cấu trúc hội thoại chuẩn.
2. Ngược lại, hàm mục tiêu `MASKED_TARGET` giúp tập trung toàn bộ trọng số gradient vào việc lựa chọn đáp án đúng, đồng thời duy trì sự ổn định của các tầng chú ý ngôn ngữ gốc.
3. Vì vậy, `MASKED_TARGET` được chính thức đóng băng (freeze) làm hàm mục tiêu cơ sở cho toàn bộ các nhánh so sánh tiếp theo ($B_3 = R_{00} = C_0$).

---

## 3. Ranh Giới Giữa Học Tri Thức Thời Gian và Ghi Nhớ Ngắn Hạn

Khi áp dụng `MASKED_TARGET` trong bối cảnh học tăng dần theo dòng thời gian (Temporal Continual Learning):
- Mô hình tiếp nhận các câu hỏi trắc nghiệm lịch sử theo từng kỷ nguyên (1947 $\rightarrow$ 1984 $\rightarrow$ 2022).
- Nếu chỉ dựa vào $\mathcal{L}_{\text{target}}$, mô hình chỉ học ánh xạ tĩnh giữa câu hỏi và đáp án mà không có ràng buộc nội tại nào buộc không gian ẩn phải bảo toàn thứ tự thời gian tuyến tính.
- Đây chính là động lực khoa học dẫn đến việc xem xét nhánh $L_2$ (bổ sung số hạng phạt thời gian $\mathcal{L}_{\text{temporal}}$ từ ECONET để định hướng không gian ẩn mà vẫn giữ nguyên lõi `MASKED_TARGET`).
