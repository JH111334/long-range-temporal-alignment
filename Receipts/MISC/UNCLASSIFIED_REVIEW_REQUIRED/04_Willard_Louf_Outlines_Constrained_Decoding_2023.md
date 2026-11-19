# CƠ SỞ KHOA HỌC: GIẢI MÃ CÓ RÀNG BUỘC (CONSTRAINED DECODING)

## 1. Thông Tin Trích Dẫn Chuẩn Mực
1. **Outlines: Flexible and Fast Text Generation with Grammar Constraints**
   - *Tác giả:* Brandon Willard, Rémi Louf
   - *Công bố:* arXiv preprint (tháng 7 năm 2023)
   - *Định danh số:* arXiv:2307.09702 [cs.CL]
   - *Mã nguồn chính thức:* https://github.com/dottxt-ai/outlines
2. **PICARD: Parsing Incrementally for Constrained Auto-regressive Decoding from Language Models**
   - *Tác giả:* Torsten Scholak, Nathan Schucher, Dzmitry Bahdanau
   - *Hội nghị công bố:* Empirical Methods in Natural Language Processing (EMNLP 2021)
   - *Định danh số:* arXiv:2109.05093 [cs.CL] / DOI: 10.18653/v1/2021.emnlp-main.779

---

## 2. Cơ Sở Toán Học: Otomata Hữu Hạn Tất Định (DFA) và Mặt Nạ Logits

Khác với Option Scoring (chỉ chấm điểm tĩnh trên các chuỗi cố định), Giải mã có ràng buộc (Constrained Decoding) duy trì toàn vẹn quá trình sinh văn bản tự hồi quy nhưng kiểm soát tập hợp token được phép xuất hiện tại mỗi bước giải mã.

### 2.1. Biến đổi ngữ pháp thành DFA
Ngữ pháp đích (ví dụ: chỉ cho phép xuất hiện một trong bốn ký tự đáp án `A`, `B`, `C`, `D`, hoặc định dạng JSON hợp lệ) được biên dịch thành một Otomata hữu hạn tất định (Deterministic Finite Automaton - DFA):
$$\mathcal{M} = (Q, \Sigma, \delta, q_0, F)$$
Trong đó:
- $Q$ là tập hợp các trạng thái.
- $\Sigma$ là bảng chữ cái (tập token của mô hình).
- $\delta: Q \times \Sigma \rightarrow Q$ là hàm chuyển trạng thái.
- $q_0$ là trạng thái bắt đầu, $F$ là tập trạng thái kết thúc.

### 2.2. Cơ chế áp mặt nạ trên Logits (Logit Masking)
Tại bước sinh $t$, với trạng thái hiện tại $q_{t-1}$ của Otomata, tập hợp các token hợp lệ được xác định bởi:
$$\mathcal{V}_{\text{valid}}(q_{t-1}) = \{ v \in \mathcal{V} \mid \delta(q_{t-1}, v) \text{ xác định}\} $$

Bộ giải mã biến đổi vector phân phối xác suất logits $z_t \in \mathbb{R}^{|\mathcal{V}|}$ trước khi đưa qua hàm Softmax:
$$z'_{t, v} = \begin{cases} z_{t, v} & \text{nếu } v \in \mathcal{V}_{\text{valid}}(q_{t-1}) \\ -\infty & \text{ngược lại} \end{cases}$$

Xác suất sinh token tiếp theo được tính toán:
$$P(x_t = v \mid x_{<t}) = \frac{\exp(z'_{t, v})}{\sum_{u \in \mathcal{V}} \exp(z'_{t, u})}$$

---

## 3. Lợi Ích Đối Với Đề Tài `Long-Range-Representation`

1. **Bảo toàn giao thức sinh tự hồi quy:** Vẫn giữ nguyên luồng suy luận tự nhiên của mô hình, không làm thay đổi cách biểu diễn bài toán.
2. **Khắc phục triệt để lỗi cú pháp ChatML:** Khi mô hình hoàn thành phần nội dung đáp án, Otomata chỉ cho phép sinh duy nhất token kết thúc chuỗi `<|im_end|>`, ngăn chặn tình trạng lặp từ bất tận (endless repetition loop) hoặc bỏ dở câu.
3. **Chi phí tính toán gần như bằng 0:** Nhờ việc tính toán trước bảng tra cứu DFA (pre-indexed vocabulary mask), thời gian sinh văn bản không bị chậm lại đáng kể so với sinh tự do không ràng buộc.
