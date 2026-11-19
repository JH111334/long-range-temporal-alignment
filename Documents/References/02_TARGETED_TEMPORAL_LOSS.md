---
loại: "tham-chiếu"
nguồn: "local-file:References/02_TARGETED_TEMPORAL_LOSS.md"
độ_tin_cậy: "exploratory"
chủ_đề: "02_TARGETED_TEMPORAL_LOSS"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: false
cập_nhật_lần_cuối: "2026-10-12"
---

# Hàm Mục Tiêu Huấn Luyện Căn Chỉnh Thời Gian Định Hướng (Targeted Temporal Loss)

Tài liệu này trình bày chi tiết cơ sở toán học, phân tích đạo hàm và quy chuẩn triển khai của hàm mục tiêu huấn luyện **Targeted Masked Loss (`MASKED_TARGET`)** trong bài toán học thích ứng tham số và học liên tục theo thời gian trên tập dữ liệu TempUN Small.

---

## 1. Ý Nghĩa Bản Chất và Động Lực Nghiên Cứu

### Hạn chế của Next-Token Prediction trên toàn bộ chuỗi (Full-Sequence NWP)
Trong các quy trình huấn luyện trước đây (điển hình như nhánh thí nghiệm chưa áp dụng mặt nạ $B_1^M$), hàm mục tiêu dự đoán từ tiếp theo được tính đồng đều trên mọi token của chuỗi văn bản:
$$\mathcal{L}_{\text{full}} = -\frac{1}{L} \sum_{i=1}^L \log P(x_i \mid x_{<i})$$
Cách tiếp cận này gặp phải vấn đề nghiêm trọng về **suy thoái trọng tâm thông tin (Objective Information Dilution)**:
- Một mẫu dữ liệu trong TempUN Small có dạng:
  - Prompt: `"In 1947, the period life expectancy at birth in Australia was"` (12 tokens).
  - Target: `"68.52"` (3 tokens).
- Nếu tính loss trên toàn bộ 15 tokens, hơn $80\%$ lượng gradient được dành để tối ưu hóa việc dự đoán các từ ngữ ngữ pháp phổ quát trong prompt (ví dụ: *"the"*, *"at"*, *"in"*).
- Mô hình bị lãng phí dung lượng thích ứng để học văn phong dẫn nhập thay vì tập trung liên kết giữa **mốc năm 1947 + chỉ số kỳ vọng sống tại Australia $\to$ giá trị 68.52**.

### Sáng kiến MASKED_TARGET
Hàm mục tiêu `MASKED_TARGET` thiết lập một ranh giới giám sát nghiêm ngặt:
- **Đóng băng hoàn toàn giám sát trên Prompt:** Toàn bộ các token thuộc câu dẫn nhập được gán nhãn $-100$, loại trừ hoàn toàn khỏi hàm tính toán mất mát.
- **Tập trung gradient vào Target và EOS:** Gradient chỉ được sinh ra từ các token mang giá trị tri thức lịch sử đích thực kèm theo token kết thúc câu `<|im_end|>`.
- Giúp adapter học chính xác ánh xạ nhân quả từ bối cảnh thời gian sang đáp án, đồng thời rèn luyện mô hình khả năng dừng câu chuẩn xác trong các tác vụ trắc nghiệm và hỏi đáp.

---

## 2. Công Thức Toán Học và Kích Thước Tensor

### 2.1. Cấu trúc Chuỗi Token và Mặt Nạ Nhãn
Một chuỗi huấn luyện hoàn chỉnh bao gồm ba đoạn token liên tiếp:
$$X = [x_1, \dots, x_P, x_{P+1}, \dots, x_{P+T}, x_{\text{eos}}]$$
trong đó:
- $[x_1, \dots, x_P]$: Chuỗi token dẫn nhập (Prompt), độ dài $P$.
- $[x_{P+1}, \dots, x_{P+T}]$: Chuỗi token đáp án vàng (Target), độ dài $T$.
- $x_{\text{eos}}$: Token kết thúc chuỗi đặc biệt (End-of-Sequence, đối với Qwen là `<|im_end|>`).
- Tổng độ dài chuỗi: $L = P + T + 1$.

Vector nhãn mục tiêu $Y = [y_1, y_2, \dots, y_L]$ được xây dựng bằng quy tắc mặt nạ:
$$y_i = \begin{cases} -100 & \text{với } 1 \le i \le P \quad \text{(Vùng bỏ qua gradient)} \\ x_i & \text{với } P < i \le L \quad \text{(Vùng giám sát tích cực)} \end{cases}$$

### 2.2. Hàm Mất Mát Causal Cross-Entropy Dịch Chuyển (Shift Loss)
Trong mô hình ngôn ngữ nhân quả tự hồi quy, vị trí token thứ $i-1$ có nhiệm vụ dự đoán token thứ $i$. Do đó, ma trận logits $\mathbf{Z} \in \mathbb{R}^{L \times V}$ và nhãn $Y \in \mathbb{R}^L$ phải được dịch chuyển một vị trí:
$$\mathbf{Z}_{\text{shift}} = \mathbf{Z}_{1 : L-1, :} \in \mathbb{R}^{(L-1) \times V}$$
$$Y_{\text{shift}} = Y_{2 : L} \in \mathbb{R}^{L-1}$$

Xác suất dự đoán của token mục tiêu $y_i$ ($i \in \{2, \dots, L\}$) qua hàm Softmax:
$$P(x_i = k \mid x_{<i}) = \frac{\exp(\mathbf{Z}_{i-1, k})}{\sum_{v=1}^V \exp(\mathbf{Z}_{i-1, v})}$$

Hàm mất mát `MASKED_TARGET` là trung bình cộng Cross-Entropy trên tập các vị trí không bị che mặt nạ:
$$\mathcal{L}_{\text{MASKED\_TARGET}} = -\frac{1}{T + 1} \sum_{i = P + 1}^L \log P(x_i = y_i \mid x_{<i})$$

### 2.3. Bảng Kích Thước Tensor (Tensor Shapes)
Bảng kích thước tensor tại bước tính hàm mất mát với kích thước batch $B=1$:

| Tensor | Ký hiệu | Kích thước | Mô tả kỹ thuật |
| :--- | :---: | :---: | :--- |
| Chuỗi token đầu vào | `input_ids` | $(B, L)$ | Mảng số nguyên `int64` chứa ID token |
| Nhãn mục tiêu | `labels` | $(B, L)$ | Mảng số nguyên chứa $-100$ và token ID |
| Đầu ra Logits của mô hình | `logits` | $(B, L, 151936)$ | Điểm số chưa chuẩn hóa trên từ vựng |
| Logits dịch chuyển | `shift_logits` | $(B, L-1, 151936)$ | Loại bỏ vị trí cuối cùng của chuỗi |
| Nhãn dịch chuyển | `shift_labels` | $(B, L-1)$ | Loại bỏ vị trí đầu tiên của chuỗi |
| Mẫu số chuẩn hóa | `denominator` | Vô hướng ($T+1$) | Số lượng token tham gia tính mất mát |
| Giá trị hàm mất mát | `loss` | Vô hướng $(1,)$ | Giá trị mất mát số thực dùng để backward |

---

## 3. Cơ Chế Lan Truyền Gradient và Động Lực Học Học Tập

### 3.1. Phân bố Gradient trên các vị trí Token
Đạo hàm của hàm mất mát đối với giá trị logit tại vị trí $i-1$ cho từ vựng thứ $k$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{Z}_{i-1, k}} = \begin{cases} 0 & \text{khi } y_i = -100 \quad (i \le P) \\ \frac{1}{T+1} \left( P(x_i = k \mid x_{<i}) - \mathbb{I}(x_i = k) \right) & \text{khi } y_i \neq -100 \quad (i > P) \end{cases}$$

Nhận xét quan trọng:
1. **Triệt tiêu gradient trực tiếp tại prompt:** Đầu ra logit của tất cả các token dẫn nhập nhận đạo hàm bằng $0$. Mô hình không bị trừng phạt nếu logit tại các vị trí này không ưu tiên tái hiện câu hỏi.
2. **Dòng gradient gián tiếp qua cơ chế Causal Self-Attention:** Mặc dù logit tại vị trí prompt không nhận gradient trực tiếp, nhưng trạng thái ẩn của các token target $\mathbf{h}_{P+1}, \dots, \mathbf{h}_L$ được tính toán thông qua cơ chế chú ý hướng về quá khứ:
   $$\mathbf{h}_t = \sum_{j=1}^t \operatorname{Softmax}\left(\frac{q_t k_j^T}{\sqrt{d_k}}\right) v_j$$
   Do đó, gradient từ các token target lan truyền ngược về toàn bộ các vector biểu diễn của câu dẫn nhập $j \le P$, và đặc biệt là lan truyền về vector $\mathbf{h}_{14}$ (nơi tiêm vector thời gian $\mathbf{z}_{40}$).

### 3.2. Hiệu ứng tối ưu hóa liên kết thời gian
Nhờ cơ chế này, ma trận thích ứng $\Delta W = B F(\theta, z) A$ và lớp chiếu tiêm $W_{\text{seam}}$ được tối ưu hóa theo tiêu chí:
*"Làm thế nào để với thông tin thời gian $(\theta, z)$ và ngữ cảnh câu hỏi, vector ẩn tại các bước cuối có thể phân biệt chính xác nhất token số học của chỉ số mục tiêu."*

---

## 4. Ánh Xạ Vào Mã Nguồn Thực Thi (Code Mapping)

### 4.1. Mã hóa nhãn trong `tempun_dataset.py`
Hàm `tokenize_masked_target_example` thực hiện chia tách và gán nhãn chính xác:
```python
prompt_ids = tokenizer(example.prompt, add_special_tokens=True)["input_ids"]
target_ids = tokenizer(example.target, add_special_tokens=False)["input_ids"]
eos_id = tokenizer.eos_token_id

full_ids = prompt_ids + target_ids + [eos_id]
labels = ([-100] * len(prompt_ids)) + target_ids + [eos_id]
```

### 4.2. Dịch chuyển và tính mất mát trong `run_m0_fc40.py`
Quá trình tính toán diễn ra tuần tự:
```python
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()

loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
)
```

---

## 5. Phương Pháp và Khế Ước Kiểm Chứng Thực Nghiệm

1. **Kiểm tra Mẫu Số Hợp Lệ (Denominator Audit):**
   Trong mọi bước huấn luyện, số lượng phần tử có nhãn khác $-100$ trong `shift_labels` phải luôn nằm trong khoảng $[2, 16]$. Nếu mẫu số bằng $0$, hàm mất mát trở thành NaN và kích hoạt cổng ngắt an toàn ngay lập tức.
2. **Kiểm tra Tính Hữu Hạn (Finite Loss Gate):**
   Giá trị mất mát ban đầu của mô hình Qwen 0.5B trên TempUN Small thường dao động trong khoảng $6{,}5 \le \mathcal{L} \le 8{,}5$ (tương ứng với $\ln(151936) \approx 11{,}9$). Nếu $\mathcal{L} > 15{,}0$ hoặc $\mathcal{L} < 0{,}0$, quá trình huấn luyện bị từ chối do tràn số.
3. **Độ dốc hội tụ theo Stage:**
   Trên mỗi chặng thời gian (1947, 1984, 2022), loss trung bình phải giảm dần sau mỗi epoch, khẳng định mô hình đang tiếp thu tri thức mục tiêu thành công.

---

## 6. Các Cạm Bẫy Kỹ Thuật Thường Gặp (Pitfalls & Failure Modes)

1. **Lỗi Dịch Chuyển Lệch Pha (Off-by-one Alignment Bug):**
   Nếu người lập trình tính loss trực tiếp giữa `logits` và `labels` mà không cắt bỏ vị trí đầu và cuối, mô hình sẽ học cách dự đoán token hiện tại thay vì token tiếp theo. Khi suy luận, mô hình chỉ lặp lại token vừa sinh ra (infinite loop).
2. **Bỏ Quên Token EOS (`<|im_end|>`):**
   Nếu không đưa `eos_id` vào vùng giám sát tích cực, mô hình không bao giờ nhận được tín hiệu kết thúc câu. Khi thực hiện suy luận trắc nghiệm sinh mã, mô hình sẽ tiếp tục sinh văn bản rác vô tận cho đến khi chạm giới hạn `max_new_tokens`, khiến bộ phân tích cú pháp không thể trích xuất được chữ cái đáp án A, B, C, D (Syntax Error Rate $> 90\%$).
3. **Che Nhãn Nhầm Lẫn Do Tokenizer Prefix:**
   Một số tokenizer tự động thêm dấu khoảng trắng vào đầu từ nếu không đặt `add_special_tokens=False` khi tokenize phần `target`. Điều này làm lệch ID token của đáp án so với phân phối tự nhiên trong văn bản, gây hiện tượng loss không thể hội tụ.
