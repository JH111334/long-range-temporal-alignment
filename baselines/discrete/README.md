# Phân Vùng Baseline Rời Rạc & Text Prompting (Discrete Controls Specification)

Thư mục `baselines/discrete/` là phân vùng quy chuẩn và triển khai cho các mô hình đối chứng kiểm soát căn nguyên:
1. **Learned 60 Can Chi Discrete Embedding Baseline**: Bảng tra cứu nhúng rời rạc 60 chu kỳ Can Chi được huấn luyện tự do qua gradient.
2. **Natural Language Text Tag Prompting Baseline**: Tiêm nhãn thời gian Can Chi tự nhiên vào văn bản prompt mà không can thiệp tầng nhúng hay trọng số mô hình.

---

## 1. Cơ Sở Lý Luận & Câu Hỏi Nghiên Cứu Đối Chứng (Scientific Rationale)

Khi nghiên cứu các kiến trúc biểu diễn thời gian liên tục như **Ticktack** (hình học cực xoắn ốc Archimedes kết hợp hàm điều hòa, Han et al., 2025) hay **FoNE** (chuỗi sóng đa tần số Fourier, Khang 2026), một câu hỏi phản biện cốt tử từ Hội đồng Khoa học là:

> *"Liệu các phép biến đổi hình học và lượng giác liên tục phức tạp có thực sự mang lại lợi thế biểu diễn, hay việc tra cứu nhúng rời rạc trực tiếp theo chu kỳ 60 năm Can Chi hoặc thậm chí chỉ cần nhắc nhở bằng văn bản (Text Tag Prompting) là đã đủ để mô hình nắm bắt quy luật thời gian?"*

Để tách bạch độc lập các yếu tố này, nhánh `discrete/` thiết lập 2 cấu hình đối chứng chuẩn mực:

```
                            ┌───────────────────────────────┐
                            │ Không gian biểu diễn thời gian│
                            └───────────────┬───────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
         Kiến trúc Liên tục (Continuous)                 Kiến trúc Rời rạc (Discrete)
    ┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐
    │ - Ticktack (Eq. 1-4 Spiral)          │        │ - Learned Discrete 60 Embedding     │
    │ - FoNE (Fourier Series)              │        │ - Text Tag Prompting (Natural Text)  │
    └──────────────────────────────────────┘        └──────────────────────────────────────┘
```

---

## 2. Đặc Tả Kỹ Thuật: Learned 60 Can Chi Discrete Embedding

### 2.1. Công Thức & Kiến Trúc
Ánh xạ năm dương lịch $t$ sang chỉ số chu kỳ Can Chi rời rạc $\tau(t) \in \{0, 1, \dots, 59\}$ theo công thức Eq. 1:
$$\tau(t) = (t - 3) \bmod 60$$

Bảng nhúng khả vi được định nghĩa độc lập:
$$\mathbf{E}_{\text{CanChi}} \in \mathbb{R}^{60 \times d} \quad (d = 896 \text{ cho Qwen2.5-0.5B})$$

Vector đặc trưng cho năm $t$:
$$\mathbf{v}_{\text{CanChi}}(t) = \text{EmbeddingLookup}(\tau(t), \mathbf{E}_{\text{CanChi}})$$

### 2.2. Phương Thức Tiêm Vào Mô Hình (Injection Protocols)
- **Phương án 1 (Localized Injection — Kế thừa C2)**: Chỉ cộng $\mathbf{v}_{\text{CanChi}}(t)$ tại đúng vị trí date tokens trong chuỗi, không cộng broadcast để bảo vệ RMSNorm:
  $$\mathbf{E}'(w_{\text{date}}, t) = \mathbf{E}(w_{\text{date}}) + \mathbf{v}_{\text{CanChi}}(t)$$
- **Phương án 2 (Soft Prompt Token Prefix)**: Bổ sung 1 token thời gian ảo vào đầu chuỗi:
  $$\mathbf{H}_0 = [\mathbf{v}_{\text{CanChi}}(t), \mathbf{E}(w_1), \dots, \mathbf{E}(w_S)]$$

### 2.3. Khởi Tạo & Tối Ưu Hóa
- Khởi tạo: Phân phối chuẩn $\mathcal{N}(0, 0.02)$ khớp với phân phối tự nhiên của `embed_tokens` của Qwen2.5.
- Tối ưu hóa: Cùng với bộ điều hợp LoRA thông qua $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NTP}} + \sigma \mathcal{L}_{\text{intra}}$.

---

## 3. Đặc Tả Kỹ Thuật: Text Tag Prompting (In-Context Prompting Baseline)

### 3.1. Cơ Chế Văn Bản Thuần Túy
Không can thiệp vào tầng embedding hay cấu trúc mô hình. Thay vào đó, bộ sinh prompt sẽ dịch năm số học sang danh xưng Can Chi bằng ngôn ngữ tự nhiên và bổ sung vào câu hỏi:

```
[Prompt Gốc]:
Question: What happened in 1984?
Answer:

[Prompt Có Text Tag Can Chi]:
Question: [Năm Giáp Tý (1984)] What happened in 1984?
Answer:
```

### 3.2. Mục Tiêu Kiểm Định
- Kiểm tra xem mô hình gốc đã có tri thức tiềm ẩn về Can Chi trong kho ngữ liệu tiền huấn luyện hay chưa.
- Đo lường mức độ thiên vị chữ cái (Letter Bias) khi prompt có chứa từ khóa ngữ nghĩa tiếng Việt / Hán Việt.

---

## 4. Kế Hoạch Triển Khai & Kiểm Thử Đối Chứng (Roadmap)

1. **Module `baselines/discrete/models.py`**:
   - `DiscreteCanChiEmbedding(num_cycles=60, hidden_dim=896)`
   - Hỗ trợ khởi tạo chuẩn hóa biên độ tự động.
2. **Module `baselines/discrete/prompting.py`**:
   - Bộ ánh xạ bảng 60 Can Chi: `{0: "Quý Hợi", 1: "Giáp Tý", 2: "Ất Sửu", ..., 59: "Nhâm Tuất"}`.
   - Hàm sinh prompt chuẩn hóa đa ngôn ngữ (Tiếng Việt & Tiếng Anh).
3. **Thực nghiệm Ghép Cặp (Paired McNemar Tests)**:
   - So sánh trực tiếp giữa `Discrete 60 Can Chi` vs `Ticktack Scaled` vs `FoNE` trên cùng tập kiểm thử khóa cứng 180 mẫu TempUN ($N=540$ qua 3 seeds).
