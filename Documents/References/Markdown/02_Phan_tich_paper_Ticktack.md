# Phân Tích Paper Ticktack và Nghiên Cứu Mở Rộng Biểu Diễn Thời Gian

Nguồn: [*Temporal Alignment of LLMs through Cycle Encoding for Long-Range Time Representations*, arXiv:2503.04150v3](https://arxiv.org/abs/2503.04150), nộp 2025-03-06, sửa 2025-10-21. Phân loại: preprint.

Rà soát lại ngày 2026-08-26: chưa xác minh được release chính thức của code hoặc TempLS có license/revision/checksum. Không tải dữ liệu.

---

## 1. Bản chất Mental Model và Phân định với Hệ thống Truy xuất (Retrieval)

Điểm khác biệt cốt lõi giữa nghiên cứu Ticktack và các hệ thống tìm kiếm truy xuất ngữ cảnh (như LegalQA) nằm ở vị trí can thiệp tri thức:

```text
Hệ thống truy xuất ngữ cảnh (LegalQA):
Tài liệu -> Chunks -> BM25/FAISS Index -> Retriever -> Evidence -> Generator

Nghiên cứu Căn chỉnh Thời gian Nội tại (Ticktack):
Mốc thời gian -> Biểu diễn chu kỳ thời gian -> Tọa độ cực/Descartes -> Temporal Encoding (TE)
             -> Inject vào Input Embedding h' = h + TE
             -> Post-train LoRA qua Loss Alignment + EWC
             -> Không gian trọng số LLM đã căn chỉnh -> Generator
```

Không có FAISS, BM25, vector DB hay bộ retriever nào trong lõi phương pháp Ticktack. Mục tiêu là biến đổi biểu diễn thời gian bên trong LLM, sau đó post-train để không gian trọng số gắn kết tri thức với niên đại chính xác.

Trong project này, Ticktack là prior và baseline của **Track B — Internal
Temporal Model Representation / Alignment**. **Track A — External Temporal
Knowledge Retrieval / Evidence** là nhánh song song: BM25/TF-IDF là baseline,
interval filter và temporal reranking là candidate intervention. Hai track gặp
nhau ở controlled reasoning; không dùng retrieval để nhận công trạng cho
representation hoặc ngược lại. R0–R3 được định nghĩa trong `References/MAP.md`.

---

## 2. Kiến trúc Hệ thống Hai Pha (Phase A & Phase B)

### Phase A — Offline Temporal Alignment Build
1. **Trích xuất thời gian:** Trích xuất mốc năm Dương lịch $t_{\text{AD}}$ từ văn bản.
2. **Chuyển đổi Lục thập hoa giáp:** Ánh xạ $t_{\text{cycle}} = (t_{\text{AD}} - 4) \pmod{60} \in \{0, \dots, 59\}$.
3. **Biểu diễn tọa độ cực:** $\theta = \frac{2\pi \cdot t_{\text{cycle}}}{60}$ (vị trí chu kỳ), $r = \alpha + \beta \cdot \text{epoch}(t_{\text{AD}})$.
4. **Chuyển đổi Descartes:** $x = r \cos \theta, y = r \sin \theta$.
5. **Mã hóa sóng thời gian:** $\text{TE}(x), \text{TE}(y)$ dạng sin/cos; cộng vào vector nhúng gốc tại span biểu thức thời gian: $h' = h + \text{TE}(x) + \text{TE}(y)$.
6. **Huấn luyện mô hình:** Đóng băng base LLM, cập nhật adapter LoRA bằng hàm mất mát tổng hợp:
   $$\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{NTP}} + \sigma \left[ \mathcal{L}_T + \frac{\lambda}{2} \sum_i F_i (\theta_{T,i} - \theta_{G,i})^2 \right]$$
   với $\mathcal{L}_T = \delta \mathcal{L}_{\text{intra}} + (1 - \delta) \mathcal{L}_{\text{inter}}$ và EWC chống quên lãng thảm khốc.

### Phase B — Online Inference
Câu hỏi thời gian $\rightarrow$ Trích xuất mốc thời gian $\rightarrow$ Chu kỳ $\rightarrow$ Tọa độ cực $\rightarrow$ Descartes $(x, y) \rightarrow \text{TE}(x), \text{TE}(y) \rightarrow$ Cộng vào embedding $h' \rightarrow$ Đưa qua LLM đã căn chỉnh ($\theta_T$) $\rightarrow$ Sinh câu trả lời.

---

## 3. Track B: Phân Tích Candidate Mở Rộng Sang Độ Hạt Tháng

### Phân tích phản biện: giả thuyết tháng cần kiểm chứng
- **Dữ liệu và nhu cầu:** Nếu ngày thưa hoặc năm gộp các sự kiện nội năm, tháng có thể là điểm cân bằng tốt hơn; tỷ lệ thiếu, coverage và nhu cầu người dùng phải đo trong data dossier, không dùng con số chưa có nguồn.
- **Phần cứng:** Kênh tháng có vẻ nhẹ hơn model mới, nhưng overhead và khả năng chạy trên RTX 2050 4 GB chỉ được kết luận sau resource probe có cấu hình cụ thể.

### Cấu trúc toán học Chu kỳ kép (Dual-Cycle Polar Geometry)
1. **Tầng Năm (Macro-cycle 60 năm):**
   $$\theta_{\text{year}} = \frac{2\pi \cdot t_{\text{cycle}}}{60}, \qquad r_{\text{year}} = \alpha_Y + \beta_Y \cdot \left\lfloor \frac{t_{\text{AD}} - t_0}{60} \right\rfloor$$
   $$x_{\text{year}} = r_{\text{year}} \cos \theta_{\text{year}}, \qquad y_{\text{year}} = r_{\text{year}} \sin \theta_{\text{year}}$$
2. **Tầng Tháng trong Năm (Meso-cycle 12 tháng, $m \in \{1, \dots, 12\}$):**
   $$\theta_{\text{month}} = \frac{2\pi \cdot (m - 1)}{12}, \qquad r_{\text{month}} = \alpha_M + \beta_M \cdot \left( \frac{m}{12} \right)$$
   $$x_{\text{month}} = r_{\text{month}} \cos \theta_{\text{month}}, \qquad y_{\text{month}} = r_{\text{month}} \sin \theta_{\text{month}}$$

### Cơ chế Precision Masking fail-closed
Sử dụng $\mathbf{m}_{\text{precision}} = [m_{\text{year}}, m_{\text{month}}] \in \{0, 1\}^2$:
$$h'_i = h_i + m_{\text{year}} \cdot \mathbf{W}_{\text{proj},Y} [\text{TE}(x_{\text{year}}) + \text{TE}(y_{\text{year}})] + m_{\text{month}} \cdot \mathbf{W}_{\text{proj},M} [\text{TE}(x_{\text{month}}) + \text{TE}(y_{\text{month}})]$$
Khi văn bản chỉ có năm, $m_{\text{month}} = 0$, triệt tiêu hoàn toàn thành phần tháng; tuyệt đối không tự ý gán giá trị mặc định tháng 1.

---

## 4. Năm Điểm Cải Thiện Kỹ Thuật Phụ Bắt Buộc từ Rà Soát Bài Báo Gốc (Ticktack v3)

1. **Khắc phục hàm bán kính chống va chạm alias (Strict Radial Monotonicity):** Paper dùng $r = \alpha + \beta \theta$. Vì $\theta$ tuần hoàn theo chu kỳ 60 năm, hai năm cách nhau 60 năm có thể trùng tọa độ. Candidate sửa là $r(t_{\text{AD}}) = \alpha + \beta \cdot \lfloor (t_{\text{AD}} - t_0)/60 \rfloor$; cần test trước khi kết luận.

2. **Chuẩn hóa hàm mất mát theo batch (Batch-Normalized Cosine Loss):** Thay Eq. 8 unnormalized bằng mất mát trung bình trên tập cặp duy nhất có margin:
   $$\mathcal{L}_{\text{intra}} = \frac{1}{|P_{\text{same}}|} \sum_{(i,j) \in P_{\text{same}}} [1 - \cos(z_i, z_j)], \quad \mathcal{L}_{\text{inter}} = \frac{1}{|P_{\text{diff}}|} \sum_{(i,k) \in P_{\text{diff}}} \max(0, \cos(z_i, z_k) - m)$$
3. **Khôi phục lực đẩy phân tách chu kỳ ($\delta < 1$):** Giữ thành phần inter-pair; hệ số phải khóa trong config và ablate.

4. **Tiêm thời gian cấp span (Span-Level Injection):** Chỉ tiêm $\text{TE}$ tại token thuộc biểu thức thời gian xác định.

5. **Định vị tham số EWC cho LoRA Adapter:** Chỉ dùng khi artifact ghi rõ current/reference/Fisher và trainable parameters.

---

## 5. Tích Hợp Mô Hình và Kỹ Thuật Mới 2025–2026

- **Mô hình nền tảng nhỏ gọn (0.5B–1.5B):** Qwen2.5-0.5B hoặc Qwen3-0.6B là candidate để probe; chưa có tuyên bố VRAM hoặc chất lượng trước khi đo.
- **RoPE / Attention Modulation:** Tích hợp vector thời gian vào các tầng self-attention theo kỹ thuật của *Date Fragments* (EMNLP 2025).
- **Temporal Hard Negative Sampling:** Có thể tạo cặp cùng thực thể nhưng sai tháng theo schema/qrels; chỉ mở khi error slice tự nhiên chứng minh cần thiết.

---

## 6. Các Phương Án Đề Xuất Phụ, Đối Chứng và Mở Rộng Khác

Các hướng kỹ thuật được gắn đúng vai trò:
1. **Track A baseline:** BM25/TF-IDF + interval table; temporal filter/rerank chỉ là candidate intervention, không phải baseline và chưa là contribution.
2. **Track B baseline:** Scalar/Fourier và Ticktack `published-equation`; tháng/multi-granularity là proposed candidate.
3. **Tokenizer/calendar probe:** Chỉ mở khi lỗi tự nhiên đủ lớn và mapping có nguồn/version/reviewer.
4. **Historical GraphRAG:** Future work; không đưa vào critical path hoặc R0–R3 hiện tại.
