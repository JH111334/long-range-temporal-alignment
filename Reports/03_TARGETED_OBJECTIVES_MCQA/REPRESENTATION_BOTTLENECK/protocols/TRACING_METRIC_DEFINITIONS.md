# Định Nghĩa Toán Học Các Độ Đo Truyền Dẫn (Tracing Metric Definitions)

Tài liệu này thiết lập hệ thống định nghĩa toán học chuẩn xác cho chuỗi truyền dẫn nhân quả thời gian và phân tách rạch ròi 5 cấp độ đo lường cùng 2 trục tọa độ vật lý.

## 1. Chuỗi Truyền Dẫn Nhân Quả

Quá trình can thiệp thời gian được mô hình hóa theo chuỗi hàm truyền:
$$\text{Prompt}(x, t) \xrightarrow{\text{parse}} t \xrightarrow{\text{encode}} z(t) \xrightarrow{\text{inject}} h_{\text{seam}}^{(0)} \xrightarrow{\text{core}} F_t \xrightarrow{\text{adapter}} \Delta W(t) \xrightarrow{\text{forward}} h^{(\ell)}[p] \xrightarrow{\text{head}} \text{logits}(y) \xrightarrow{\text{decide}} \hat{y}.$$

### Phân Biệt Hai Trục Tọa Độ Đo Đạc
- **Trục Tầng Mạng (Layer Index $\ell \in \{0, 1, \dots, L-1\}$)**: Với $L = 24$ (trên Qwen2.5-0.5B), mỗi tầng $\ell$ chứa các khối Self-Attention và MLP kèm bộ điều hợp `DualCycleMaTLoRA`.
- **Trục Vị Trí Chuỗi Token (Token Position Index $p \in \{0, 1, \dots, T-1\}$)**:
  - $p = p_{\text{seam}} = 14$: Vị trí cổng tiêm Seam Injection trên khuôn mẫu prompt chuẩn Stage 03.
  - $p = p_{\text{comp}} = -1$: Vị trí token cuối cùng của prompt, nơi mô hình tính toán phân phối logit để sinh token đáp án $\hat{y}$.
  - Tuyệt đối không nhầm lẫn tầng $\ell = 14$ với vị trí token $p = 14$.

---

## 2. Hệ Thống Năm Cấp Độ Đo Lường Độc Lập

### Cấp Độ 1: Phân Phối Tín Hiệu (Delivery)
Xác nhận tín hiệu thời gian được mã hóa, tiêm thành công và làm thay đổi trọng số nội tại của mô hình:
1. **Độ lệch vector biểu diễn ($D_z$)**:
   $$D_z(t_1, t_2) = \|z(t_1) - z(t_2)\|_2.$$
2. **Độ biến thiên Dynamic Core ($D_F$)**:
   $$D_F(t_1, t_2) = \|F_{t_1} - F_{t_2}\|_F = \sqrt{\sum_{i=1}^r \sum_{j=1}^r (F_{t_1}[i, j] - F_{t_2}[i, j])^2}.$$
3. **Độ biến thiên ma trận trọng số thích ứng ($D_W$)**:
   $$D_W(t_1, t_2) = \|\Delta W(t_1) - \Delta W(t_2)\|_F = \left\|\frac{\gamma}{r} B (F_{t_1} - F_{t_2}) A\right\|_F.$$
- **Tiêu chuẩn đạt**: $D_z > 10^{-4}$, $D_F > 10^{-4}$, $D_W > 10^{-5}$.

### Cấp Độ 2: Độ Nhạy Trạng Thái Ẩn và Logit (Sensitivity)
Đo lường mức độ phản ứng của kích hoạt ẩn và logit đầu ra khi thời gian điều kiện thay đổi:
1. **Độ nhạy kích hoạt tại tầng $\ell$ và vị trí $p$ ($S_{\ell, p}$)**:
   $$S_{\ell, p}(t_1, t_2) = \frac{\|h^{(\ell)}[p; t_1] - h^{(\ell)}[p; t_2]\|_2}{\frac{1}{2}\left(\|h^{(\ell)}[p; t_1]\|_2 + \|h^{(\ell)}[p; t_2]\|_2\right)}.$$
2. **Độ tương đồng Cosine trạng thái ẩn ($\text{CosineSim}_{\ell, p}$)**:
   $$\text{CosineSim}_{\ell, p}(t_1, t_2) = \frac{\langle h^{(\ell)}[p; t_1], h^{(\ell)}[p; t_2] \rangle}{\|h^{(\ell)}[p; t_1]\|_2 \cdot \|h^{(\ell)}[p; t_2]\|_2}.$$
3. **Độ lệch logit tuyệt đối trung bình ($\Delta \text{logits}$)**:
   $$\Delta \text{logits}(t_1, t_2) = \frac{1}{|V|} \sum_{v \in V} |\ell_v(t_1) - \ell_v(t_2)|.$$
4. **Biến thiên biên độ đáp án mục tiêu ($M(t)$ và $\Delta M$)**:
   $$M(t) = \text{logit}(y_{\text{target}}) - \max_{y' \ne y_{\text{target}}} \text{logit}(y'),$$
   $$\Delta M(g_2) = \text{logit}_{t_2}(g_2) - \text{logit}_{t_1}(g_2).$$
- **Tiêu chuẩn đạt**: $S_{14, 14} > 0.005$, $\Delta \text{logits} > 0.01$.

### Cấp Độ 3: Tính Chọn Lọc Nhân Quả (Selectivity)
Đảm bảo biến thiên của mô hình là phản ứng đặc thù với tín hiệu thời gian thực thụ, không phải phản ứng tổng quát trước bất kỳ nhiễu số học nào:
1. **Biên độ chọn lọc thời gian so với nhiễu chuẩn hóa ($\text{SelectivityMargin}$)**:
   $$\Delta_{\text{time}} = \Delta \text{logits}(t_1, t_2),$$
   $$\Delta_{\text{noise}} = \Delta \text{logits}(t_1, z_{\text{rand}}), \quad \text{với } \|z_{\text{rand}}\|_2 = \|z(t_2)\|_2, \ z_{\text{rand}} \sim \mathcal{N}(0, I),$$
   $$\text{SelectivityMargin} = \Delta_{\text{time}} - \Delta_{\text{noise}}.$$
- **Tiêu chuẩn đạt**: $\text{SelectivityMargin} > 0.02$.

### Cấp Độ 4: Tính Đúng Đắn Ngữ Nghĩa Lịch Sử (Semantic Correctness)
Đo lường sự chuyển dịch nhãn trên tập phản thực nghiệm đối chứng ($t_1 \ne t_2$):
1. **Tỷ lệ đúng trọn vẹn cả cặp (Pair-Level Both Correct)**:
   $$\text{BothCorrect} = \mathbb{I}(\hat{y}(t_1) = y^*(t_1) \land \hat{y}(t_2) = y^*(t_2)).$$
2. **Tỷ lệ lật nhãn trên tập Gold-Flip**:
   $$\text{FlipRate}_{\text{flip}} = \mathbb{P}(\hat{y}(t_2) \ne \hat{y}(t_1) \mid y^*(t_2) \ne y^*(t_1)).$$
3. **Tỷ lệ giữ nguyên nhãn trên tập Gold-Same (Negative Control)**:
   $$\text{Stability}_{\text{same}} = \mathbb{P}(\hat{y}(t_2) = \hat{y}(t_1) \mid y^*(t_2) = y^*(t_1)).$$
- **Tiêu chuẩn đạt**: $\text{BothCorrectRate} \ge 30\%$, $\text{FlipRate}_{\text{flip}} \ge 40\%$, $\text{Stability}_{\text{same}} \ge 90\%$.

### Cấp Độ 5: Độ Bền Vững và Toàn Vẹn Hệ Thống (Robustness)
1. **Bảo toàn khả năng sinh từ kế tiếp (NTP Perplexity)**:
   $$\text{PerplexityRatio} = \frac{\text{PPL}_{\text{adapted}}}{\text{PPL}_{\text{base}}} \le 1.15.$$
2. **Ngăn chặn sụp đổ phân phối nhãn (Label Diversity & Entropy)**:
   $$H(\hat{Y}) = - \sum_{k} p_k \log p_k, \quad \frac{H(\hat{Y}_{\text{adapted}})}{H(\hat{Y}_{\text{base}})} \ge 0.75.$$
