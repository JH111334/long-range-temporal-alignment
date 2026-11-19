# Báo Cáo Nghiên Cứu Khoa Học: Biểu Diễn và Căn Chỉnh Thời Gian Dài Hạn

**Mã tài liệu:** `DOCS-NCKH-TEMPORAL-2026-V2`  
**Phiên bản đồng bộ HTML:** [bao_cao_nckh_temporal_alignment.html](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Documents/References/bao_cao_nckh_temporal_alignment.html)  
**Phân loại nghiên cứu:** Nghiên cứu khoa học (NCKH) / Xử lý ngôn ngữ tự nhiên  
**Mô hình nền tảng:** Ticktack (*arXiv:2503.04150v3*)  
**Mức bằng chứng:** Tách bạch `published-equation`, `operational-assumption` và `proposed-method`

---

## Tuyên Ngôn Mental Model Cốt Lõi

Nghiên cứu có hai track ưu tiên chạy song song. Hai priority chỉ là ưu tiên đầu
tư; BM25 là baseline của Track A, Ticktack là prior/baseline của Track B, còn
GraphRAG/vector DB nằm ngoài critical path.

```text
Track A — External Temporal Knowledge Retrieval / Evidence:
[Corpus] -> [BM25/TF-IDF candidate pool] -> [Interval filter/rerank] -> [Evidence] -> [Controlled reasoning]

Track B — Internal Temporal Model Representation / Alignment:
[Mốc thời gian] -> [Chu kỳ] -> [Temporal Encoding TE] -> [Input Embedding h' = h + TE]
                 -> [Ticktack prior hoặc proposed alignment] -> [LLM] -> [Answer]
```

> **Nguyên tắc phân định:** Lõi Ticktack không sử dụng retriever ngoài; Ticktack
> là baseline/prior của Track B. Track A được đánh giá riêng, sau đó mới ghép ở
> controlled reasoning. Không suy ra causal contribution từ overall score.

---

## Phần 1: Khảo Sát Dữ Liệu Thời Gian (Data Survey)

### 1.1. Bản chất phân bố lệch và thách thức biểu diễn dài hạn

Tập ngữ liệu tiền huấn luyện của các mô hình ngôn ngữ lớn có phân bố thời gian cực kỳ mất cân đối:
- **Giai đoạn cổ đại (75.000 TCN – 0 SCN):** Cực kỳ thưa thớt (sparse), số lượng văn bản rất thấp.
- **Giai đoạn hiện đại (1900 – 2025 SCN):** Cực kỳ dày đặc (dense), tài nguyên văn bản số áp đảo.

```text
Trực quan phân bố thời gian trong Pretraining Corpus:
75.000 TCN ---------------------------------------------------- 2025 SCN
[ Cực kỳ thưa thớt (sparse) ]                      [ Cực kỳ dày đặc (dense) ]
```

Khi mô hình chỉ tiếp nhận mốc thời gian dưới dạng các token số rời rạc (`-5000`, `500`, `1965`, `2025`), không gian biểu diễn ẩn không tự động phản ánh khoảng cách niên đại và tính chu kỳ. Điều này dẫn đến hiện tượng sai lệch biểu diễn thời gian (*temporal misalignment*) ở các khoảng thời gian dài và các vùng dữ liệu ít tài nguyên.

### 1.2. Chuẩn hóa mẫu dữ liệu thời gian (Canonical Temporal Samples)

Dữ liệu văn bản thô được trích xuất và lưu trữ độc lập dưới dạng mẫu chuẩn hóa:

```json
{
  "sample_id": "VN_HIST_1945_001",
  "text": "Chủ tịch Hồ Chí Minh đọc Tuyên ngôn Độc lập tại Quảng trường Ba Đình vào ngày 2 tháng 9 năm 1945.",
  "gregorian_year": 1945,
  "month": 9,
  "day": 2,
  "time_span": [1945, 1945],
  "granularity": "day",
  "source_domain": "vietnamese_history",
  "provenance": "canonical_verified"
}
```

### 1.3. Chiến lược phân chia dữ liệu chống rò rỉ (Leakage-Resistant Splits)

Nghiêm cấm phân chia ngẫu nhiên (random split) để tránh rò rỉ thông tin từ các thực thể và sự kiện gần trùng lặp (*near-duplicates*). Dữ liệu được cô lập theo 4 trục:

| Trục phân chia | Mục tiêu kỹ thuật | Biện pháp kiểm soát |
|---|---|---|
| **Source Split** | Đo lường khả năng khái quát hóa xuyên nguồn tài liệu | Cô lập hoàn toàn các nguồn biên khảo khác nhau giữa tập train và test |
| **Entity Split** | Ngăn mô hình học vẹt mối liên hệ thực thể – niên đại | Thực thể lịch sử ở tập test không xuất hiện trong tập train |
| **Event Split** | Kiểm tra năng lực suy luận niên đại sự kiện mới | Toàn bộ mô tả về cùng một sự kiện được gom chung vào một tập chia |
| **Time-Range Split** | Đánh giá suy luận trên từng giai đoạn lịch sử riêng biệt | Thiết lập các tập test độc lập theo kỷ nguyên (BCE, 0–500, 500–1500, 1500–2000, >2000) |

### 1.4. Bốn lớp Artifact độc lập

1. **Canonical Temporal Data:** `train.jsonl`, `validation.jsonl`, `test.jsonl`.
2. **Temporal Representation Artifact:** `temporal_config.json` (kích thước chu kỳ, quy tắc chuyển đổi, tham số tỷ lệ).
3. **Trained Model Artifact:** Checkpoint LoRA, base model revision, training config, ma trận Fisher cho EWC.
4. **Evaluation Artifact:** `predictions.jsonl`, `metrics.json`, `metrics_by_era.json`, `year_similarity.npy`, biểu đồ t-SNE.

---

## Phần 2: Kiến Trúc Quy Trình Hệ Thống (System Pipeline)

### 2.1. Sơ đồ luồng dữ liệu hai track

```text
SHARED DATA CONTRACT
Nguồn -> rights/evidence gate -> canonical time -> quality/dedup -> split
  -> cùng corpus release, parser, prompt, evaluator và answer policy

TRACK A — EXTERNAL RETRIEVAL / EVIDENCE
  -> BM25/TF-IDF candidate pool
  -> interval filter / bounded temporal rerank
  -> evidence provenance / abstention

TRACK B — INTERNAL REPRESENTATION / ALIGNMENT
  -> scalar/Fourier -> Ticktack published-equation
  -> proposed year-to-month representation (chỉ sau gate)

RẼ NHÁNH Ở CONTROLLED REASONING
  -> answer hoặc abstention có evidence ID
```

### 2.2. Chi tiết các bước kỹ thuật trong Phase A (Mô hình Ticktack chuẩn)

1. **Trích xuất năm:** Trích xuất giá trị số nguyên $t_{\text{AD}}$ từ văn bản.
2. **Ánh xạ chu kỳ 60 năm:** $t_{\text{cycle}} = (t_{\text{AD}} - 4) \pmod{60}, \quad t_{\text{cycle}} \in \{0, \dots, 59\}$.
3. **Tọa độ cực:** $\theta = \frac{2\pi \cdot t_{\text{cycle}}}{60}$.
4. **Tọa độ Descartes:** $x = r \cos \theta, \quad y = r \sin \theta$.
5. **Mã hóa sóng thời gian (Temporal Encoding):**
   $$h'_i = h_i + \text{TE}(x_i) + \text{TE}(y_i)$$
6. **Hàm mất mát căn chỉnh và EWC:**
   $$\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{NTP}} + \sigma \left[ \mathcal{L}_T + \frac{\lambda}{2} \sum_i F_i (\theta_{T,i} - \theta_{G,i})^2 \right]$$
   - $\mathcal{L}_{\text{NTP}}$: Duy trì năng lực ngôn ngữ tự nhiên.
   - $\mathcal{L}_T = \delta \mathcal{L}_{\text{intra}} + (1 - \delta) \mathcal{L}_{\text{inter}}$: Tối ưu khoảng cách tương đồng cosine giữa các biểu diễn cùng lớp và khác lớp chu kỳ.
   - EWC với ma trận Fisher $F$: Ngăn chặn hiện tượng quên lãng thảm khốc (*catastrophic forgetting*).

### 2.3. Bậc thang thử nghiệm bóc tách (Ablation Ladder)

- **E0 (Base Model):** Qwen2.5 / LLaMA gốc, không chỉnh sửa.
- **E1 (Normal Post-train):** Post-training chỉ dùng $\mathcal{L}_{\text{NTP}}$ trên cùng dữ liệu (đo hiệu ứng tăng dữ liệu).
- **E2 (TE Only):** Tích hợp $\text{TE}(x) + \text{TE}(y)$ với $\mathcal{L}_{\text{NTP}}$ (đo inductive bias hình học chu kỳ).
- **E3 (+ Alignment):** Cấu hình E2 bổ sung mất mát căn chỉnh $\mathcal{L}_T$ (intra/inter).
- **E4 (Full System):** Cấu hình E3 bổ sung điều hòa EWC (bảo tồn tri thức ngôn ngữ chung).

### 2.4. Ma trận causal R0–R3

R0–R3 dùng cùng corpus, candidate pool, parser, split, prompt, evaluator và
answer policy:

| Run | Retrieval | Representation | Vai trò |
|---|---|---|---|
| `R0` | BM25/TF-IDF, không filter/rerank | Base model | Control chung |
| `R1` | R0 + interval filter/temporal rerank/evidence policy | Giữ R0 | Retrieval-only |
| `R2` | Giữ R0 | Proposed representation; Ticktack báo riêng | Representation-only |
| `R3` | Như R1 | Như R2 | Combined/interaction |

Chỉ kết luận contribution bằng paired delta theo case/group và lát cắt. Candidate
pool hoặc contract đổi phải ghi `confounded`; không thay thế evidence bằng
overall score.

### 2.5. Đánh giá phân rã theo kỷ nguyên (Macro Temporal Metric)

$$\text{Macro Score} = \frac{1}{|B|} \sum_{b \in B} \text{Score}_b, \quad B = \{\text{BCE}, 0\text{--}500, 500\text{--}1000, 1000\text{--}1500, 1500\text{--}2000, >2000\}$$

Triệt tiêu độ thiên lệch đánh giá khi mô hình đạt điểm cao ở thời hiện đại nhờ dữ liệu dày đặc nhưng thất bại ở các giai đoạn lịch sử cổ xưa.

---

## Phần 3: Hai Track Ưu Tiên Và Giả Thuyết Mở Rộng Độ Hạt Tháng

### 3.1. Phân tích phản biện: tháng là candidate cần kiểm chứng

Để bảo đảm tính khoa học nghiêm cẩn, dự án đặt câu hỏi: **liệu độ hạt tháng có
giúp phân biệt sự kiện nội năm với chi phí phù hợp hay không?** Đây là giả thuyết,
chỉ được chấp nhận sau data dossier, R2/R3 và resource probe; chưa gọi là Pareto
hoặc kết quả đã đo.

1. **Phục vụ con người và bản chất dữ liệu lịch sử:**
   - Nếu chọn **Ngày (Day)**: Dữ liệu lịch sử cổ–trung đại có tỷ lệ thiếu ngày lên tới >85%. Ép độ hạt ngày trên toàn bộ không gian lịch sử sẽ dẫn đến suy thoái biểu diễn do dữ liệu giả mạo/nhiễu.
   - Nếu giữ nguyên **Năm (Year)**: Mất hoàn toàn trật tự quan hệ nhân quả nội năm. Trong các sự kiện cận–hiện đại (ví dụ năm 1945: tháng 3 đảo chính, tháng 8 khởi nghĩa, tháng 9 tuyên ngôn), mô hình Ticktack-year gộp chung về cùng một tọa độ $(x_{1945}, y_{1945})$, khiến mô hình không thể trả lời các câu hỏi thực tế của con người về quan hệ trước/sau trong cùng một năm.
   - **Độ hạt Tháng (Month)** là candidate cân bằng; coverage và lợi ích thực tế phải được đo theo source, thời kỳ và lát cắt tiếng Việt.
2. **Khả thi theo compute profile:**
   - Overhead tọa độ tháng và khả năng chạy base model 0,5B–1,5B phải được đo trên
     profile đích; local host chỉ là `local_validation`, không phải architecture cap.
3. **Tính trung thực khoa học nhờ Precision Masking:**
   - Cơ chế Precision Masking bảo đảm nguyên tắc fail-closed: khi dữ liệu chỉ có năm, kênh tháng được ngắt bằng mask 0; tuyệt đối không tự tiện gán tháng 1 làm sai lệch lịch sử.

### 3.2. Đặc tả Toán học Kiến trúc Chu kỳ kép (Dual-Cycle Polar Geometry)

Thiết lập không gian biểu diễn hai tầng kế thừa:
1. **Tầng Năm (Macro-cycle — Chu kỳ 60 năm):**
   $$\theta_{\text{year}} = \frac{2\pi \cdot t_{\text{cycle}}}{60}, \qquad r_{\text{year}} = \alpha_Y + \beta_Y \cdot \text{epoch}(t_{\text{AD}})$$
   $$x_{\text{year}} = r_{\text{year}} \cos \theta_{\text{year}}, \qquad y_{\text{year}} = r_{\text{year}} \sin \theta_{\text{year}}$$
2. **Tầng Tháng trong Năm (Meso-cycle — Chu kỳ 12 tháng, $m \in \{1, \dots, 12\}$):**
   $$\theta_{\text{month}} = \frac{2\pi \cdot (m - 1)}{12}, \qquad r_{\text{month}} = \alpha_M + \beta_M \cdot \left( \frac{m}{12} \right)$$
   $$x_{\text{month}} = r_{\text{month}} \cos \theta_{\text{month}}, \qquad y_{\text{month}} = r_{\text{month}} \sin \theta_{\text{month}}$$

Chu kỳ 12 tháng riêng biệt tạo ra độ phân tách góc $30^\circ$ giữa mỗi tháng liền kề (vượt trội so với việc chỉ cộng góc lẻ vào vòng tròn 60 năm khiến góc lệch giữa các tháng chỉ đạt $\approx 0.5^\circ$, không đủ độ nhạy để Transformer nhận diện).

### 3.3. Cơ chế Precision Masking Bảo vệ Tính Trung thực Dữ liệu

Sử dụng vector mặt nạ độ chính xác $\mathbf{m}_{\text{precision}} = [m_{\text{year}}, m_{\text{month}}] \in \{0, 1\}^2$:

$$h'_i = h_i + m_{\text{year}} \cdot \mathbf{W}_{\text{proj},Y} [\text{TE}(x_{\text{year}}) + \text{TE}(y_{\text{year}})] + m_{\text{month}} \cdot \mathbf{W}_{\text{proj},M} [\text{TE}(x_{\text{month}}) + \text{TE}(y_{\text{month}})]$$

- Khi văn bản chỉ có thông tin năm: $m_{\text{year}} = 1, m_{\text{month}} = 0$. Thành phần tháng bị triệt tiêu hoàn toàn.
- Khi văn bản có cả năm và tháng: $m_{\text{year}} = 1, m_{\text{month}} = 1$. Cả hai kênh phối hợp định vị chính xác mốc thời gian.

### 3.4. Hàm Mất Mát Căn Chỉnh Đa Độ Hạt (Multi-Granularity Alignment Loss)

$$\mathcal{L}_{T,\text{multi}} = \mathcal{L}_{T,\text{year}} + \gamma \cdot \mathcal{L}_{T,\text{month}}$$

Thành phần $\mathcal{L}_{T,\text{month}}$ kéo gần biểu diễn của các sự kiện diễn ra cùng tháng và đẩy xa các sự kiện khác tháng trong cùng một năm (chống hiện tượng suy biến biểu diễn nội năm).

### 3.5. Năm Điểm Cải Thiện Kỹ Thuật Phụ Bắt Buộc từ Rà Soát Paper Gốc (Ticktack v3)

Rà soát kỹ thuật bài báo gốc chỉ ra 5 khiếm khuyết toán học cần được hoàn thiện:

1. **Sửa lỗi hàm bán kính gây va chạm alias (Strict Radial Monotonicity):**
   Paper Ticktack v3 dùng $r = \alpha + \beta \theta$. Vì $\theta$ tuần hoàn theo chu kỳ 60 năm, hai năm cách nhau 60 năm (như 1965 và 2025) có cùng $\theta \implies$ cùng bán kính $r \implies$ trùng lặp tọa độ hoàn toàn.  
   *Giải pháp:* Thay bằng hàm bán kính đơn điệu theo kỷ nguyên tuyệt đối:
   $$r(t_{\text{AD}}) = \alpha + \beta \cdot \left\lfloor \frac{t_{\text{AD}} - t_0}{60} \right\rfloor$$
2. **Chuẩn hóa hàm mất mát theo batch (Batch-Normalized Cosine Loss):**
   Thay thế tổng cặp không chuẩn hóa ở Eq. 8 bằng mất mát trung bình trên các cặp duy nhất:
   $$\mathcal{L}_{\text{intra}} = \frac{1}{|P_{\text{same}}|} \sum_{(i,j) \in P_{\text{same}}} [1 - \cos(z_i, z_j)], \quad \mathcal{L}_{\text{inter}} = \frac{1}{|P_{\text{diff}}|} \sum_{(i,k) \in P_{\text{diff}}} \max(0, \cos(z_i, z_k) - m)$$
3. **Khôi phục lực đẩy phân tách chu kỳ ($\delta < 1$):**
   Đặt $\delta = 0.5$ thay vì $\delta = 1$ để bảo toàn thành phần $(1 - \delta)\mathcal{L}_{\text{inter}}$, giúp các lớp chu kỳ khác nhau phân tách rõ ràng trong không gian đặc trưng.
4. **Cơ chế tiêm thời gian cấp span (Temporal Span-Level Injection):**
   Chỉ tiêm vector $\text{TE}$ tại các token thuộc biểu thức thời gian (ví dụ: token của cụm `tháng 8 năm 1945`), không tiêm vào toàn bộ câu gây loãng ngữ nghĩa.
5. **Định vị ma trận Fisher EWC trên trọng số LoRA:**
   Tính ma trận Fisher Information trực tiếp trên các tham số adapter LoRA được cập nhật, bảo đảm ràng buộc toán học chuẩn xác khi base model bị đóng băng.

### 3.6. Ứng Dụng Tiến Bộ Kỹ Thuật và Mô Hình 2025–2026

- **Mô hình nền tảng nhỏ gọn (Compact Base Models):** **Qwen2.5-0.5B** hoặc
  **Qwen3-0.6B** chỉ là candidate. Ước lượng VRAM không chứng minh khả năng chạy; cần
  resource probe trên `local_validation` hoặc chuyển workload sang approved
  `external_gpu` với cùng contract.
- **RoPE Modulation / Attention-level Injection:** Tích hợp vector thời gian vào lớp attention hoặc RoPE dimension (*Time Is Encoded in the Weights* ACL 2024, *Date Fragments* EMNLP 2025) để ngăn tín hiệu thời gian bị triệt tiêu qua các tầng sâu.
- **Lấy mẫu âm tính khó theo thời gian (Temporal Hard Negative Sampling):** Kế thừa từ *ETRQA* (ACL 2025) và *ChronoSense* (ACL 2025), tạo các mẫu đối chứng cùng thực thể nhưng sai tháng để tối ưu hóa năng lực phân biệt niên đại.

### 3.7. Cổng Kiểm Soát Chất Lượng Suy Giảm Tri Thức Năm (Year-Retention Gate)

Mô hình đa độ hạt (Year + Month) chỉ được chấp nhận khi:
1. Đạt mức cải thiện có ý nghĩa thống kê trên tập kiểm thử có độ hạt tháng ($p < 0.05$).
2. Hiệu năng trên tập kiểm thử thuần năm (Year-only held-out benchmark) không bị suy giảm vượt quá ngưỡng cho phép $\Delta_{\text{tol}} \le 1.0\%$ so với Ticktack-year gốc.

---

## Phần 4: Vai Trò Baseline, Stress Test Và Future Work

Hai priority không biến mọi kỹ thuật thành contribution. Vai trò phải được giữ
đúng trong R0–R3:

### 4.1. Track A — Baseline và candidate retrieval
- **Bản chất:** BM25 / TF-IDF kết hợp lọc khoảng thời gian trên bảng SQLite interval.
- **Vai trò trong NCKH:** BM25/TF-IDF là baseline; interval filter, bounded rerank và evidence policy là candidate intervention, chỉ thành contribution khi R1 vượt R0 trên cùng candidate pool/qrels/evaluator.

### 4.2. Track B — Prior và candidate representation
- **Bản chất:** Ticktack `published-equation` là prior/baseline; dual-cycle, precision mask và year-to-month là candidate representation.
- **Vai trò trong NCKH:** Tách bằng R2/R3; tokenizer/calendar chỉ là stress test khi error slice tự nhiên đủ lớn và mapping có nguồn, version, uncertainty, reviewer.

### 4.3. Future work — Historical GraphRAG
- **Bản chất:** Trích xuất thực thể, quan hệ và mốc thời gian thành đồ thị tri thức (Knowledge Graph) để hỗ trợ truy xuất đa chặng (*multi-hop reasoning*).
- **Vị trí lộ trình:** Ngoài R0–R3 và critical path; chỉ mở bằng ADR mới khi có blocker đa chặng được chứng minh, resource budget và rollback.
