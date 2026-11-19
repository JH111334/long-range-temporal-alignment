# Khảo Sát Sau Ticktack Từ 03/2025 Đến 26/08/2026

## 1. Bản gốc và ranh giới tái lập

arXiv:2503.04150 được nộp ngày 06/03/2025 và sửa thành v3 ngày 21/10/2025. Ticktack báo cáo TempLS gồm 137.090 QA tiếng Trung, phạm vi 75.000 BCE–2025 CE và thí nghiệm trên tám A100; các số này là `paper_reported`. Lần rà soát hiện tại chưa xác minh được release chính thức có license/revision/checksum cho code hoặc TempLS ([Ticktack v3](https://arxiv.org/abs/2503.04150)).

Điểm cần xác lập chính xác: Ticktack chỉ xử lý **độ hạt năm**; tháng và ngày nằm
trong phần hạn chế/hướng tương lai. Project giữ hai track ưu tiên: **Track A —
External Temporal Knowledge Retrieval / Evidence** và **Track B — Internal
Temporal Model Representation / Alignment**. Track A dùng BM25/TF-IDF làm
baseline rồi mới thử interval filter/rerank; Track B dùng scalar/Fourier và
Ticktack làm prior/baseline rồi mới thử `year → month`. `point|interval` là trục
cấu trúc độc lập với granularity. R0–R3 trong `References/MAP.md` dùng để tách
causal contribution; không gộp baseline thành đóng góp.

---

## 2. Cải tiến liên quan từ 2025 đến 2026

| Hướng | Nguồn sơ cấp | Điều được cải thiện | Phần chuyển giao hẹp |
|---|---|---|---|
| Interval và event reasoning | [ETRQA](https://aclanthology.org/2025.findings-acl.1198/), [ChronoSense](https://aclanthology.org/2025.acl-short.46/) | Timespan, câu hỏi ghép, quan hệ sự kiện và 16 task Allen | Dùng point/interval, temporal hard negative sampling và relation metrics |
| Tokenization của ngày | [Date Fragments](https://aclanthology.org/2025.emnlp-main.159/) | Date fragmentation và benchmark ngày hiếm | Tokenizer probe trước fine-tune; RoPE modulation cho biểu thức thời gian |
| Temporal error và answerability | [TempAnswerQA](https://aclanthology.org/2025.findings-emnlp.637/) | EM không cho biết đáp án sai xa bao nhiêu | Báo distance/endpoint error, abstention và invalid output |
| Ngôn ngữ, tokenizer và task | [MultiTempBench](https://arxiv.org/abs/2603.19017), [TDBench](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2cbeff05cd646fcc09f6c3c48ba56f74-Abstract-Conference.html) | Mở rộng ngôn ngữ, format/lịch và task; chẩn đoán temporal competence | Dùng taxonomy/slice; đánh giá phân rã theo kỷ nguyên |
| QA thời gian tiếng Việt | [VLSP TemporalQA](https://aclanthology.org/2025.vlsp-1.35/), [QDoRA system](https://aclanthology.org/2025.vlsp-1.40/) | Date arithmetic/duration; parser, solver và canonicalization có ích | Benchmark kỹ năng phụ; cấu hình 7B/2×5090 không phải local path |
| Retrieval lịch sử Việt Nam | [HisGraphRAG](https://aclanthology.org/2025.paclic-1.49/) | Entity alignment, temporal retrieval/rerank và lọc context | Giữ retrieval/filter baseline; tách gain retriever khỏi representation |
| Không trả lời khi thiếu bằng chứng | [When Silence Is Golden](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7718914dfe7d5a657bf6261b5f431021-Abstract-Conference.html) | Huấn luyện và đánh giá khả năng từ chối trong temporal QA | Đưa abstention/risk-coverage vào output contract |

---

## 3. Hai track ưu tiên và giả thuyết mở rộng tháng

### 3.1. Phân tích phản biện: giả thuyết cần data/resource gate
- Ngày thưa và năm gộp sự kiện nội năm là các giả thuyết phải đo theo source,
  thời kỳ và provenance; chưa dùng tỷ lệ thiếu hoặc nhãn Pareto khi chưa có
  manifest.
- Tháng là candidate của Track B vì giữ được cấu trúc `year → month`; mức độ
  hữu ích phải thể hiện qua R2/R3 và lát cắt tiếng Việt, không qua overall score.
- Overhead kênh tháng, model 0,5B–1,5B và QLoRA trên RTX 2050 `4.096 MiB` chỉ
  được kết luận sau resource probe.

### 3.2. Cấu trúc Chu kỳ kép và Precision Masking
1. **Chu kỳ kép (Dual-Cycle):**
   - Vòng vĩ mô 60 năm: $\theta_{\text{year}} = \frac{2\pi t_{\text{cycle}}}{60}, r_{\text{year}} = \alpha_Y + \beta_Y \lfloor \frac{t_{\text{AD}} - t_0}{60} \rfloor \implies (x_{\text{year}}, y_{\text{year}})$.
   - Vòng trung mô 12 tháng: $\theta_{\text{month}} = \frac{2\pi (m - 1)}{12}, r_{\text{month}} = \alpha_M + \beta_M (\frac{m}{12}) \implies (x_{\text{month}}, y_{\text{month}})$.
2. **Cơ chế Precision Masking:** Vector $\mathbf{m}_{\text{precision}} = [m_{\text{year}}, m_{\text{month}}]$. Khi dữ liệu chỉ có năm, $m_{\text{month}} = 0$; tuyệt đối không gán mặc định tháng 1 (nguyên tắc fail-closed).

### 3.3. Năm điểm cải thiện kỹ thuật phụ từ bài báo gốc (Ticktack v3)
1. **Sửa lỗi hàm bán kính (Strict Radial Monotonicity):** Sửa $r = \alpha + \beta \theta$ thành hàm đơn điệu ngặt theo kỷ nguyên để triệt tiêu va chạm alias giữa các năm cùng lớp chu kỳ.
2. **Chuẩn hóa mất mát theo batch (Batch-Normalized Cosine Loss):** Dùng Mean Unique-Pair Loss với margin thay cho tổng cặp unnormalized.
3. **Khôi phục lực đẩy phân tách chu kỳ:** Đặt $\delta = 0.5$ thay vì $\delta = 1$ để không triệt tiêu $(1 - \delta)\mathcal{L}_{\text{inter}}$.
4. **Tiêm thời gian cấp span (Span-Level Injection):** Chỉ tiêm $\text{TE}$ tại token thuộc biểu thức thời gian xác định.
5. **Định vị Fisher EWC trên LoRA:** Tính ma trận Fisher trực tiếp trên các tham số LoRA trainable.

---

## 4. Vai trò baseline, stress test và future work

1. **Track A baseline:** BM25/TF-IDF + bảng interval SQLite. Filter/rerank là
   candidate intervention, không phải baseline hay contribution mặc định.
2. **Track B baseline:** Scalar/Fourier và Ticktack `published-equation`;
   dual-cycle/precision mask là proposed candidate.
3. **Tokenizer/calendar probe:** Chỉ mở khi error slice tự nhiên đủ lớn; mapping
   can-chi/âm lịch/niên hiệu cần nguồn, version, uncertainty và reviewer.
4. **Historical GraphRAG:** Future work; vector DB và hạ tầng nặng không nằm
   trong R0–R3 hoặc critical path hiện tại.
