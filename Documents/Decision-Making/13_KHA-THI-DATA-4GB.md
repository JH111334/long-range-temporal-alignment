# Khả Thi Bài Toán, Dữ Liệu Và Compute

**Bối cảnh:** [Biểu diễn thời gian dài và truy xuất lịch sử](<C:/Users/GF63/Documents/long-range-temporal-alignment/Documents/Decision-Making/12_PROJECT-CONTEXT.md>). File này cung cấp tiền đề dữ liệu/phương pháp cho đề tài; từng nguồn chỉ có vai trò trong phép đánh giá được ghi, không tự cấp quyền tải hay chứng minh hiệu quả.

Tên file legacy được giữ để không phá validator và link lịch sử; policy hiện hành thuộc
`15_CAPABILITY-BASED-COMPUTE.md`, không còn khóa architecture vào 4 GB VRAM.

Ngày rà soát: 26/08/2026. Mọi con số quy mô dưới đây là **mục tiêu hoạch định**, chưa phải dữ liệu đã thu thập hay kết quả đo.

## Kết luận để thuyết phục nhóm

Hướng P1 khả thi nhất là **biểu diễn đa độ hạt + temporal evidence retrieval + historical QA có abstention** trên pilot tiếng Việt nhỏ nhưng sạch. Tính mới tập trung vào bước `year → month/day`; point/interval là trục cấu trúc độc lập. Precision-aware contract, split chống rò rỉ và evaluator quan trọng hơn tăng kích thước model.

| Gate | Mức tối thiểu để tiếp tục | Dừng/đổi hướng khi |
|---|---|---|
| Người dùng | Có quyết định thực cần hỗ trợ và tiêu chí lỗi chấp nhận được | Chỉ nhận phản hồi “hay” nhưng không có use case |
| Quyền dữ liệu | Mỗi nguồn có URL, terms/license, snapshot/revision, reviewer | Không được phép lưu/chia sẻ/biến đổi ở mức cần thiết |
| Gold/evaluator | Schema ổn định, double-review tập test, agreement và adjudication | Câu hỏi không có đáp án duy nhất hoặc evaluator phụ thuộc parser lỗi |
| Baseline | CPU chạy tái lập trên split đóng băng | Không tạo được baseline không-model có ý nghĩa |
| Cải tiến | Gain paired trên held-out và ít nhất một error slice quan trọng | Gain chỉ có ở random split/best seed |
| Tài nguyên | Profile đích có measured RAM/VRAM/time/cost budget và fallback | OOM, latency/chi phí vượt budget đã khóa |

## Dữ liệu cần có, theo thứ tự

### 1. Data-source dossier trước khi tải

Với từng nguồn, ghi: chủ sở hữu; URL chính thức; ngày truy cập; license/terms; phạm vi sử dụng; khả năng trích dẫn đoạn; snapshot/revision/checksum; ngôn ngữ; thời kỳ; granularity; calendar system; độ tin cậy; xung đột giữa nguồn; PII; kích thước; định dạng; chi phí thu thập; người rà soát; retention/deletion. Trang tổng hợp, mirror hoặc Kaggle không thay thế nguồn gốc.

Ưu tiên khảo sát: nguồn mở có đoạn dẫn ổn định và metadata rõ; Wikipedia/Wikidata chỉ là candidate có attribution theo item; sách giáo khoa/đề thi/scan chỉ dùng khi quyền phù hợp. Paper HisGraphRAG cho thấy bài toán lịch sử Việt Nam là khả thi về ý tưởng, không cấp quyền tái sử dụng corpus của họ.

### 2. Minimum viable data

- **P0 – contract set:** 30–50 ví dụ do nhóm tự viết từ nguồn được phép, chỉ để kiểm schema/parser; mang nhãn fixture, không đo chất lượng.
- **P1 – natural pilot:** mục tiêu 150–300 query và 500–2.000 evidence unit từ ít nhất 3 source family/2 thời kỳ; phủ `year|month|day`, `point|interval`, no-answer/conflict và hard negative. Quy mô cùng phân bổ chốt sau đo thời gian annotation.
- **P2 – benchmark nội bộ:** chỉ mở rộng khi agreement, leakage audit và baseline ổn; tăng theo lát cắt thiếu chứ không tăng đồng đều.
- Tập test cần double annotation; train có thể single annotation + review theo mẫu. Mọi item lưu `source_span`, `accessed_on`, annotator/reviewer và lý do adjudication.

### 3. Schema tối thiểu

`example_id, query, answer_set, raw_time_expression, interval, era, calendar_system, granularity, temporal_kind, precision_mask, uncertainty, event_id, entity_ids, source_id, source_span, evidence_ids, no_answer_reason, provenance, annotation_status, split_group`.

Không lưu năm âm để đại diện BCE/CE. `HistoricalYear` thực hiện ánh xạ hiển thị không có năm 0; khoảng dùng quy ước `[start,end)` và giữ độ chắc chắn. Năm thiếu tháng/ngày là toàn khoảng năm, không tự gán `01/01`. Giữ chuỗi gốc bên cạnh dạng canonical để audit.

## Thiết kế query và qrels

Lấy mẫu có chủ đích theo: BCE/CE; triều đại/thời kỳ; point/interval/duration/order; một hay nhiều evidence; khoảng cách tới train; alias tên người/địa danh; câu không đủ bằng chứng; nguồn đồng thuận/xung đột. Hard negative nên cùng thực thể nhưng sai thời điểm, cùng mốc nhưng sai sự kiện, hoặc cùng sự kiện từ nguồn yếu hơn.

Mỗi qrel phải trả lời: evidence nào đủ, evidence nào chỉ liên quan, evidence nào mâu thuẫn, và hệ thống phải abstain khi nào. Đo agreement riêng cho answer và evidence; không dùng agreement cao do nhãn “không liên quan” chiếm đa số.

## Split và chống rò rỉ

1. Đóng băng raw manifest và định nghĩa group trước preprocessing học từ dữ liệu.
2. Cluster near-duplicate theo normalized text + fingerprint; human review vùng ngưỡng.
3. Giữ cùng event/entity/source family trong một split khi nguy cơ ghi nhớ cao.
4. Có ít nhất source-held-out và time-held-out; random split chỉ là diagnostic, không làm kết quả chính.
5. Lưu split manifest/hash và candidate-pool hash. Không so metric giữa pool/qrels khác nhau.
6. Audit contamination thủ công trên top error và query có wording giống nguồn phổ biến.

## Baseline và cải tiến thông minh

| Bậc | Phương pháp | Câu hỏi cần trả lời |
|---|---|---|
| B0 | Majority/rule + BM25/TF-IDF | Dữ liệu/evaluator có hoạt động và retrieval lexical mạnh tới đâu? |
| B1 | Scalar year + interval filter | Chỉ thêm trục đơn điệu có giảm evidence sai thời điểm không? |
| B2 | Fourier/cycle features | Chu kỳ có giúp nhưng gây alias ở khoảng cách 60 năm không? |
| B3 | `published-equation` Ticktack | Công thức công bố có hơn B1/B2 khi giữ cùng corpus/parser không? |
| P1 | Precision-aware monotonic–multi-cycle cho `year|month|day` + point/interval evaluator + bounded rerank | Gain đến từ biểu diễn đa độ hạt hay từ filter/rerank/data cleaning? |
| P2 | Frozen small encoder | Semantic feature có lợi trên hard negative sau khi cache không? |
| P3 | PEFT/service | Chỉ mở khi B0–P2 chỉ ra lỗi model capacity thực sự |

Ablation bắt buộc: bỏ precision mask; bỏ interval filter; bỏ monotonic channel; bỏ cycle channel; đổi candidate pool; giữ nguyên prompt/parser. Báo EM/F1, absolute day/month error, endpoint/IoU/Allen F1, evidence Recall@k/nDCG, wrong-time, abstention, invalid-output, latency và peak RAM/VRAM; dùng nhiều seed/CI khi có stochastic training.

## Runbook theo capability profile

- `local_validation`: sample/bounded batch/small index; đo CPU/RAM rồi GPU nếu cần.
- `external_gpu`: full training hoặc large embedding/index qua immutable request có
  approval, input hash và expected output.
- `full_experiment`: chỉ mở multi-seed/long ablation sau probe có tín hiệu và local replay.
- Không suy khả năng chạy từ số tham số; profile đích phải ghi batch, input length và
  measured resource.
- Frozen inference: `eval()`, no-grad/inference mode, tuần tự, cache embedding theo model/data hash; kiểm normalization và dimension.
- Chỉ cân nhắc FP16 khi backend hỗ trợ ổn; quantization/offload là biến thí nghiệm, không là “miễn phí”.
- Rerank top 20–100, không cross-encode toàn corpus. Exact search trước ANN; ANN chỉ khi latency/RAM đo được là bottleneck.
- OOM phải giảm input/top-k/model và ghi artifact; không âm thầm chuyển CPU/provider/model khác.
- Mỗi probe lưu profile, host/GPU, driver, Python/package, precision, batch, length,
  p50/p95, peak VRAM/RAM, fallback và hash.

## Lộ trình 10 tuần có thể co giãn

1–2: phỏng vấn, problem/data contract, quyền nguồn. 3–4: pilot + annotation audit. 5: split/evaluator. 6: B0–B2 CPU. 7: error analysis và data repair. 8: Ticktack equation/frozen probe nếu gate đạt. 9: proposed method + ablation nhỏ. 10: clean rerun, report, Streamlit chỉ đọc artifact.

Nếu thiếu dữ liệu tự nhiên ở tuần 3, thu hẹp thành nghiên cứu **protocol/evaluator và temporal evidence retrieval**, không bù bằng synthetic rồi tuyên bố ứng dụng thực tế.
