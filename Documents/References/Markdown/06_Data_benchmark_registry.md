# Data Và Benchmark Registry

## Ma trận quyết định

| Dataset | Phạm vi | Availability | License/terms | Vai trò dự kiến | Quyết định |
|---|---|---|---|---|---|
| TempLS | 75.000 BCE–2025 CE, Chinese MCQA, 137.090 cặp theo paper | Chưa thấy release chính thức | Chưa xác minh | Reproduction target | Hold |
| TAQA | 2000–2023, 20.148 time-sensitive QA | Repo/HF tác giả | Apache-2.0 hiển thị | Probe temporal alignment | Candidate; freeze revision |
| TempUN | Khoảng dài, dữ liệu Our World in Data | Paper có; repo hiện chưa xác nhận | Cần kiểm tra | Long-range MCQA | Hold |
| TempLAMA | 2010–2020, temporal factual QA | Paper/release cần đối chiếu | Cần kiểm tra | Benchmark ngắn hạn | Hold |
| ToT | Synthetic temporal reasoning | HF tác giả | Cần kiểm tra card/file | Contamination-controlled test | Candidate |
| DateAugBench | Historical/future date stress test | Paper 2025 | Release cần xác minh | Tokenizer/date robustness | Candidate metadata |
| TRD | Multilingual synthetic, 5 mức khó | Repo Amazon Science | Xác minh trước ingest | Reasoning slices | Candidate |
| VLSP 2025 TemporalQA | 9.000 date arithmetic + 2.873 duration QA tiếng Việt | Organizer/ACL paper | Dataset license/release checksum chưa xác minh | Benchmark kỹ năng phụ | Hold |
| Vietnam multi-granularity history pilot | Nguồn Việt Nam phủ `year|month|day`, `point|interval`, no-answer/conflict | HisGraphRAG/VLSP chỉ là prior art | Quyền tái sử dụng/chia sẻ chưa xác minh | Primary use case | Hold; lập source dossier trước |
| HistoryBankQA | 10M+ event, 10 ngôn ngữ, không có tiếng Việt | ACL paper + code/data links | Exact release terms/checksum cần khóa | International stress test | Hold |
| ComplexTempQA | 100M+ QA, 1987–2023 | ACL paper + repo | Quá lớn cho pilot; license/revision cần khóa | Taxonomy/multi-hop stress | Metadata only |

## Gold data tối thiểu nếu phải tự xây

1. Định nghĩa task và input/output trước thu thập.
2. Mẫu từ ít nhất ba source family/hai thời kỳ, phủ `year|month|day`, `point|interval` và no-answer/conflict.
3. Mỗi item có raw expression, interval `[start,end)`, `granularity`, `temporal_kind`, precision mask, uncertainty, source span, URL, ngày truy cập, answer, annotator và reviewer.
4. Double annotation cho tập đánh giá; adjudication cho bất đồng.
5. Split theo source/entity/event/time; deduplicate trước gán nhãn nhưng sau khi đóng băng raw manifest.
6. Pilot nhỏ để đo agreement/error taxonomy trước mở rộng.
7. Với lịch sử Việt Nam, lưu cả chuỗi gốc, TCN/SCN chuẩn, interval, calendar system, niên hiệu/can-chi nếu có, uncertainty và nguồn conversion.

## Không làm trong giai đoạn đầu

- Không scrape hàng loạt trước khi chốt task.
- Không dùng LLM sinh đáp án rồi gọi là gold nếu không human review.
- Không đưa cùng một event/template vào train và test chỉ vì wording khác.
- Không trộn prompt/parser khác nhau giữa baseline và phương pháp.
