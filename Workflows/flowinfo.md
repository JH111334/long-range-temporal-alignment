# Workflow Nghiên Cứu

1. Chốt problem contract cho historical temporal QA tiếng Việt; ghi giả thuyết người dùng chưa xác minh.
2. Ghi nguồn vào registry; chỉ thu thập khi URL, terms, revision/checksum, reviewer và role đạt.
3. Giữ raw snapshot bất biến; chuẩn hóa TCN/SCN, interval, calendar system, uncertainty và provenance.
4. Quality/dedup trước; chia group/event/entity/source/time trước preprocessing học từ dữ liệu.
5. Khóa hai track: A là external temporal retrieval/evidence; B là internal temporal representation/alignment.
6. Chạy baseline đúng vai trò: BM25/TF-IDF cho A; scalar/Fourier và Ticktack `published-equation` cho B.
7. Chạy R0–R3 với cùng corpus, parser, split, candidate pool, evaluator và answer policy.
8. Báo evidence recall, wrong-time, temporal order, year-retention, invalid/abstention và latency theo lát cắt.
9. Chạy resource probe trước frozen model/LoRA; khóa model/tokenizer revision, precision và quantization.
10. Mỗi run xuất config/data/code hash, resource peak, seeds/CI, paired delta và warnings; release chỉ trả answer có evidence.
11. GraphRAG/vector DB/production chỉ mở bằng ADR mới sau khi error evidence chứng minh blocker.
12. Cập nhật registry, ADR, risk register, error analysis và brief trước buổi họp GVHD.

## Gate chuyển profile

- Khảo sát → pilot: problem contract được duyệt; dataset qua evidence gate.
- `local_validation` → `external_gpu`: test/validator đạt; split/evaluator ổn định;
  hypothesis cần compute lớn; input public/licensed cùng request hash/approval đã khóa.
- `external_gpu` → `full_experiment`: probe cho tín hiệu, artifact được local replay và
  giả thuyết cần multi-seed/long confirmation.
- Model → ứng dụng thử: release card, evidence/abstention contract, rollback và drift metrics hoàn chỉnh.

Mọi profile giữ cùng schema, failure semantics, artifact và evaluator. Source ngoài đi
qua `SourceAPI` validation/normalization trước data lifecycle.
