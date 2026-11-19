---
title: "Objective, EWC, LoRA và ablation"
lang: vi-VN
---

# Ranh giới bằng chứng

Objective in từ paper được triển khai dưới nhãn `published-equation`, kể cả khi có nhạy batch. Mọi normalization, margin, weighting hoặc adapter-aware EWC là `proposed-method`/`operational-assumption` cho tới khi được benchmark.

Objective/EWC/LoRA chỉ là ablation phụ của hướng multi-granularity. Không mở PEFT nếu lexical–interval baseline, data slices và representation comparison chưa khóa; kết quả âm tính hợp lệ không được dùng để tăng model size vô hạn.

# Published temporal alignment loss

Paper dùng các tổng intra/inter pair và một hệ số $\delta$ chịu ảnh hưởng batch. Việc đặt $\delta=1$ có thể triệt hệ số $(1-\delta)$ của inter-class term theo cách đọc phương trình hiện tại. Các chi tiết self-pair, ordered/unordered pair và normalization chưa đủ rõ.

Baseline phải lưu riêng total, intra, inter, số cặp, self-pair policy và class count. Full-batch duplication test xác nhận loss sum đổi khi số cặp đổi.

# Proposed normalized objective

Phương án nghiên cứu dùng mean trên unique unordered pairs, loại self-pair, ghi class weighting và inter-margin trong config. Mục tiêu là giảm phụ thuộc batch/class imbalance, không phải mặc định tăng accuracy.

Gate hỗ trợ giả thuyết:

- loss/gradient ổn định khi nhân đôi batch;
- không collapse class hoặc zero gradient;
- downstream gain giữ trên cùng data/split/parser/evaluator;
- coefficient sensitivity không chỉ chọn một điểm tốt nhất trên test.

# EWC với LoRA

EWC chỉ hợp lệ khi mỗi parameter có current value, reference value và Fisher tương thích. Nếu base frozen và chỉ adapter trainable, penalty trên base không tạo cập nhật. Adapter-aware EWC cần checkpoint/reference adapter cùng parameter names và Fisher estimator được mô tả.

Artifact bắt buộc ghi:

- trainable/frozen parameter names và fraction có gradient;
- checkpoint tham chiếu và checksum;
- Fisher dataset/split, estimator, sample count và normalization;
- $\lambda_{EWC}$, penalty theo parameter group;
- retention metric và temporal metric.

Nếu penalty gần zero hoặc retention không hơn L2/early stopping, loại EWC khỏi phương pháp cuối.

# LoRA contract

Ghi model/tokenizer revision, rank, alpha, dropout, target modules, bias policy, precision, quantization, optimizer, learning rate, seed, sequence length, micro-batch, accumulation, checkpointing, peak RAM/VRAM và latency. LoRA ít trainable parameter không đồng nghĩa activation vừa 4 GB VRAM.

# Ma trận ablation tối thiểu

| ID | Retrieval/encoding | Objective | EWC | Vai trò |
|---|---|---|---|---|
| A | BM25/raw time | Không | Không | Deterministic/retrieval baseline |
| B | Semantic/raw time | Next-token | Không | Model control |
| C | Published Ticktack | Next-token | Không | Encoding-only |
| D | Published Ticktack | Published loss | Không | Alignment contribution |
| E | Published Ticktack | Published loss | Có | Operational reproduction |
| F | Interval retrieval + multi-granularity hybrid | Normalized | Không | Proposed core |
| G | Interval retrieval + multi-granularity hybrid | Normalized | Có | Retention trade-off |

Mọi run dùng cùng corpus, data revision, frozen split, top-k, prompt, parser, evaluator và compute budget. Retrieval ablation phải tách time filter khỏi evidence filtering vì HisGraphRAG cho thấy temporal/rerank đứng riêng có thể kéo nhiễu ([Nguyen et al., 2025](https://aclanthology.org/2025.paclic-1.49/)).

# Điều kiện mở PEFT

Chỉ mở khi dataset/evaluator approved, deterministic và retrieval baselines đã chạy, tokenizer/frozen probe có paired held-out signal, resource probe đạt và GVHD chấp nhận giả thuyết. Kết quả âm tính được giữ nếu artifact đầy đủ.
