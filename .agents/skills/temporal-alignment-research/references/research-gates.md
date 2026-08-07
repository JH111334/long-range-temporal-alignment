# Research Gates

## Dataset gate

- URL chính thức; ngày truy cập; license/terms.
- Revision/checksum; reviewer; train/validation/test role.
- Group keys, time boundary, source boundary và near-duplicate audit.
- Nguồn do LLM sinh phải ghi model/prompt/revision và tỷ lệ human review.

## Representation gate

- Year-zero convention và BCE/CE round-trip test.
- Dimension cố định, finite values, không zero-vector ngoài thiết kế có giải thích.
- Alias-period collision và chronological-distance correlation.
- So sánh với scalar và Fourier trước biểu diễn phức tạp.

## Objective gate

- Self-pair/ordered-pair policy; mean hay sum; class weighting.
- Sensitivity khi nhân đôi batch và khi class imbalance.
- Delta/margin/coefficient được ghi trong config.
- EWC chỉ áp dụng lên tham số có current/reference/Fisher tương thích.

## Model/resource gate

- Model ID, revision, license và tokenizer revision.
- Peak RAM/VRAM đo trên sequence length và micro-batch cụ thể.
- Precision, quantization, gradient checkpointing và offload được ghi.
- Không suy ra “vừa 4 GB VRAM” từ số tham số hoặc lời quảng cáo.

## Evaluation gate

- Prompt/parser/evaluator cố định giữa phương pháp.
- Báo invalid-output rate tách khỏi accuracy/F1/ROUGE.
- Ít nhất time-held-out và group/source-held-out khi dữ liệu cho phép.
- Multi-seed, confidence interval và error slices trước tuyên bố cải thiện.
