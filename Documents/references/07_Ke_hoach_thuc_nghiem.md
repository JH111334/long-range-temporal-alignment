# Kế Hoạch Thực Nghiệm

## P0. Reproduction audit

- Chốt paper v3, công thức, bảng kết quả và trường còn thiếu.
- Chạy unit test cho BCE/CE, modulo 60, alpha/beta, self-pair và delta.
- Đầu ra: bảng “specified / missing / assumption / test”.

## P1. Representation benchmark

So sánh scalar, Fourier, published Ticktack, operational radial và hybrid trên synthetic grid rộng. Metric: alias collision, chronological-distance correlation, dimension, finite-value, BCE/CE boundary.

## P2. Data/evaluator pilot

Chọn một dataset public đã qua gate. Khóa revision/checksum, xây group/time split và parser. Báo invalid-output rate riêng.

## P3. Frozen model probe

Chạy Qwen2.5-0.5B candidate sau resource probe. So raw hidden representation với temporal feature injection; chưa fine-tune.

## P4. PEFT ablation

| Run | Encoding | Temporal loss | EWC | Mục đích |
|---|---|---|---|---|
| A | Không | Không | Không | Base |
| B | Không | Không | Không | Next-token post-train control |
| C | Published | Không | Không | Encoding-only |
| D | Published | Published Eq. 8 | Không | Alignment-only |
| E | Published | Published Eq. 8 | Có | Ticktack operational reproduction |
| F | Hybrid | Normalized | Không | Proposed core |
| G | Hybrid | Normalized | Có | Retention test |

## P5. Robustness

- Ít nhất 3 seed khi chi phí cho phép.
- Time/source/entity-held-out; BCE/CE; near/far years; granularity; language.
- Bootstrap confidence interval hoặc paired test trên cùng examples.
- Retention task để kiểm tra general capability.

## Điều kiện dừng

Dừng tăng model size nếu hiệu ứng không vượt baseline Fourier trên split chống leakage, evaluator chưa ổn định, hoặc resource probe vượt ngân sách.
