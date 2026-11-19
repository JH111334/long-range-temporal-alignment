---
title: "Biểu diễn thời gian và chuẩn hóa lịch sử"
lang: vi-VN
---

# Internal time

API dùng `HistoricalYear(era, year)`, `year ≥ 1`. Astronomical numbering chỉ là biểu diễn nội bộ:

$$
y_{astro}=\begin{cases}y,&\text{CE}\\1-y,&\text{BCE}\end{cases}
$$

Do đó 1 BCE ánh xạ 0, 2 BCE ánh xạ -1 và 1 CE ánh xạ 1; không có display year 0. Khoảng dùng `[start, end_exclusive)` để tránh nhập nhằng inclusive/exclusive.

# Baseline bắt buộc

## Gregorian scalar

$$e_{scalar}(y)=y_{astro}/S$$

`S` là scale ghi trong config. Baseline kiểm tra thứ tự và extrapolation, nhưng một chiều có thể khó tích hợp với embedding model.

## Fourier đa chu kỳ

Với chu kỳ $p_i$:

$$e_{Fourier}(y)=\left[\sin(2\pi y/p_i),\cos(2\pi y/p_i)\right]_{i=1}^{k}$$

Các $p_i$ là `operational_assumption`; phải freeze trong config. Fourier là baseline mạnh, trung lập hơn một lịch văn hóa duy nhất nhưng vẫn có alias khi tổ hợp chu kỳ lặp.

# Published Ticktack

Sexagenary class dùng modulo 60 và góc $\theta=2\pi c/60$. Paper mô tả $r=\alpha+\beta\theta$, sau đó $x=r\cos\theta$, $y=r\sin\theta$ và temporal encoding. Nếu $\alpha,\beta$ là hằng số theo class, các năm cách 60 năm có thể trùng representation; code giữ hành vi này để audit.

`OperationalRadialTicktackEncoder` thêm cycle index vào bán kính. Nó không phải exact reproduction vì paper không đặc tả đầy đủ hàm bán kính.

# Proposed multi-granularity hybrid

Hybrid nối trục tuyệt đối đơn điệu với các thành phần chu kỳ và precision mask. Với năm-only, month/day không được bịa; record biểu diễn toàn khoảng năm. Với month/day, dùng absolute month/day index cùng month, day-of-month và day-of-year cycle khi hợp lệ:

$$e_{multi}(t,g)=\left[e_{absolute}(t);\ e_{cycle}(t);\ e_{interval}(t);\ m_g\right].$$

`g` là granularity `year|month|day` và $m_g$ là precision mask; point/interval được mã hóa bằng trường cấu trúc riêng. Đây là `proposed-method`, không phải công thức Ticktack. Chỉ giữ nếu month/day và interval slice cải thiện tách biệt trên held-out, collision giảm và year-only không suy giảm quá ngưỡng đã khóa. Period/scale/dimension phải được ablate; không chọn theo test score.

# Interval representation

Một interval cần start, end-exclusive, duration, endpoint granularity, uncertainty và relation features; không thay bằng midpoint duy nhất. `temporal_kind` tách point/interval/duration/relation khỏi độ chính xác lịch. ChronoSense cho thấy các quan hệ Allen, kể cả quan hệ đối xứng, vẫn khó và có dấu hiệu memorization ([Islakoglu & Kalo, 2025](https://aclanthology.org/2025.acl-short.46/)).

# Chuẩn hóa tiếng Việt

Tier 0 chỉ parse nhãn TCN/SCN tường minh. Các trường hợp sau phải từ chối hoặc chuyển review queue:

- “năm 938” không có era trong contract nghiêm ngặt;
- niên hiệu, can-chi hoặc âm lịch chưa có mapping table approved;
- khoảng mơ hồ như “đầu thế kỷ”, “khoảng năm…” thiếu uncertainty policy;
- ngày/tháng bị thiếu nhưng pipeline cố nâng thành `01/01`;
- calendar reform/timezone khi độ hạt tháng/ngày chưa hỗ trợ.

# Tokenizer probe

Date Fragments cho thấy BPE có thể chia ngày thành fragment thiếu nghĩa và liên hệ với giảm accuracy ở ngày hiếm ([Jiang et al., 2025](https://aclanthology.org/2025.emnlp-main.159/)). Trước fine-tune phải đo token count, fragmentation ratio, format invariance và layer-wise representation cho TCN/SCN, năm dài, khoảng và nhiều cách viết tiếng Việt.

# Metric representation

- dimension, finite values, norm distribution;
- alias collision ở ±60 và các least-common-period;
- chronological-distance correlation;
- local smoothness ở ±1 và BCE/CE boundary;
- interval-relation separability;
- hierarchy consistency giữa year/month/day và downstream evidence/answer metrics trên cùng retrieval/parser.

Embedding plot chỉ là minh họa. Không dùng t-SNE cluster để thay metric hoặc ablation.
