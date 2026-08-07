# ADR 0002: Evidence Gate Cho Dữ Liệu

## Quyết định

Không ingest dataset thật khi thiếu một trong các trường: nguồn chính thức, ngày truy cập, giấy phép/điều khoản, revision, checksum, người rà soát, data role và split policy.

Candidate và hold chỉ phục vụ khảo sát metadata. Dữ liệu tổng hợp được phép dùng để kiểm thử hợp đồng nhưng phải mang nhãn `synthetic` hoặc `smoke_test`.

## Lý do

Temporal QA có nguy cơ leakage theo nguồn, thực thể, sự kiện và near-duplicate. Random split có thể đo ghi nhớ mẫu thay vì khái quát theo thời gian.
