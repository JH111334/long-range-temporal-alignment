# Báo Cáo Kiểm Toán Tập Dữ Liệu Phản Thực Nghiệm (Counterfactual Gold Audit)

## 1. Tóm Tắt Tổng Quan

- **Tổng số cặp kiểm toán**: 54 cặp (108 câu hỏi đơn lẻ).
- **Cặp Gold-Flip (Primary Endpoint)**: 36 / 36 cặp được công nhận (`admitted_primary: true`).
- **Cặp Gold-Same (Negative Controls)**: 18 / 18 cặp được công nhận (`admitted_primary: true`).
- **Cặp loại trừ / Vi phạm quy chuẩn**: 0 cặp.
- **Số lượng cụm thời gian (Temporal Clusters)**: 23.
- **Số lượng thực thể lịch sử độc lập (Unique Entities)**: 34.

## 2. Tiêu Chí Thẩm Định Độc Lập

1. **Tính hợp lệ của năm lịch sử**: Toàn bộ mốc năm là số nguyên khác 0, tuân thủ đúng quy ước phân tách BCE/CE theo quy chuẩn dự án.
2. **Tính nghiêm ngặt của nhãn chân lý**:
   - Nhánh **Gold-Flip**: Bắt buộc thỏa mãn $y^*(t_1) \ne y^*(t_2)$ qua sự kiện lịch sử xác định (ví dụ: trước/sau thống nhất đất nước, tan rã liên bang, gia nhập tổ chức).
   - Nhánh **Gold-Same**: Bắt buộc thỏa mãn $y^*(t_1) = y^*(t_2)$ (ví dụ: thực thể đã tồn tại ổn định suốt khoảng thời gian quan sát) làm đối chứng âm nhằm phát hiện xu hướng lật giả do nhạy cảm với bất kỳ sự thay đổi năm nào.
3. **Tính kiểm chứng của xuất xứ (Provenance)**: Mỗi cặp đều gắn liền với văn kiện, hiệp ước hoặc mốc lịch sử xác thực cụ thể.

## 3. Bảng Phân Bổ Theo Cụm Lịch Sử

| Cụm Thời Gian (Temporal Cluster) | Số Cặp Flip | Số Cặp Same | Tổng Số Cặp |
|---|---|---|---|
| `21ST_CENTURY_NATIONHOOD` | 2 | 0 | 2 |
| `AVIATION_HISTORY` | 1 | 0 | 1 |
| `CIVIL_RIGHTS_AFRICA` | 1 | 0 | 1 |
| `COLD_WAR_COLLAPSE` | 3 | 0 | 3 |
| `CONSTITUTIONAL_TRANSITIONS` | 2 | 0 | 2 |
| `EUROPEAN_INTEGRATION` | 1 | 0 | 1 |
| `EUROZONE_MONETARY_HISTORY` | 4 | 0 | 4 |
| `GLOBAL_TRADE_GOVERNANCE` | 3 | 0 | 3 |
| `HUMAN_RIGHTS_EUROPE` | 1 | 0 | 1 |
| `INFRASTRUCTURE_HISTORY` | 1 | 0 | 1 |
| `NORTH_AFRICA_DECOLONIZATION` | 1 | 0 | 1 |
| `POST_COLD_WAR_AFRICA` | 1 | 0 | 1 |
| `POST_WAR_EUROPE` | 1 | 0 | 1 |
| `SOUTHEAST_ASIA_COLD_WAR` | 1 | 0 | 1 |
| `SOUTHEAST_ASIA_DECOLONIZATION` | 2 | 0 | 2 |
| `SOUTHEAST_ASIA_REGIONALISM` | 1 | 0 | 1 |
| `SOUTH_ASIA_DECOLONIZATION` | 1 | 0 | 1 |
| `SPACE_EXPLORATION` | 2 | 0 | 2 |
| `STABLE_HISTORICAL_CONTROL` | 0 | 18 | 18 |
| `SUFFRAGE_AMERICAS` | 1 | 0 | 1 |
| `SUFFRAGE_EUROPE` | 1 | 0 | 1 |
| `UN_MEMBERSHIP_MILESTONES` | 3 | 0 | 3 |
| `VDEM_SUBSTRATE` | 2 | 0 | 2 |

## 4. Kết Luận Kiểm Toán

Tập dữ liệu phản thực nghiệm đạt chuẩn kiểm định khoa học (`PASS`), đủ điều kiện làm ground truth cho tiến trình bóc tách điểm nghẽn nhân quả Stage 03 mà không cần bổ sung hoặc tạo lập thêm dữ liệu tổng hợp.
