# Phân Vùng Kiểm Toán Điểm Nghẽn Biểu Diễn Thời Gian (Representation Bottleneck)

Tài liệu này xác lập phạm vi, cấu trúc và nguyên tắc vận hành của không gian làm việc `REPRESENTATION_BOTTLENECK` thuộc Stage 03 (`03_TARGETED_OBJECTIVES_MCQA`).

## 1. Mục Tiêu Nghiên Cứu

Giai đoạn Stage 03 tập trung kiểm chứng giả thuyết về tính khả phân nhân quả của biểu diễn thời gian dài hạn trước khi tiến hành mở rộng quy mô dữ liệu hoặc huấn luyện lại toàn diện (`NO-GO SCALE`).
Mục tiêu cốt lõi bao gồm:
1. Xác định chính xác vị trí xuất hiện sự suy giảm tín hiệu trên chuỗi truyền dẫn:
   $$\text{HistoricalYear/time parse} \to z(t) \to \text{projection} \to \text{Seam injection} \to F_t \to \Delta W(t) \to h[\ell, p] \to \text{scores} \to \text{semantic flip}.$$
2. Tách bạch 5 cấp độ đo lường độc lập:
   - **Delivery**: Tín hiệu $z(t)$ và trọng số thích ứng $\Delta W(t)$ có được tính toán và đưa vào mô hình hay không.
   - **Sensitivity**: Kích hoạt ẩn $h[\ell, p]$ và phân phối logit có biến thiên theo thời gian điều kiện $t$ hay không.
   - **Selectivity**: Biến thiên logit của tín hiệu thời gian có vượt trội so với nhiễu ngẫu nhiên cùng chuẩn độ lớn ($\|z_{\text{rand}}\| = \|z(t)\|$) hay không.
   - **Semantic Correctness**: Sự thay đổi logit có dẫn đến việc chuyển dịch nhãn phân loại đúng với sự thật lịch sử (Gold Semantic Flip) hay không.
   - **Robustness**: Mô hình có bảo toàn khả năng mô hình hóa ngôn ngữ tự nhiên (NTP perplexity) và không bị sụp đổ nhãn thiên lệch (Label Collapse) hay không.
3. Phân biệt tường minh hai trục vật lý:
   - Trục tầng (Layer index $\ell \in \{0 \dots 23\}$).
   - Trục vị trí chuỗi (Token position $p$, phân biệt vị trí cổng Seam $p = 14$ và vị trí sinh từ tiếp theo $p = -1$).

## 2. Cấu Trúc Thư Mục

Phân vùng được chuẩn hóa theo cấu trúc:
- `config/`: Chứa cấu hình thực nghiệm sweep, tham số OpenTelemetry/Phoenix, schema giao diện biểu diễn và ngưỡng kiểm định.
- `contracts/`: Chứa các cam kết phân định trách nhiệm liên nhánh và quy chuẩn tương thích giao diện.
- `protocols/`: Chứa định nghĩa toán học các độ đo, hợp đồng đánh giá nhân quả và báo cáo thẩm định tập phản thực nghiệm.
- `runs/`: Lưu trữ các file log tổng hợp và chỉ mục trace của các lượt chạy kiểm định cục bộ.
- `reports/`: Chứa ma trận quyết định điểm nghẽn và báo cáo tổng hợp chi tiết cho GVHD.
- `writebacks/`: Sổ cái nối tiếp ghi nhận lịch sử đồng bộ kết quả kiểm định sang các hồ sơ tri thức trung tâm.

## 3. Ranh Giới Bất Biến

1. **Thư mục đối tác**: `C:\Users\GF63\Documents\Long-Range-Representation` duy trì chế độ **chỉ đọc nghiêm ngặt (STRICT READ-ONLY)**. Mọi giao tiếp thực hiện thông qua giao diện dữ liệu và checkpoint đã khóa.
2. **Khóa dừng mở rộng**: Không mở rộng tập huấn luyện hoặc khởi chạy huấn luyện tham số quy mô lớn khi chưa vượt qua cổng kiểm định tính chọn lọc và lật nhãn nhân quả.
3. **Fail-Open Tracing**: Hệ thống OpenTelemetry và Arize Phoenix hoạt động ở chế độ fail-open; sự cố kết nối collector không làm gián đoạn tiến trình đánh giá cục bộ.
