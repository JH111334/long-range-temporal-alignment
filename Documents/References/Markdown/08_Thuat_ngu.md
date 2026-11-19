# Thuật Ngữ Cốt Lõi

| Thuật ngữ | Nghĩa dùng trong dự án |
|---|---|
| Temporal alignment | Điều chỉnh representation/weight/output để tri thức gắn đúng mốc hoặc khoảng thời gian |
| Long-range | Phạm vi vượt internet age, có thể gồm BCE và nhiều thế kỷ |
| Temporal representation | Vector mã hóa năm/tháng/ngày/interval |
| Sexagenary cycle | Chu kỳ 60 lớp can-chi; có tính chu kỳ nên dễ alias |
| Alias collision | Hai mốc khác nhau cho vector giống hoặc gần như giống vì cùng chu kỳ |
| Monotonic axis | Thành phần thay đổi có thứ tự theo thời gian, giúp phân biệt các chu kỳ lặp |
| Fourier features | Cặp sin/cos tại nhiều chu kỳ |
| Temporal QA | Hỏi đáp mà đáp án phụ thuộc mốc/khoảng thời gian |
| Extrapolation | Đánh giá ngoài vùng thời gian đã thấy khi train |
| Leakage | Thông tin test lọt vào train qua duplicate, source, event, entity hoặc preprocessing |
| EWC | Penalty theo Fisher nhằm giữ tham số quan trọng của task cũ |
| LoRA | Fine-tuning bằng các ma trận low-rank, base weights thường đóng băng |
| PEFT | Nhóm kỹ thuật fine-tuning ít tham số |
| Operational assumption | Chi tiết nhóm phải bổ sung vì paper chưa nêu |
| Provenance | Nguồn gốc, revision, checksum và quá trình tạo artifact |
| Invalid-output rate | Tỷ lệ output không parse được theo contract, báo riêng khỏi accuracy |
| Ablation | Bỏ/thêm từng thành phần để xác định nguồn hiệu ứng |
| Smoke test | Kiểm tra hệ thống chạy và contract đúng; không phải đo chất lượng |
| Measured result | Kết quả có artifact, config/data/code revision, seed và resource thực |
| Evidence-grounded QA | Câu trả lời bắt buộc liên kết tới passage/source ID có provenance |
| Abstention | Từ chối trả lời khi evidence, calendar mapping hoặc confidence không đạt gate |
| Date fragmentation ratio | Mức tokenizer chia thành phần ngày/năm thành các fragment thiếu nghĩa |
| Allen relation | Quan hệ giữa hai khoảng như before, overlaps, during, meets |
| Temporal distance error | Độ xa giữa mốc dự đoán và gold; bổ sung cho exact match |
| Data lineage | Quan hệ raw → derived → split → artifact kèm script/config/hash |
| Release gate | Điều kiện model/data/evaluator/resource phải đạt trước khi ứng dụng dùng một release |
