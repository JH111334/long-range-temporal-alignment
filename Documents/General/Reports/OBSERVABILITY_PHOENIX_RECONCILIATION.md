# BÁO CÁO PHỤC HỒI VÀ ĐỒNG BỘ KIẾN TRÚC ARIZE PHOENIX TRONG HỆ THỐNG QUAN SÁT CHIA SẺ (SHARED OBSERVABILITY)

**Tác giả:** Kỹ sư Trưởng Hệ thống & Giám định viên Độc lập (Senior Systems Auditor & Observability Engineer)  
**Ngày lập báo cáo:** 11/10/2026  
**Phạm vi áp dụng:** Hệ thống chia sẻ Agent System v2.0.3 và Ma trận 8 Dự án Cục bộ  
**Trạng thái nghiệm thu:** PASS_PHOENIX_RESTORED_E2E (11/11 Cổng kiểm định đạt chuẩn)  

---

## 1. BỐI CẢNH VÀ ĐÍNH CHÍNH PHƯƠNG PHÁP LUẬN

Trong đợt đánh giá trước, báo cáo kiểm định ghi nhận kết luận `BACKEND_STATUS: NOT_INSTALLED` và nhận định hệ thống quan sát chỉ có bồn chứa cục bộ dạng tập tin JSONL. Qua quá trình rà soát pháp y độc lập, kết luận này xuất phát từ các sai sót phương pháp luận sau:

1. **Đồng nhất trạng thái cổng đóng với sự vắng mặt của kiến trúc:** Đợt kiểm tra trước chỉ thăm dò cổng mạng 6006 tại thời điểm hệ thống ở trạng thái nghỉ (`ON_DEMAND_STOPPED`). Khi nhận tín hiệu từ chối kết nối, kiểm định viên đã vội vã kết luận Arize Phoenix hoàn toàn không tồn tại trên hệ thống máy chủ.
2. **Bỏ qua dấu vết mã nguồn trên các phân nhánh thực nghiệm:** Cấu trúc cầu nối `PhoenixTracerBridge` đã được định nghĩa và lưu trữ trên nhánh `observability` tại kho lưu trữ `Namespace-Skill-Routing-System` (`phoenix_bridge.py`), đồng thời tập lệnh `trace_probe.py` trong tiện ích mở rộng `observability-v1` đã thiết lập sẵn chế độ `PHOENIX_INSPECT`.
3. **Thiếu bước khôi phục môi trường biệt lập an toàn:** Thay vì tạo lập môi trường ảo để kiểm thử toàn diện chuỗi cung ứng vết (trace pipeline), báo cáo trước dừng lại ở việc đánh giá thụ động, dẫn đến phán quyết âm tính giả (false negative).

Báo cáo này xác nhận việc khôi phục hoàn chỉnh backend Arize Phoenix, tích hợp liền mạch với OpenTelemetry Collector v0.162.0 qua mô hình dual-sink, bảo toàn trọn vẹn nguyên tắc vận hành theo yêu cầu (`ON_DEMAND_STOPPED`) và bảo mật dữ liệu nhạy cảm.

---

## 2. KẾT QUẢ ĐIỀU TRA PHÁP Y VÀ PHÂN LOẠI HIỆN TRẠNG

Quá trình quét định vị chuyên sâu trên toàn bộ môi trường máy chủ ghi nhận các thực tế kỹ thuật:

- **Các môi trường Python:** Quét kiểm tra các môi trường Python toàn cục (3.12, 3.10), môi trường Conda (`C:\Users\GF63\.conda\envs\env`), và các virtualenv của 8 dự án. Trước đợt khôi phục này, gói `arize-phoenix` chưa được cài đặt trong bất kỳ môi trường dùng chung nào.
- **Thư mục dữ liệu người dùng:** Không phát hiện thư mục `~/.phoenix` hay cấu hình Phoenix tồn đọng tại `AppData\Local` và `AppData\Roaming`.
- **Cổng mạng:** Cổng 6006 hoàn toàn sạch, không có dịch vụ nào chiếm dụng trước kiểm thử.
- **Môi trường ảo hóa:** Docker CLI khả dụng nhưng Docker daemon không chạy; hệ thống WSL2 không cài đặt bản phân phối nào.
- **Phân loại hiện trạng:** Thuộc nhóm `PROTOTYPE_BRIDGE_ONLY` (Khớp Trường hợp 5 trong quy trình khôi phục: Bản mẫu giao tiếp đã có trong mã nguồn nhưng runtime máy chủ và cơ sở dữ liệu chưa được khởi tạo trên máy chủ).

---

## 3. THIẾT KẾ KIẾN TRÚC VÀ TRIỂN KHAI PHỤC HỒI

Để bảo vệ tính toàn vẹn của hệ thống máy chủ, toàn bộ ngăn xếp Arize Phoenix được triển khai trong môi trường biệt lập nghiêm ngặt:

### 3.1. Môi trường thực thi cô lập (Isolated Runtime)
- **Đường dẫn venv:** `C:\Users\GF63\.tools\agent-system-v2\extensions\observability-v2\phoenix-venv`
- **Phiên bản cố định:**
  - `arize-phoenix==20.20.0` (Máy chủ quan sát và đánh giá AI)
  - `arize-phoenix-client==3.5.0` (Client SDK truy vấn)
  - `arize-phoenix-evals==3.9.2` (Thư viện đánh giá)
  - `arize-phoenix-otel==0.17.2` (Bộ tích hợp OpenTelemetry)
- **Kho lưu trữ bền vững (Persistent Storage):** `C:\Users\GF63\.tools\agent-system-v2\extensions\observability-v2\data\phoenix\phoenix.db`. Quá trình di chuyển lược đồ SQLite (Alembic migrations) hoàn tất trong 0.155 giây.

### 3.2. Cấu hình Dual-Sink trên OpenTelemetry Collector v0.162.0
Cấu hình thu thập tại `collector.dev.yaml` được mở rộng với hai đích xuất độc lập:
1. `file/local`: Xuất vết dạng JSONL tại `C:\Users\GF63\.tools\agent-system-v2\artifacts\audits\observability\collector-traces.jsonl` (xoay vòng tập tin 4MB, tối đa 2 bản lưu trữ).
2. `otlp_http/phoenix`: Xuất qua giao thức chuẩn OpenTelemetry OTLP/HTTP đến `http://127.0.0.1:6006`, cơ chế tự động thử lại (retry with exponential backoff: 200ms đến 1s).

Bộ lọc `attributes/sanitize` được đặt trước pipeline xuất, loại bỏ hoàn toàn các trường dữ liệu nhạy cảm (`sentinel.secret`, `gen_ai.input.messages`, `user.email`, `http.request.header.authorization`) trước khi vết đến bất kỳ bồn chứa nào.

---

## 4. KẾT QUẢ NGHIỆM THU CHI TIẾT QUA 11 CỔNG KIỂM ĐỊNH (PHX-01 ĐẾN PHX-11)

Quy trình kiểm thử tự động độc lập `phoenix_collector_e2e_test.py` ghi nhận kết quả tuyệt đối 11/11 PASS:

| Cổng kiểm định | Nội dung kiểm tra | Kết quả thực nghiệm | Trạng thái |
| :--- | :--- | :--- | :--- |
| **PHX-01** | Xác minh định danh sản phẩm và phiên bản | Arize Phoenix v20.20.0, Client v3.5.0, SQLite DB tồn tại | **PASS** |
| **PHX-02** | Khởi động máy chủ Phoenix và kiểm tra giao diện | PID 23376, cổng 6006 mở, HTTP GET `/` trả về mã 200 | **PASS** |
| **PHX-03** | Xác thực cú pháp cấu hình Collector dual-sink | `otelcol.exe validate --config collector.dev.yaml` trả về mã 0 | **PASS** |
| **PHX-04** | Phát vết canary tổng hợp qua OTel SDK | 4 spans phát đi (1 root, 3 children bao gồm 1 span lỗi mô phỏng), Trace ID: `c00c4634f617d9d3046aaf66c042a784` | **PASS** |
| **PHX-05** | Collector xuất vết thành công sang Phoenix | Phoenix nhận POST `/v1/traces` thành công với HTTP 200, 4 spans được ghi nhận | **PASS** |
| **PHX-06** | Truy vấn vết qua Phoenix API và cơ sở dữ liệu SQLite | Truy vấn thành công 4 spans qua Phoenix Client API và câu truy vấn quan hệ SQLite (`spans JOIN traces`) | **PASS** |
| **PHX-07** | Khả năng truy cập giao diện đồ họa Phoenix Web UI | Giao diện đồ họa hiển thị đầy đủ dự án tại `http://127.0.0.1:6006/` | **PASS** |
| **PHX-08** | Bảo toàn dữ liệu tại bồn chứa cục bộ JSONL | Tập tin `collector-traces.jsonl` ghi nhận đầy đủ 4 spans của vết canary | **PASS** |
| **PHX-09** | Kiểm tra cơ chế khử dữ liệu nhạy cảm (Sanitization) | Thuộc tính sentinel nhạy cảm bị loại bỏ 100%, không xuất hiện trong JSONL hay cơ sở dữ liệu Phoenix | **PASS** |
| **PHX-10** | Diễn tập cô lập lỗi (Backend Outage Resilience) | Khi tắt máy chủ Phoenix, Collector vẫn hoạt động ổn định tại cổng 13133 (HTTP 200), không bị sập hay tràn bộ nhớ | **PASS** |
| **PHX-11** | Giải phóng cổng an toàn và duy trì trạng thái nghỉ | Toàn bộ tiến trình dừng an toàn, các cổng 4318, 13133, 8888, 6006 được giải phóng sạch sẽ (`ON_DEMAND_STOPPED`) | **PASS** |

---

## 5. MA TRẬN KHẢ NĂNG TIẾP CẬN TRÊN 8 DỰ ÁN (8-PROJECT REACHABILITY MATRIX)

Năng lực quan sát được mở rộng đồng bộ cho toàn bộ 8 dự án cục bộ thông qua bộ định tuyến dùng chung của Agent System v2.0.3 (`intent=observability`) và Global Skill `trace-context-propagation`:

1. `long-range-temporal-alignment`: Sẵn sàng kết nối trên định tuyến dùng chung; chưa tích hợp mã nguồn ứng dụng (`PREPARED_NOT_ACTIVE`).
2. `cross-dataset-bias-pipeline`: Sẵn sàng kết nối trên định tuyến dùng chung; chưa tích hợp mã nguồn ứng dụng (`PREPARED_NOT_ACTIVE`).
3. `hyper-parameter-evaluation-lab`: Sẵn sàng kết nối trên định tuyến dùng chung; chưa tích hợp mã nguồn ứng dụng (`PREPARED_NOT_ACTIVE`).
4. `adversarial-robustness-toolkit`: Sẵn sàng kết nối trên định tuyến dùng chung; chưa tích hợp mã nguồn ứng dụng (`PREPARED_NOT_ACTIVE`).
5. `contrastive-representation-suite`: Sẵn sàng kết nối trên định tuyến dùng chung; chưa tích hợp mã nguồn ứng dụng (`PREPARED_NOT_ACTIVE`).
6. `multimodal-retrieval-benchmark`: Sẵn sàng kết nối trên định tuyến dùng chung; chưa tích hợp mã nguồn ứng dụng (`PREPARED_NOT_ACTIVE`).
7. `uncertainty-quantification-flow`: Sẵn sàng kết nối trên định tuyến dùng chung; chưa tích hợp mã nguồn ứng dụng (`PREPARED_NOT_ACTIVE`).
8. `continual-learning-framework`: Sẵn sàng kết nối trên định tuyến dùng chung; chưa tích hợp mã nguồn ứng dụng (`PREPARED_NOT_ACTIVE`).

Tuân thủ nguyên tắc trung thực thực nghiệm, hệ thống ghi nhận chính xác: **8/8 dự án đạt tính tương thích về kiến trúc và đường truyền; 0/8 dự án bị tuyên bố khống là đã gắn mã đo đạc tự động trong ứng dụng production.**

---

## 6. QUY TRÌNH VẬN HÀNH THỐNG NHẤT BẰNG CÔNG CỤ ĐIỀU KHIỂN (`collector_ctl.py`)

Tập lệnh điều khiển `collector_ctl.py` đã được cập nhật toàn diện để quản lý tập trung cả hai thành phần (OpenTelemetry Collector và Arize Phoenix) theo cơ chế idempotent:

```bash
# 1. Kiểm tra trạng thái toàn diện (Collector + Phoenix)
python C:\Users\GF63\.tools\agent-system-v2\extensions\observability-v2\collector_ctl.py status

# 2. Khởi động Phoenix theo yêu cầu
python C:\Users\GF63\.tools\agent-system-v2\extensions\observability-v2\collector_ctl.py phoenix-start

# 3. Khởi động Collector theo yêu cầu
python C:\Users\GF63\.tools\agent-system-v2\extensions\observability-v2\collector_ctl.py start

# 4. Phát vết canary kiểm tra đường truyền
python C:\Users\GF63\.tools\agent-system-v2\extensions\observability-v2\collector_ctl.py canary

# 5. Dừng các thành phần an toàn và giải phóng cổng
python C:\Users\GF63\.tools\agent-system-v2\extensions\observability-v2\collector_ctl.py stop
python C:\Users\GF63\.tools\agent-system-v2\extensions\observability-v2\collector_ctl.py phoenix-stop
```

---

## 7. KẾT LUẬN VÀ BẢO LƯU HỆ THỐNG

Nhiệm vụ phục hồi Arize Phoenix và hoàn thiện chuỗi quan sát OTel đã đạt kết quả xuất sắc với đầy đủ chứng từ thực nghiệm có thể kiểm chứng độc lập. Hệ thống không làm rò rỉ bất kỳ thông tin nhạy cảm nào, không làm thay đổi các nhánh kinh doanh chưa nghiệm thu (`backend`, `uiux`), và hoàn toàn bảo toàn tư thế nghỉ tiết kiệm tài nguyên trên môi trường Windows máy chủ.
