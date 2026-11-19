---
loại: "quyết-định"
mã: "LRT-D004"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
thay_thế: []
bị_thay_thế_bởi: []
tham_chiếu: ["[[References/Markdown/EXTERNAL-SERVICES-GUIDELINE]]"]
được_xác_thực_bởi: []
cập_nhật_lần_cuối: "2026-09-28"
---

# Tích Hợp TypeSafe Jev System One Vào Dự Án Long-Range Temporal Alignment

## 1. Trạng Thái Vận Hành (Status)

- `JEV_MODE = ADVISORY`
- `AUTO_ACTION = FALSE`
- `CAPABILITY_STATUS = PROVISIONAL_ADVISORY`

Mô hình Jev System One hoạt động thuần túy dưới vai trò bộ tư vấn xác suất có chặn biên (bounded probabilistic decision advisor). Jev đưa ra các gợi ý định tuyến hoặc xếp hạng ứng viên trong không gian rời rạc hữu hạn. Hệ thống không cho phép Jev tự động thực thi bất kỳ đột biến mã nguồn, ghi đè tệp tin, thao tác Git hay quyết định khoa học nào mà không có sự phê duyệt của System 2 hoặc con người.

---

## 2. Mục Đích Tồn Tại của Jev Trong Dự Án (Why Jev Exists in this Project)

Trong quá trình nghiên cứu và phát triển hệ thống canh chỉnh thời gian dài hạn (`long-range-temporal-alignment`), nhiều bài toán mang tính chất phán đoán mờ nhưng không gian nghiệm hoàn toàn bị chặn biên và hữu hạn:
- Lựa chọn bộ tiền xử lý chuẩn hóa chuỗi thời gian phù hợp giữa các cấu trúc niên đại (ví dụ: ISO direct parser vs. Astronomical HistoricalDate parser).
- Định tuyến truy vấn thời gian vào nhánh chỉ mục từ vựng (lexical) hoặc ngữ nghĩa (dense semantic) dựa trên đặc tính chuỗi văn bản.
- Sàng lọc độ ưu tiên thực thi của các tác vụ kiểm định khi tài nguyên tính toán bị giới hạn.

Nếu sử dụng toàn bộ mô hình ngôn ngữ lớn (System 2: GPT-4o, Claude 3.5, Gemini 1.5 Pro) cho các quyết định này, hệ thống sẽ gánh chịu độ trễ cao (2–10 giây) và tiêu tốn token không cần thiết. Ngược lại, nếu chỉ dùng mã tĩnh (CODE), hệ thống không xử lý được các đặc trưng mờ của ngôn ngữ tự nhiên. Jev đóng vai trò là tầng trung gian (System 1) giải quyết bài toán với độ trễ đại diện đo lường được trong thực nghiệm khoảng ~0.9–1.4 giây cho các yêu cầu cô đọng (tùy thuộc vào phương sai mạng và tải mô hình) cùng đầu ra định kiểu nghiêm ngặt (typed output).

---

## 3. Các Lớp Quyết Định Được Phép (Allowed Bounded Decision Classes)

Jev chỉ được phép tiếp nhận các bài toán thuộc các lớp sau:

1. **`temporal_router_candidate_selection`**: Lựa chọn giữa các chiến lược truy vấn hoặc biểu diễn thời gian rời rạc đã định nghĩa sẵn trong hệ thống (`choice`).
2. **`preprocessing_strategy_choice`**: Gợi ý parser hoặc tokenizer phù hợp nhất cho định dạng thời gian cổ/hiện đại (`choice`).
3. **`data_leakage_risk_heuristic`**: Đánh giá sơ bộ xem một chuỗi truy vấn có dấu hiệu rò rỉ nhãn thời gian hay không trước khi chạy audit chuyên sâu (`noul`).
4. **`task_execution_priority_scoring`**: Chấm điểm mức độ ưu tiên của một thí nghiệm hoặc lát cắt đánh giá trên thang đo rời rạc từ 1 đến 5 (`score`).

---

## 4. Các Lớp Quyết Định Bị Cấm Tuyệt Đối (Forbidden Decision Classes)

Tuyệt đối cấm sử dụng Jev cho các mục đích sau:

1. **Kiểm tra tất định (Deterministic Verification)**: Kiểm tra file tồn tại, đếm dòng, tính checksum SHA-256, kiểm tra mã thoát (exit code), cú pháp AST, unit test. Các tác vụ này bắt buộc phải do mã Python thuần (`CODE`) thực hiện.
2. **Phê duyệt khoa học (Scientific Acceptance)**: Không dùng Jev để kết luận một giả thuyết nghiên cứu là `supported`, `refuted` hoặc `inconclusive`. Quyết định này chỉ được đưa ra dựa trên số liệu thực nghiệm ghép cặp có kiểm định thống kê.
3. **Đột biến tệp và hạ tầng (File & Infrastructure Mutation)**: Cấm dùng Jev để quyết định xóa tệp, ghi đè mã nguồn, commit Git, đẩy mã lên remote server.
4. **Sinh văn bản hoặc suy luận mở (Open-ended Text Generation)**: Jev không phải là mô hình tự hồi quy (non-autoregressive); cấm dùng Jev để sinh báo cáo, viết code hay tóm tắt tài liệu.

---

## 5. Yêu Cầu Đối Với Gói Dữ Liệu Quyết Định (Decision Packet Requirements)

Mọi yêu cầu gửi tới Jev bắt buộc phải được đóng gói theo hợp đồng `Decision Packet` chuẩn hóa và vượt qua kiểm định tiền trạm (preflight gate):

```json
{
  "task": "Mô tả ngắn gọn mục tiêu quyết định",
  "facts": { "observation_key": "deterministic_value" },
  "evidence_refs": ["Documents/Decision-Making/16_SOURCEAPI-BOUNDARY.md#L10-L20"],
  "unknowns": ["danh sách biến số chưa biết"],
  "inferences": ["danh sách suy luận tách biệt khỏi facts"],
  "constraints": { "budget_ms": 500 },
  "risk_class": "low",
  "questions": {
    "question_name": {
      "type": "choice | noul | score",
      "instructions": "Yêu cầu đánh giá cụ thể",
      "criteria": {}
    }
  }
}
```

Ràng buộc bảo mật dữ liệu:
- Dung lượng payload JSON không vượt quá 32 KB.
- Áp dụng nguyên tắc bằng chứng tối thiểu (Minimum Necessary Evidence).
- Quét tự động và từ chối toàn bộ payload chứa pattern API key, private key hoặc mật khẩu.

---

## 6. Phân Lớp Rủi Ro (Risk Classes)

Hệ thống phân loại rủi ro theo bốn cấp độ:

- **`low`**: Quyết định tư vấn thứ tự đọc tài liệu, gợi ý ứng viên parser không gây đột biến trạng thái. Jev được phép tư vấn; System 2 có thể tham khảo.
- **`medium`**: Gợi ý phân nhánh thuật toán thử nghiệm trong môi trường cô lập. Yêu cầu ghi nhận biên nhận đầy đủ trước khi thực thi.
- **`high`**: Quyết định liên quan đến cấu trúc pipeline hoặc phân bổ tài nguyên tính toán lớn. Preflight tự động chặn Jev và chuyển giao trực tiếp cho System 2.
- **`critical`**: Thay đổi cấu trúc bảo mật, phát hành báo cáo, xóa dữ liệu hoặc thay đổi hợp đồng cốt lõi. Cấm Jev tham gia; chỉ con người hoặc System 2 có thẩm quyền.

---

## 7. Chính Sách Độ Tin Cậy và Chuyển Cấp (Confidence & Escalation Policy)

- **Ngưỡng độ tin cậy tạm thời (Provisional Threshold)**: Thiết lập mặc định ở mức `0.80`.
  > Lưu ý: Ngưỡng này được ghi nhận dưới dạng `PROVISIONAL - NOT CALIBRATED ON THIS WORKFLOW`.
- **Cơ chế chuyển cấp (Escalation)**:
  - Khi `confidence < 0.80` đối với `choice` hoặc `score`: Trả về trạng thái `ADVISORY_LOW_CONFIDENCE`, yêu cầu System 2 xác nhận lại.
  - Khi độ chênh lệch xác suất của `noul` so với 0.5 nhỏ hơn 0.3 (`|p - 0.5| < 0.3`): Trả về `ADVISORY_LOW_CONFIDENCE`.
  - Khi gặp lỗi mạng, HTTP status khác 200, timeout 15s hoặc schema không hợp lệ: Wrapper tự động chuyển giao fail-closed thành `ESCALATE_TO_SYSTEM2`.

---

## 8. Nguyên Tắc Thẩm Định Tất Định (Deterministic Verification Rule)

- Mọi khuyến nghị của Jev chỉ có giá trị tham khảo.
- Mã nguồn kiểm thử tất định (`scripts/validate_project.py`, `scripts/validate_knowledge_vault.py`, test suite) luôn có quyền tối thượng: nếu khuyến nghị của Jev dẫn đến vi phạm bất kỳ chốt kiểm định tất định nào, hành động đó lập tức bị hủy bỏ và ghi nhận vào log lỗi.

---

## 9. Biên Nhận Quyết Định và Phản Hồi Kết Quả (Receipt & Outcome Feedback)

- Mỗi lần gọi Jev thành công, wrapper tự động xuất một biên nhận JSON thu nhỏ tại `artifacts/receipts/jev-receipt-*.json`.
- Cấu trúc biên nhận gồm: `id`, `timestamp`, `policy_version`, `model`, `state_hash` (SHA-256), `question_types`, `answers`, `usage`, `outcome = null`, `correct = null`.
- Tuyệt đối không lưu trữ khóa API hoặc toàn bộ ngữ cảnh thô trong biên nhận.
- Khi một nhánh quyết định được thực thi và có kết quả thực nghiệm kiểm chứng, trường `outcome` và `correct` sẽ được cập nhật để phục vụ công tác hiệu chuẩn (calibration).

---

## 10. Chính Sách Hiệu Chuẩn Mô Hình (Calibration Policy)

> [!NOTE]
> **PROVISIONAL PROMOTION / CALIBRATION POLICY — NOT ESTABLISHED MODEL TRUTH**
> Các ngưỡng 50 mẫu, 96% độ chính xác, ECE/Brier gates và ngưỡng tin cậy 0.80 dưới đây là chính sách hiệu chuẩn tạm thời phục vụ nghiên cứu và kiểm thử, không phải chân lý bất biến của mô hình. Hệ thống tuyệt đối duy trì `JEV_MODE = ADVISORY` và `AUTO_ACTION = FALSE`.

- Đội ngũ phát triển định kỳ đối chiếu giữa độ tin cậy dự báo (`confidence`) và tính chính xác thực nghiệm (`empirical correctness`) dựa trên tập biên nhận đã gán nhãn `outcome`.
- Việc điều chỉnh ngưỡng độ tin cậy từ mức tạm thời (0.80) sang ngưỡng chính thức chỉ được xem xét khi có tối thiểu 50 biên nhận được kiểm chứng trên cùng một lớp quyết định.

---

## 11. Tiêu Chí Thăng Cấp Lên Tự Động Hóa (Promotion Criteria for Automation)

> [!IMPORTANT]
> **PROVISIONAL POLICY — AUTOMATION DISABLED**
> Hiện tại, không có bất kỳ lớp quyết định nào được cấp quyền tự động hóa. Mọi quyết định của Jev chỉ mang tính chất khuyến nghị (advisory).

Để một lớp quyết định trong tương lai được xem xét thăng cấp từ `ADVISORY` lên `AUTOMATIC_BRANCHING` (cho phép agent tự động đi theo nhánh khuyến nghị ở mức rủi ro thấp), lớp quyết định đó phải thỏa mãn đồng thời:
1. Tối thiểu 50 biên nhận thực tế với tỷ lệ chính xác kiểm chứng đạt trên 96%.
2. Brier score hoặc Expected Calibration Error (ECE) dưới 0.05 trên phân phối quyết định đó.
3. Không ghi nhận bất kỳ sự cố phá vỡ chốt kiểm định tất định nào xuất phát từ khuyến nghị của Jev.
4. Có phê duyệt chính thức bằng một bản sửa đổi trong Decision-Making.


---

## 12. Tiêu Chí Thăng Cấp Lên MCP Server (MCP Promotion Criteria)

Hiện tại, Jev được triển khai dưới dạng công cụ chia sẻ (`Shared Tool Wrapper`). Chỉ xem xét đóng gói Jev thành MCP server độc lập khi xuất hiện đồng thời các nhu cầu sau:
- Có từ 3 client hoặc tiến trình agent độc lập cùng chia sẻ phiên kết nối đồng thời.
- Cần cơ chế khám phá công cụ động (dynamic tool discovery) qua giao thức chuẩn JSON-RPC.
- Cần quản lý vòng đời xác thực tập trung và giám sát lưu lượng tập trung giữa nhiều kho mã nguồn.

---

## 13. Tiêu Chí Thăng Cấp Lên Local Skill (Local Skill Promotion Criteria)

Chỉ tạo Local Skill riêng cho Jev khi:
- Xuất hiện quy trình làm việc đặc thù chỉ có ở dự án này mà không thể tái sử dụng ở các dự án khác.
- Đã hoàn thành giai đoạn hiệu chuẩn và lớp quyết định được phê duyệt tự động hóa một phần.

---

## 14. Điều Kiện Thu Hồi và Vô Hiệu Hóa (Rollback & Disable Conditions)

Hệ thống lập tức vô hiệu hóa Jev (`enabled = false`) khi xuất hiện một trong các tình huống sau:
1. Phát hiện sự cố rò rỉ khóa bí mật hoặc thông tin nhạy cảm vào payload viễn thông.
2. Tỷ lệ lỗi API (timeout, 5xx) vượt quá 10% trong 20 lần gọi liên tiếp.
3. Bất kỳ khuyến nghị nào của Jev dẫn đến sai lệch dữ liệu hoặc phá vỡ các chốt kiểm định tất định của Living Knowledge Vault.
4. Nhà cung cấp TypeSafe thay đổi hợp đồng API mà wrapper chưa kịp thích ứng.
