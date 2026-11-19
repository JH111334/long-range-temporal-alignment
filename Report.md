# Báo cáo định hướng nghiên cứu thời gian dài

Tổng hợp **đề xuất**, dựa trên [ngữ cảnh bền vững](Documents/Decision-Making/12_PROJECT-CONTEXT.md), [thiết kế](Documents/Decision-Making/04_Quyet_dinh_thiet_ke.md), [kế hoạch đánh giá](Documents/Decision-Making/07_Ke_hoach_thuc_nghiem.md) và [chuẩn hóa Goal Ladder ADR 23](Documents/Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105.md). Không là bằng chứng đã triển khai enterprise stack hay đã huấn luyện mô hình production.

## Vấn đề, Hai Track và Dải Mục Tiêu (G1 → G2 → G3)

Hỏi đáp lịch sử tiếng Việt có bằng chứng và khả năng từ chối đòi hỏi sự kết hợp đồng thời giữa hai nhánh kỹ thuật (Track A và Track B) để từng bước hiện thực hóa dải mục tiêu nghiên cứu:

| Mục tiêu (`GOAL_ID`) | Trọng tâm & Độ hạt | Vai trò Track A (Truy xuất bên ngoài) | Vai trò Track B (Biểu diễn nội tại) |
|---|---|---|---|
| **GOAL 1 (`G1`)** | Trắc nghiệm đóng mốc Năm lịch sử (MCQA 4 lựa chọn + Nhị phân Yes/No) | Cung cấp ngữ cảnh tham chiếu đối chứng; cách ly shortcut văn bản | **Trọng tâm hiện hành**: Điều hòa biểu diễn thời gian dài hạn (FoNE, Ticktack, MaT-LoRA) để tối ưu độ chính xác trả lời đóng |
| **GOAL 2 (`G2`)** | Trắc nghiệm đóng kết hợp Năm + Tháng | Lọc và tái xếp hạng ngữ cảnh theo khoảng thời gian năm-tháng kết hợp | Tích hợp biểu diễn đa độ hạt, mặt nạ phân giải thời gian và triệt tiêu ảo giác tháng |
| **GOAL 3 (`G3`)** | General Temporal QA (Hỏi đáp đóng + mở + từ chối trả lời) | **Hợp nhất đầy đủ**: Hệ thống truy xuất bằng chứng lịch sử pháp lý Việt Nam | **Hợp nhất đầy đủ**: Mô hình sinh câu trả lời có trích dẫn căn cứ và từ chối khi thiếu dữ kiện |

| Track | Câu hỏi | Baseline và candidate |
|---|---|---|
| A — truy xuất bên ngoài | Evidence đúng nội dung/thời gian có giúp answer? | BM25/TF-IDF; interval filter, temporal rerank và policy evidence/abstention |
| B — biểu diễn bên trong | Biểu diễn dài và phân biệt year→month có giảm lỗi? | Scalar/Fourier, Ticktack đúng published-equation; multi-granularity/precision mask là candidate |

Không nhận BM25/Ticktack là đóng góp mới. Dual-Cycle/delta 0,5, LoRA/EWC hay tokenizer bonus chỉ là proposal khi chưa có run/gate; không ép mọi candidate vào baseline. Công thức chi tiết ở [phân tích Ticktack](Documents/References/Markdown/02_Phan_tich_paper_Ticktack.md).

## Dữ liệu và hợp đồng

[Khả thi dữ liệu/compute](Documents/Decision-Making/13_KHA-THI-DATA-4GB.md) xác định corpus/pilot; không lấy URL paper làm release. Nguồn phải có quyền, ngày truy cập, revision/hash, reviewer và role; qrels/evidence cần annotation. Nối từng corpus về Track A/B và phép đo trước admission.

Input giữ câu hỏi, raw time, source/release, interval `[start,end)`, granularity year/month/day, kind point/interval/duration/relation và uncertainty. Output answer/abstain, evidence span/ID, interval, provenance/warning. HistoricalYear/Date tách BCE/CE; thiếu tháng/ngày không gán 01/01; niên hiệu/âm lịch không rõ thì fail-closed.

## Thực nghiệm phân biệt nguyên nhân

R0 lexical+base; R1 chỉ thay retrieval; R2 chỉ thay representation; R3 kết hợp. Giữ corpus/pool/parser/split/prompt/evaluator/policy/case budget. Đổi contract thì `confounded`, mở family mới. Tách group/source/entity/event/time; fit chỉ train; validation chọn tham số.

Khóa anchor/parent/hypothesis/gold/evaluation contract hash. Báo paired delta, CI theo nhóm, denominator, invalid/abstention, alias, error slices và phản ví dụ; kết luận supported/refuted/inconclusive. Run hợp lệ nhưng kém là REJECT/REPEAT, invalid/gate FAIL mới là Error.

## Khả thi và hoàn thành

CPU local cho canonicalization/lexical/evaluator và bounded probe; external GPU chỉ sau immutable request/quyền/resource gate, giữ cùng contract. Không suy fit từ model size hay fixture. GraphRAG/vector DB và enterprise ứng dụng nằm ngoài critical path, đọc [nhánh hoãn](Documents/References/LATER-DEV.md) khi được giao.

Đầu ra cần corpus/query/qrels đã duyệt, frozen protocol, baseline/candidate có artifact/config/data/code hash, seed/CI/resource, phân tích lỗi và replay. Chưa có bằng chứng đo thì chỉ bàn giao design/fixture; dừng khi thiếu quyền/evaluator, gain mất held-out hoặc vượt ngân sách.
