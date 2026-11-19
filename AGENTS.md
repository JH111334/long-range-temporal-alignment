# Quy Ước Dự Án Temporal Alignment

## Phạm vi

- Bắt đầu bằng baseline tái lập được; không gọi scaffold, smoke test hoặc dữ liệu tổng hợp là kết quả nghiên cứu.
- Tách rõ `published-equation`, `operational-assumption` và `proposed-method` trong cấu hình, artifact và báo cáo.
- Nhánh hiện hành `v2-targeted-continual-alignment` / Goal `G1`: YEAR-only closed QA, chuyển giao Zero-Shot từ train NWP sang eval MCQA/Rel; **CHỈ can thiệp các tầng MLP Linear (`gate_proj`, `up_proj`, `down_proj`), tuyệt đối KHÔNG can thiệp Attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) và không dùng DoRA/DLoRA**.
- Không tải dataset/model thật nếu registry chưa có URL chính thức, ngày truy cập, giấy phép hoặc điều khoản, revision/checksum, người rà soát và vai trò train/validation/test.

## Dữ liệu và đánh giá

- Biểu diễn năm lịch sử bằng `HistoricalYear`; không truyền số 0 mơ hồ giữa BCE và CE.
- Dùng `HistoricalDate` cho tháng/ngày; tách `granularity=year|month|day` khỏi `TemporalKind=point|interval|duration|relation`.
- Chia dữ liệu theo nhóm nguồn/thực thể/sự kiện và thời gian; không random split nếu có nguy cơ near-duplicate.
- Báo riêng invalid-output rate, leakage audit, số seed và khoảng tin cậy.
- Mọi kết quả phải gắn config hash, data manifest hash, code revision và mức bằng chứng.
- Trước run so sánh, khóa `anchor_checkpoint`, `parent_checkpoint`, hypothesis, evaluation role, gold status và evaluation contract hash.
- Chỉ kết luận gain khi so sánh ghép cặp trên cùng evaluation contract; báo delta theo mẫu và lát cắt, denominator, case budget cùng counterexample.
- Tách quan sát khỏi nguyên nhân. Kết luận giả thuyết bằng `supported|refuted|inconclusive` và nêu minimal next experiment cùng stop gate.
- Run hợp lệ nhưng kém hơn vẫn là checkpoint với quyết định `REJECT|REPEAT`; chỉ tạo Error khi run không hợp lệ hoặc quality gate đã định trước thất bại.

## Mã nguồn

- Core giữ thuần Python để chạy được trên CPU; tích hợp Torch/Transformers/PEFT nằm sau adapter tùy chọn.
- `SourceAPI/` là boundary external provider → validation → normalized temporal record
  + provenance; raw payload và credential không đi vào core.
- Compute theo capability: local chạy cùng pipeline ở scale nhỏ; Kaggle chỉ là optional
  burst compute có request/hash/approval và không được fork codebase.
- Hàm phải nhỏ, có type hint, docstring ngắn và lỗi fail-closed.
- Không dùng emoji, sticker, banner ký hiệu hoặc thông báo ăn mừng trong mã, test, log hay CLI; chỉ giữ ký hiệu có nghĩa kỹ thuật hoặc dữ liệu protocol đã ghi rõ.
- Mọi thay đổi encoder, objective, split hoặc schema phải có unit test tương ứng.

## Tài liệu

- Viết tiếng Việt có dấu, viết hoa đầu câu, ngắn gọn và đặt nguồn gần mệnh đề; không chép debug narrative vào tài liệu canonical.
- Tích hợp bối cảnh Việt Nam vào cùng chuỗi lập luận vấn đề–dữ liệu–phương pháp–đánh giá–ứng dụng; không tách thành mục trang trí thiếu evidence hoặc gate khả thi.
- Không đưa PDF paper, model weights, dataset có điều khoản riêng vào Git nếu chưa được phép.
- Chạy `python scripts/validate_project.py` và test trước khi commit.

<!-- BEGIN living-knowledge-vault:v1 -->

## Living Knowledge Vault

- Mở Documents làm vault root; bắt đầu tại Documents/HOME/Trang chủ.md.
- Trước khi đổi knowledge, đọc Documents/HOME/Quy tắc tri thức.md và General/RULE.md.
- Evidence mới vào checkpoint/error hợp lệ trước; chỉ evidence đã validate mới được
  update hoặc supersede Decision-Making, rồi propagate tới dependency thật.
- Khi tạo file mới trong Decision-Making, References hoặc General, chạy
  scripts/ingest_knowledge_file.py <path>; command sẽ phân loại, seed metadata an
  toàn và chạy lại validator.
- Chạy interpreter của project với scripts/validate_knowledge_vault.py cùng project
  gates hiện hữu trước khi bàn giao.
<!-- END living-knowledge-vault:v1 -->

## Điều hướng agent theo sự kiện

- IF bắt đầu task hoặc đổi phạm vi THEN dùng `project-context-router` để đọc ngữ cảnh vừa đủ và dừng đọc khi đã rõ hành động hợp lệ.
- IF sửa code, config, test hoặc hành vi runtime THEN dùng `project-change-verification` trước và sau thay đổi.
- IF có milestone, lỗi hoặc quyết định đã kiểm chứng THEN dùng `project-knowledge-writeback`; không tự nâng trạng thái từ tên file hoặc config.
- Chọn tool theo `.agents/capabilities.json`; capability thiếu hoặc chưa xác minh phải fail-closed. Các quy tắc này vẫn áp dụng khi hook chưa được trust hoặc không bao phủ tool tương ứng.

## Điều phối context nhiều nhánh

- Đọc `Documents/Global/Branch-Protocol.md` và registry trước task có nhánh. Chọn `ACTIVE_BRANCH` rõ ràng; không suy từ file mở gần nhất.
- Chỉ ghi namespace đã chọn; shared state và promotion qua Joint Gate. Nếu lineage hoặc evaluator mâu thuẫn, dừng trước thực thi.

## Khung vận hành Agent System v2

- Cấu hình tại `.agents/agent-system-v2.json`; engine dùng chung tại `C:/Users/GF63/.skills/global/agent-system-v2/`.
- Bảy pha: `RESEARCH`, `IMPLEMENT`, `AUDIT`, `EXECUTE`, `VALIDATE`, `CHANGES`, `REPORTS&DOCUMENTS`.
- Phạm vi bảo trì hạ tầng hiện hành: `ACTIVE_BRANCH=GLOBAL_MAINTENANCE`. Các nhánh nghiên cứu (`track-a`, `track-b`) giữ nguyên trạng thái chưa kích hoạt.

## Documenting

Tầng tổng hợp phục vụ trình bày/hồi cứu; đọc `Documents/Documenting/RULE.md` khi cần. Giữ cổng ý nghĩa, provenance và thẩm quyền nguồn; không ghi lỗi vụn hay nạp tầng này mặc định khi lập trình.

## Tài liệu gọn và có ngữ cảnh

Nêu vấn đề/đề tài, vai trò và liên kết tiền đề; dữ liệu nối đúng câu hỏi/phép đánh giá. Giữ ý quyết định, gate và giới hạn; dẫn nguồn thay chép lại. Áp dụng quy tắc global về tiếng Việt, provenance và lịch sử; ranh giới project ở trên vẫn ưu tiên.

<!-- inference-notebook-governance:v1 -->

## Gate inference và notebook

- Dùng reference `inference-run-safety` của global `python-code-standard` khi sửa loader/treatment; `notebook-submission-hygiene` của `documenting-governance` khi chuẩn bị notebook nộp bài. Không clear toàn bộ output hoặc biến canary thành kết quả nghiên cứu.
- Shared gate: `C:/Users/GF63/.skills/global/agent-system-v2/governance_runner.py`; local overlay chỉ giữ policy. Scope liên project cần grant đúng phiên và quyền host thực tế.
