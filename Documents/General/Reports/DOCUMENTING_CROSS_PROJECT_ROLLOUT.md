# Báo Cáo Triển Khai Tầng Diễn Ngôn Phái Sinh Trên Toàn Bộ 8 Dự Án (Cross-Project Documenting Layer Rollout)

## 1. Tổng Quan Triển Khai
Báo cáo này ghi nhận việc hoàn tất triển khai tầng diễn ngôn phát triển phái sinh (`Documents/Documenting/`) trên toàn bộ 8 dự án thuộc hệ sinh thái nghiên cứu và học tập. Tầng tài liệu mới được thiết kế nhằm tổng hợp quá trình tiến hóa kiến trúc, các bước ngoặt thực nghiệm, những giả thuyết sai lầm đã bị loại bỏ và các bài học kỹ thuật cốt lõi, phục vụ trực tiếp cho bốn mục đích: thuyết trình báo cáo khoa học (presentation), xây dựng câu chuyện bài báo (paper-story), hoàn thiện hồ sơ năng lực kỹ thuật (portfolio) và học tập hồi tưởng (retrospective learning).

## 2. Nguyên Tắc Thẩm Quyền Và Ranh Giới Quản Trị
Tầng `Documents/Documenting/` được định vị là nguồn thông tin thứ cấp, mang tính tổng hợp phái sinh và tuyệt đối không phải là nguồn sự thật gốc (non-authoritative layer). Quy tắc phân định thẩm quyền được thống nhất:
1. Mã nguồn và runtime đã kiểm chứng (`src/`, test suites, hashes, caches).
2. Các quyết định kỹ thuật chuẩn hóa (`Documents/Decision-Making/`).
3. Tài liệu tham chiếu và cơ sở lý thuyết (`Documents/Reference/`).
4. Bằng chứng kiểm định lịch sử và biên bản lỗi (`Documents/General/Checkpoints/`, `Documents/General/Errors/`).
5. Tầng diễn ngôn phái sinh (`Documents/Documenting/`).

Đồng thời, quy chế điều hướng (Router / Governance) trên cả 8 dự án đã được cập nhật: Tuyến `DOCUMENTING` chỉ được kích hoạt khi có yêu cầu tường minh về thuyết trình, bài báo, hồ sơ năng lực hoặc bài học hồi tưởng; tuyệt đối không nạp theo mặc định trong các tác vụ lập trình hay sửa lỗi thông thường.

## 3. Phân Tầng Độ Sâu Nghiên Cứu (Depth Tier Classification)

### Tier A (Deepest Empirical Research — 2 Dự Án)
1. **`DSC-TinDipLaPo`**: Cuộc thi UIT Data Science Challenge 2026 (Task 1 LegalIR và Task 2 LegalQA).
   - *Tệp tạo lập*: `README.md` (66 dòng), `Project-Evolution.md` (10 mục chuẩn hóa), và 8 component files chuyên sâu:
     - `Task1-LegalIR.md`: Lexical retrieval, regex số hiệu văn bản và candidate bounding.
     - `Task2-Corpus-Data.md`: Parser phân cấp Điều/Khoản, Silver V2 corpus và chống rò rỉ dữ liệu.
     - `Hybrid-Retrieval.md`: Kết hợp BM25 và Dense Embeddings qua Reciprocal Rank Fusion (RRF $k=60$).
     - `Reranking.md`: Mô hình cross-encoder `AITeamVN/Vietnamese_Reranker` và đóng băng cache 160.046 cặp candidate.
     - `Evidence-Packing.md`: Thuật toán Greedy-Fit E9P0, bảo toàn cấu trúc Điều/Khoản và tối ưu hóa context cap 768-1280 token.
     - `Generator-SFT.md`: Huấn luyện LoRA trên `ViLegalQwen3-1.7B-Base`, greedy decoding, METEOR 0.6211.
     - `NLI-Auditing.md`: Kiểm định mâu thuẫn nội dung, xác thực trích dẫn và chính sách từ chối an toàn.
     - `Kaggle-Runtime.md`: Kỹ thuật chạy offline trong hạn mức 9 giờ trên GPU P100/T4, Beautiful Soup 4 Parity.
2. **`long-range-temporal-alignment`**: Đề tài NCKH trọng điểm về biểu diễn thời gian dài cho lịch sử Việt Nam.
   - *Tệp tạo lập*: `README.md` (54 dòng), `Project-Evolution.md` (10 mục chuẩn hóa), và 5 component files cốt lõi:
     - `Problem-Formulation.md`: Hệ tọa độ thiên văn Astronomical Numbering, loại bỏ năm 0, phân tách granularity và temporal kind.
     - `Temporal-Representation.md`: Kết hợp công thức chu kỳ Ticktack 60 năm với thành phần bán kính tuyến tính triệt tiêu va chạm thế kỷ.
     - `Targeted-Learning.md`: Huấn luyện LoRA có ràng buộc Elastic Weight Consolidation (EWC) bảo vệ tri thức cổ đại.
     - `Retrieval-Evaluation.md`: Giao thức Time-Held-Out Split và chỉ số sai số thời gian Time-MAE.
     - `Infrastructure-Capabilities.md`: Kiến trúc CPU-first dưới 4 GB RAM, ranh giới SourceAPI và cố vấn Jev Advisory.

### Tier B (Capstone & Portfolio Scaffolds — 2 Dự Án)
1. **`Evidence-Grounded-Text-Agent`**: Đồ án Capstone cá nhân về tác nhân văn bản cục bộ tiếng Việt - Anh.
   - *Tệp tạo lập*: `README.md` (55 dòng), `Project-Evolution.md` (10 mục chuẩn hóa), và 3 component files:
     - `Access-Controlled-Retrieval.md`: Phân quyền truy cập tài liệu và tiền lọc trước khi truy vấn.
     - `Citation-Integrity.md`: Kiểm tra toàn vẹn trích dẫn 100% bằng mã xác định và chính sách từ chối an toàn.
     - `Local-Execution-Engine.md`: Pipeline thuần CPU, tiêu thụ dưới 1.5 GB RAM, không phụ thuộc dịch vụ ngoài.
2. **`general-reasoning-fact-checking`**: Portfolio / Scaffold nghiên cứu độc lập về kiểm chứng phát biểu dựa trên bằng chứng có phiên bản.
   - *Tệp tạo lập*: `README.md` (54 dòng), `Project-Evolution.md` (10 mục chuẩn hóa), và 3 component files:
     - `Prefilter-Retrieval.md`: Lọc bằng chứng theo quyền hạn và cửa sổ thời gian hiệu lực `[valid_from, valid_to]`.
     - `Proof-Graph-Engine.md`: Động cơ đồ thị chứng minh hữu hạn DAG, triệt tiêu lập luận vòng quanh.
     - `Conservative-Verdict.md`: Cơ chế ra phán quyết bảo thủ ba trạng thái (SUPPORTED / REFUTED / NOT_ENOUGH_INFO).

### Tier C (Lightweight Course Scaffolds — 4 Dự Án)
1. **`DS111-Retrieval-Data-Analysis`**: Đồ án môn Phân tích dữ liệu.
   - *Tệp tạo lập*: `README.md` (52 dòng), `Project-Evolution.md` (10 mục chuẩn hóa), và 2 component files:
     - `Data-Audit-Pipeline.md`: Kiểm định schema, missingness và phát hiện rò rỉ trùng lặp gần (near-duplicate).
     - `Slice-Metric-Evaluation.md`: Đánh giá hiệu năng truy xuất theo lát cắt kèm khoảng tin cậy 95% Bootstrap.
2. **`DS201-Temporal-Retrieval-Learning`**: Đồ án môn Deep Learning trong khoa học dữ liệu.
   - *Tệp tạo lập*: `README.md` (52 dòng), `Project-Evolution.md` (10 mục chuẩn hóa), và 2 component files:
     - `Temporal-Ranking-Model.md`: Mô hình Deep Learning học xếp hạng theo thời gian với Margin Ranking Loss.
     - `Temporal-Split-Protocol.md`: Giao thức phân chia Time-Held-Out loại bỏ 100% rò rỉ thời gian.
3. **`DS314-Evidence-Grounded-Retrieval`**: Đồ án môn Rút trích và truy vấn thông tin.
   - *Tệp tạo lập*: `README.md` (52 dòng), `Project-Evolution.md` (10 mục chuẩn hóa), và 2 component files:
     - `Lexical-Fusion-Engine.md`: Động cơ gộp hạng BM25 và TF-IDF qua Reciprocal Rank Fusion ($k=60$).
     - `Abstention-Controller.md`: Bộ kiểm soát từ chối trả lời theo ngưỡng tin cậy và phạm vi Active Corpus.
4. **`DS335-VLM-Research`**: Chuyên đề nghiên cứu Mô hình Ngôn ngữ Thị giác (Vision-Language Models).
   - *Tệp tạo lập*: `README.md` (52 dòng), `Project-Evolution.md` (10 mục chuẩn hóa), và 2 component files:
     - `Cross-Modal-Alignment.md`: Cơ chế căn chỉnh đa phương thức qua 2-layer MLP Projector có đóng băng backbone.
     - `Visual-Temporal-Benchmarking.md`: Khung đánh giá suy luận thời gian thị giác loại bỏ text shortcut.

## 4. Bảng Tổng Hợp Kiểm Định Hồi Quy (Regression & Validation Matrix)

| Dự án | Phân tầng | Số tệp tạo | Số dòng README | Kiểm định Project Gate | Kiểm định Vault Gate |
|---|---|---|---|---|---|
| `DSC-TinDipLaPo` | Tier A | 10 tệp | 66 dòng | N/A (Thi đấu) | PASS (163 docs, 80 managed) |
| `long-range-temporal-alignment` | Tier A | 7 tệp | 54 dòng | PASS (66 required paths) | PASS (72 docs, 21 managed) |
| `Evidence-Grounded-Text-Agent` | Tier B | 5 tệp | 55 dòng | PASS (7 paths, 12 hooks) | N/A |
| `general-reasoning-fact-checking` | Tier B | 5 tệp | 54 dòng | PASS (22 paths, 4 checks) | N/A |
| `DS111-Retrieval-Data-Analysis` | Tier C | 4 tệp | 52 dòng | PASS (43 files, 10 ckpts) | PASS (43 docs) |
| `DS201-Temporal-Retrieval-Learning` | Tier C | 4 tệp | 52 dòng | PASS (43 files, 10 ckpts) | PASS (43 docs) |
| `DS314-Evidence-Grounded-Retrieval` | Tier C | 4 tệp | 52 dòng | PASS (43 files, 10 ckpts) | PASS (43 docs) |
| `DS335-VLM-Research` | Tier C | 4 tệp | 52 dòng | PASS (Contracts verified) | PASS (43 docs) |

## 5. Kết Luận
Toàn bộ 8 dự án đã được trang bị đầy đủ tầng diễn ngôn phát triển phái sinh đúng tiêu chuẩn kỹ thuật, không làm sai lệch các nguồn sự thật gốc, không can thiệp vào mã nguồn nghiệp vụ hay hệ thống hook, và vượt qua 100% các chốt kiểm định tự động hiện hành.
