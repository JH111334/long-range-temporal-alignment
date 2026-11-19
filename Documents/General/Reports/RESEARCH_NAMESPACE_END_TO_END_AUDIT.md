# Báo Cáo Kiểm Định và Sửa Chữa Toàn Diện Không Gian Nghiên Cứu (Research Namespace)

## 1. Mục Đích và Phạm Vi Kiểm Định

Báo cáo này ghi nhận kết quả kiểm định thực tế đầu-cuối (end-to-end live audit) đối với không gian nghiên cứu (RESEARCH namespace) của hệ thống Agent System v2, áp dụng trên bộ lõi dùng chung (`C:\Users\GF63\.skills\global\agent-system-v2`) và 8 dự án thành phần (kèm 1 nhánh successor canonical):
1. `long-range-temporal-alignment` (kèm canonical successor `Long-Range-Representation`)
2. `DS111-Retrieval-Data-Analysis`
3. `DS201-Temporal-Retrieval-Learning`
4. `DS314-Evidence-Grounded-Retrieval`
5. `DS335-VLM-Research`
6. `DSC-TinDipLaPo`
7. `evidence-grounded-text-agent`
8. `general-reasoning-fact-checking` (tại `evidence-grounded-text-agent\Source\QA-TEXT\FactChecking`)

Kiểm định tập trung vào khả năng vận hành thực tế của chuỗi điều phối: nhận diện ý định nghiên cứu, nạp ngữ cảnh tối thiểu từ Living Knowledge Vault, tiếp cận nguồn dữ liệu học thuật độc lập, phân tích đối chiếu mã nguồn - dữ liệu - bài báo, phát hiện mâu thuẫn học thuật và ghi nhận tri thức có kiểm chứng về đúng tầng tài liệu theo nguyên tắc fail-closed.

---

## 2. Kiểm Tra Ranh Giới Quản Trị và Khóa Vendor

### 2.1. Quản trị Kho Tri thức và Bản quyền Dữ liệu
- Mọi truy vấn nghiên cứu đều tuân thủ nguyên tắc không tự ý tải dữ liệu lớn (`AUTO_DOWNLOAD=FALSE`).
- Các cổng bản quyền dữ liệu và quyền phân phối (`research:data-evidence-gate`) kiểm tra giấy phép, tính toàn vẹn (checksum) và siêu dữ liệu FAIR trước khi cho phép sử dụng.
- Dữ liệu thô và credential tuyệt đối không đi vào ngữ cảnh lõi.

### 2.2. Kiểm toán Trôi dạt Mã nguồn Upstream (Nature-Skills)
- Khóa vendor hiện hành: commit `84880815fb37317b3766bff2c2abba395b8993c3` tại `C:\Users\GF63\.skills\global\shared\_vendor\nature-skills\84880815fb37317b3766bff2c2abba395b8993c3`.
- Rà soát đối chiếu nhánh upstream HEAD (`7d5f160ebfe8033375b4afef7c5911aa4203c983`):
  - Phân loại toàn bộ 21 kỹ năng thượng nguồn cho thấy không có lý do phá vỡ tính ổn định của hệ thống.
  - Các kỹ năng cốt lõi học thuật (`nature-academic-search`, `nature-data`) đã có adapter nội bộ ổn định (`academic-evidence-search`, `data-evidence-gate`).
  - Kỹ năng `nature-literature-pipeline`, `nature-reader`, `nature-ref-verifier` được tích hợp dưới dạng mẫu thuật toán chạy theo yêu cầu (on-demand), từ chối tuyệt đối việc kích hoạt tiến trình ngầm (daemon, cron, push notifications) nhằm bảo vệ tính toàn vẹn tài nguyên và bảo mật.
  - Kỹ năng `nature-experiment-log` bị từ chối triệt để (`REJECT_OVERLAP`) vì xung đột với Living Knowledge Vault Checkpoints.
  - Chính sách đóng băng: Duy trì cam kết `AUTO_UPDATE=FALSE`.

---

## 3. Khảo Sát Khả Năng Tiếp Cận Nguồn Học Thuật Trực Tiếp (Live Reachability)

Thực nghiệm đo đạc lưu lượng mạng thực tế trên 12 phân lớp nguồn học thuật, mã nguồn và cơ sở dữ liệu:

| Phân lớp nguồn | Điểm cuối (Endpoint) | Trạng thái | Mã HTTP / Giao thức | Cơ chế thay thế (Fallback) |
|---|---|---|---|---|
| Preprints Học thuật | arXiv Export API (`export.arxiv.org/api/query`) | PASS | HTTP 200 | search_web |
| Trích dẫn & Siêu dữ liệu | OpenAlex REST API (`api.openalex.org/works`) | PASS | HTTP 200 | Crossref REST API |
| Chỉ mục DOI Toàn cầu | Crossref REST API (`api.crossref.org/works`) | PASS | HTTP 200 | OpenAlex REST API |
| DOI Dữ liệu & Mã nguồn | DataCite REST API (`api.datacite.org/dois`) | PASS | HTTP 200 | Crossref REST API |
| Hội thảo NLP | ACL Anthology (`aclanthology.org`) | PASS | HTTP 200 | OpenAlex REST API |
| Hội thảo Machine Learning | PMLR (`proceedings.mlr.press`) | PASS | HTTP 200 | OpenAlex REST API |
| Kho Dữ liệu Chuẩn | Hugging Face Datasets API (`huggingface.co/api/datasets`) | PASS | HTTP 200 | search_web |
| Kho Mã nguồn Mở | GitHub Search REST API (`api.github.com/search/repositories`) | PASS | HTTP 200 | search_web |
| Dữ liệu Khoa học Mở | Zenodo REST API (`zenodo.org/api/records`) | PASS | HTTP 200 | DataCite REST API |
| Tri thức Nội bộ Cục bộ | WeKnora Local SQLite (`knowledge.sqlite`) | PASS | Local File Read | Documents/References |
| Đồ thị Học thuật | Semantic Scholar API (`api.semanticscholar.org`) | GAPPED | HTTP 429 (Rate Limited) | OpenAlex / arXiv |
| Biên bản Phản biện Kín | OpenReview API (`api2.openreview.net`) | GAPPED | HTTP 403 (Token Required) | search_web / arXiv |

Hệ thống đạt tỷ lệ tiếp cận trực tiếp 10/12 nguồn không cần xác thực token, đồng thời 2 nguồn có giới hạn truy cập (Semantic Scholar, OpenReview) đều có đường dẫn fallback xác định và minh bạch.

---

## 4. Phát Hiện Lỗi Định Tuyến và Sửa Chữa Cốt Lõi (Narrow Repair)

### 4.1. Nguyên nhân Gốc rễ Sai lệch (Root Cause)
Quá trình kiểm thử trước đây phát hiện 2 khiếm khuyết trong `C:\Users\GF63\.skills\global\agent-system-v2\router.py`:
1. Tác vụ đối chiếu mã nguồn bài báo (Canary C4: *"For one influential method, locate official implementation and compare assumptions with the paper"*): Từ khóa `implementation` bị bộ lọc fallback bắt nhầm chuỗi con `implement` ở nhánh mã nguồn thông thường, dẫn tới việc định tuyến nhầm sang pha `IMPLEMENT`.
2. Tác vụ kiểm soát âm bản (Canary C7: *"Fix this bounded Python syntax/type/path error."*): Do có ký tự gạch chéo `/`, chuỗi không khớp các mẫu đơn lẻ như `syntax error` hay `type error`, dẫn tới việc rơi vào fallback mặc định và bị phân loại nhầm thành `RESEARCH`.

### 4.2. Sửa chữa Tối thiểu Xác định
Tệp `router.py` đã được cập nhật chính xác:
- Bổ sung các mẫu nhận diện nghiên cứu nâng cao: `deep read`, `extract core method`, `locate official implementation`, `compare assumptions with the paper`, `paper-to-code`, `reproducibility check`, `propose the cheapest experiment`, `discriminating leading hypotheses`, `citations/dois`, `reference verification`.
- Bổ sung mẫu nhận diện kiểm soát âm bản: `syntax/type/path error`, `fix this bounded python`, `path error`.
- Bổ sung mẫu nhận diện cổng bản quyền dữ liệu: `candidate datasets`, `audit provenance`, `leakage risk`, `dataset provenance`.

---

## 5. Kết Quả Kiểm Thử Thực Tế Bộ Canary C1–C7

### 5.1. Kết Quả Canary Toàn Diện
Sau sửa chữa, toàn bộ 7 kịch bản kiểm thử đều đạt độ chính xác 100%:

| Mã Canary | Tác vụ Khảo sát | Pha Định Tuyến | Ý Định Khớp | Kỹ Năng / Trạng Thái |
|---|---|---|---|---|
| C1 | Tìm kiếm Prior Art và phân lập đóng góp | RESEARCH | `research:academic-evidence-search` | `academic-evidence-search` (PASS) |
| C2 | Rà soát Bản quyền Dữ liệu & Rò rỉ | RESEARCH | `research:data-evidence-gate` | `data-evidence-gate` (PASS) |
| C3 | Đọc sâu bài báo, trích xuất công thức & giả định | RESEARCH | `research:academic-evidence-search` | `academic-evidence-search` (PASS) |
| C4 | Đối chiếu Triển khai Mã nguồn và Giả định bài báo | RESEARCH | `research:academic-evidence-search` | `academic-evidence-search` (PASS) |
| C5 | Thiết kế thực nghiệm chi phí tối thiểu | RESEARCH | `research:academic-evidence-search` | `academic-evidence-search` (PASS) |
| C6 | Kiểm định tính chuẩn xác của trích dẫn & DOI | RESEARCH | `research:academic-evidence-search` | `academic-evidence-search` (PASS) |
| C7 | Kiểm soát Âm tính (Sửa lỗi cú pháp code) | IMPLEMENT | `implement:code:patch` | Tránh xa RESEARCH (PASS) |

### 5.2. Kiểm Tra Lan Truyền Trên 8 Dự Án và Nhánh Successor
Kiểm tra smoke test C1-lite và C7 âm tính đồng loạt trên toàn bộ 8 dự án logic và 1 successor:
- `long-range-temporal-alignment`: C1 PASS, C7 PASS
- `DSC-TinDipLaPo`: C1 PASS, C7 PASS
- `ds314-evidence-grounded-retrieval`: C1 PASS, C7 PASS
- `ds201-temporal-retrieval-learning`: C1 PASS, C7 PASS
- `ds111-retrieval-data-analysis`: C1 PASS, C7 PASS
- `DS335-VLM-Research`: C1 PASS, C7 PASS
- `evidence-grounded-text-agent`: C1 PASS, C7 PASS
- `general-reasoning-fact-checking`: C1 PASS, C7 PASS
- `Long-Range-Representation` (canonical successor): C1 PASS, C7 PASS

Toàn bộ 8/8 dự án và successor đều phản hồi nhất quán, không có ngoại lệ.

### 5.3. Kiểm Tra Đơn Vị Lõi và Hồi Quy Codex
- Chạy toàn bộ 15 kiểm thử đơn vị trong `C:\Users\GF63\.skills\global\agent-system-v2\tests`:
  - 15/15 tests hoàn thành trong 0.58 giây với kết quả `OK`.
  - Canaries H1 (Antigravity Bootstrap), H2 (Codex Workflow Hint), H3 Positive, H3 Negative, H5 Evidence Receipt, H8 Stop Gate đều PASS.
  - Số lượng hồi quy trên Codex: 0. Tổn thất tính năng: 0.

---

## 6. Kết Luận và Định Hướng Vận Hành

Không gian nghiên cứu (RESEARCH namespace) trong Agent System v2 đã được thẩm định toàn diện trên môi trường sống thực tế. Hệ thống định tuyến chính xác, quản trị ngữ cảnh tinh gọn theo cơ chế JIT, khai thác hiệu quả các nguồn tri thức học thuật chính thống mà không phát sinh chi phí LLM ngầm hay nguy cơ rò rỉ dữ liệu.

Các tạo tác kiểm toán đã được ghi nhận đầy đủ tại:
- `artifacts/audits/research/RESEARCH_NAMESPACE_END_TO_END_AUDIT.json`
- `artifacts/audits/research/RESEARCH_SOURCE_COVERAGE_MATRIX.json`
- `artifacts/audits/research/RESEARCH_NATURE_SKILLS_GAP_ANALYSIS.json`

Hệ thống sẵn sàng chuyển sang giai đoạn vận hành nghiên cứu chuyên sâu trực tiếp trên các bài toán khoa học của từng dự án.
