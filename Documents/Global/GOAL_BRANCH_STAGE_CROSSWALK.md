---
loại: "điều-hướng"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
cập_nhật_lần_cuối: "2026-11-05"
---

# Bản Đồ Ánh Xạ Mục Tiêu — Nhánh — Chặng Triển Khai (Goal-Branch-Stage Crosswalk)

> [!NOTE] CĂN CỨ PHÁP LÝ & QUẢN TRỊ
> Tài liệu này chuẩn hóa mối quan hệ ánh xạ giữa Dải Mục Tiêu Nghiên Cứu (`GOAL_ID`), Không Gian Nhánh (`BRANCH_ID`), Chặng Triển Khai (`STAGE_ID`) và Biến Thể Thực Nghiệm (`ARM_ID`), tuân thủ nghiêm ngặt [ADR 23: GOAL LADDER AND ACCEPTANCE](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Documents/Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105.md) ban hành ngày 05/11/2026.

---

## 1. Bảng Ánh Xạ Toàn Diện (Comprehensive Crosswalk Matrix)

| Thuộc Tính | GOAL 1 (`G1`) | GOAL 2 (`G2`) | GOAL 3 (`G3`) |
| :--- | :--- | :--- | :--- |
| **Tiêu đề mục tiêu** | YEAR-only Closed-Answer Temporal QA | YEAR + MONTH Closed-Answer Temporal QA | General Temporal QA (Closed + Open-ended) |
| **Độ hạt thời gian (`granularity`)** | `YEAR` (Năm lịch sử, BCE/CE) | `YEAR` + `MONTH` (Năm và Tháng kết hợp) | `YEAR` + `MONTH` + `OPEN_QA` (Đa độ hạt + Sinh tự do) |
| **Định dạng câu trả lời** | MCQA 4 phương án (`A/B/C/D`) + Nhị phân (`Yes/No`) | MCQA 4 phương án (`A/B/C/D`) + Nhị phân (`Yes/No`) | Closed QA (`A/B/C/D`, `Yes/No`) + Open-ended có trích dẫn nguồn + Abstention |
| **Trọng tâm kiểm chứng** | Nhận diện mốc năm, suy luận trật tự năm, loại bỏ shortcut thực thể | Mặt nạ độ hạt, nhất quán xuyên cấp độ (tháng $\subset$ năm), triệt tiêu ảo giác tháng | Sinh câu trả lời có căn cứ lịch sử Việt Nam, từ chối khi thiếu dữ kiện, bảo toàn năng lực |
| **Các nhánh hỗ trợ (`branches_supporting`)** | `v1-dual-cycle-mat-lora` (Baseline Anchor), `v2-targeted-continual-alignment` (Active) | `v2-targeted-continual-alignment` (Chưa kích hoạt cho G2) | `v2-targeted-continual-alignment` (Chưa kích hoạt cho G3), Track-A + Track-B Joint |
| **Các chặng hỗ trợ (`stages_supporting`)** | `Stage 01`, `Stage 02`, `Stage 03` | `Stage 04` (Được đề xuất sau khi G1 đóng hoàn toàn) | `Stage 05` (Tổng hợp Track A + B) |
| **Các Arms hỗ trợ (`arms_supporting`)** | `Arm 1` (Base), `Arm 2` (MaT-LoRA), `Arm 3` (Temporal), `Arm 4` (Static), `Arm 5` (Hợp nhất) | Các Arms mở rộng tiếp theo (Chưa cấp phát) | Toàn bộ hệ thống hoàn chỉnh |
| **Tập dữ liệu liên kết** | Train: `data/tempun_full_clean/train_104k_clean.jsonl.gz`<br/>Eval: `data/tempun_full_clean/eval_5292_clean.jsonl.gz` (Dev Benchmark) | Tệp dữ liệu tháng có chứng thực nguồn gốc (Chưa khởi tạo) | Benchmark lịch sử Việt Nam mở rộng + Căn cứ pháp lý |
| **Phạm vi công bố (`claim_scope`)** | Năng lực trả lời đóng về mốc năm lịch sử; không ngoại suy sang tháng hay sinh tự do | Năng lực phân giải năm kết hợp tháng; không ngoại suy sang sinh tự do mở | Năng lực tổng quát hoá toàn diện trên tri thức lịch sử Việt Nam |
| **Trạng thái bằng chứng (`evidence_status`)** | `ACTIVE_CURRENT_SCOPE`<br/>`QUARANTINED_PENDING_PROVENANCE`<br/>`METRIC_ARITHMETIC_ONLY` (5 epochs) | `NEXT_SCOPED_GOAL`<br/>`NOT_YET_MEASURED` | `ULTIMATE_RESEARCH_GOAL`<br/>`TARGET_NOT_ACHIEVED` |
| **Cổng kiểm định tiếp theo (`next_gate`)** | `GPU_REPLAY_VERIFICATION` (Đo kiểm độc lập trên GPU mẫu đại diện) | Hoàn thành đóng `G1` + Phê duyệt Data Card cho độ hạt Tháng | Hoàn thành đóng `G2` + Thiết lập Gold Test Set sinh tự do có đối sánh |

---

## 2. Nguyên Tắc Vận Hành và Giới Hạn Quản Trị

1. **Bảo toàn Nhánh và Chặng**:
   - Việc xác lập thang mục tiêu $G_1 \to G_2 \to G_3$ là tái định hình tiêu chí nghiệm thu năng lực đầu ra, **không xóa bỏ, không phân rã** các branch (`v1`, `v2`) và các stage (`Stage 01`, `Stage 02`, `Stage 03`) đã niêm phong.
   - Nhánh `v2-targeted-continual-alignment` và `Stage 03` hiện phục vụ độc quyền việc hoàn tất nghiệm thu cho `GOAL 1`.

2. **Ranh Giới Bằng Chứng Nghiêm Ngặt**:
   - Không được công bố bất kỳ kết quả nào cho `GOAL 2` khi dữ liệu mới chỉ chứa độ hạt Năm (`YEAR`).
   - Không được tuyên bố đạt được `GOAL 3` khi chưa có tập đánh giá Open-ended QA có thẩm định độc lập.
   - Các kết quả 5 epochs hiện tại của `Stage 03` duy trì nhãn `METRIC_ARITHMETIC_ONLY` và trạng thái `QUARANTINED_PENDING_PROVENANCE` cho đến khi thực hiện Replay GPU đối chứng.
