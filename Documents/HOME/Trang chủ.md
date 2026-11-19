---
loại: "điều-hướng"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
cập_nhật_lần_cuối: "2026-11-19"
---

# long-range-temporal-alignment

Nghiên cứu biểu diễn thời gian dài và truy xuất bằng chứng lịch sử tiếng Việt theo hai track có gate.

Trạng thái và gate Stage 03 hiện hành xem [[Branches/v2-targeted-continual-alignment/Current-State|trạng thái nhánh V2]].

## Khung Mục Tiêu Nghiên Cứu Ba Tầng (Goal Ladder)

Theo quyết định chuẩn hóa ngày 05/11/2026 tại [[Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105|ADR 23]]:
- **GOAL 1 (`G1`)**: YEAR-only Closed-Answer Temporal QA (MCQA 4 lựa chọn + binary Yes/No). Độ chính xác đáp án đóng là tiêu chuẩn nghiệm thu bắt buộc.
  - *Nhánh & Mô thức hiện hành (`v2-targeted-continual-alignment`, [[Decision-Making/25_DECISION_ZERO_SHOT_CROSS_TASK_TRANSFER|ADR 25]]):* Zero-Shot Cross-Task Transfer (huấn luyện NWP thuần số học 104k $\to$ đánh giá MCQA/Rel). **Ràng buộc kiến trúc bất biến:** chỉ can thiệp các tầng MLP Linear (`gate_proj`, `up_proj`, `down_proj`, 108 modules), **tuyệt đối không can thiệp Attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) và không dùng DoRA/DLoRA**. Trạng thái: `ACTIVE_CURRENT_SCOPE`.
- **GOAL 2 (`G2`)**: YEAR + MONTH Closed-Answer Temporal QA (Mở rộng độ hạt tháng, nhất quán xuyên cấp độ). Trạng thái: `NEXT_SCOPED_GOAL / NOT_YET_MEASURED`.
- **GOAL 3 (`G3`)**: General Temporal QA (Closed QA + Open-ended QA + Abstention ngữ cảnh Việt Nam). Trạng thái: `ULTIMATE_RESEARCH_GOAL / TARGET_NOT_ACHIEVED`.

## Quyết định đang chi phối

- [[Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105|Khung mục tiêu nghiên cứu ba tầng và tiêu chuẩn nghiệm thu đo lường (ADR 23)]]
- [[Decision-Making/12_PROJECT-CONTEXT|Ngữ cảnh bền vững]]
- [[Decision-Making/04_Quyet_dinh_thiet_ke|Quyết định thiết kế]]
- [[Decision-Making/07_Ke_hoach_thuc_nghiem|Kế hoạch thực nghiệm]]
- [[Decision-Making/15_CAPABILITY-BASED-COMPUTE|Compute theo capability]]
- [[Decision-Making/16_SOURCEAPI-BOUNDARY|Ranh giới SourceAPI]]
- [[Decision-Making/18_STAGE03_MECHANISM_BEFORE_SCALE|Quyết định tinh chỉnh cơ chế trước khi scale Stage 03]]
- [[Decision-Making/19_STAGE03_REPRESENTATION_FIRST_CONTRACT|Khế ước thẩm định biểu diễn Stage 03 (Đã bị thay thế một phần bởi ADR 23)]]

## Reference trọng yếu

- [[References/Research-References|Thư viện tham chiếu tùy chọn và đối chiếu Stage 03]]

- [[References/Markdown/01_Danh_muc_nguon|Danh mục nguồn]]
- [[References/Markdown/02_Phan_tich_paper_Ticktack|Phân tích Ticktack]]

## Điều phối

- [[HOME/Bản đồ tri thức|Bản đồ tri thức]]
- [[HOME/Quy tắc tri thức|Quy tắc tri thức]]
- [[HOME/Chuẩn metadata|Chuẩn metadata]]
- [[HOME/Nhật ký thay đổi tri thức|Nhật ký thay đổi tri thức]]
- [[General/RULE|Quy tắc log General]]

## Điều phối nhiều nhánh

- [[Global/Current-State|Trạng thái chung]]
- [[Global/Branch-Registry|Registry nhánh]]
- [[Global/Branch-Protocol|Giao thức nhánh]]
- [[Global/Joint-Gates|Joint Gate]]
- [[Global/GOAL_BRANCH_STAGE_CROSSWALK|Bản đồ ánh xạ mục tiêu (Crosswalk G1/G2/G3)]]
- [[Branches/v1-dual-cycle-mat-lora/HOME|Nhánh V1: Starting Anchor (Dual-Cycle MaT-LoRA)]]
- [[Branches/v2-targeted-continual-alignment/HOME|Nhánh V2: Active Working Branch (Targeted Continual Alignment)]]
