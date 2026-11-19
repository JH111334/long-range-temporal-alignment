---
loại: "điều-hướng"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
cập_nhật_lần_cuối: "2026-11-05"
---

# Registry nhánh

Registry chỉ chứa danh tính và pointer. Trạng thái, config, kết quả nằm trong nhánh tương ứng. Tham chiếu chuẩn hóa mục tiêu tại [[Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105|ADR 23: GOAL LADDER]] và [[Global/GOAL_BRANCH_STAGE_CROSSWALK|Bản đồ ánh xạ mục tiêu]].

| Branch ID | Trọng tâm nghiên cứu | Mục tiêu (`GOAL_ID`) | Trạng thái | Parent | State pointer | Gate kế tiếp |
|---|---|:---:|---|---|---|---|
| `v1-dual-cycle-mat-lora` | Điểm xuất phát chuẩn tắc (Starting Anchor): Periodic Dual-Cycle MaT-LoRA trên FIXED_FONE_CANCHI_V1 40D qua Targeted Continual Pretraining 3 chặng | `G1` (Anchor) | ĐÃ HOÀN TẤT ĐÓNG BĂNG | NONE | `Branches/v1-dual-cycle-mat-lora/Current-State.md` | PROMOTE_TO_V2_BASELINE |
| `v2-targeted-continual-alignment` | Nhánh làm việc hiện hành (Active Working Branch): Tối ưu hóa cơ chế tiếp nhận biểu diễn thời gian dài hạn, chống quên và bảo toàn độ chính xác câu trả lời đóng | `G1` (Phục vụ hiện hành)<br/>`G2`, `G3` (Chưa kích hoạt) | ĐANG HOẠT ĐỘNG (ACTIVE) | `v1-dual-cycle-mat-lora` | `Branches/v2-targeted-continual-alignment/Current-State.md` | GPU_REPLAY_VERIFICATION |

`ĐÃ HOÀN TẤT ĐÓNG BĂNG`: Nhánh V1 đã chạy xong đầy đủ benchmark và khóa chặt làm baseline đối chứng.  
`ĐANG HOẠT ĐỘNG (ACTIVE)`: Nhánh V2 là không gian làm việc duy nhất tiếp nhận các phát triển và thử nghiệm mới, hiện tập trung giải quyết dứt điểm `GOAL 1`.
