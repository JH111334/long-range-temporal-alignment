---
loại: "chuẩn-metadata"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
cập_nhật_lần_cuối: "2026-09-06"
---

# Chuẩn metadata

Chỉ gắn YAML cho HOME, decision/specification hiện hành và reference trọng yếu. Không retrofit log, raw output hoặc generated artifact.

## Trường chung

| Trường | Quy tắc |
|---|---|
| loại | Vai trò của node tri thức |
| dự_án | Đúng project owner |
| trạng_thái | Decision chỉ dùng đề-xuất, đang-thử-nghiệm, đang-dùng, bị-thay-thế, ngừng-sử-dụng |
| chuẩn_hiện_hành | Boolean; Reference luôn false |
| cập_nhật_lần_cuối | ISO date YYYY-MM-DD |

Decision thêm mã, thay_thế, bị_thay_thế_bởi, tham_chiếu và được_xác_thực_bởi. Reference thêm nguồn, độ_tin_cậy và chủ_đề; độ tin cậy chỉ dùng canonical, high, medium, exploratory.

~~~yaml
---
loại: "quyết-định"
mã: "PROJECT-D001"
dự_án: "project"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
thay_thế: []
bị_thay_thế_bởi: []
tham_chiếu: []
được_xác_thực_bởi: []
cập_nhật_lần_cuối: "2026-09-06"
---
~~~

Validator dùng một YAML subset dependency-free: scalar một dòng, boolean và JSON-style inline list.
