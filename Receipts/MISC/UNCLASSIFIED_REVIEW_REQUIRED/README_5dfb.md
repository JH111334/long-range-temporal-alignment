# Bằng Chứng Thực Nghiệm Nghiên Cứu (Research Evidence)
*Khối nghiên cứu: Targeted Temporal Continual Training + MLP-Only LoRA (C0 / B1 / B2)*

Thư mục này lưu trữ hệ thống bằng chứng khoa học tinh gọn, bất biến và có thể kiểm chứng độc lập cho buổi báo cáo tiến độ nghiên cứu với Giảng viên Hướng dẫn (GVHD).

---

## 1. Cấu Trúc Các Tệp Dữ Liệu Bằng Chứng

| Tệp | Định dạng | Nội dung và vai trò kiểm chứng |
| :--- | :---: | :--- |
| `research_provenance.json` | JSON | Nguồn gốc phương pháp luận, chuỗi liên kết các pha nghiên cứu từ Week 1 đến Rescue 103, xác nhận định danh $B_3 = R_{00} = C_0$, và các ranh giới nhân quả được bảo vệ. |
| `experiment_contracts.json` | JSON | Khế ước thực nghiệm đóng băng: revision mô hình `Qwen2.5-0.5B-Instruct`, tập dữ liệu `TempUN_Small`, 3 giai đoạn lịch sử, 72 module MLP LoRA ($r=2, \alpha=4$), 180 mẫu đánh giá, siêu tham số và cấu hình giải mã greedy. |
| `branch_results.json` | JSON | Bảng kết quả định lượng chi tiết của bốn nhánh ($C_0, B_1, B_2, N_0$) theo từng seed độc lập và giá trị trung bình kèm độ lệch chuẩn (mean ± SD). |
| `paired_case_outcomes.jsonl` | JSONL | 540 bản ghi ghép cặp ($180\text{ mẫu} \times 3\text{ seed}$) lưu trữ kết quả đầu ra chi tiết (`parseable`, `outcome`, `strict_correct`) của từng trường hợp kiểm thử cho cả bốn nhánh. |
| `mcnemar_verified.json` | JSON | Kết quả tái tính toán kiểm định thống kê ghép cặp McNemar độc lập: bảng liên hợp 2×2, số lượng mẫu bất đồng $n_{\text{disc}}$, phương pháp kiểm định (Exact Binomial hoặc Edwards Asymptotic), trị số $p$, tỷ số chênh (Odds Ratio) và hiệu chỉnh đa giả thuyết Holm-Bonferroni. |
| `MANIFEST.sha256` | SHA256 | Mã băm toàn vẹn của tất cả các tệp trong thư mục. |

---

## 2. Lệnh Tái Kiểm Tra Tính Toàn Vẹn và Thống Kê (One-Liner Verification)

Người đọc có thể tái kiểm tra tính toán toàn bộ bảng liên hợp và các phép kiểm định McNemar trực tiếp từ `paired_case_outcomes.jsonl` bằng lệnh sau:

```bash
python -c "import json; lines=[json.loads(l) for l in open('paired_case_outcomes.jsonl', encoding='utf-8')]; print('Tong so ban ghi:', len(lines))"
```

Toàn bộ các tệp dự đoán gốc và trọng số mô hình được lưu trữ tại các bản phát hành bất biến:
- `artifacts/releases/WEEK1TARGET/`
- `artifacts/releases/REP_ABLATION_102/`
