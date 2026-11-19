# Báo Cáo Quyết Định Can Thiệp Một Biến và Quản Trị Cổng Nghiên Cứu (Phase D Report)

**Dự án**: `long-range-temporal-alignment`  
**Nhánh thực hiện**: `v2-targeted-continual-alignment`  
**Thời điểm khóa sổ**: `2026-10-20`  
**Trạng thái cổng hiện hành**: `NO_GO_SCALE`, `GO_CPU_DIAGNOSTIC`, `ALTERNATIVE_OBJECTIVE_PROPOSAL_ONLY`

---

## 1. Khóa Ranh Giới Chức Năng Của Module Option Permutation

Module [`OnlineOptionPermutation`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/src/temporal_alignment/training/option_permutation.py) đã được tích hợp vào kiến trúc huấn luyện với cờ mặc định `enabled=False`. Bộ unit test độc lập tại [`tests/test_stage03_option_permutation.py`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/tests/test_stage03_option_permutation.py) đã kiểm chứng tính toàn vẹn toán học và tính bất biến ánh xạ đáp án (3/3 unit tests PASS).

### Ranh Giới Phạm Vi Bắt Buộc
1. **Phạm vi áp dụng hợp lệ**: Module chỉ hoạt động trên các câu hỏi trắc nghiệm 4 phương án (MCQA) có cấu trúc danh sách `[A, B, C, D]`. Mục đích duy nhất là phân tán đồng đều phân bố nhãn mục tiêu qua các vị trí chữ cái khác nhau trong quá trình huấn luyện, nhằm loại bỏ shortcut thiên kiến vị trí (hiện tượng mô hình co cụm 78.2% dự đoán vào phương án 'b' hoặc 'c').
2. **Ranh giới phủ định tuyệt đối**: Module **hoàn toàn không thể giải quyết và không được kỳ vọng giải quyết hiện tượng sụp đổ nhãn All-No** trên tác vụ nhị phân Yes/No. Câu hỏi Yes/No không có danh sách phương án hoán vị. Việc nhầm lẫn giữa hai hiện tượng này sẽ dẫn tới các can thiệp sai lệch về mặt phương pháp luận.

---

## 2. Nguyên Tắc Can Thiệp Đơn Biến (Single-Variable Protocol)

Mọi can thiệp trong tương lai bắt buộc phải tuân thủ nghiêm ngặt nguyên tắc một biến (single-variable intervention). Tuyệt đối không thay đổi đồng thời kiến trúc encoder, hàm mục tiêu, tốc độ học và kích thước dữ liệu trong cùng một lần chạy.

### Hai Lộ Trình Thử Nghiệm Tách Biệt
- **Lộ trình 1: Thử nghiệm giải tỏa thiên kiến vị trí MCQA (Option Permutation Smoke)**:
  - Can thiệp: Bật `OnlineOptionPermutation(enabled=True)` trên tác vụ huấn luyện MCQA.
  - Điều kiện kiểm chứng: Đo lường độ entropy của chữ cái dự đoán trên tập dev MCQA (kỳ vọng entropy tăng từ ~0.9 bits lên >1.5 bits, độ chính xác dưới phép quay 4 hướng tăng từ 24.2% lên mức cân bằng ngẫu nhiên hoặc tốt hơn).
  - Trạng thái: Sẵn sàng ở mức mã nguồn (`READY_FOR_APPROVED_SMALL_RUN`), chờ phê duyệt cổng.
- **Lộ trình 2: Thử nghiệm cân bằng gradient nhị phân (Binary Loss Calibration)**:
  - Can thiệp: Thay thế Cross-Entropy thuần bằng Focal Loss hoặc Balanced Margin Loss trên tác vụ Relation để ngăn chặn hiện tượng gradient overshooting từ các mẫu "No".
  - Trạng thái: Chỉ dừng ở mức đề xuất thuật toán (`ALTERNATIVE_OBJECTIVE_PROPOSAL_ONLY`), chưa được cấp phép huấn luyện.

---

## 3. Phán Quyết Quản Trị và Cam Kết Nghiên Cứu

1. **Duy trì phán quyết `NO_GO_SCALE`**: Mọi bằng chứng từ Phase A, Phase B và Phase C đều xác nhận rằng nút nghẽn cốt lõi nằm ở khả năng nhận diện cơ chế (identifiability of mechanism) và sự sụp đổ nhãn của mô hình, không phải do thiếu quy mô dữ liệu hay số bước tối ưu. Do đó, việc mở rộng dữ liệu huấn luyện hoặc chạy GPU quy mô lớn ở thời điểm hiện tại là hoàn toàn bị cấm.
2. **Bảo toàn ranh giới phân công nghiên cứu**: Workspace của đối tác tại `C:\Users\GF63\Documents\Long-Range-Representation` được duy trì ở trạng thái **STRICT READ-ONLY (0 byte can thiệp)**. Mọi hoạt động nghiên cứu chỉ diễn ra trong phạm vi mã nguồn mục tiêu và đánh giá của nhánh hiện tại.
