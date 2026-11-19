# Báo Cáo Quyết Định Tiếp Nhận Mô Hình Nền Qwen2.5-3B-Instruct

**Dự án**: `long-range-temporal-alignment` — Learning Track — Stage 03  
**Nhánh thực hiện**: `v2-targeted-continual-alignment`  
**Thời điểm ban hành**: `2026-10-20`  
**Căn cứ pháp lý & hướng dẫn**: Chỉ đạo định hướng nghiên cứu của Giảng viên Hướng dẫn (GVHD) ngày 20/10/2026  
**Trạng thái quyết định**: `APPROVED_FOR_PREPARATION` (Chỉ chuẩn bị, chưa cấp phép huấn luyện quy mô lớn)

---

## 1. Bối Cảnh và Quyết Định Định Hướng Của GVHD

1. **Nhận định của GVHD**:
   - Qua các chu kỳ kiểm toán pháp y tại Stage 03, mô hình `Qwen2.5-0.5B-Instruct` đã bộc lộ những hạn chế nghiêm trọng về mặt dung lượng tham số ($0.49\text{B}$) và thiên kiến tiên nghiệm cố hữu (co cụm về 'b' trên MCQA $78.2\%$ và co cụm về 'No' trên câu hỏi xác nhận $100\%$).
   - GVHD định hướng chuyển đổi sang thử nghiệm trên mô hình **`Qwen2.5-3B-Instruct`** thuộc cùng họ kiến trúc Qwen2.5 nhằm kiểm chứng xem dung lượng tham số lớn hơn ($3.09\text{B}$) có giải tỏa được sự sụp đổ nhãn và nâng cao năng lực tiếp nhận tín hiệu thời gian hay không.
2. **Yêu cầu bất biến về dữ liệu (Data Integrity Directive)**:
   - **Giữ nguyên tập dữ liệu gốc**: Tuyệt đối không xáo trộn, sửa đổi, lọc bỏ hay tái tạo dữ liệu để "cứu điểm số" (accuracy).
   - **Không áp dụng kỹ thuật đảo vị trí phương án (No Option Permutation)**: Khóa toàn bộ các module hoán vị chữ cái A/B/C/D trong cả pha huấn luyện và pha đánh giá.
   - **Bản chất khoa học**: Đây là một *quyết định định hướng thực nghiệm* (empirical hypothesis), không phải lời khẳng định chắc chắn rằng mô hình 3B sẽ giải quyết được toàn bộ các nút nghẽn. Nghiên cứu tuyệt đối không biến tướng thành bài toán tối ưu điểm MCQA bằng cách can thiệp vào dữ liệu.

---

## 2. Thông Tin Thẩm Định Mô Hình Qwen2.5-3B-Instruct

- **Tên mô hình chính thức**: `Qwen/Qwen2.5-3B-Instruct`
- **Mã băm commit revision cố định (Pinned Revision)**: `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- **Đường dẫn Hub**: `https://huggingface.co/Qwen/Qwen2.5-3B-Instruct`
- **Kiến trúc mô hình**: `Qwen2ForCausalLM`
- **Kích thước số học**:
  - Số lớp ẩn (Layers): $36$
  - Chiều ẩn (Hidden Size): $2048$
  - Chiều trung gian MLP (Intermediate Size): $11008$
  - Số đầu chú ý Query: $16$ (Head Dimension: $128$)
  - Số đầu chú ý Key/Value: $2$ (Grouped-Query Attention, GQA)
  - Quy mô từ vựng (Vocab Size): $151936$ (Tokenizer vocab: $151665$)
  - Tổng số tham số: $3.087.815.680$ (~$3.09\text{B}$)

---

## 3. Kiểm Toán Giấy Phép Sử Dụng (License Audit)

1. **Điều khoản cấp phép**:
   - Khác với Qwen2.5-0.5B được phát hành dưới giấy phép `Apache-2.0` (cho phép thương mại hóa), Qwen2.5-3B-Instruct được quản lý theo thỏa thuận **`Qwen RESEARCH LICENSE AGREEMENT`** (ngày ban hành: 19/09/2024).
   - Điều khoản 2.a quy định rõ: Giấy phép chỉ cấp quyền sử dụng, sao chép, tạo tác phẩm phái sinh **CHO MỤC ĐÍCH PHI THƯƠNG MẠI (NGHIÊN CỨU VÀ ĐÁNH GIÁ HỌC THUẬT DUY NHẤT)**.
2. **Đánh giá tính tuân thủ của dự án**:
   - Dự án `long-range-temporal-alignment` là công trình nghiên cứu học thuật của sinh viên và GVHD tại Trường Đại học, hoàn toàn không có hoạt động kinh doanh hoặc khai thác thương mại.
   - Trọng số mô hình và các adapter chỉ được lưu trữ nội bộ trong bộ nhớ đệm (cache) phục vụ suy luận đánh giá.
   - **Kết luận thẩm định**: Dự án hoàn toàn tuân thủ các điều khoản của `Qwen RESEARCH LICENSE AGREEMENT`. Nghiêm cấm phân phối công khai trọng số mô hình hoặc các bản phái sinh ra ngoài phạm vi nghiên cứu khi chưa có sự chấp thuận của Alibaba Cloud.

---

## 4. Kế Hoạch Triển Khai Phân Kỳ (Phased Execution Roadmap)

Nghiên cứu tuân thủ lộ trình phân kỳ nghiêm ngặt nhằm bảo đảm ngân sách tính toán và rào chắn an toàn:

1. **Giai đoạn Hiện Tại — Chuẩn Bị & Khóa Hợp Đồng (`PRE_3B_READY`)**:
   - Hoàn tất toàn bộ tài liệu đặc tả hình học, kiểm toán giấy phép, ma trận cắt bỏ và unit tests ở chế độ metadata-only.
   - **Không tải đa-gigabyte trọng số, không huấn luyện, không cấp phát GPU**.
2. **Giai đoạn Kế Tiếp — Đo Lường Cơ Bản (`GO_BASE3B_INFERENCE`)**:
   - Khi được người dùng cấp phép, tải trọng số mô hình gốc và thực thi suy luận thuần túy cánh tay `Q3_A0_BASE` trên tập kiểm thử 54 cặp để đo lường thiên kiến tiên nghiệm gốc của 3B.
3. **Giai đoạn Đánh Giá Tinh Chỉnh (`GO_SMOKE_3B`)**:
   - Chỉ khi cánh tay `Q3_A0_BASE` cung cấp dữ liệu cơ sở hợp lệ và được cấp kinh phí tính toán, mới khởi tạo một đợt chạy thử nghiệm nhỏ (smoke run) đối chứng giữa `Q3_A1` (Static LoRA) và `Q3_A3` (Full Temporal) trên Kaggle GPU.
   - Duy trì trạng thái **`NO_GO_LARGE_TRAINING`** làm rào chắn mặc định.
