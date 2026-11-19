---
loại: "quyết-định"
mã: "DECISION-STAGE03-QWEN25-3B-SCALE-TRANSITION-109"
dự_án: "long-range-temporal-alignment"
trạng_thái: "bị-thay-thế"
chuẩn_hiện_hành: false
thay_thế: []
bị_thay_thế_bởi: ["[[21_STAGE03_QWEN25_3B_BASE_DUAL_DISPATCH]]"]
tham_chiếu: ["18_STAGE03_MECHANISM_BEFORE_SCALE", "19_STAGE03_REPRESENTATION_FIRST_CONTRACT"]
được_xác_thực_bởi: []
cập_nhật_lần_cuối: "2026-10-20"
---

# Quyết Định Chuyển Đổi Mô Hình Nền Sang Qwen2.5-3B-Instruct và Khóa Rào Chắn Tiền Triển Khai

## 1. Tóm Tắt Định Hướng Khoa Học và Chỉ Đạo Của GVHD

Ngày 20/10/2026, Giảng viên Hướng dẫn (GVHD) đã đưa ra chỉ đạo định hướng quan trọng đối với Giai đoạn 03 (Stage 03):
1. **Chuyển đổi quy mô dung lượng mô hình**: Mô hình `Qwen2.5-0.5B-Instruct` bộc lộ giới hạn dung lượng ($0.49\text{B}$) và thiên kiến tiên nghiệm cố hữu (co cụm về 'b' trên MCQA $78.2\%$ và co cụm về 'No' trên câu hỏi xác nhận $100\%$). GVHD định hướng thử nghiệm mô hình nền lớn hơn trong cùng họ: **`Qwen/Qwen2.5-3B-Instruct`** (~$3.09\text{B}$ tham số) để kiểm chứng giả thuyết về dung lượng mô hình (**Capacity-Scale Hypothesis**).
2. **Nguyên tắc bảo tồn dữ liệu tuyệt đối (Zero Data Mutation)**: Giữ nguyên vẹn $100\%$ tập dữ liệu gốc, không xáo trộn, sửa đổi hay tái tạo dữ liệu để "cứu điểm số" (accuracy).
3. **Khóa chức năng hoán vị lựa chọn (No Option Permutation)**: Giữ nguyên định dạng và thứ tự phương án ban đầu, khóa toàn bộ các module hoán vị A/B/C/D (`OnlineOptionPermutation.enabled = False`).
4. **Bản chất khoa học**: Đây là một *quyết định định hướng thực nghiệm*, không phải lời khẳng định 3B chắc chắn tốt hơn. Nghiên cứu tuyệt đối không biến tướng thành bài toán tối ưu điểm MCQA bằng cách can thiệp vào dữ liệu.

---

## 2. Thông Số Thẩm Định Qwen2.5-3B-Instruct và Giấy Phép Nghiên Cứu

1. **Thông số kiến trúc chuẩn tắc**:
   - Tên định danh: `Qwen/Qwen2.5-3B-Instruct`.
   - Mã băm commit revision cố định: `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
   - Kích thước số học: $36$ tầng transformer, $d_{\text{model}} = 2048$, $d_{\text{intermediate}} = 11008$, $16$ query heads, $2$ KV heads (GQA).
   - Tổng số module MLP cần can thiệp LoRA: **$108$ modules** ($36 \times 3$), được phát hiện động tại runtime.
2. **Kiểm toán giấy phép (License Audit)**:
   - Quản lý theo thỏa thuận **`Qwen RESEARCH LICENSE AGREEMENT`** (ngày ban hành: 19/09/2024).
   - Thẩm định phạm vi: Giấy phép chỉ cấp quyền cho mục đích nghiên cứu và đánh giá học thuật phi thương mại. Dự án của sinh viên và GVHD hoàn toàn tuân thủ. Nghiêm cấm phân phối thương mại hoặc chuyển giao trọng số ra ngoài phạm vi nghiên cứu.

---

## 3. Ranh Giới Kỹ Thuật và Giao Thức So Sánh Liên Quy Mô

1. **Khóa cấm tải chéo trọng số (Zero Cross-Scale Weight Loading)**:
   - Do kích thước chiều ẩn thay đổi ($896 \to 2048$), tuyệt đối không nạp/reuse trọng số LoRA, projection 40D hay optimizer state của 0.5B vào 3B.
   - Thí nghiệm 3B là một mô hình nền mới (`NEW BASE BACKBONE`), khởi tạo adapter mới từ đầu với seed cố định.
2. **Ánh xạ độ sâu tương đối Gated Seam**:
   - Điểm tiêm Gated Seam được chuẩn hóa tại **Tầng 21 trên 36 tầng** (tương đương độ sâu tương đối $21/36 \approx 0.5833$, bảo toàn vị trí hình học của tầng 14/24 trên 0.5B).
   - Vị trí token tiêm được xác định động tại runtime prompt parser, không hardcode token 14.
3. **Ma trận cắt bỏ 5 cánh tay đăng ký trước**:
   - `Q3_A0_BASE`: Mô hình 3B đóng băng 100%, đo lường thiên kiến và năng lực tự nhiên gốc.
   - `Q3_A1_STATIC_MLP_LORA`: Thích ứng LoRA tĩnh, đo lường tác động thích ứng phi thời gian.
   - `Q3_A2_TEMPORAL_CORE_NO_SEAM`: Thích ứng MaT-LoRA lõi động, Seam OFF.
   - `Q3_A3_FULL_TEMPORAL`: Thích ứng MaT-LoRA lõi động, Seam ON tại Tầng 21 (phương pháp đề xuất).
   - `Q3_A4_TICKTACK_INSPIRED`: Checkpoint đối chứng do đối tác cung cấp độc lập.

---

## 4. Quản Trị Cổng Thực Thi và Rào Chắn An Toàn

1. **Trạng thái cổng hiện hành**:
   - **`PRE_3B_READY`**: Đã hoàn tất $14/15$ cổng chuẩn bị (`G01` – `G10`, `G12` – `G15` đạt `PASS`).
   - Cổng `G11_GPU_VRAM_PRECISION_MEASUREMENT` được ghi nhận chuẩn xác là **`NOT_VERIFIED`** do chưa có GPU cấp phát trong phiên chuẩn bị.
2. **Rào chắn khóa dừng mặc định**:
   - Duy trì nghiêm ngặt: **`GO_3B_PREPARATION`**, **`NO_GO_DATA_MUTATION`**, **`NO_GO_OPTION_PERMUTATION`**, **`NO_GO_LARGE_TRAINING`**.
   - Tuyệt đối không tải đa-gigabyte trọng số, không huấn luyện GPU lớn khi chưa có phê duyệt chính thức từ người dùng.
3. **Thao tác tiếp theo được cấp phép**:
   - Chỉ khi người dùng ra lệnh rõ ràng, bước thực thi đầu tiên duy nhất được phép kích hoạt là **suy luận mốc cơ sở `Q3_A0_BASE`** để thu thập phân bố dự đoán tự nhiên của Qwen2.5-3B-Instruct trên 54 cặp phản thực nghiệm.
