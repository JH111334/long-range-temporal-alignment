# Documents

Thư mục này chỉ giữ `README.md` ở cấp gốc. Tài liệu được phân theo vai trò để quyết định, nguồn tham chiếu và nhật ký vận hành không bị trộn lẫn.

## Cấu trúc

- `Decision-Making/`: problem contract, quyết định phương pháp, kế hoạch thí nghiệm, vòng đời dữ liệu–mô hình và kinh nghiệm lỗi.
- `References/Markdown/`: khảo sát paper, dataset, model, thuật ngữ và hướng dẫn dịch vụ ngoài.
- `References/DOCX/`: mười tài liệu Word giao nộp; nguồn duy trì nằm tại `../docs/converted/`.
- `References/HTML/`: báo cáo tiến độ tự chứa cho giảng viên hướng dẫn.
- `References/bao_cao_nckh_temporal_alignment.html`: Báo cáo khoa học tự chứa toàn diện chuẩn hóa 3 phần (Khảo sát Data, System Pipeline, Gợi ý Hướng cải thiện độ hạt tháng).
- `General/`: chỉ `RULE.md` và nhật ký `*_Checkpoints_DD-MM-YY_Vn.md` hoặc `*_Error_DD-MM-YY_Vn.md`.

Thông tin môn học, MCP, workflow và hướng dẫn nằm trong `References/Markdown/`; biên bản và phạm vi hoãn nằm trong `Decision-Making/`. Xem `General/RULE.md` trước khi tạo log.

Điểm vào cho phiên mới là `Decision-Making/12_PROJECT-CONTEXT.md`; phạm vi ba hướng nằm tại `03_Ma_tran_khoang_trong_va_cau_hoi.md`; báo cáo tổng hợp đầy đủ là `References/bao_cao_nckh_temporal_alignment.html` (bản Markdown compact tương ứng tại `../docs/bao_cao_nckh_temporal_alignment.md`).

## Mức bằng chứng

- `paper_reported`: số liệu do nguồn công bố, nhóm chưa tái lập.
- `scaffold`: sườn import được hoặc fail-closed; không phải kết quả mô hình.
- `smoke_test`: kiểm tra hợp đồng và luồng nhỏ; không phải benchmark.
- `measured`: chỉ dùng khi có revision dữ liệu/mô hình, split, seed, config/data/code hash và artifact tài nguyên.

PDF paper, model weights và dataset có điều khoản riêng không được đưa vào Git. Paper gốc: [Ticktack, arXiv:2503.04150](https://arxiv.org/abs/2503.04150).

## Tra cứu khoa học

[Thư viện tham chiếu theo câu hỏi](References/Research-References.md). Có thể mở PDF, mục lục và ghi chú Stage 03 theo nhu cầu.
