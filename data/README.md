# Data Boundary

`synthetic/` chứa fixture tự tạo để kiểm thử schema, BCE/CE, split, encoder và evaluator. Không suy ra chất lượng mô hình từ fixture này.

Dataset thật chỉ được đặt dưới `data/raw/` sau khi registry có đủ:

- URL chính thức và ngày truy cập;
- giấy phép hoặc điều khoản sử dụng;
- revision và checksum;
- người rà soát;
- vai trò train/validation/test;
- chính sách group/time/source split và kiểm tra near-duplicate.

`data/raw/`, `data/interim/`, model weights và checkpoints không được commit.
