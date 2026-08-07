# ADR 0001: Ranh Giới Tái Lập Ticktack

## Quyết định

Giữ ba nhãn bắt buộc:

- `published-equation`: đúng phần công thức paper đã in;
- `operational-assumption`: bổ sung để chạy được nhưng paper không đặc tả;
- `proposed-method`: giả thuyết mới của nhóm.

`PublishedTicktackEncoder` cố ý giữ va chạm giữa các năm cách nhau 60 năm do bán kính trong công thức chỉ phụ thuộc góc khi alpha/beta là hằng số. `OperationalRadialTicktackEncoder` là một cách đặc tả bổ sung, không được gọi là tái lập chính xác.

## Lý do

Paper không công bố mã nguồn/TempLS đã xác minh và không nêu hàm alpha/beta theo từng năm, LoRA rank/target modules, lambda EWC, Fisher estimation, seed hoặc evaluator đầy đủ. Trộn giả định vào baseline sẽ làm kết quả không thể quy trách nhiệm.
