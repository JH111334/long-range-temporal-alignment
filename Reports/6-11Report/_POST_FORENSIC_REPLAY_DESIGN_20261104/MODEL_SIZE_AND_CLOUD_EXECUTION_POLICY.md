# Policy cloud-first cho temporal

Tiền đề là [ma trận host](HOST_CAPABILITY_PARITY_MATRIX.json). Policy và CLI chuẩn nằm trong shared Agent System v2. Overlay temporal pin `Qwen/Qwen2.5-3B` ở revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` để đối chiếu metadata công khai. 3.085.938.688 tham số vượt ngưỡng 3.000.000.000. Nhãn 3B giữ vai trò định tuyến workflow cloud. [Model card Qwen](https://huggingface.co/Qwen/Qwen2.5-3B)

- `model-preflight` nhận request operation, model ID, revision và target. Quyền compute chi phối riêng. Metadata và adapter tensor-only CPU giới hạn có đường kiểm riêng.
- `model-storage` nhận action inventory, archive, verify, dry_run, evict_eligible hoặc restore trên file khai báo. Archive stage tạo identity để connector tải lên. Verify/restore cần ID, byte SHA và restore proof. Eviction ở dry-run vì chưa có retention được duyệt.
- Native status là PARTIAL. Dùng [canary adapter](CLOUD_EXECUTION_GUARD_CANARIES.json) để kiểm wire và [replay plan](REPLAY_EXECUTION_PLAN.md) cho ranh giới scientific.
