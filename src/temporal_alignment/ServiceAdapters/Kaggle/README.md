# Kaggle GPU adapter — Long-range Temporal Alignment

Adapter này là công cụ compute tùy chọn cho Tier 3, không phải baseline và không tự
chứng minh quality. Mặc định chỉ `preflight`; `execute` chỉ hợp lệ sau data/model
registry, resource probe, paired multi-seed/CI gate và approval cho dữ liệu `public`.

CLI chỉ có hai lệnh:

```powershell
python -m temporal_alignment.ServiceAdapters.kaggle_runtime contract
python -m temporal_alignment.ServiceAdapters.kaggle_runtime run --request request.json
```

Request giữ toàn bộ config. Không ghi credential vào JSON; Kaggle CLI đọc từ
`KAGGLE_CONFIG_DIR` hoặc `%USERPROFILE%\.kaggle\kaggle.json`. `preflight` kiểm hash,
privacy, approval, kích thước và tạo bundle. `execute` mới push kernel, theo dõi và tải
expected outputs về `run_root/outputs`.

`probe_worker.py` chỉ đo môi trường GPU, không đọc dữ liệu. Copy nó vào project artifact,
tính SHA-256 và dùng làm worker đầu tiên. Mọi worker nghiên cứu thật phải tự pin model,
revision, dependency, seed và xuất metric/slice/case artifact theo evaluation contract.

Không dùng adapter cho data chưa rõ license, PII/private data hoặc để né gate local-first.
