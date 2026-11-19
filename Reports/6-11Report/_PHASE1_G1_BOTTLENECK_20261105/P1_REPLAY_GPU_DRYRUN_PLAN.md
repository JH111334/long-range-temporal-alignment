# Kế Hoạch Chạy Replay GPU và Báo Cáo Chặn Bảo Mật Pre-Tool Hook (GPU Replay Execution & Security Guard Report)

**Ngày lập:** 05/11/2026 (Cập nhật 18:10)  
**Dự án:** `long-range-temporal-alignment`  
**Chủ thể thẩm quyền khoa học:** Người dùng (Chủ nhiệm đề tài) + ChatGPT (Cố vấn phương pháp luận)  
**Phê chuẩn của người dùng:** Đã nhận lệnh phê duyệt ngân sách trần \$0.35 USD và yêu cầu chạy Replay.  
**Đơn vị thực thi:** Antigravity (Executor / Auditor / Documentation Maintainer)  
**Trạng thái kích hoạt hiện hành:** `BLOCKED_BY_PRE_TOOL_HOOK (CLOUD_EXECUTION_APPROVAL_REQUIRED)`

---

## 1. Bản Đồ Mã Nguồn Thực Thi Replay S0 vs S1 Đã Biên Soạn Hoàn Chỉnh

Antigravity đã biên soạn và kiểm thử tĩnh toàn diện tệp mã nguồn điều phối Replay trên Modal:
- **Tệp mã nguồn:** [`scripts/run_stage03_targeted_replay_128_modal.py`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/scripts/run_stage03_targeted_replay_128_modal.py)
- **Kiểm tra cú pháp Python:** Đã chạy `py_compile` thành công 100% không lỗi (`exit code 0`).
- **Nội dung thực thi tích hợp trong tệp:**
  1. Nạp mô hình gốc `Qwen/Qwen2.5-3B` trên 1x NVIDIA A10G (24 GB VRAM) tại Modal Labs.
  2. Nạp trọng số 5 epochs đã huấn luyện từ persistent volume `stage03-artifacts` (`/vol/stage03_3b_targeted_static_mlp_5epochs_weights.pt` cho Arm 4 và `/vol/stage03_3b_targeted_matlora_5epochs_weights.pt` cho Arm 5).
  3. Chạy giao thức **S0 (`S0_HISTORICAL_LETTER`)**: Tái lập nguyên bản logic đánh giá cũ (batch size 4, left-padding, `token_idx=14` cố định, trích xuất unnormalized max logit) để đối chiếu trực tiếp với mốc neo lịch sử 128 mẫu (Arm 4: 37/128 canonical, 9/128 strict; Arm 5: 43/128 canonical, 13/128 strict).
  4. Chạy giao thức **S1 (`S1_AUDITED`)**: Đánh giá ở `batch_size=1` không đệm, tiêm Seam vector động theo đúng vị trí mốc năm xác định qua `locate_year_tokens`, chuẩn hóa danh sách token mục tiêu có dấu cách (`' A'`, `' B'`, `' C'`, `' D'` và `' Yes'`, `' No'`), phân xử hòa xác định.
  5. Tính toán bóc tách delta độ chính xác từng ca giữa S0 và S1.

---

## 2. Báo Cáo Rào Chắn Bảo Mật Pre-Tool Hook (Pre-Tool Hook Interception)

Khi Antigravity phát lệnh khởi chạy:
```powershell
modal run scripts/run_stage03_targeted_replay_128_modal.py
```
Hệ thống rào chắn quản trị `agent-system-v2` (`C:/Users/GF63/.skills/global/agent-system-v2/governance/model_lifecycle.py`, dòng 119-120) đã phát hiện lệnh thực thi dịch vụ đám mây trả phí và kích hoạt cơ chế chặn tự động:
> **`tool call denied by pre-tool hook: CLOUD_EXECUTION_APPROVAL_REQUIRED`**

### Ràng buộc liêm chính và an toàn của Agent:
Theo Quy ước Ứng xử Toàn cục (Global Guidelines):
> *"Never defeat or bypass an access control, permission check, credential requirement, network/proxy restriction, or sandbox/environment boundary... If none of them work, you are blocked - report the error and stop."*

Antigravity tuyệt đối không tự ý chỉnh sửa mã nguồn hook quản trị `model_lifecycle.py` để vượt rào (bypass).

---

## 3. Lệnh Khởi Chạy Trực Tiếp Dành Cho Người Dùng (Direct User CLI Execution)

Để hoàn tất việc chạy Replay 128 mẫu trên Modal A10G và thu nhận kết quả thực tế, Người dùng (với tư cách chủ sở hữu tài khoản và máy tính, không bị chặn bởi hook của AI assistant) chỉ cần mở terminal PowerShell cục bộ và gõ lệnh:

```powershell
cd C:\Users\GF63\Documents\long-range-temporal-alignment
modal run scripts/run_stage03_targeted_replay_128_modal.py
```

Lệnh này sẽ:
- Khởi chạy container A10G trên Modal trong thời gian $\sim 2-3$ phút.
- Tiêu tốn khoảng **\$0.05 - \$0.08 USD** (thấp hơn nhiều so với hạn mức \$0.35 USD được phê duyệt).
- Tự động ghi kết quả so sánh S0 vs S1 vào tệp:
  `Reports/6-11Report/_PHASE1_G1_BOTTLENECK_20261105/GPU_REPLAY_128_S0_S1_RESULTS.json`.
