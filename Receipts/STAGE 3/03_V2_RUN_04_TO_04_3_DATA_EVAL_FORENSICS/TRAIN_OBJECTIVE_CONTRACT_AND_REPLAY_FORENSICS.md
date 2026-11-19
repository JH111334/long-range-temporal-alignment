# Kiểm Toán Pháp Y Mục Tiêu Huấn Luyện và Hiện Trạng Replay (Run 04.1)

## 1. Bản Chất Toán Học của Hàm Mục Tiêu Huấn Luyện (Loss Arithmetic Verification)

Khế ước đăng ký sơ bộ quy định hàm mất mát kết hợp đa nhiệm:
$$\mathcal{L}_{\text{joint}} = 0{,}5 \cdot \mathcal{L}_{\text{relation}} + 0{,}5 \cdot \mathcal{L}_{\text{mcqa}}$$

Trong mã nguồn thực thi thực tế (`scripts/kaggle_v2_run04_executor.py`, dòng 368–371):
```python
out = base_model(input_ids=inp_t, labels=lbl_t)
loss_i = out.loss / batch_size
loss_i.backward()
batch_loss += loss_i.item() * batch_size
```

### Phân tích đại số tương đương:
Tại mỗi bước tối ưu hóa, `batch_size = 2` bao gồm chính xác 2 mẫu tuần tự:
- Mẫu 1: Bài toán quan hệ thời gian (`task_type = "relation"`), tính ra giá trị $\mathcal{L}_{\text{relation}}$. Gradient tích lũy vào tham số là:
  $$g_1 = \nabla \left(\frac{\mathcal{L}_{\text{relation}}}{2}\right) = 0{,}5 \cdot \nabla \mathcal{L}_{\text{relation}}$$
- Mẫu 2: Bài toán trắc nghiệm 4 lựa chọn (`task_type = "mcqa"`), tính ra giá trị $\mathcal{L}_{\text{mcqa}}$. Gradient tích lũy tiếp tục vào tham số là:
  $$g_2 = \nabla \left(\frac{\mathcal{L}_{\text{mcqa}}}{2}\right) = 0{,}5 \cdot \nabla \mathcal{L}_{\text{mcqa}}$$

Tổng gradient bộ tối ưu hóa AdamW nhận được trước khi gọi `optimizer.step()` là:
$$g_{\text{total}} = g_1 + g_2 = 0{,}5 \cdot \nabla \mathcal{L}_{\text{relation}} + 0{,}5 \cdot \nabla \mathcal{L}_{\text{mcqa}} = \nabla \left(0{,}5 \mathcal{L}_{\text{relation}} + 0{,}5 \mathcal{L}_{\text{mcqa}}\right)$$

**Kết luận khoa học:** Phép cộng dồn gradient tuần tự với hệ số chia `batch_size = 2` tương đương đại số hoàn toàn với mục tiêu đa nhiệm có trọng số cân bằng $0{,}5 / 0{,}5$. Không có lỗi sai lệch hàm mục tiêu ở nhánh $R4\_B$ và $R4\_C$.

---

## 2. Pháp Y Hiện Trạng Nhánh Replay $R4\_D$ (Forensic Attribution of R4_D)

Mục tiêu thiết kế sơ bộ dự kiến nhánh $R4\_D$ huấn luyện với 3 thành phần:
$$\mathcal{L}_{R4\_D} = 0{,}4 \cdot \mathcal{L}_{\text{relation}} + 0{,}4 \cdot \mathcal{L}_{\text{mcqa}} + 0{,}2 \cdot \mathcal{L}_{\text{replay}}$$

Tuy nhiên, kiểm toán mã nguồn thực thi thực tế (`kaggle_v2_run04_executor.py`, dòng 331–336) phát hiện:
```python
elif arm == "R4_D_CLEAN_JOINT_REPLAY":
    # Multi-task + Replay (1 relation + 1 MCQA, with prompt format preservation)
    step_items = [
        ("relation", train_rel_pairs[rel_indices[step % len(train_rel_pairs)]]),
        ("mcqa", train_mcqa_clean[mcqa_indices[step % len(train_mcqa_clean)]]),
    ]
```

### Bằng chứng thực nghiệm:
1. Danh sách `step_items` của $R4\_D$ giống hệt $100\%$ danh sách `step_items` của $R4\_B$.
2. Không có tập dữ liệu replay nào được nạp vào bộ nhớ hoặc lấy mẫu.
3. Không có gradient replay nào được sinh ra.
4. Mã băm tensor của 508 trọng số khả huấn giữa $R4\_B$ và $R4\_D$ trùng khớp tuyệt đối (`max_abs_diff = 0.0`).
5. Độ dịch tham số so với $C_1$ giống nhau tuyệt đối ($\Delta W = 1{,}593727$).
6. Điểm số đánh giá trên cả 4 bảng điểm giống nhau tuyệt đối từng ca một.

**Phán quyết khoa học:** Nhánh $R4\_D$ được phân loại chuẩn tắc là **`REPLAY_NOT_TESTED`** (điều kiện replay chưa được cài đặt trong thực tế). Tuyệt đối không diễn giải kết quả này là "kỹ thuật replay thất bại trong việc cứu retention", mà là giả thuyết replay chưa từng được đưa vào kiểm chứng thực nghiệm.

---

## 3. Kiểm Tra Tính Bất Biến của Mô Hình Nền (Backbone Freeze & Gradient Isolation)

Trong suốt 30 bước huấn luyện trên cả 4 nhánh:
1. Tham số mô hình nền Qwen2.5-0.5B được khóa bằng `p.requires_grad = False`.
2. Kiểm tra assert dòng 375–378 xác nhận:
   $$\forall \theta_{\text{backbone}},\quad \|\nabla \theta_{\text{backbone}}\|_1 = 0{,}0$$
3. Gradient chỉ chảy độc quyền vào:
   - `lora_A` và `lora_B` trên 72 module MLP ($r=8, \alpha=16{,}0$).
   - Các ma trận cơ sở của lõi động $F(\theta, z)$ (`cos_basis`, `sin_basis`, `epoch_linear`).
   - Ma trận chiếu `GatedResidualProjection` ($40 \to 896$) và hệ số cổng $\alpha$.
