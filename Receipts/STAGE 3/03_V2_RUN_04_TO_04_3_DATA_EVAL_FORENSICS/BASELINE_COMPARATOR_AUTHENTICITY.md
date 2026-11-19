# Tính Xác Thực Của Mô Hình Đối Chứng Và Độ Chuẩn Xác Bài Báo (Baseline Comparator Authenticity & Paper Fidelity)

**Phạm vi:** Phân định rạch ròi bản chất kỹ thuật, vai trò và giới hạn của các biến thể mô hình trong nghiên cứu.

---

## 1. Danh Mục Các Biến Thể Và Nhãn Gọi Chuẩn Tắc

| Mã Biến Thể | Vai Trò | Kiến Trúc & Cơ Chế | Dữ Liệu & Huấn Luyện | Cách Gọi Chuẩn Tắc (Canonical Label) |
|---|---|---|---|---|
| **`B0_RAW_QWEN05B`** | Baseline Cơ sở | `Qwen/Qwen2.5-0.5B-Instruct`, không có adapter, không có vector thời gian | Pretrained weights nguyên bản | **`base (inference only)`** |
| **`B1_PARTNER_PT_JOINT_NTP_QV`** | Đối chứng PT của đối tác | Frozen backbone + Trainable LoRA trên $W_q, W_v$ (48 modules) | Huấn luyện Next Token Prediction (NTP) đồng thời (Joint) trên dữ liệu PT | **`partner PT baseline`** |
| **`B2_PARTNER_TT_RAW_BROADCAST`** | Đối chứng TickTack của đối tác (Gốc) | Frozen backbone + $W_q, W_v$ LoRA + TE broadcast toàn bộ token | Huấn luyện Joint NTP + Temporal Contrastive Loss | **`partner implementation (unscaled broadcast)`** |
| **`B3_PARTNER_TT_SCALED_BROADCAST`** | Đối chứng TickTack của đối tác (Chuẩn hóa) | Như B2 nhưng vector TE được nhân tỷ lệ $1/d = 1/896$ để tránh Cosine Collapse | Huấn luyện Joint NTP + Temporal Contrastive Loss | **`partner scaled control`** |
| **`B4_YEAR_TOKEN_ALIGNED`** | Biến thể diễn giải bài báo | Chỉ tiêm TE vào token năm theo Eq.(5) | Huấn luyện trên clone độc lập | **`experimental interpretation of paper`** |
| **`M0_FC40_V1`** | Phương pháp chính của dự án (Learning Track) | Frozen backbone + 72 modules MLP MaT-LoRA + Gated Seam token 14 + FoNE32–CanChi8 40D | Huấn luyện học tăng tiến (Continual Learning stages) | **`our primary method (Learning Track)`** |

---

## 2. Ranh Giới Nghiên Cứu Và Bảo Toàn Phương Pháp Hai Bên

1. **Tính độc lập của hai hướng tiếp cận:**
   - Hướng của đối tác tập trung vào việc tái hiện cơ chế dạng bài báo TickTack với attention LoRA và huấn luyện đồng thời (Joint NTP).
   - Hướng của dự án (Learning Track) tập trung vào học tăng tiến thời gian dài hạn (Long-range continual temporal alignment) bằng `FoNE-CanChi 40D` và `Dual-Cycle MaT-LoRA` trên các khối MLP.
   - Hai hướng hoàn toàn độc lập, không thay thế lẫn nhau và không sao chép mã nguồn của nhau.
2. **Khác biệt về số lượng module LoRA:**
   - LoRA của đối tác tác động lên $W_q, W_v$ (Attention) gồm **48 modules** trên Qwen-0.5B ($24 \text{ layers} \times 2 = 48$).
   - MaT-LoRA của dự án tác động lên $W_{gate}, W_{up}, W_{down}$ (MLP) gồm **72 modules** trên Qwen-0.5B ($24 \text{ layers} \times 3 = 72$).
   - Việc so sánh điểm số giữa hai hướng là **so sánh toàn diện phương pháp (whole-method comparison)**, bao gồm cả kiến trúc module, không gian biểu diễn thời gian và cơ chế huấn luyện.
3. **Giới hạn số liệu của bài báo gốc (arXiv:2503.04150v3):**
   - Bài báo chỉ công bố số liệu cho **LLaMA2-7B/13B** và **Qwen2.5-3B/7B**.
   - Bài báo hoàn toàn **không có số liệu cho Qwen2.5-0.5B**. Mọi số liệu đo trên 0.5B đều là kết quả thực nghiệm nội bộ, tuyệt đối không được gán mác `paper_reported`.
