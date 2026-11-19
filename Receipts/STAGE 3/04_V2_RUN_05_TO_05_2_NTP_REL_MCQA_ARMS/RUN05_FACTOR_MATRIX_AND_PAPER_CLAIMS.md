# Ma Trận Phân Tích Yếu Tố Và Phân Định Tuyên Bố Bài Báo (Factor Matrix & Paper Claims)

**Tài liệu tham chiếu chính:** Xue Han et al., *Temporal Alignment of LLMs through Cycle Encoding for Long-Range Time Representations*, arXiv:2503.04150v3 (§3.3–3.5, §4.2, Eq. 1–10).

---

## 1. Phân Định Rạch Ròi Giữa Bài Báo Gốc Và Phương Pháp Đề Xuất Của Dự Án

| Khía Cạnh Khoa Học | Bài Báo Gốc (Han et al., arXiv:2503.04150v3) | Dự Án Cá Nhân (Learning Track Run 05) | Phân Loại Pháp Lý Nghiên Cứu |
|---|---|---|---|
| **Mô hình nền (Backbone)** | LLaMA2-7B/13B, Qwen2.5-3B/7B. **Không có 0.5B.** | `Qwen/Qwen2.5-0.5B-Instruct` (24 layers, $d=896$, đóng băng $100\%$). | `PROJECT_EXTRAPOLATION` (Khảo sát mô hình nhỏ). |
| **Không gian biểu diễn thời gian** | Polar-Cartesian multi-cycle (Year, Month, Day) 96D. 1-based index `(year-3)%60`. | `FoNE-CanChi 40D` (FoNE 32D + Can-Chi 8D), 0-based index `(year-4)%60`. | `PROJECT_INDEPENDENT_ENCODER` (Không gian nhúng riêng). |
| **Vị trí và cơ chế thích ứng (LoRA)** | LoRA trên các khối chú ý tự thân (Attention $W_q, W_v$, 48 modules). | `Dual-Cycle MaT-LoRA` trên các khối MLP ($W_{gate}, W_{up}, W_{down}$, 72 modules). | `PROJECT_INDEPENDENT_ADAPTER` (Can thiệp mạng nơ-ron truyền thẳng). |
| **Điểm tiêm vector thời gian** | Eq.(5): Cộng trực tiếp vào token embedding của token mốc năm ($x \in \mathcal{T}$). | Gated Seam tại token index 14 trong pha prefill với cổng học được (learnable gate). | `PROJECT_SEAM_ARCHITECTURE`. |
| **Hàm mất mát tổng thể** | Eq.(10): $\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{NTP}} + \sigma \mathcal{L}_{\text{temporal}}$ (kèm Fisher/EWC). | $\mathcal{L}_A = \mathcal{L}_{\text{NTP}} + \lambda_R \mathcal{L}_{\text{REL}}$<br>$\mathcal{L}_B = \mathcal{L}_{\text{NTP}} + \lambda_R \mathcal{L}_{\text{REL}} + \lambda_M \mathcal{L}_{\text{MCQA}}$ | `PAPER_INSPIRED_NTP_PLUS_TARGETED_OBJECTIVES`. |
| **NTP Causal Loss** | Causal Next-Token-Prediction trên kho ngữ liệu tổng quát. | Causal Next-Token-Prediction (shift 1 vị trí chuẩn tắc) trên văn bản thời gian sạch `train_nwp.jsonl`. | `OPERATIONAL_CAUSAL_NTP_LOSS`. |

---

## 2. Cam Kết Khoa Học Và Giới Hạn Tuyên Bố

1. **Không mạo nhận tái lập nguyên bản (No Exact Paper Claim):**
   - Dự án công bố rõ ràng: Cánh tay huấn luyện Run 05 sử dụng **"hàm mất mát NTP lấy cảm hứng từ bài báo kết hợp các mục tiêu quan hệ thời gian đặc thù của dự án"** (`paper-inspired NTP term + project-specific targeted objectives`).
   - Dự án không tuyên bố thực thi đầy đủ ma trận thông tin Fisher hay hàm phạt EWC phức tạp của Eq.(8)–(9) khi chưa có mã nguồn chính thức từ tác giả.
2. **Bảo toàn tính độc lập của 3 cánh tay:**
   - **Arm A (`R5_A_NTP_REL`):** Huấn luyện độc lập bằng $\mathcal{L}_{\text{NTP}} + \mathcal{L}_{\text{REL}}$, không chứa nhãn giám sát trắc nghiệm MCQA.
   - **Arm B (`R5_B_NTP_REL_MCQA`):** Huấn luyện độc lập bằng $\mathcal{L}_{\text{NTP}} + \mathcal{L}_{\text{REL}} + \mathcal{L}_{\text{MCQA}}$ với 400 câu MCQA sạch.
   - **Arm C (`R5_C_B_CONDITIONING_INTERVENTIONS`):** Hoàn toàn **không dùng optimizer**, thực hiện đo đạc can thiệp suy luận trên checkpoint đóng băng của Arm B để bóc tách tính phụ thuộc vào tọa độ thời gian và đường dẫn adapter.
