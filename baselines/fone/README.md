# Phân Vùng Baseline FoNE (Fourier Neural Embeddings — LOGGING 102)

Tài liệu này tổng hợp hồ sơ kỹ thuật, công thức toán học, liên kết artifact và kết quả thực nghiệm của cấu hình đối chứng **FoNE (Fourier Neural Embeddings)** do Nghiên cứu viên Khang (`JH111334`) triển khai trong chiến dịch thực nghiệm **LOGGING 102**.

---

## 1. Bản Chất Toán Học & Cơ Chế Can Thiệp

FoNE tiếp cận bài toán biểu diễn thời gian số học bằng cách ánh xạ đại lượng năm dương lịch $t = \text{canonical\_year} \in \mathbb{R}$ thành chuỗi sóng điều hòa đa tần số Fourier:

$$\phi_k(t) = \left[ \sin\left(\frac{2\pi f_k t}{T}\right), \cos\left(\frac{2\pi f_k t}{T}\right) \right], \quad k \in [1, K]$$

Vector đa tần số sau đó được đưa qua mạng nơ-ron truyền thẳng (MLP projection) để nâng lên không gian ẩn $d$ của mô hình ngôn ngữ lớn:
$$\mathbf{e}_{\text{FoNE}}(t) = \text{MLP}(\phi(t)) \in \mathbb{R}^d$$

Phương thức can thiệp:
- **Additive Input Conditioning**: Vector $\mathbf{e}_{\text{FoNE}}(t)$ được cộng trực tiếp vào token embedding đầu vào tại tầng nhúng của `Qwen/Qwen2.5-0.5B-Instruct`:
  $$\mathbf{E}'(w_i, t) = \mathbf{E}(w_i) + \mathbf{e}_{\text{FoNE}}(t)$$

---

## 2. Thiết Lập Thực Nghiệm & Nguyên Tắc Đơn Biến Can Thiệp

Chiến dịch LOGGING 102 thực thi theo quy chuẩn đơn biến can thiệp (*single intervention factorial design*):
- **Cấu hình F0 (Đối chứng - Control)**: Huấn luyện học liên tục (Continual Learning) qua 3 chặng thời gian (`1947` $\rightarrow$ `1984` $\rightarrow$ `2022`) với PEFT MLP-only LoRA ($829.440$ tham số), **hoàn toàn không tiêm vector FoNE**.
- **Cấu hình F1 (Can thiệp - Intervention)**: Giữ nguyên $100\%$ siêu tham số, dữ liệu và hạt nhân LoRA của F0, **chỉ kích hoạt thêm vector nhúng FoNE** tại tầng nhúng.
- **Phần cứng thực thi**: Kaggle GPU Tesla T4 (FP16 / FP32 hybrid).
- **Hạt giống ngẫu nhiên**: Độc lập qua 3 seeds (Seed 1, Seed 2, Seed 3).

---

## 3. Kết Quả Đo Đạc Thực Nghiệm & Thẩm Định Thống Kê (LOGGING 102)

### 3.1. Bảng Hiệu Năng Học Liên Tục (Continual Learning Benchmark)

| Cấu hình | Seed | ACC (Chính xác TB) | BWT (Truyền tri thức) | Forgetting 1947 | Forgetting 1984 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **F0 (Baseline không FoNE)** | Seed 1 | 37.05% | +0.38% | -5.48% | +4.72% |
| **F1 (FoNE Numeric Year)** | Seed 1 | 26.73% | +2.10% | +7.61% | -11.81% |
| *Chênh lệch ($\Delta_{\text{F1 - F0}}$)* | *Seed 1* | *-10.32 pp* | *+1.72 pp* | *+13.09 pp* | *-16.53 pp* |
| **F0 (Baseline không FoNE)** | Seed 2 | 36.61% | +7.70% | -11.11% | -4.28% |
| **F1 (FoNE Numeric Year)** | Seed 2 | 27.72% | -2.79% | -7.20% | +12.77% |
| *Chênh lệch ($\Delta_{\text{F1 - F0}}$)* | *Seed 2* | *-8.90 pp* | *-10.48 pp* | *+3.91 pp* | *+17.05 pp* |
| **F0 (Baseline không FoNE)** | Seed 3 | 30.63% | -5.50% | +10.08% | +0.92% |
| **F1 (FoNE Numeric Year)** | Seed 3 | 30.00% | -5.66% | +7.69% | +3.64% |
| *Chênh lệch ($\Delta_{\text{F1 - F0}}$)* | *Seed 3* | *-0.63 pp* | *-0.16 pp* | *-2.39 pp* | *+2.72 pp* |
| **F0 Tổng hợp (Mean ± Std)** | **3 Seeds** | **34.76% ± 3.59%** | **+0.86% ± 6.64%** | **-2.17% ± 10.98%** | **+0.45% ± 4.52%** |
| **F1 Tổng hợp (Mean ± Std)** | **3 Seeds** | **28.15% ± 1.67%** | **-2.12% ± 3.91%** | **+2.70% ± 8.57%** | **+1.53% ± 12.51%** |
| **Chênh Lệch Trung Bình** | **Mean $\Delta$** | **-6.61 pp** | **-2.98 pp** | **+4.87 pp** | **+1.08 pp** |

### 3.2. Kiểm Định Ghép Cặp McNemar Trên 180 Mẫu MCQA (`after_02`)

| Hạt nhân | $a$ (Cả 2 đúng) | $b$ (F0 đúng, F1 sai) | $c$ (F0 sai, F1 đúng) | $d$ (Cả 2 sai) | Thống kê $\chi^2_{\text{Edwards}}$ | Giá trị $p$ | Kết luận thống kê |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Seed 1** | 14 | 28 | 11 | 127 | 6.5641 | **0.0104** | F0 vượt trội có ý nghĩa ($p < 0.05$) |
| **Seed 2** | 15 | 25 | 9 | 131 | 6.6176 | **0.0101** | F0 vượt trội có ý nghĩa ($p < 0.05$) |
| **Seed 3** | 12 | 19 | 17 | 132 | 0.0278 | **0.8676** | Không có khác biệt ($p > 0.05$) |

### 3.3. Kết Luận Khoa Học
Giả thuyết $H_1$ ban đầu cho rằng *"FoNE input conditioning cải thiện biểu diễn trật tự thời gian và giảm hiện tượng quên lãng"* chính thức bị **BÁC BỎ (REFUTED)**:
- Tích hợp FoNE làm suy giảm trung bình **-6.61 điểm phần trăm** độ chính xác.
- Làm trầm trọng hóa hiện tượng quên lãng tri thức lịch sử cũ (**-2.98 pp** BWT).
- Căn nguyên: Tiêm vector nhân tạo tần số cao gây lệch phân phối tiềm ẩn (Latent Representation Drift), xung đột với cơ chế mã hóa vị trí RoPE vốn có của backbone Transformer.

---

## 4. Hồ Sơ Artifacts & Chứng Chỉ Xác Thực (Receipts)

Mọi tệp biên nhận, nhật ký thực thi và mã nguồn FoNE được bảo toàn nguyên vẹn tại các đường dẫn:
- **Báo cáo phân tích chuyên sâu**: `Reports/Teacher/LOGGING102_FoNE_Result.md`
- **Biên nhận thực thi Smoke Test**: `artifacts/runs/LOGGING102_FONE_SMOKE_RECEIPT.json`
- **Biên nhận điều phối Kaggle**: `artifacts/dispatch/fone_f1/FONE_KAGGLE_BUNDLE_RECEIPT.json`
- **Kiểm toán thống kê McNemar**: `artifacts/audits/MCNEMAR_F0_F1_ANALYSIS.json`
- **Kiểm toán tính đối xứng cặp dữ liệu**: `artifacts/audits/LOGGING102_DATA_PAIR_PARITY.json`
- **Mã nguồn triển khai kiến trúc FoNE**:
  * `SourceCode/long_range_representation/representation/fone_can_chi.py`
  * `SourceCode/ticktack_v2/models/fone_can_chi.py`
