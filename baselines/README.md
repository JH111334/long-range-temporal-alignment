# Thư Viện Mô Hình Đối Chứng Nghiên Cứu (Baselines Index)

Thư mục `baselines/` là phân vùng trung tâm lưu trữ toàn bộ các mã nguồn triển khai mô hình đối chứng (baselines), các ma trận kiểm soát căn nguyên (control matrices), mã nguồn sinh Kaggle notebook, động cơ đánh giá khử thiên vị (debiasing evaluation engines), và các tệp dữ liệu kết quả thực nghiệm chuẩn tắc (`.json`) phục vụ đề tài nghiên cứu **Long-Range Temporal Alignment & Can Chi Cycle Representation**.

---

## 1. Sơ Đồ Ánh Xạ Cấu Trúc Thư Mục (Directory Mapping)

```
baselines/
├── __init__.py                         # Khởi tạo package Python baselines
├── README.md                           # Bản chỉ mục toàn diện (tài liệu này)
│
├── ticktack/                           # Nhánh baseline Ticktack (Han et al., 2025)
│   ├── __init__.py                     # Package init & re-export các hàm chuẩn
│   ├── models.py                       # Mã hóa thời gian tuần hoàn (Eq 1-4), scale modes ('raw', 'scaled', 'gated'), LoRA
│   ├── losses.py                       # Hàm mất mát nội chu kỳ Can Chi (Eq 8), loss kết hợp (Eq 7)
│   ├── date_mask.py                    # Trích xuất date-token mask & cơ chế tiêm cục bộ (Localized Injection)
│   ├── eval_debiasing.py               # Động cơ Choice-Logit & 4-fold Permutation Rotation (Khử thiên vị chữ cái)
│   ├── build_kaggle_notebook.py        # Mã nguồn sinh Kaggle notebook độc lập cho Tesla T4 (0.5B)
│   ├── build_kaggle_notebook_3b_full.py # Mã nguồn sinh Kaggle notebook độc lập cho Tesla T4 (3B + TempUN Full)
│   ├── notebooks/                      # Sổ tay Jupyter thực nghiệm hoàn chỉnh
│   │   ├── ticktack_paper_rebuild.ipynb
│   │   └── ticktack_3b_full.ipynb
│   └── outputs/                        # Kết quả benchmark thực nghiệm chính thức từ Kaggle T4
│       ├── TICKTACK_CONTROL_MATRIX_RESULTS.json   # Kết quả ma trận đối chứng C1-C4 (N=540)
│       ├── TICKTACK_REBUILD_BENCHMARK_RESULTS.json # Kết quả tái lập Bảng 1 gốc
│       └── ticktack-paper-rebuild-baseline.log    # Toàn bộ log thực thi phần cứng Kaggle
│
├── fone/                               # Nhánh baseline Fourier Neural Embeddings (LOGGING 102)
│   └── README.md                       # Hồ sơ chỉ mục, công thức, thực nghiệm và bằng chứng bác bỏ H1
│
└── discrete/                           # Nhánh baseline đối chứng rời rạc (Upcoming Control)
    └── README.md                       # Thiết kế kỹ thuật nhúng 60 Can Chi rời rạc & Text Prompting
```

---

## 2. Bảng Tổng Hợp 8 Cấu Hình Mô Hình Đối Chứng (Baselines Inventory)

| STT | Tên Cấu Hình Đối Chứng | Phân Nhánh | Mô Tả & Phương Thức Can Thiệp | Hàm Mất Mát | Hiện Trạng Thực Nghiệm / Kết Quả |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Base (Zero-Shot)** | `ticktack/` | `Qwen/Qwen2.5-0.5B-Instruct` đông băng, đánh giá trực tiếp trên prompt nguyên bản qua Choice-Logit. | Không huấn luyện | **Canonical ACC**: $26.67\%$<br>**Debiased Mean ACC**: $25.56\%$<br>**Consistency**: $3.89\%$<br>*Thiên vị chữ cái `'A'`: $65.0\%$* |
| **2** | **PT (NTP-only LoRA)** | `ticktack/` | Tinh chỉnh LoRA ($r=8, \alpha=16$ trên $q, v$) với bài toán dự đoán token tiếp theo, không có vector thời gian ($\mathrm{TE} = 0, \sigma = 0$). | $\mathcal{L}_{\text{NTP}}$ | **Canonical ACC**: $29.07\% \pm 2.74\%$<br>**Debiased Mean ACC**: $26.85\% \pm 1.48\%$<br>**Consistency**: $3.89\% \pm 0.32\%$ |
| **3** | **Ticktack Raw (C1)** | `ticktack/` | Mã hóa thời gian xoắn ốc cực $\mathrm{TE}_{\text{raw}}$ cộng broadcast vào mọi token; $\sigma = 0.5$. (Công thức chuẩn của bài báo gốc). | $\mathcal{L}_{\text{NTP}} + 0.5 \mathcal{L}_{\text{intra}}$ | **Canonical ACC**: $22.41\% \pm 1.16\%$<br>**Debiased Mean ACC**: $26.85\% \pm 1.16\%$<br>**Consistency**: $5.19\% \pm 1.16\%$<br>*Sụp đổ biểu diễn, 100% dự đoán `'A'`* |
| **4** | **Ticktack Scaled** | `ticktack/` | Mã hóa thời gian chuẩn hóa biên độ năng lượng: $\mathrm{TE}_{\text{scaled}} = \frac{1}{d} \mathrm{TE}_{\text{raw}}$ hoặc qua hệ số gating $\gamma = 0.01$. | $\mathcal{L}_{\text{NTP}} + 0.5 \mathcal{L}_{\text{intra}}$ | Triệt tiêu hiện tượng sụp đổ cosine (giảm $\cos$ từ $>0.999$ về $<0.15$), bảo tồn phương sai ngữ nghĩa của token. |
| **5** | **Ticktack Date-Token (C2)** | `ticktack/` | Mã hóa thời gian thô $\mathrm{TE}_{\text{raw}}$ chỉ cộng cục bộ tại các vị trí token đại diện cho năm ($y$); các token khác giữ nguyên; $\sigma = 0.5$. | $\mathcal{L}_{\text{NTP}} + 0.5 \mathcal{L}_{\text{intra}}$ | **Canonical ACC**: $29.07\% \pm 1.79\%$<br>**Debiased Mean ACC**: $26.44\% \pm 1.29\%$<br>**Consistency**: $3.70\% \pm 0.32\%$<br>*Ngăn ngừa RMSNorm triệt tiêu token ngữ nghĩa* |
| **6** | **FoNE (Continuous Fourier)** | `fone/` | Vector nhúng chuỗi Fourier điều hòa cộng tính (`additive input conditioning`) trên đại lượng số học `canonical_year`. | $\mathcal{L}_{\text{continual}}$ | **ACC**: $28.15\% \pm 1.67\%$ (thua kém đối chứng F0 $34.76\%$, $\Delta = -6.61$ pp)<br>**BWT**: $-2.12\%$<br>*Bác bỏ giả thuyết $H_1$ (LOGGING 102)* |
| **7** | **Learned Discrete Year / Can Chi** | `discrete/` | Bảng tra cứu nhúng rời rạc `nn.Embedding(60, d)` trực tiếp theo chỉ số chu kỳ $\tau(t) = (t - 3) \bmod 60$, tối ưu hóa tự do bằng gradient. | $\mathcal{L}_{\text{NTP}} + \sigma \mathcal{L}_{\text{intra}}$ | Đối chứng câu hỏi: Liệu phép chiếu hình học liên tục có cần thiết hay chỉ cần nhúng rời rạc chu kỳ Can Chi? |
| **8** | **Text Tag Prompting** | `discrete/` | Tiêm tiền tố nhãn Can Chi tự nhiên bằng văn bản (ví dụ: `[Năm Giáp Tý]`) vào prompt đầu vào, không can thiệp kiến trúc mô hình. | Không hoặc SFT chuẩn | Đối chứng đánh giá năng lực suy luận niên đại thuần túy từ tri thức tiền huấn luyện (In-context Reasoning). |

---

## 3. Công Thức Toán Học Chuẩn Mực

### 3.1. Chu kỳ Can Chi 60 năm (Han et al., 2025, Eq. 1)
Ánh xạ năm dương lịch $t \in \mathbb{Z}$ sang chỉ số chu kỳ Can Chi $\tau(t) \in [0, 59]$:
$$\tau(t) = (t - 3) \bmod 60$$
*(Chỉ số $0$ ứng với Giáp Tý; Năm 4 SCN là chỉ số 1; Năm 1984 là chỉ số 1; Năm 3 SCN là chỉ số 0).*

### 3.2. Tọa độ cực Archimedes & Chiếu Descartes (Eq. 2–3)
$$\theta = \tau(t) \times \frac{6\pi}{180}, \quad r = \alpha + \beta \theta \quad (\alpha = 0.5, \beta = 0.5)$$
$$x = r \cos \theta, \quad y = r \sin \theta$$

### 3.3. Sinusoidal Temporal Encoding (Eq. 4)
Với số chiều ẩn $d$:
$$\omega_k = \frac{1}{10000^{2k/d}}, \quad k \in \left[0, \frac{d}{2}-1\right]$$
$$\text{Embed}(v) = \left[ \sin(v\omega_0), \cos(v\omega_0), \dots, \sin(v\omega_{d/2-1}), \cos(v\omega_{d/2-1}) \right]$$
$$\mathrm{TE}(t) = \text{Embed}(x) + \text{Embed}(y) \in \mathbb{R}^d$$

### 3.4. Các Chế Độ Chuẩn Hóa Biên Độ (Scale Normalization Modes)
1. **Raw Mode**: $\mathrm{TE}_{\text{raw}}(t) = \mathrm{TE}(t) \quad (\|\mathrm{TE}_{\text{raw}}\|_2 \approx 41.2)$. Do $\|E(w)\|_2 \approx 0.45$, tỉ lệ chênh lệch $\approx 93\times$ dẫn đến sụp đổ cosine $>0.999$.
2. **Scaled Mode**: $\mathrm{TE}_{\text{scaled}}(t) = \frac{1}{d} \mathrm{TE}_{\text{raw}}(t) \quad (\|\mathrm{TE}_{\text{scaled}}\|_2 \approx 0.046)$, giữ tỉ trọng năng lượng hài hòa với token embedding.
3. **Gated Mode**: $\mathrm{TE}_{\text{gated}}(t) = \gamma \cdot \mathrm{TE}_{\text{raw}}(t)$ với $\gamma \in (0, 1]$.

### 3.5. Hàm Mất Mát Nội Chu Kỳ Can Chi (Eq. 8)
$$\mathcal{L}_{\text{intra}} = 1 - \frac{1}{|\mathcal{P}|} \sum_{(i, j) \in \mathcal{P}} \cos(\mathbf{h}_i, \mathbf{h}_j)$$
Trong đó tập cặp dương $\mathcal{P} = \{(i, j) \mid \tau(t_i) = \tau(t_j), i \neq j\}$. Hai năm cách nhau bội số 60 năm (ví dụ $1924$ và $1984$) thuộc cùng chu kỳ ($\tau = 1$) tạo thành cặp dương.

Hàm mất mát tổng quát:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NTP}} + \sigma \mathcal{L}_{\text{intra}} \quad (\sigma = 0.5)$$

---

## 4. Giao Thức Đánh Giá Khử Thiên Vị & Kiểm Định Thống Kê

1. **Choice Logit Scoring**: Đo trực tiếp logit của các token lựa chọn $\{a, b, c, d\}$ và $\{yes, no\}$, đạt tỷ lệ lỗi bất hợp lệ (**Invalid Rate = 0.0%**).
2. **4-Fold Option Cyclic Permutation Rotation**:
   - Rot 0: `ABCD` (trật tự chuẩn)
   - Rot 1: `BCDA`
   - Rot 2: `CDAB`
   - Rot 3: `DABC`
3. **Chỉ số đánh giá**:
   - **Canonical Accuracy**: Điểm chính xác trên thứ tự gốc (bị ảnh hưởng bởi thiên vị chữ cái).
   - **Debiased Mean Accuracy**: Trung bình điểm chính xác qua 4 phép xoay vòng.
   - **Consistency (Strict Accuracy)**: Tỷ lệ mẫu mà mô hình trả lời đúng ở **toàn bộ cả 4 phép xoay** (độ vững chắc thực chất).
4. **Kiểm toán thống kê ghép cặp McNemar**:
   - Thống kê Edwards có hiệu chỉnh tính liên tục: $\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$ với bậc tự do $df = 1$.
   - Tính toán trên từng seed ($N=180$) và kiểm định gộp (Pooled $N=540$).

---

## 5. Hướng Dẫn Tái Lập (Reproduction Commands)

### 5.1. Chạy Bộ Kiểm Thử Tự Động Toàn Diện
```powershell
pytest SourceCode/tests/ -v
```

### 5.2. Sinh Kaggle Notebook Thực Nghiệm Tái Lập Độc Lập
```powershell
python scripts/build_kaggle_pure_ticktack_baseline.py
# Hoặc trực tiếp qua module package:
python -m baselines.ticktack.build_kaggle_notebook
```

### 5.3. Sử Dụng Các Module Trong Mã Nguồn Nghiên Cứu
```python
import torch
from baselines.ticktack import (
    compute_canchi_cycle,
    compute_batch_te,
    compute_intra_loss,
    find_date_token_mask,
    inject_te_embeddings,
    generate_option_permutations,
)

# 1. Tính chu kỳ Can Chi (1984 -> Giáp Tý = 1)
cycle_1984 = compute_canchi_cycle(1984)
assert cycle_1984 == 1

# 2. Sinh vector mã hóa thời gian chuẩn hóa
years = torch.tensor([1984, 1924, 2022])
te_vecs = compute_batch_te(years, hidden_dim=896, scale_mode='scaled')
print("TE shape:", te_vecs.shape)  # [3, 1, 896]

# 3. Tạo 4 hoán vị khử thiên vị
perms = generate_option_permutations("What occurred in 1984? Options: (a) X (b) Y (c) Z (d) W", "b")
print("Number of rotations:", len(perms))  # 4
```
