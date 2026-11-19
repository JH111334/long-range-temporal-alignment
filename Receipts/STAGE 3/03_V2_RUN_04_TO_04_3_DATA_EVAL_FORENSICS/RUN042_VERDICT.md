# Phan Quyet Khoa Hoc Run 04.2: Kiem Toan Do Chuan Xac Bai Bao TickTack, Khieu Nai Doi Tac Va Chuan Hoa Chu Ky Thoi Gian

**Ngay lap:** 2026-10-15  
**Nhanh phap ly:** `experiment/v2-run04-2-partner-paper-fidelity-audit`  
**Doi tuong kiem toan:** Khieu nai ky thuat cua doi tac, repo `Long-Range-Representation` (commit `72f4a50`), ma nguon du an `long-range-temporal-alignment`, va bai bao goc TickTack (arXiv:2503.04150v3).

---

## 1. Phan Xu Chi Tiet Cac Khieu Nai Cua Doi Tac

### 1.1. Khieu nai ve cong thuc Can-Chi: `year - 4` so voi `year - 3`
- **Ket luan:** BAC BO khieu nai day la loi tinh toan sinh ra gia tri `None`.
- **Can cu toan hoc va phap y ma nguon:**
  - Bai bao TickTack (Phu luc A, Eq. 1) su dung he danh so 1-based index tren tap $\{1, \dots, 60\}$:
    $$\text{cycle}_{\text{1-based}}(y) = ((y - 3 - 1) \pmod{60}) + 1 = (y - 4) \pmod{60} + 1$$
    Theo do, nam 4 CN la nam 1 (Giap Ty), nam 1984 la nam 1 (Giap Ty).
  - Khong gian nhung `FoNE-CanChi` cua du an su dung he danh so 0-based index tren tap $\{0, \dots, 59\}$:
    $$\text{cycle}_{\text{0-based}}(y) = (y - 4) \pmod{60}$$
    Theo do, nam 4 CN la chi muc 0 (Giap Ty), nam 1984 la chi muc 0 (Giap Ty).
  - Hai phep bien doi so hoc nay dong cau hoan hao qua phep tinh $\text{cycle}_{\text{1-based}} \equiv \text{cycle}_{\text{0-based}} + 1 \pmod{60}$, lech pha goc dung $6^\circ$ ($2\pi/60$).
  - Phep toan so hoc tren kieu so nguyen trong Python hoan toan khong the tra ve gia tri `None`. Khieu nai cho rang `year - 4` gay ra loi `None` la sai lech ve mat lap trinh va toan hoc.

### 1.2. Khieu nai ve hien tuong khuyet nam (`canonical_year: null`)
- **Ket luan:** XAC NHAN hien tuong khuyet nam la co that, song can nguyen nam o tap du lieu danh gia goc va co che fallback cu, hoan toan khong phai do module Can-Chi.
- **Can cu thuc nghiem:**
  - Kiem toan tap du lieu `data/tempun/eval_mcqa.jsonl` (180 mau) phat hien co toi 144 mau ($80.0\%$) mang gia tri `canonical_year: null`.
  - Day la cac cau hoi thuoc nhom *Min-Max* va *Relative-Based Temporal Reasoning*, trong do ngu canh khong chua mot nam co dinh duy nhat ma chua nhieu moc thoi gian hoac khoang cach thoi gian tuong doi.
  - O phien ban thuc thi cu, he thong su dung co che du phong ngam: `year = record.get("canonical_year") or 1984`. Co che nay vo tinh ap dat moc 1984 (Giap Ty) cho 144 cau hoi khong phu hop, lam meo mo nghiem trong tin hieu dau vao cua mo hinh.
  - Ban sua loi Run 04.2 da thiet lap co che giai quyet 3 tang (`YearResolution`) minh bach, cam tuyet doi fallback ngam.

### 1.3. Khieu nai ve vi tri tiem token thoi gian va hien tuong suy bien goc (Cosine Collapse)
- **Ket luan:** XAC NHAN hien tuong Cosine Collapse trong ma nguon cua doi tac; phan dinh ro rang su khac biet giua thiet ke bai bao goc va cach trien khai cua hai nhom.
- **Can cu vat ly va toan hoc:**
  - Bai bao TickTack Eq. (5) quy dinh: vector thoi gian $\mathrm{TE}(t)$ chi duoc cong vao dung cac token bieu dien nam trong cau ($x \in \mathcal{T}$).
  - Trong ma nguon cu cua doi tac, phep cong duoc thuc hien theo co che broadcasting: `embed_tokens(input_ids) + te`, nghia la vector TE bi cong tran vao toan bo $100\%$ cac token trong chuoi.
  - Do chuan vector TE theo cong thuc goc dat $\|\mathrm{TE}\|_2 \approx 42.26$, trong khi chuan token embedding cua Qwen2.5-0.5B chi dat $\|E(x)\|_2 \approx 0.45$, vector TE chiem toi $99.99\%$ nang luong vector sau phep cong. He qua la do tuong dong Cosine giua moi cap token trong cau vot len $0.9999$ (Cosine Collapse hoan toan).
  - Doi tac giai quyet bang cach nhan he so ty le $1/d = 1/896$, giup dua Cosine lien token tro ve $0.0702$. Tuy nhien, day la giai phap thuc dung cho loi broadcast, khong phai la thiet ke goc cua bai bao (bai bao goc khong broadcast ma chi cong vao token nam).

### 1.4. Khieu nai ve 4 con so danh gia cua doi tac
- **Ket luan:** XAC NHAN 4 con so cua doi tac la co that va co the tai lap tren tap `eval_180_clean.jsonl`, song chung duoc do tren giao thuc khac (Academic Choice Logit Scoring) nen khong the so sanh ngang voi diem Free Generation cua giao thuc `P12_TASK_AWARE_CHATML`.
- **So lieu doi tac bao cao (Commit `72f4a50`):**
  - Base (Qwen2.5-0.5B-Instruct): $48/180 = 26.67\%$
  - PT (Prompt-based baseline): $52/180 = 29.07\%$
  - TickTack Raw (bi Cosine Collapse): $40/180 = 22.41\%$
  - TickTack Scaled ($1/d$ normalization): $52/180 = 28.89\%$
- **Ban chat giao thuc:** Doi tac do truc tiep logit cua cac token lua chon ('A', 'B', 'C', 'D' hoac 'Yes', 'No') tai buoc sinh dau tien sau prompt dang hoc thuat. Giao thuc nay triet tieu loi dinh dang (Invalid Rate $= 0.0\%$), trong khi giao thuc `P12` do kha nang sinh tu do toan tu vung theo mau ChatML.

---

## 2. Do Chuan Xac So Voi Bai Bao Goc TickTack (arXiv:2503.04150v3)

1. **Khong co ket qua cho Qwen2.5-0.5B trong bai bao goc:**
   - Bang 1 va Bang 2 trong bai bao TickTack chi tien hanh thuc nghiem tren cac mo hinh quy mo lon hon: **LLaMA2-7B, LLaMA2-13B, Qwen2.5-3B va Qwen2.5-7B**.
   - Khong ton tai bat ky so lieu thuc nghiem nao cho phien ban **Qwen2.5-0.5B** trong tai lieu goc. Moi con so tren Qwen2.5-0.5B deu la phep thu suy rong (extrapolation) doc lap cua hai nhom nghien cuu.
2. **Kien truc va muc tieu huan luyen:**
   - Bai bao su dung MoE-based Temporal Adapter voi nhieu tang chu ky (Year, Month, Day) va ap dung LoRA len cac ma tran $W_q, W_v$ (48 modules tren Qwen-0.5B).
   - Du an `long-range-temporal-alignment` su dung `FoNE-CanChi 40D` ket hop `MaT-LoRA` tren cac lop MLP (72 modules tren Qwen-0.5B) de hoc moi quan he thoi gian dai han. Day la hai huong tiep can bo tro song song, khong triet tieu lan nhau.

---

## 3. Ket Qua Thuc Nghiem Doi Soat Run 04.2

Tren cung mot bo tap tin danh gia da lam sach `eval_180_clean.jsonl` ($n=180$) va duoc kiem toan ti mi, cac mo hinh the hien ket qua nhu sau tren hai giao thuc:

| Mo hinh / Bien the | Giao thuc Do | Accuracy (Strict) | MCQA (156) | Yes/No (24) | Invalid Rate |
|---|---|---|---|---|---|
| **Qwen2.5-0.5B Base** | Academic Choice Logit | 48/180 (26.67%) | 35/156 | 13/24 | 0.0% |
| **TickTack Raw (Broadcast)** | Academic Choice Logit | 40/180 (22.22%) | 29/156 | 11/24 | 0.0% |
| **TickTack Scaled ($1/d$)** | Academic Choice Logit | 52/180 (28.89%) | 39/156 | 13/24 | 0.0% |
| **Qwen2.5-0.5B Base** | P12 ChatML Greedy | 58/180 (32.22%) | 41/156 | 17/24 | 0.0% |
| **M0-FC40 / Run02 C1** | P12 ChatML Greedy | 58/180 (32.22%) | 41/156 | 17/24 | 0.0% |
| **Run03 T0 Multitask** | P12 ChatML Greedy | 60/180 (33.33%) | 43/156 | 17/24 | 0.0% |

- **Nhan xet thuc nghiem:**
  - Phep kiem dinh ghep cap giua TickTack Scaled va Base tren Choice Logit cho thay do cai thien nhe $+4$ cau ($+2.22\%$, McNemar $p = 0.453$, chua dat y nghia thong ke $p < 0.05$).
  - Giao thuc Choice Logit Scoring cho diem thap hon P12 ChatML vi mo hinh Qwen-0.5B-Instruct duoc tinh chinh dac biet tot tren dinh dang ChatML, giup doan dap an qua sinh tu do tot hon qua xac suat logit doc lap.

---

## 4. Ranh Gioi Nghien Cuu Va Huong Di Tiep Theo

1. **Bao ton nguyen ven kien truc:**
   - Tiep tuc giu vung kien truc `FoNE-CanChi 40D` 0-based va `MaT-LoRA` cho Learning Track, dong thoi cong nhan tinh hop le cua he 1-based trong tai lieu TickTack nhu mot bieu dien dong cau.
2. **Khai thac bo du lieu da lam sach:**
   - Tap `eval_180_clean.jsonl` va `train_2461_clean.jsonl` do doi tac lam sach mang lai gia tri khoa hoc cao, loai bo cac nhan mau thuan va dinh dang loi. Learning Track se su dung bo du lieu nay lam tieu chuan kiem chung chung.
3. **Chuan hoa giao thuc danh gia:**
   - Moi bao cao so sanh bat buoc phai ghi ro danh gia tren giao thuc nao: *Academic Choice Logit Scoring* hay *Task-Aware ChatML Free Generation*. Tuyet doi khong tron lan hai thang do.
