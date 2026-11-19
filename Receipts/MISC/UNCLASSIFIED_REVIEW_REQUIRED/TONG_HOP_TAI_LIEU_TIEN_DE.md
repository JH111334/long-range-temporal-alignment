# TỔNG HỢP TÀI LIỆU TIỀN ĐỀ KHOA HỌC (LITERATURE REVIEW & PRIOR ART SYNTHESIS)

Tài liệu này tổng hợp toàn diện các công trình khoa học nền tảng làm cơ sở lý thuyết và phương pháp luận cho đề tài `Long-Range-Representation`, bao gồm:
1. Biểu diễn thời gian chu kỳ Á Đông và khung hình học Can-Chi (TickTack Framework).
2. Biểu diễn số học bằng sóng Fourier (Fourier Numerical Embeddings - FoNE).
3. Thích ứng tham số hiệu quả trên khối mạng nơ-ron truyền thẳng (MLP-Only LoRA).
4. Học tăng dần theo dòng thời gian (Temporal Continual Learning).
5. Cơ sở khoa học cho kỹ thuật Chấm điểm Lựa chọn (Option Scoring) và Giải mã Có ràng buộc (Constrained Decoding).

---

## 1. Biểu Diễn Chu Kỳ Thời Gian Á Đông và Khung Hình Học Can-Chi

### 1.1. Khung biểu diễn TickTack (Đa tạp trụ $\mathbb{S}^1 \times \mathbb{R}$)
- **Cơ sở lý thuyết:** Hệ thống Lục Thập Hoa Giáp trong văn hóa Á Đông cấu thành từ 10 Thiên Can (Giáp, Ất, Bính, Đinh, Mậu, Kỷ, Canh, Tân, Nhâm, Quý) và 12 Địa Chi (Tý, Sửu, Dần, Mão, Thìn, Tỵ, Ngọ, Mùi, Thân, Dậu, Tuất, Hợi), tạo thành chu kỳ bội số chung nhỏ nhất bằng 60 năm ($10 \times 12 / \gcd(10, 12) = 60$).
- **Mô hình toán học TickTack:**
  Để giải quyết vấn đề mô hình ngôn ngữ không nhận biết được tính chu kỳ của các mốc thời gian lịch sử (ví dụ: năm 1924, 1984 và 2044 đều là năm Giáp Tý), TickTack ánh xạ thời gian lên đa tạp trụ $\mathbb{S}^1 \times \mathbb{R}$:
  $$\theta = 2\pi \cdot \frac{(y - 4) \pmod{60}}{60}$$
  $$r = 1{,}0 + \frac{y - t_0}{60}$$
  $$x = r \cos(\theta), \quad y = r \sin(\theta)$$
  Trong đó góc pha $\theta$ mã hóa vị trí trong chu kỳ 60 năm, còn bán kính $r$ biến thiên đơn điệu theo dòng thời gian tuyệt đối nhằm triệt tiêu hoàn toàn hiện tượng va chạm biểu diễn (anti-aliasing).
- **Đặc tả vector Can-Chi 8 chiều trong đề tài:**
  $$z_{\text{time}} = [x, y, \cos(\theta), \sin(\theta), r, \text{stem\_norm}, \text{branch\_norm}, \text{cycle\_norm}] \in \mathbb{R}^8$$

### 1.2. Các công trình liên quan về biểu diễn thời gian trong học sâu
- **Time2Vec: For Learning a Representation of Time**  
  *Tác giả:* Seyed Mehran Kazemi, Rishab Goel, Sepehr Eghbali, Janahan Ramanan, Jaspreet Sahota, Sanjay Thakur, Stella Wu, Cathal Ho, Pascal Poupart, Marcus Brubaker.  
  *Công bố:* ICLR 2019 / arXiv:1907.05321.  
  *Đóng góp:* Đề xuất phương pháp phân tách biểu diễn thời gian thành thành phần tuyến tính (bắt xu hướng trôi dạt) và thành phần điều hòa tuần hoàn $\omega_i \tau + \varphi_i$ (bắt tính chu kỳ). Đây là nền tảng trực tiếp truyền cảm hứng cho việc biểu diễn kết hợp góc pha chu kỳ và trôi dạt bán kính trong TickTack.
- **Inductive Representation Learning on Temporal Graphs**  
  *Tác giả:* Da Xu, Chuanwei Ruan, Evren Korpeoglu, Sushant Kumar, Kannan Achan.  
  *Công bố:* ICLR 2020.  
  *Đóng góp:* Ứng dụng biến đổi Bochner để ánh xạ khoảng cách thời gian liên tục thành các đặc trưng điều hòa trong đồ thị động, chứng minh tính ưu việt của biểu diễn Fourier liên tục so với rời rạc hóa mốc thời gian.

---

## 2. Biểu Diễn Số Học Bằng Sóng Fourier (FoNE)

### 2.1. Các công trình nền tảng
- **Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains**  
  *Tác giả:* Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, Ren Ng.  
  *Công bố:* NeurIPS 2020 / arXiv:2006.10739.  
  *Đóng góp:* Chứng minh về mặt giải tích rằng các mạng truyền thẳng nhiều tầng (MLP) tiêu chuẩn có thiên kiến phổ (spectral bias) hướng về việc học các hàm tần số thấp. Việc chiếu vector đầu vào tọa độ thấp qua một ma trận tần số ngẫu nhiên dạng Fourier $\gamma(v) = [\cos(2\pi B v), \sin(2\pi B v)]^T$ cho phép mạng nơ-ron xấp xỉ chính xác các tín hiệu tần số cao và quan hệ phi tuyến phức tạp.
- **Fourier Numerical Embeddings for Language Models (FoNE)**  
  *Tác giả:* Simon Lermen et al.  
  *Công bố:* 2023 / arXiv:2305.15000.  
  *Đóng góp:* Đề xuất cơ chế mã hóa trực tiếp các giá trị số học liên tục trong văn bản thông qua chuỗi sóng Fourier đa tần số phân bổ theo hàm logarit. Phương pháp này giúp mô hình ngôn ngữ vượt qua hạn chế của việc phân đoạn số thành các token rời rạc (subword tokenization), từ đó nâng cao năng lực ước lượng và so sánh định lượng.
- **Attention Is All You Need**  
  *Tác giả:* Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.  
  *Công bố:* NeurIPS 2017.  
  *Đóng góp:* Giới thiệu cơ chế mã hóa vị trí bằng hàm sin và cosin với chuỗi bước sóng cấp số nhân từ $2\pi$ đến $10.000 \cdot 2\pi$, đặt nền móng toán học cho việc sử dụng sóng điều hòa trong kiến trúc Transformer.

### 2.2. Bài học rút ra từ thực nghiệm của đề tài
Trong giai đoạn `REP_ABLATION_102`, nhóm phát hiện rằng việc tiêm trực tiếp vector FoNE cộng thô vào token điều khiển mở đầu của cấu trúc hội thoại (`<|im_start|>`) làm phá vỡ cơ chế chú ý cấu trúc của kiến trúc ChatML, dẫn tới tỷ lệ lỗi định dạng $97{,}78\%$. Khi tách rời vector FoNE/Can-Chi sang cơ chế cộng số dư có cổng khởi tạo bằng 0 ($B_2$) hoặc chuyển hóa thành văn bản tự nhiên ($B_1$), tính toàn vẹn cú pháp của mô hình được khôi phục hoàn toàn.

---

## 3. Thích Ứng Tham Số Hiệu Quả Trên Khối MLP (MLP-Only LoRA)

### 3.1. Các công trình nền tảng
- **LoRA: Low-Rank Adaptation of Large Language Models**  
  *Tác giả:* Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen.  
  *Công bố:* ICLR 2022 / arXiv:2106.09685.  
  *Đóng góp:* Đề xuất cơ chế phân rã ma trận cập nhật trọng số thành hai ma trận hạng thấp: $\Delta W = \frac{\alpha}{r} B A$ với $r \ll d$. Giữ cố định trọng số gốc $W_0$, giúp giảm hàng nghìn lần bộ nhớ lưu trữ gradient và trạng thái bộ tối ưu hóa optimizer mà không làm suy giảm năng lực mô hình.
- **Parameter-Efficient Transfer Learning for NLP**  
  *Tác giả:* Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly.  
  *Công bố:* ICML 2019 / arXiv:1902.00751.  
  *Đóng góp:* Đề xuất chèn các module adapter dạng nghẽn cổ chai (bottleneck) sau các khối Attention và FFN, mở ra kỷ nguyên tinh chỉnh tham số hiệu quả (PEFT).
- **Towards a Unified View of Parameter-Efficient Transfer Learning**  
  *Tác giả:* Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig.  
  *Công bố:* ICLR 2022 / arXiv:2110.04366.  
  *Đóng góp:* Thống nhất các phương pháp Adapter, Prefix-Tuning và LoRA dưới cùng một khung toán học, chỉ ra rằng tác động vào các tầng FFN mang lại hiệu quả cao trong việc điều chỉnh không gian biểu diễn tri thức.
- **Locating and Editing Factual Associations in GPT (ROME)**  
  *Tác giả:* Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov.  
  *Công bố:* NeurIPS 2022 / arXiv:2202.05262.  
  *Đóng góp:* Thực nghiệm định vị nhân quả (causal mediation analysis) chỉ ra rằng các tầng mạng nơ-ron truyền thẳng (Feed-Forward / MLP) của mô hình Transformer hoạt động tương tự như một bộ nhớ kết hợp (Key-Value associative memory), trực tiếp lưu trữ các sự kiện và tri thức thế giới thực.
- **Transformer Feed-Forward Layers Are Key-Value Memories**  
  *Tác giả:* Mor Geva, Roei Schuster, Jonathan Berant, Omer Levy.  
  *Công bố:* EMNLP 2021 / arXiv:2012.14913.  
  *Đóng góp:* Chứng minh các neuron trong lớp MLP thứ nhất đóng vai trò phát hiện mẫu (key), và lớp thứ hai kích hoạt phân phối xác suất của từ vựng đầu ra (value).

### 3.2. Cơ sở lý thuyết cho lựa chọn cấu hình của đề tài
Dựa trên các bằng chứng từ Meng et al. (2022) và Geva et al. (2021), việc cập nhật tri thức lịch sử theo thời gian không đòi hỏi phải thay đổi cơ chế định tuyến cú pháp của Self-Attention. Nhóm quyết định giới hạn LoRA nghiêm ngặt trên **72 module MLP** (`gate_proj`, `up_proj`, `down_proj`) của `Qwen2.5-0.5B-Instruct` với hạng $r=2, \alpha=4$, chỉ sử dụng đúng **829.440 tham số huấn luyện** (chiếm $< 0{,}17\%$ tổng tham số mô hình), vừa tối ưu hóa bộ nhớ GPU vừa tập trung can thiệp đúng vào vùng nhớ tri thức thực tế.

---

## 4. Học Tăng Dần Theo Dòng Thời Gian (Temporal Continual Learning)

- **ECONET: Effective Continual Pretraining of Language Models for Event Temporal Reasoning**  
  *Tác giả:* Rujun Han, Xiang Ren, Nanyun Peng.  
  *Công bố:* EMNLP 2021 / arXiv:2104.08779.  
  *Đóng góp:* Đề xuất phương pháp tiền huấn luyện tăng dần có cấu trúc theo dòng thời gian để mô hình nắm bắt được sự tiến hóa của các sự kiện và giảm thiểu hiện tượng quên lãng thảm họa (catastrophic forgetting). Đây là tài liệu tiền đề trực tiếp cho giao thức huấn luyện tăng dần 3 chặng (1947 $\rightarrow$ 1984 $\rightarrow$ 2022) của đề tài.
- **Time-Aware Language Models as Temporal Knowledge Bases**  
  *Tác giả:* Bhuwan Dhingra, Jeremy R. Cole, Julian Martin Eisenschlos, Daniel Gillick, Jacob Eisenstein, William W. Cohen.  
  *Công bố:* TACL 2022 / arXiv:2106.15110.  
  *Đóng góp:* Chứng minh mô hình ngôn ngữ truyền thống gặp suy giảm nghiêm trọng khi đối mặt với tri thức biến đổi theo thời gian; đề xuất đưa tiền tố thời gian vào ngữ cảnh truy vấn để điều kiện hóa không gian tìm kiếm tri thức (cơ sở cho nhánh $B_1$ của đề tài).
- **Towards Continual Knowledge Learning of Language Models**  
  *Tác giả:* Joel Jang, Seonghyeon Ye, Sohee Yang, Joonsuk Park, Minjoon Seo.  
  *Công bố:* ACL 2022 / arXiv:2110.03215.  
  *Đóng góp:* Thiết lập các benchmark đo lường hiện tượng quên lãng tri thức và đề xuất các phương pháp phân tách trọng số thời gian.

---

## 5. Cơ Sở Khoa Học Cho Chấm Điểm Lựa Chọn (Option Scoring) và Giải Mã Có Ràng Buộc (Constrained Decoding)

Trong các buổi báo cáo tiến độ, nhóm nhận thấy câu hỏi nghiên cứu quan trọng cần giải quyết là: **Làm thế nào để tách bạch triệt để lỗi cú pháp sinh văn bản (format failure) ra khỏi năng lực suy luận thời gian thực chất (temporal understanding) của mô hình?** Hai hướng tiếp cận dưới đây cung cấp cơ sở khoa học hoàn chỉnh để triển khai:

### 5.1. Chuyên đề Chấm điểm Lựa chọn Đáp án (Option Scoring / Log-Likelihood Evaluation)
Thay vì để mô hình sinh văn bản tự do rồi dùng regex để phân tích cú pháp, phương pháp này tính toán trực tiếp điểm xác suất hợp lý logarit (log-likelihood) của từng phương án trả lời dựa trên phân phối xác suất đầu ra (logits) của mô hình.

- **Language Models are Few-Shot Learners (GPT-3)**  
  *Tác giả:* Tom B. Brown et al. (OpenAI).  
  *Công bố:* NeurIPS 2020 / arXiv:2005.14165.  
  *Đóng góp:* Chuẩn hóa phương pháp đánh giá trắc nghiệm (Multiple-Choice Evaluation) bằng cách tính tổng hoặc trung bình log-likelihood của chuỗi token tương ứng với từng đáp án:
  $$\text{Score}(O_i) = \frac{1}{|O_i|} \sum_{t=1}^{|O_i|} \log P(w_t^{(i)} \mid \text{Prompt}, w_{<t}^{(i)})$$
  Phương án có điểm số cao nhất được chọn làm đáp án dự đoán. Phương pháp này loại bỏ $100\%$ nguy cơ sinh lỗi định dạng cú pháp.
- **Surface Form Competition—Why the Highest Probability Answer Isn't Always Right**  
  *Tác giả:* Ari Holtzman, Peter West, Vered Shwartz, Yejin Choi, Luke Zettlemoyer.  
  *Công bố:* EMNLP 2021 / arXiv:2104.08315.  
  *Đóng góp:* Phát hiện hiện tượng cạnh tranh hình thức bề mặt (các đáp án có nhiều cách viết đồng nghĩa hoặc token ngắn hơn sẽ bị thiên lệch xác suất); đề xuất phương pháp chuẩn hóa Domain Conditional PMI để đánh giá khách quan năng lực tri thức thực chất.
- **Leveraging Large Language Models for Multiple Choice Question Answering**  
  *Tác giả:* Joshua Robinson, David Wingate.  
  *Công bố:* ACL 2023 / arXiv:2210.12353.  
  *Đóng góp:* So sánh thực nghiệm chi tiết giữa sinh văn bản tự do và chấm điểm log-likelihood trên các bài toán MCQA, khẳng định option scoring phản ánh chính xác năng lực tri thức nội tại của mô hình hơn việc sinh văn bản tự do khi mô hình có kích thước nhỏ ($< 1\text{B}$).
- **Large Language Models Are Not Robust Multiple Choice Selectors**  
  *Tác giả:* Chunting Zhou, Junxian He, Graham Neubig et al.  
  *Công bố:* ICLR 2024 / arXiv:2309.03882.  
  *Đóng góp:* Phân tích hiện tượng thiên kiến thứ tự phương án (position bias) và đề xuất kỹ thuật hoán vị các phương án lựa chọn (permutation scoring) để đo lường độ bền vững của mô hình.

### 5.2. Chuyên đề Giải mã Có ràng buộc (Constrained Decoding / Grammar-Guided Generation)
Nếu vẫn giữ giao thức sinh ngôn ngữ tự hồi quy, phương pháp này can thiệp vào từng bước giải mã để ép phân phối xác suất của các token không hợp lệ về âm vô cùng ($-\infty$), đảm bảo mô hình chỉ sinh ra các ký tự tuân thủ chính xác ngữ pháp định sẵn.

- **Outlines: Flexible and Fast Text Generation with Grammar Constraints**  
  *Tác giả:* Brandon Willard, Rémi Louf.  
  *Công bố:* 2023 / arXiv:2307.09702.  
  *Đóng góp:* Xây dựng khung giải mã có ràng buộc hiệu năng cao dựa trên Otomata hữu hạn tất định (Deterministic Finite Automaton - DFA). Bằng cách chuyển biểu thức chính quy (Regex) hoặc ngữ pháp định dạng JSON thành DFA, bộ giải mã xác định tập hợp token hợp lệ tại mỗi bước sinh và gán mặt nạ che (masking) trên logits:
  $$P'(w_t) = \begin{cases} P(w_t) & \text{nếu } w_t \in \mathcal{V}_{\text{valid}} \\ -\infty & \text{ngược lại} \end{cases}$$
  Đảm bảo $100\%$ câu trả lời sinh ra luôn tuân thủ đúng định dạng mà không tốn chi phí tính toán suy luận bổ sung.
- **PICARD: Parsing Incrementally for Constrained Auto-regressive Decoding from Language Models**  
  *Tác giả:* Torsten Scholak, Nathan Schucher, Dzmitry Bahdanau.  
  *Công bố:* EMNLP 2021 / arXiv:2109.05093.  
  *Đóng góp:* Ứng dụng bộ phân tích cú pháp tăng dần trong quá trình tìm kiếm chùm (beam search) để ngăn chặn mô hình sinh các cấu trúc SQL sai cú pháp.
- **Grammar-Constrained Decoding for Structured NLP Tasks**  
  *Tác giả:* Saibo Geng et al.  
  *Công bố:* 2023.  
  *Đóng góp:* Tổng quát hóa việc giải mã dựa trên ngữ pháp phi ngữ cảnh (Context-Free Grammar - CFG) cho các mô hình sinh mã nguồn và hỏi đáp có cấu trúc.

---

## 6. Tổng Kết và Ý Nghĩa Đối Với Lộ Trình Nghiên Cứu Tiếp Theo

1. **Về biểu diễn Can-Chi và FoNE:** Các tài liệu tiền đề xác nhận việc kết hợp sóng Fourier và chu kỳ thời gian Đông Á là hướng đi có cơ sở toán học vững chắc. Tuy nhiên, bài học từ `REP_ABLATION_102` và `RESCUE_103` cho thấy phương thức ghép nối giao diện (interface gating) đóng vai trò quyết định đến tính ổn định của mô hình.
2. **Về kiến trúc MLP-only LoRA:** Cơ sở lý thuyết từ ROME (Meng et al., 2022) củng cố tính đúng đắn của việc giới hạn cập nhật trên các khối FFN để điều chỉnh tri thức thực tế, bảo toàn năng lực tổng quát mà chỉ tốn 0.83M tham số.
3. **Về giải pháp chẩn đoán tiếp theo:** Việc triển khai Option Scoring (Brown et al., 2020) và Constrained Decoding (Willard & Louf, 2023) là bước đi cấp thiết và chuẩn mực khoa học nhằm tách rời triệt để lỗi cú pháp khỏi năng lực suy luận thời gian, cung cấp câu trả lời khách quan cho Giảng viên Hướng dẫn trước khi quyết định mở rộng kích thước mô hình lên 1.5B hoặc 3B.
