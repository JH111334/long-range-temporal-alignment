# Bản đồ project học kỳ 2026

Cập nhật: 2026-08-24. Bốn repository dưới đây là scaffold độc lập, private, MIT và dùng nhánh `main`.

| Môn | Repository | Bài toán chính | Baseline đầu tiên |
|---|---|---|---|
| CS406 | [CS406-Visual-Asset-Retrieval](https://github.com/JH111334/CS406-Visual-Asset-Retrieval) | Truy xuất ảnh tương tự có provenance cho kho ảnh văn hóa/công ích | pHash → histogram màu → HOG exact search |
| DS111 | [DS111-Retrieval-Data-Analysis](https://github.com/JH111334/DS111-Retrieval-Data-Analysis) | Phân tích chất lượng dữ liệu/run truy xuất, leakage, drift và CI | Schema/quality audit → paired bootstrap |
| DS201 | [DS201-Temporal-Retrieval-Learning](https://github.com/JH111334/DS201-Temporal-Retrieval-Learning) | Học biểu diễn thời gian dài cho truy xuất lịch sử tiếng Việt | Lexical → scalar → Fourier → published Ticktack |
| DS314 | [DS314-Evidence-Grounded-Retrieval](https://github.com/JH111334/DS314-Evidence-Grounded-Retrieval) | Truy xuất tài liệu tiếng Việt có citation, access scope và abstention | BM25 → TF-IDF → RRF |

## Bộ khung dùng chung

- `Documents/{Decision-Making,References,General}`; cấp gốc chỉ có `README.md`.
- Problem contract, data/model/evaluation gates và `ERROR-EXPERIENCE.md` theo toàn vòng đời.
- Python core offline; `ServiceAdapters` cho Hugging Face, Kaggle, nguồn domain và artifact store đều fail-closed.
- `requirements.txt` là kế hoạch, chưa cài; registries candidate đều `hold`.
- Streamlit chỉ đọc trạng thái/artifact; không tải data/model hoặc training.
- Markdown nguồn, DOCX đã render QA, HTML tự chứa, local skill và GitHub Actions `quality`.

## Kiểm tra mỗi repository

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m unittest discover -s tests -v
py -3.12 scripts/validate_project.py
```

Kết quả bàn giao: mỗi repo đạt 4 unit tests, validator 37 required files, local skill validation và DOCX rendered QA; bốn workflow GitHub đều `success`. Đây là bằng chứng scaffold/quality gate, không phải benchmark mô hình.

## Tái tạo scaffold

Từ repository temporal này, chạy `py -3.12 scripts/bootstrap_course_repositories.py --output-root <thu_muc_moi>`. Script từ chối overwrite. Sau khi sinh, build `Documents/References/PROJECT-KNOWLEDGE.docx` từ Markdown bằng skill `author-validate-docx`, rồi mới chạy validator đầy đủ. Không dùng lệnh tái tạo trên bốn thư mục đã có.
