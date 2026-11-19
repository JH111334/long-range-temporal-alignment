"""Catalog for four compact, Vietnam-first course repositories."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Source:
    """Describe one primary or official reference."""

    label: str
    url: str
    use: str


@dataclass(frozen=True, slots=True)
class Candidate:
    """Describe one held candidate for data or models."""

    name: str
    url: str
    role: str
    gate: str


@dataclass(frozen=True, slots=True)
class CourseProject:
    """Contain all project-specific choices used by the scaffold generator."""

    repo_name: str
    package_name: str
    course_code: str
    course_name: str
    schedule: tuple[str, ...]
    title: str
    short_title: str
    primary_user: str
    supported_decision: str
    problem: str
    vietnam_value: str
    global_transfer: str
    input_contract: tuple[str, ...]
    output_contract: tuple[str, ...]
    exclusions: tuple[str, ...]
    stages: tuple[str, ...]
    baselines: tuple[str, ...]
    candidates: tuple[str, ...]
    metrics: tuple[str, ...]
    split_policy: str
    falsification: tuple[str, ...]
    hardware: tuple[str, ...]
    data_candidates: tuple[Candidate, ...]
    model_candidates: tuple[Candidate, ...]
    sources: tuple[Source, ...]
    curriculum_facts: tuple[str, ...]
    curriculum_inferences: tuple[str, ...]
    curriculum_exclusions: tuple[str, ...]
    curriculum_evidence_limit: str
    curriculum_sources: tuple[Source, ...]
    research_dependencies: tuple[str, ...]
    input_type: str
    output_type: str
    contracts_code: str
    domain_adapter_module: str
    domain_adapter_class: str
    domain_adapter_provider: str


CS406 = CourseProject(
    repo_name="CS406-Visual-Asset-Retrieval",
    package_name="cs406_visual_retrieval",
    course_code="CS406",
    course_name="Xử lý ảnh và ứng dụng",
    schedule=(
        "CS406.R12 — LT — Thứ 4, tiết 6–8, B6.04 — GV: Cáp Phạm Đình Thăng — 07/09/2026–26/12/2026",
        "CS406.R12.1 — HT2 — lịch/phòng chưa công bố — GV: Cáp Phạm Đình Thăng — 07/09/2026–26/12/2026",
    ),
    title="Truy xuất ảnh tương tự có provenance cho kho ảnh văn hóa và công ích Việt Nam",
    short_title="Visual Asset Retrieval",
    primary_user="Cán bộ số hóa, sinh viên và người quản lý kho ảnh cần tìm ảnh tương tự/ảnh trùng biến đổi.",
    supported_decision="Xác định nhanh các ảnh liên quan để rà soát bản ghi, bản quyền, trùng lặp và phục vụ tra cứu.",
    problem="Xếp hạng ảnh trong một corpus đã cấp phép theo mức tương tự với ảnh truy vấn, đồng thời trả nguồn, phiên bản đặc trưng và trạng thái lỗi rõ ràng.",
    vietnam_value="Hỗ trợ số hóa và quản trị các bộ sưu tập di sản, giáo dục và truyền thông công ích còn phân tán; nhu cầu thực tế phải được xác nhận bằng phỏng vấn người dùng pilot.",
    global_transfer="Hợp đồng query-by-image, provenance, duplicate audit và benchmark landmark chuyển giao được sang mọi kho ảnh có qrels.",
    input_contract=(
        "Ảnh truy vấn hợp lệ, URI cục bộ hoặc asset ID; không tự crawl từ Internet.",
        "Corpus manifest gồm asset_id, source URI, license/attribution, checksum và corpus version.",
        "Top-k, feature version và run identity bất biến.",
    ),
    output_contract=(
        "Danh sách VisualHit theo rank với asset_id, score, source URI và feature version.",
        "Warning/invalid-image status thay vì âm thầm bỏ lỗi.",
        "Run artifact có config hash, corpus manifest hash, code revision, latency và evidence class.",
    ),
    exclusions=(
        "Không nhận truy vấn văn bản, không sinh caption/OCR và không huấn luyện detector trong baseline.",
        "Không tuyên bố semantic retrieval khi mới dùng pHash, histogram màu hoặc HOG.",
        "Không dùng ảnh không rõ quyền tái sử dụng.",
    ),
    stages=(
        "manifest",
        "decode_validate",
        "normalize",
        "feature_extract",
        "exact_search",
        "evaluate",
        "artifact",
        "ui",
    ),
    baselines=(
        "pHash cho near-duplicate",
        "histogram màu",
        "HOG + cosine/L2 exact search",
    ),
    candidates=(
        "frozen DINOv2 embedding sau data/resource gate",
        "learned metric head chỉ sau benchmark cố định",
    ),
    metrics=(
        "mAP",
        "Recall@1/5/10",
        "nDCG@10",
        "invalid-image rate",
        "p50/p95 latency",
        "peak RAM/VRAM",
    ),
    split_policy="Chia theo landmark/sự kiện/nguồn và nhóm near-duplicate; không random split theo từng file ảnh.",
    falsification=(
        "Baseline không vượt kiểm tra metadata/duplicate đơn giản trên qrels held-out.",
        "Gain của frozen embedding biến mất khi loại near-duplicate hoặc đổi nguồn.",
        "Corpus pilot không có quyền sử dụng/attribution đủ rõ.",
    ),
    hardware=(
        "Tier 0 CPU: decode, pHash, histogram/HOG, exact search trên corpus nhỏ.",
        "Tier 1 RTX 2050 4 GB: frozen embedding, batch nhỏ sau khi đo peak VRAM.",
        "Tier 2 lab/cloud: chỉ cho ablation quy mô lớn; không phải mặc định.",
    ),
    data_candidates=(
        Candidate(
            "Revisited Oxford/Paris",
            "https://cmp.felk.cvut.cz/revisitop/",
            "benchmark quốc tế",
            "Rà điều khoản và tải từ trang chính thức",
        ),
        Candidate(
            "Wikimedia Commons pilot",
            "https://commons.wikimedia.org/w/api.php",
            "corpus Việt Nam candidate",
            "Lưu license/attribution theo từng file; thiết kế qrels riêng",
        ),
    ),
    model_candidates=(
        Candidate(
            "DINOv2",
            "https://github.com/facebookresearch/dinov2",
            "frozen feature candidate",
            "Pin code/weight revision và đo VRAM; chưa tải",
        ),
    ),
    sources=(
        Source(
            "Revisited Oxford and Paris",
            "https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html",
            "Protocol landmark retrieval và độ khó query",
        ),
        Source(
            "Revisited benchmark project",
            "https://cmp.felk.cvut.cz/revisitop/",
            "Dataset/evaluation source chính thức",
        ),
        Source(
            "DINOv2 official repository",
            "https://github.com/facebookresearch/dinov2",
            "Frozen representation candidate; phải rà model card",
        ),
        Source(
            "ILIAS CVPR 2025",
            "https://openaccess.thecvf.com/content/CVPR2025/html/Kordopatis-Zilos_ILIAS_Instance-Level_Image_retrieval_At_Scale_CVPR_2025_paper.html",
            "Hard distractor và protocol instance retrieval quy mô lớn; không tải full dataset P1",
        ),
        Source(
            "GLR 2019 winning system",
            "https://openaccess.thecvf.com/content_CVPRW_2020/html/w61/Yokoo_Two-Stage_Discriminative_Re-Ranking_for_Large-Scale_Landmark_Retrieval_CVPRW_2020_paper.html",
            "Mẫu two-stage retrieval + bounded rerank; không sao chép ensemble",
        ),
        Source(
            "Faiss index guidance",
            "https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index",
            "Exact Flat trước ANN; cosine qua normalized inner product",
        ),
        Source(
            "MediaWiki Action API",
            "https://www.mediawiki.org/wiki/API:Main_page",
            "Metadata access; API không thay thế license audit",
        ),
    ),
    curriculum_facts=(
        "UIT công bố CS406 là học phần đang mở, 3 tín chỉ lý thuyết và 1 tín chỉ thực hành; mục tiêu là nền tảng xử lý ảnh và hệ thống thông tin dựa trên ảnh.",
        "UIT công bố CS335 — Tìm kiếm ảnh và video như một học phần riêng; retrieval trong project CS406 phải phục vụ việc chứng minh năng lực xử lý ảnh, không thay thế nó.",
        "HCMUS tách xử lý ảnh/video, thị giác máy tính, truy vấn thông tin thị giác và học sâu thành các lớp năng lực liên quan nhưng khác nhau.",
        "HCMUT đặt xử lý ảnh và thị giác máy tính cạnh học máy, học sâu, Toán và lập trình trong định hướng Khoa học máy tính.",
    ),
    curriculum_inferences=(
        "Đóng góp chính nên là pipeline ảnh có kiểm soát: decode, màu, resize, normalization, đặc trưng và phân tích độ bền trước biến đổi.",
        "Retrieval giữ phạm vi query-by-image; báo mAP/Recall@k và lát cắt theo ánh sáng, tỉ lệ, nền, nguồn và near-duplicate.",
        "Ưu tiên corpus ảnh Việt Nam công khai về giáo dục, di sản, nông nghiệp hoặc giao thông khi quyền tái sử dụng và attribution rõ.",
    ),
    curriculum_exclusions=(
        "Không đưa truy vấn văn bản, caption, OCR hoặc học đa thể thức vào core.",
        "Không dùng encoder lớn để che thiếu sót ở image contract, qrels hoặc phân tích biến đổi ảnh.",
        "Không mở rộng thành hệ nhận dạng tổng quát nếu evaluator vẫn chỉ đo retrieval.",
    ),
    curriculum_evidence_limit="Đối chiếu HCMUS/HCMUT là tín hiệu chương trình, không phải công nhận tương đương học phần UIT.",
    curriculum_sources=(
        Source(
            "Danh mục môn học UIT",
            "https://student.uit.edu.vn/danh-muc-mon-hoc-dai-hoc",
            "Trạng thái, tín chỉ và ranh giới CS406/CS335",
        ),
        Source(
            "Đề án ngành Trí tuệ nhân tạo UIT",
            "https://student.uit.edu.vn/sites/daa/files/202412/de_an_nganh_mo_nganh_tri_tue_nhan_tao_uit.pdf",
            "Mô tả chính thức Xử lý ảnh và ứng dụng",
        ),
        Source(
            "Đào tạo Thị giác máy tính HCMUS",
            "https://www.fit.hcmus.edu.vn/vn/Default.aspx?language=vi-VN&tabid=709",
            "Cấu trúc nền tảng và học phần ứng dụng thị giác",
        ),
        Source(
            "Bộ môn Khoa học máy tính HCMUT",
            "https://cse.hcmut.edu.vn/bo-mon-khoa-hoc-may-tinh",
            "Các trục đào tạo ảnh, ML/DL, Toán và lập trình",
        ),
    ),
    research_dependencies=(
        "numpy>=2,<3",
        "Pillow>=11,<13",
        "opencv-python-headless>=4.10,<5",
        "scikit-image>=0.25,<1",
        "torch>=2.6,<3",
        "torchvision>=0.21,<1",
    ),
    input_type="ImageQuery",
    output_type="VisualHit",
    contracts_code='''@dataclass(frozen=True, slots=True)\nclass ImageQuery:\n    """Identify one image-to-image retrieval request."""\n\n    query_id: str\n    image_uri: str\n    corpus_version: str\n    top_k: int = 10\n\n    def __post_init__(self) -> None:\n        _require(self.query_id, "query_id")\n        _require(self.image_uri, "image_uri")\n        _require(self.corpus_version, "corpus_version")\n        if self.top_k < 1 or self.top_k > 100:\n            raise ProjectContractError("top_k must be in [1, 100]")\n\n\n@dataclass(frozen=True, slots=True)\nclass VisualHit:\n    """Return one provenance-bearing visual result."""\n\n    asset_id: str\n    score: float\n    rank: int\n    source_uri: str\n    feature_version: str\n\n    def __post_init__(self) -> None:\n        _require(self.asset_id, "asset_id")\n        _require(self.source_uri, "source_uri")\n        _require(self.feature_version, "feature_version")\n        if self.rank < 1:\n            raise ProjectContractError("rank must be positive")\n''',
    domain_adapter_module="wikimedia",
    domain_adapter_class="WikimediaCommonsAdapter",
    domain_adapter_provider="wikimedia-commons",
)


DS111 = CourseProject(
    repo_name="DS111-Retrieval-Data-Analysis",
    package_name="ds111_retrieval_analytics",
    course_code="DS111",
    course_name="Phân tích dữ liệu",
    schedule=(
        "DS111.R12 — LT — Thứ 2, tiết 6–9, B8.08 — GV: Phạm Thế Sơn — 07/09/2026–28/11/2026",
        "DS111.R12.1 — HT2 — lịch/phòng chưa công bố — GV: Phạm Thế Sơn — 14/09/2026–26/12/2026",
    ),
    title="Phân tích chất lượng dữ liệu và độ tin cậy của hệ thống truy xuất",
    short_title="Retrieval Data Analysis",
    primary_user="Nhóm nghiên cứu hoặc vận hành cần biết một thay đổi retrieval có thật sự tốt, ổn định và không rò rỉ dữ liệu.",
    supported_decision="Chấp nhận, bác bỏ hoặc điều tra thêm một corpus/run trước khi tốn chi phí training và triển khai.",
    problem="Kiểm định schema, missingness, duplicate/leakage, drift, latency và metric theo lát cắt trên run/qrels được version hóa; xuất báo cáo có khoảng tin cậy.",
    vietnam_value="Giúp các nhóm dữ liệu nhỏ tại Việt Nam tránh quyết định dựa trên một điểm trung bình hoặc dashboard không có provenance.",
    global_transfer="Run schema, paired bootstrap, slice audit và drift checks không phụ thuộc ngôn ngữ hay mô hình.",
    input_contract=(
        "Run table gồm query_id, item_id, rank, score, latency và run version.",
        "Qrels/corpus manifest tùy bài toán, có source/group/time split và checksum.",
        "Data dictionary nêu kiểu, đơn vị, missing policy và khóa duy nhất.",
    ),
    output_contract=(
        "AnalysisReport gồm metric, khoảng tin cậy, sample size, slice và warning.",
        "Leakage/quality findings có severity, evidence rows và owner.",
        "Artifact Parquet/JSON/HTML chỉ đọc, gắn config/data/code hash.",
    ),
    exclusions=(
        "Không train retrieval model trong baseline phân tích dữ liệu.",
        "Không dùng p-value đơn lẻ để thay effect size, confidence interval và kiểm tra giả định.",
        "Không đưa dữ liệu riêng/organizer data vào Git.",
    ),
    stages=(
        "manifest",
        "schema_validate",
        "quality_audit",
        "leakage_audit",
        "descriptive",
        "paired_inference",
        "slice_drift",
        "report",
    ),
    baselines=(
        "schema/constraint checks",
        "descriptive statistics",
        "paired bootstrap trên per-query metric",
        "rule-based drift/slice audit",
    ),
    candidates=(
        "DuckDB/Parquet cho dữ liệu lớn hơn RAM",
        "robust/non-parametric analysis sau assumption audit",
    ),
    metrics=(
        "schema-valid rate",
        "missing/duplicate rate",
        "leakage count",
        "metric delta + 95% CI",
        "effect size",
        "p50/p95 latency",
        "slice coverage",
    ),
    split_policy="Tôn trọng split source/entity/event/time của hệ nguồn; mọi so sánh run phải paired trên cùng query set.",
    falsification=(
        "Kết luận đổi dấu khi bootstrap theo group thay vì theo row.",
        "Metric gain chỉ xuất hiện ở query bị trùng hoặc slice thiếu coverage.",
        "Schema/run lineage không đủ để tái dựng mẫu phân tích.",
    ),
    hardware=(
        "Tier 0 CPU/RAM: CSV/JSONL fixture và phân tích nhỏ.",
        "Tier 1 CPU + DuckDB/Parquet: pushdown/streaming khi bảng vượt RAM.",
        "GPU không nằm trên đường găng; chỉ dùng tại project sinh run, không dùng để phân tích.",
    ),
    data_candidates=(
        Candidate(
            "Synthetic run fixture",
            "https://ir-measur.es/",
            "contract smoke",
            "Dữ liệu tổng hợp, phải gắn synthetic",
        ),
        Candidate(
            "Versioned runs từ CS406/DS201/DS314",
            "https://github.com/JH111334",
            "phân tích nội bộ candidate",
            "Chỉ ingest artifact có manifest/hash và quyền truy cập",
        ),
    ),
    model_candidates=(),
    sources=(
        Source(
            "ir-measures",
            "https://arxiv.org/abs/2111.13466",
            "Chuẩn hóa cách tính và gọi metric IR",
        ),
        Source(
            "SciPy bootstrap",
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html",
            "Khoảng tin cậy bootstrap",
        ),
        Source(
            "TREC 2025 RAG overview",
            "https://trec.nist.gov/pubs/trec34/papers/Overview_rag.pdf",
            "Run protocol, qrels/judgment coverage và tách retrieval-generation",
        ),
        Source(
            "UTokyo-HitU TREC 2025",
            "https://trec.nist.gov/pubs/trec34/papers/UTokyo.rag.pdf",
            "Winning-system ablation: BM25/small dense/RRF hữu ích; query expansion có thể drift",
        ),
        Source(
            "ViRE",
            "https://aclanthology.org/2026.findings-eacl.110/",
            "Candidate-pool size làm đổi absolute score; phải lưu pool hash",
        ),
        Source(
            "DuckDB Parquet",
            "https://duckdb.org/docs/stable/data/parquet/overview",
            "Column projection/filter pushdown và phân tích ngoài RAM",
        ),
        Source(
            "Pandas user guide",
            "https://pandas.pydata.org/docs/user_guide/index.html",
            "DataFrame/schema operations; không thay data contract",
        ),
    ),
    curriculum_facts=(
        "UIT công bố DS111 là học phần cơ sở ngành đang mở, 3 tín chỉ lý thuyết và 1 tín chỉ thực hành; kiến thức trước gồm lập trình và xác suất thống kê.",
        "Chương trình Khoa học dữ liệu UIT nhấn mạnh tìm kiếm, làm sạch, lưu trữ, truy vấn, phân tích, thống kê và diễn giải dữ liệu để hỗ trợ quyết định.",
        "Chương trình Khoa học dữ liệu HCMUS năm 2026 đặt Toán, Xác suất, Thống kê, thu thập, quản trị và phân tích dữ liệu trước mô hình thông minh.",
        "Đề cương Phân tích dữ liệu của HCMUT nhấn mạnh quy trình, thu thập, tiền xử lý, phân tích và ứng dụng; đây là đối chiếu lân cận, không phải học phần tương đương DS111.",
    ),
    curriculum_inferences=(
        "Dùng run, qrels và corpus manifest của retrieval làm đối tượng phân tích thay vì xây thêm một retriever.",
        "Sản phẩm phải có data dictionary, audit thiếu/trùng/leakage, thống kê mô tả, effect size, khoảng tin cậy, lát cắt và giới hạn kết luận.",
        "Giữ pipeline CPU và tái lập; dashboard chỉ trình bày artifact, không thay thế lập luận phân tích.",
    ),
    curriculum_exclusions=(
        "Không lấy training, model selection hoặc agent làm contribution của DS111.",
        "Không kết luận từ một điểm trung bình, p-value đơn lẻ hoặc biểu đồ thiếu denominator.",
        "Không gom toàn bộ data engineering của hệ thống khác vào project phân tích.",
    ),
    curriculum_evidence_limit="Chưa xác nhận toàn văn đề cương DS111 từ kho số; facts cấp học phần chỉ dùng catalog UIT, còn nội dung sâu dùng chuẩn đầu ra chương trình và đối chiếu được ghi nhãn.",
    curriculum_sources=(
        Source(
            "Danh mục môn học UIT",
            "https://student.uit.edu.vn/danh-muc-mon-hoc-dai-hoc",
            "Trạng thái, tín chỉ và kiến thức trước của DS111",
        ),
        Source(
            "Chương trình Khoa học dữ liệu UIT 2024",
            "https://daa.uit.edu.vn/content/cu-nhan-khoa-hoc-nganh-khoa-hoc-du-lieu-ap-dung-tu-khoa-19-2024?page=3",
            "Chuẩn đầu ra phân tích, diễn giải và ra quyết định",
        ),
        Source(
            "Chương trình Khoa học dữ liệu HCMUS 2026",
            "https://math.hcmus.edu.vn/dao-tao-dai-hoc/chuong-trinh-tcta/nganh-khoa-hoc-du-lieu",
            "Nền Toán–Thống kê, quản trị và phân tích dữ liệu",
        ),
        Source(
            "Đề cương Phân tích dữ liệu HCMUT",
            "https://grad.hcmut.edu.vn/gv/front.php/capnhatdecuong/public/printdecuongcdio?mamh=505938&manganh=8520301&option=tv",
            "Đối chiếu quy trình phân tích dữ liệu; không suy tương đương tín chỉ",
        ),
    ),
    research_dependencies=(
        "pandas>=2.2,<4",
        "pyarrow>=19,<25",
        "duckdb>=1.2,<2",
        "scipy>=1.15,<2",
        "plotly>=6,<7",
        "ir-measures>=0.3,<1",
    ),
    input_type="RetrievalRunInput",
    output_type="AnalysisReport",
    contracts_code='''@dataclass(frozen=True, slots=True)\nclass RetrievalRunInput:\n    """Identify one immutable retrieval run and optional qrels."""\n\n    run_id: str\n    run_uri: str\n    qrels_uri: str\n    schema_version: str\n\n    def __post_init__(self) -> None:\n        for name in ("run_id", "run_uri", "qrels_uri", "schema_version"):\n            _require(getattr(self, name), name)\n\n\n@dataclass(frozen=True, slots=True)\nclass AnalysisReport:\n    """Return audited metrics and warnings for one run."""\n\n    run_id: str\n    evidence_level: str\n    metrics: tuple[tuple[str, float], ...]\n    warnings: tuple[str, ...] = ()\n\n    def __post_init__(self) -> None:\n        _require(self.run_id, "run_id")\n        if self.evidence_level not in {"scaffold", "smoke_test", "measured"}:\n            raise ProjectContractError("unsupported evidence_level")\n        if not self.metrics:\n            raise ProjectContractError("metrics must not be empty")\n''',
    domain_adapter_module="open_data",
    domain_adapter_class="OpenDataPortalAdapter",
    domain_adapter_provider="official-open-data",
)


DS201 = CourseProject(
    repo_name="DS201-Temporal-Retrieval-Learning",
    package_name="ds201_temporal_learning",
    course_code="DS201",
    course_name="Deep Learning trong khoa học dữ liệu",
    schedule=(
        "DS201.R11 — LT — Thứ 3, tiết 1–3, B8.08 — GV: Nguyễn Tấn Hoàng Phước — 07/09/2026–26/12/2026",
        "DS201.R11.2 — HT1 — Thứ 6, tiết 1–5 cách tuần, B3.02 — GV: Nguyễn Hiếu Nghĩa — 28/09/2026–12/12/2026",
    ),
    title="Học biểu diễn thời gian dài cho truy xuất lịch sử tiếng Việt có bằng chứng",
    short_title="Temporal Retrieval Learning",
    primary_user="Người học, nhà nghiên cứu và biên tập viên cần tìm đoạn nguồn đúng mốc/khoảng thời gian trong tài liệu lịch sử.",
    supported_decision="Chọn bằng chứng phù hợp theo nội dung và thời gian, hoặc từ chối khi không đủ bằng chứng.",
    problem="So sánh biểu diễn thời gian scalar/Fourier/công thức Ticktack/phương pháp lai cho ranking đoạn lịch sử trên split source/time-held-out.",
    vietnam_value="Tạo protocol tái lập cho lịch sử tiếng Việt thay vì sao chép TempLS tiếng Trung chưa có release được xác minh.",
    global_transfer="HistoricalYear, interval relation, alias stress test và evidence-aware retrieval dùng được cho nhiều lịch/ngôn ngữ sau adaptation rõ ràng.",
    input_contract=(
        "TemporalQuery gồm văn bản, HistoricalYear/TemporalInterval rõ BCE/CE, corpus version và top-k.",
        "Corpus chunk có source URI, event/entity group, time interval, license và checksum.",
        "Representation/method label thuộc published-equation, operational-assumption hoặc proposed-method.",
    ),
    output_contract=(
        "TemporalHit có chunk_id, rank, content score, temporal score, evidence URI và representation version.",
        "Artifact có invalid-output/alias rate, slice metric, seed, CI, config/data/code hash và resource peak.",
        "Abstention khi interval/evidence không đủ hoặc parser lỗi.",
    ),
    exclusions=(
        "Không gọi synthetic fixture là TempLS reproduction.",
        "Không fine-tune LLM trước lexical/scalar/Fourier/frozen probe.",
        "Không dùng số 0 mơ hồ giữa BCE và CE.",
    ),
    stages=(
        "evidence_gate",
        "normalize_time",
        "quality_dedup",
        "group_time_split",
        "lexical_retrieve",
        "represent",
        "rerank",
        "evaluate",
        "registry",
        "ui",
    ),
    baselines=(
        "lexical retrieval không time feature",
        "Gregorian scalar",
        "Fourier",
        "published Ticktack equation",
    ),
    candidates=(
        "operational-assumption variant",
        "monotonic + multi-cycle hybrid",
        "frozen encoder probe",
        "LoRA nhỏ sau resource gate",
    ),
    metrics=(
        "MRR/nDCG/Recall@K",
        "temporal distance/boundary error",
        "Allen relation accuracy",
        "alias collision rate",
        "invalid-output rate",
        "seed + 95% CI",
        "peak RAM/VRAM",
    ),
    split_policy="Group theo source/entity/event rồi hold-out theo thời gian; audit near-duplicate trước mọi fit/tokenizer adaptation.",
    falsification=(
        "Gain biến mất trên source/time-held-out hoặc chỉ đến từ leakage.",
        "Biểu diễn lai không giảm alias mà làm giảm monotonicity/IR metric.",
        "Tokenizer/parser error giải thích gain tốt hơn representation.",
        "Chi phí vượt tier được duyệt.",
    ),
    hardware=(
        "Tier 0 CPU: calendar contracts, scalar/Fourier/Ticktack, synthetic smoke.",
        "Tier 1 RTX 2050 4 GB: tokenizer/frozen embedding probe sau resource measurement.",
        "Tier 2 RTX 2050 4 GB: LoRA model nhỏ, micro-batch 1 nếu probe đạt.",
        "Tier 3 lab/cloud: multi-seed/3B+; tách riêng, có ngân sách và stop condition.",
    ),
    data_candidates=(
        Candidate(
            "VLSP 2025 TemporalQA",
            "https://aclanthology.org/2025.vlsp-1.35/",
            "benchmark tiếng Việt phụ",
            "Xác minh release/terms/revision trước ingest",
        ),
        Candidate(
            "HisGraphRAG historical sources",
            "https://aclanthology.org/2025.paclic-1.49/",
            "prior art Việt Nam",
            "Không suy ra dataset được phép tải từ paper",
        ),
        Candidate(
            "ChronoSense",
            "https://aclanthology.org/2025.acl-short.46/",
            "interval stress test",
            "Xác minh data card/license/revision",
        ),
        Candidate(
            "HistoryBankQA",
            "https://aclanthology.org/2026.starsem-conference.33/",
            "international transfer",
            "Không có tiếng Việt; chỉ dùng khi data gate đạt",
        ),
    ),
    model_candidates=(
        Candidate(
            "Qwen3-0.6B",
            "https://huggingface.co/Qwen/Qwen3-0.6B",
            "small frozen/PEFT candidate",
            "Pin full commit, rà license, đo 4 GB VRAM; chưa tải",
        ),
        Candidate(
            "Qwen2.5-0.5B-Instruct",
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
            "control candidate",
            "Pin full commit, rà model card và resource; chưa tải",
        ),
    ),
    sources=(
        Source(
            "Ticktack",
            "https://arxiv.org/abs/2503.04150",
            "Published equation và claim paper_reported; code/TempLS chưa được giả định có sẵn",
        ),
        Source(
            "Date Fragments",
            "https://aclanthology.org/2025.emnlp-main.159/",
            "Tokenizer fragmentation gate",
        ),
        Source(
            "ETRQA",
            "https://aclanthology.org/2025.findings-acl.1198/",
            "Event/timespan/fine-grained evaluation",
        ),
        Source(
            "ChronoSense",
            "https://aclanthology.org/2025.acl-short.46/",
            "Allen interval relations",
        ),
        Source(
            "TempAnswerQA",
            "https://aclanthology.org/2025.findings-emnlp.637/",
            "Distance-aware temporal error",
        ),
        Source(
            "VLSP TemporalQA",
            "https://aclanthology.org/2025.vlsp-1.35/",
            "Vietnamese benchmark context",
        ),
        Source(
            "HisGraphRAG",
            "https://aclanthology.org/2025.paclic-1.49/",
            "Vietnamese historical retrieval ablation",
        ),
        Source(
            "HistoryBankQA",
            "https://aclanthology.org/2026.starsem-conference.33/",
            "Multilingual historical event QA; chỉ dùng lát nhỏ sau license/checksum gate",
        ),
        Source(
            "ComplexTempQA",
            "https://aclanthology.org/2025.emnlp-main.463/",
            "Taxonomy comparison/count/multi-hop; full scale ngoài P1",
        ),
    ),
    curriculum_facts=(
        "UIT công bố DS201 là học phần 3 tín chỉ lý thuyết + 1 tín chỉ thực hành, có học phần trước DS102.",
        "Bảng tóm tắt UIT nêu biểu diễn văn bản, ảnh, chuỗi thời gian; học có giám sát; CNN/RNN và tối ưu bằng gradient.",
    ),
    curriculum_inferences=(
        "Giữ non-neural control, mô hình nhỏ, learning curve, ablation, nhiều seed/CI và resource profile.",
        "Representation chỉ được giữ khi sửa failure slice đã đo; data/evaluator và baseline CPU đi trước PEFT.",
    ),
    curriculum_exclusions=(
        "Không lấy LLM lớn hoặc model scale làm contribution mặc định.",
        "Không gọi frozen probe, synthetic grid hoặc best seed là kết quả deep-learning tổng quát.",
    ),
    curriculum_evidence_limit="Đối chiếu chương trình không thay đề cương DS201 do giảng viên cung cấp; kiến trúc cụ thể phải cập nhật khi vào học.",
    curriculum_sources=(
        Source(
            "Danh mục môn học UIT",
            "https://student.uit.edu.vn/danh-muc-mon-hoc-dai-hoc",
            "Tín chỉ và học phần trước DS201",
        ),
        Source(
            "Bảng tóm tắt môn học UIT",
            "https://student.uit.edu.vn/content/bang-tom-tat-mon-hoc",
            "Nội dung học sâu, biểu diễn và tối ưu",
        ),
        Source(
            "Chương trình Khoa học dữ liệu HCMUS 2026",
            "https://math.hcmus.edu.vn/dao-tao-dai-hoc/chuong-trinh-tcta/nganh-khoa-hoc-du-lieu",
            "Đối chiếu nền dữ liệu, máy học và AI",
        ),
    ),
    research_dependencies=(
        "numpy>=2,<3",
        "torch>=2.6,<3",
        "transformers>=4.51,<6",
        "datasets>=3,<5",
        "accelerate>=1,<3",
        "peft>=0.15,<1",
        "safetensors>=0.5,<1",
        "psutil>=6,<8",
    ),
    input_type="TemporalQuery",
    output_type="TemporalHit",
    contracts_code='''from .normalization.historical_time import HistoricalYear\n\n\n@dataclass(frozen=True, slots=True)\nclass TemporalQuery:\n    """Identify one evidence-grounded temporal retrieval request."""\n\n    query_id: str\n    text: str\n    start: HistoricalYear\n    end: HistoricalYear\n    corpus_version: str\n    top_k: int = 10\n\n    def __post_init__(self) -> None:\n        for name in ("query_id", "text", "corpus_version"):\n            _require(getattr(self, name), name)\n        if self.start.astronomical > self.end.astronomical:\n            raise ProjectContractError("start must not be after end")\n        if self.top_k < 1 or self.top_k > 100:\n            raise ProjectContractError("top_k must be in [1, 100]")\n\n\n@dataclass(frozen=True, slots=True)\nclass TemporalHit:\n    """Return one ranked historical chunk with separate scores."""\n\n    chunk_id: str\n    rank: int\n    content_score: float\n    temporal_score: float\n    evidence_uri: str\n    representation_version: str\n\n    def __post_init__(self) -> None:\n        for name in ("chunk_id", "evidence_uri", "representation_version"):\n            _require(getattr(self, name), name)\n        if self.rank < 1:\n            raise ProjectContractError("rank must be positive")\n''',
    domain_adapter_module="official_sources",
    domain_adapter_class="OfficialSourceAdapter",
    domain_adapter_provider="official-source",
)


DS314 = CourseProject(
    repo_name="DS314-Evidence-Grounded-Retrieval",
    package_name="ds314_evidence_retrieval",
    course_code="DS314",
    course_name="Rút trích và truy vấn thông tin",
    schedule=(
        "DS314.R11 — LT — Thứ 3, tiết 6–9, C212 — GV: Nguyễn Văn Kiệt — 07/09/2026–28/11/2026",
        "DS314.R11.2 — HT1 — Thứ 7, tiết 1–5 cách tuần, B3.04 — GV: Nguyễn Thị Hoàng Anh — 28/09/2026–12/12/2026",
    ),
    title="Truy xuất tài liệu tiếng Việt có bằng chứng, phạm vi truy cập và khả năng từ chối",
    short_title="Evidence-Grounded Retrieval",
    primary_user="Người dân, sinh viên và nhân viên nghiệp vụ cần tìm đúng đoạn trong bộ tài liệu công khai đã quản trị.",
    supported_decision="Xác định tài liệu/đoạn nguồn cần đọc tiếp; hệ thống không thay thế chuyên gia hoặc đưa tư vấn pháp lý.",
    problem="Xếp hạng chunk tiếng Việt bằng lexical retrieval/fusion, bảo toàn source/access scope và chỉ cho phép câu trả lời khi citation thuộc active corpus.",
    vietnam_value="Cải thiện khả năng tra cứu tài liệu hành chính, giáo dục và hướng dẫn công ích tiếng Việt với chi phí CPU và khả năng kiểm toán.",
    global_transfer="Evidence contract, access filtering, abstention và BEIR-style evaluation chuyển giao được sang corpus/ngôn ngữ khác.",
    input_contract=(
        "RetrievalQuery gồm query_id, văn bản tiếng Việt, corpus version, access scope và top-k.",
        "Corpus manifest/chunk giữ document_id, chunk_id, source URI, version, access scope và checksum.",
        "Retriever/fusion config bất biến; query set/qrels tách khỏi tuning.",
    ),
    output_contract=(
        "EvidenceHit theo rank với document/chunk ID, score, source URI và access scope.",
        "Citation chỉ trỏ tới retrieved chunk thuộc active corpus; thiếu bằng chứng thì abstain.",
        "Run artifact báo retrieval metric, citation integrity, invalid/abstention rate, latency và hashes.",
    ),
    exclusions=(
        "Không dùng LLM/agent để che retrieval yếu; baseline không cần generation.",
        "Không truy xuất corpus riêng hoặc vượt access scope.",
        "Không tuyên bố metric/thể lệ DSC chính thức nếu chưa có organizer specification.",
    ),
    stages=(
        "manifest",
        "access_validate",
        "normalize_chunk",
        "lexical_retrieve",
        "fusion",
        "citation_validate",
        "abstain",
        "evaluate",
        "artifact",
        "ui",
    ),
    baselines=(
        "BM25",
        "word/character TF-IDF",
        "Reciprocal Rank Fusion",
        "extractive evidence + deterministic abstention",
    ),
    candidates=(
        "ViRE-style Vietnamese dense retrieval",
        "VN-MTEB-vetted embedding",
        "bounded reranker",
        "bounded agent policy sau retrieval gate",
    ),
    metrics=(
        "nDCG@10",
        "MRR/Recall@K",
        "citation integrity",
        "unauthorized-hit rate",
        "abstention precision/coverage",
        "invalid-output rate",
        "p50/p95 latency",
    ),
    split_policy="Group theo source/document/template/entity và hold-out theo thời gian khi phù hợp; audit near-duplicate trước indexing.",
    falsification=(
        "Dense/rerank gain biến mất trên source-held-out hoặc không vượt BM25/RRF ở cùng budget.",
        "Citation integrity/access scope không đạt 100% trên test bắt buộc.",
        "Agent tăng lỗi hoặc latency nhưng không tăng retrieval/citation quality.",
    ),
    hardware=(
        "Tier 0 CPU: chunking, BM25/TF-IDF/RRF, citation/access tests.",
        "Tier 1 RTX 2050 4 GB: frozen embedding/reranker probe trên candidate top-k.",
        "Tier 2 lab/cloud: fine-tuning nhỏ chỉ sau benchmark, allowlist và resource gate.",
    ),
    data_candidates=(
        Candidate(
            "Public Vietnamese document pilot",
            "https://data.gov.vn/",
            "Vietnam-first candidate",
            "Chọn bộ cụ thể có license/terms, snapshot và qrels; không crawl mặc định",
        ),
        Candidate(
            "BEIR",
            "https://openreview.net/forum?id=wCu6T5xFjeJ",
            "international evaluation protocol",
            "Dùng protocol/subset có điều khoản rõ; không thay corpus Việt Nam",
        ),
    ),
    model_candidates=(
        Candidate(
            "ViRE",
            "https://github.com/longstnguyen/ViRE",
            "Vietnamese dense candidate",
            "Xác minh model card/license/revision và benchmark parity",
        ),
        Candidate(
            "VN-MTEB candidates",
            "https://aclanthology.org/2026.findings-eacl.86/",
            "embedding selection evidence",
            "Chọn model chỉ sau task/license/resource audit",
        ),
    ),
    sources=(
        Source(
            "BEIR",
            "https://openreview.net/forum?id=wCu6T5xFjeJ",
            "Heterogeneous IR evaluation và zero-shot caution",
        ),
        Source(
            "ViRE",
            "https://aclanthology.org/2026.findings-eacl.110/",
            "Vietnamese information retrieval prior art",
        ),
        Source(
            "VN-MTEB",
            "https://aclanthology.org/2026.findings-eacl.86/",
            "Vietnamese embedding benchmark evidence",
        ),
        Source(
            "RAG-RewardBench",
            "https://aclanthology.org/2025.findings-acl.877/",
            "Taxonomy citation, abstention và bằng chứng xung đột",
        ),
        Source(
            "Ground Every Sentence",
            "https://aclanthology.org/2025.findings-naacl.55/",
            "Sentence-to-evidence mapping thay vì citation trang chung",
        ),
        Source(
            "RAG evaluation survey 2026",
            "https://aclanthology.org/2026.acl-long.1430/",
            "Khóa metric contract trước benchmark, tránh cherry-pick",
        ),
        Source(
            "ir-measures",
            "https://arxiv.org/abs/2111.13466",
            "Metric implementation consistency",
        ),
        Source(
            "Hugging Face Hub download",
            "https://huggingface.co/docs/huggingface_hub/en/guides/download",
            "Pinned revision/cache service contract",
        ),
    ),
    curriculum_facts=(
        "UIT công bố DS314 là học phần Rút trích và truy vấn thông tin, 3 tín chỉ lý thuyết + 1 tín chỉ thực hành.",
        "Nguồn công khai được rà soát chưa đủ để khẳng định toàn bộ đề cương chi tiết DS314.",
    ),
    curriculum_inferences=(
        "Giữ corpus, indexing, lexical retrieval, qrels, ranking metrics, evidence span và extraction được đánh giá riêng.",
        "Dense/rerank chỉ mở sau BM25 và hard-negative analysis; generation/agent không phải lõi trước citation gate.",
    ),
    curriculum_exclusions=(
        "Không biến DS314 thành chatbot RAG hoặc project LLM thuần.",
        "Không gộp extraction, retrieval và generation thành một metric không truy nguyên được.",
    ),
    curriculum_evidence_limit="Tên, tín chỉ và trạng thái học phần là official fact; phạm vi sâu hơn là suy luận thiết kế cho tới khi có đề cương giảng viên.",
    curriculum_sources=(
        Source(
            "Danh mục môn học UIT",
            "https://student.uit.edu.vn/danh-muc-mon-hoc-dai-hoc",
            "Tên và tín chỉ DS314",
        ),
        Source(
            "Bảng tóm tắt môn học UIT",
            "https://student.uit.edu.vn/content/bang-tom-tat-mon-hoc",
            "Đối chiếu các học phần IR gần kề; không suy thành đề cương DS314",
        ),
        Source(
            "Đào tạo Thị giác máy tính HCMUS",
            "https://www.fit.hcmus.edu.vn/vn/Default.aspx?language=vi-VN&tabid=709",
            "Cho thấy truy vấn thông tin được giữ như năng lực riêng",
        ),
    ),
    research_dependencies=(
        "numpy>=2,<3",
        "scikit-learn>=1.6,<2",
        "rank-bm25>=0.2,<1",
        "ir-measures>=0.3,<1",
        "datasets>=3,<5",
        "transformers>=4.51,<6",
        "sentence-transformers>=4,<6",
        "torch>=2.6,<3",
    ),
    input_type="RetrievalQuery",
    output_type="EvidenceHit",
    contracts_code='''@dataclass(frozen=True, slots=True)\nclass RetrievalQuery:\n    """Identify one scoped Vietnamese retrieval request."""\n\n    query_id: str\n    text: str\n    corpus_version: str\n    access_scope: str\n    top_k: int = 10\n\n    def __post_init__(self) -> None:\n        for name in ("query_id", "text", "corpus_version", "access_scope"):\n            _require(getattr(self, name), name)\n        if self.top_k < 1 or self.top_k > 100:\n            raise ProjectContractError("top_k must be in [1, 100]")\n\n\n@dataclass(frozen=True, slots=True)\nclass EvidenceHit:\n    """Return one authorized, provenance-bearing chunk."""\n\n    document_id: str\n    chunk_id: str\n    rank: int\n    score: float\n    source_uri: str\n    access_scope: str\n\n    def __post_init__(self) -> None:\n        for name in ("document_id", "chunk_id", "source_uri", "access_scope"):\n            _require(getattr(self, name), name)\n        if self.rank < 1:\n            raise ProjectContractError("rank must be positive")\n''',
    domain_adapter_module="official_documents",
    domain_adapter_class="OfficialDocumentAdapter",
    domain_adapter_provider="official-documents",
)


PROJECTS = (CS406, DS111, DS201, DS314)
