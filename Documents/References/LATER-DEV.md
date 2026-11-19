# LATER DEV — Long-range Temporal Representation

Các mục dưới đây chỉ làm sau khi research contract, pilot, baseline của hai track
và ma trận R0–R3 đã có bằng chứng. Chúng không được thay critical path trong MAP.

## Simple còn phải đóng

- Loader/validator cho pilot và gold status.
- Unit test point, interval, open interval, month/day và ambiguous time.
- Deterministic BM25/TF-IDF baseline và metric report của Track A.
- Replay Ticktack `published-equation` và scalar/Fourier control của Track B.
- One-command smoke trên CPU.
- Manifest cho source/split/config/seed/code revision.

## Development phục vụ nghiên cứu

- Training/evaluation CLI có resume và checkpoint selection cho từng track.
- Multi-seed runner, paired bootstrap/confidence interval.
- Slice/error report tự sinh.
- Equation-to-code consistency tests.
- Reproduction script cho baseline paper.
- Artifact registry nhẹ: config, metric, prediction và figure hash.
- Environment lock và clean replay.

## Chỉ làm sau khi claim đứng vững

- Demo visualization cho interval/alignment.
- API hoặc notebook tương tác để trình bày.
- Model compression, batching và deployment.
- Mở rộng domain/ngôn ngữ, continual learning hoặc online inference.
- Calendar đặc thù, recurrence phức tạp và temporal knowledge graph.

## Không cần để NCKH đạt 100%

GraphRAG, vector database, ensemble retrieval, multi-tenant auth, queue,
microservice, Kubernetes, product analytics và kiến trúc SaaS. Các phần này thuộc
capstone/product hoặc khóa luận tiếp theo, không tạo thêm scientific evidence cho
hai hypothesis hiện tại.

## Mức hoàn thiện

Simple hoàn tất khi mỗi track có data contract, baseline, metric và error case.
Development hoàn tất khi R0–R3, ablation, statistics và reproduction packet chạy
lại từ môi trường sạch. Research hoàn tất khi paper claims khớp measured artifacts
và limitations; không yêu cầu production hosting.
