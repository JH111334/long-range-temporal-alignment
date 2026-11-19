"""Render changed DOCX deliverables with LibreOffice and inspect PDFs with PyMuPDF."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz
from PIL import Image, ImageDraw
from rebuild_knowledge_library import AIC_OUTPUTS, catalog_outputs

COURSE_REPOS = (
    "CS406-Visual-Asset-Retrieval",
    "DS111-Retrieval-Data-Analysis",
    "DS201-Temporal-Retrieval-Learning",
    "DS314-Evidence-Grounded-Retrieval",
)
SOFFICE = Path(r"C:\Program Files\LibreOffice\program\soffice.com")


class RenderError(RuntimeError):
    """Report a conversion, pagination, or visual-preview failure."""


@dataclass(frozen=True, slots=True)
class RenderResult:
    """Store compact evidence for one rendered DOCX."""

    label: str
    source: str
    pages: int
    source_bytes: int
    pdf_bytes: int
    text_characters: int
    minimum_page_characters: int
    status: str


def expected_documents(library_root: Path, course_root: Path, scope: str) -> list[tuple[str, Path]]:
    """Return exactly 49 main, 15 AIC, and 4 course deliverables."""
    main = [
        ("library-main", library_root / relative)
        for relative in catalog_outputs(library_root).values()
    ]
    aic = [("library-aic", library_root / relative) for relative in map(Path, AIC_OUTPUTS.values())]
    course = [
        (
            "course",
            course_root / repo / "Documents/References/PROJECT-KNOWLEDGE.docx",
        )
        for repo in COURSE_REPOS
    ]
    documents = {"all": main + aic + course, "library": main + aic, "course": course}[scope]
    if len(main) != 49 or len(aic) != 15 or len(course) != 4:
        raise RenderError("Sai số lượng deliverable dự kiến.")
    missing = [str(path) for _, path in documents if not path.is_file()]
    if missing:
        raise RenderError("Thiếu DOCX: " + ", ".join(missing))
    return documents


def render_one(
    index: int, label: str, source: Path, output_root: Path
) -> tuple[RenderResult, Path]:
    """Render one DOCX to an isolated PDF directory and inspect every page."""
    render_dir = output_root / "pdf" / f"{index:03d}"
    render_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            str(SOFFICE),
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(render_dir),
            str(source),
        ],
        check=False,
        capture_output=True,
        timeout=120,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise RenderError(f"LibreOffice lỗi {source}: {stderr}")
    pdf = render_dir / f"{source.stem}.pdf"
    if not pdf.is_file() or pdf.stat().st_size < 1_000:
        raise RenderError(f"Không tạo được PDF hợp lệ: {source}")

    with fitz.open(pdf) as document:
        if document.page_count < 1:
            raise RenderError(f"PDF không có trang: {source}")
        page_characters = [len(page.get_text("text").strip()) for page in document]
        if any(count == 0 for count in page_characters):
            raise RenderError(f"PDF có trang trắng hoàn toàn: {source}; chars={page_characters}")
        first_page = document[0]
        pixmap = first_page.get_pixmap(matrix=fitz.Matrix(0.55, 0.55), alpha=False)
        thumbnail = output_root / "thumbs" / f"{index:03d}.png"
        thumbnail.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(thumbnail)
        result = RenderResult(
            label=label,
            source=str(source),
            pages=document.page_count,
            source_bytes=source.stat().st_size,
            pdf_bytes=pdf.stat().st_size,
            text_characters=sum(page_characters),
            minimum_page_characters=min(page_characters),
            status="PASS",
        )
    return result, thumbnail


def contact_sheets(
    results: list[RenderResult], thumbnails: list[Path], output_root: Path
) -> list[Path]:
    """Create compact first-page contact sheets for manual visual inspection."""
    sheets: list[Path] = []
    batch_size = 17
    cell_width, cell_height = 280, 390
    for batch_index, start in enumerate(range(0, len(thumbnails), batch_size), start=1):
        batch = thumbnails[start : start + batch_size]
        canvas = Image.new("RGB", (cell_width * 4, cell_height * 5), "white")
        draw = ImageDraw.Draw(canvas)
        for offset, thumbnail in enumerate(batch):
            result = results[start + offset]
            with Image.open(thumbnail) as image:
                preview = image.convert("RGB")
                preview.thumbnail((250, 330))
                x = (offset % 4) * cell_width + 15
                y = (offset // 4) * cell_height + 38
                canvas.paste(preview, (x, y))
                draw.text(
                    (x, 8 + (offset // 4) * cell_height),
                    f"{start + offset + 1:02d} {result.label} p={result.pages}",
                    fill="black",
                )
        target = output_root / f"contact-sheet-{batch_index}.png"
        canvas.save(target)
        sheets.append(target)
    return sheets


def write_report(results: list[RenderResult], sheets: list[Path], output_root: Path) -> None:
    """Write JSON and Markdown evidence without modifying source repositories."""
    (output_root / "render-results.json").write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = [
        "# Rendered DOCX QA",
        "",
        f"- Kết quả: **{len(results)}/{len(results)} PASS**.",
        f"- Phạm vi render: {len(results)} DOCX; nhóm: {', '.join(sorted({item.label for item in results}))}.",
        "- Gate: LibreOffice headless, PDF tồn tại, không có trang trắng hoàn toàn, PyMuPDF đọc được mọi trang.",
        f"- Contact sheets: {', '.join(path.name for path in sheets)}.",
        "",
        "| # | Nhóm | Trang | Min ký tự/trang | File |",
        "|---:|---|---:|---:|---|",
    ]
    for index, result in enumerate(results, start=1):
        rows.append(
            f"| {index} | {result.label} | {result.pages} | "
            f"{result.minimum_page_characters} | `{result.source}` |"
        )
    (output_root / "RENDER-QA.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse explicit roots and a workspace-local output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--course-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("all", "library", "course"), default="all")
    return parser.parse_args()


def main() -> int:
    """Render all expected deliverables and create visual QA artifacts."""
    args = parse_args()
    if not SOFFICE.is_file():
        raise RenderError(f"Không tìm thấy LibreOffice: {SOFFICE}")
    output = args.output.resolve()
    if output.exists():
        raise RenderError(f"Output QA đã tồn tại, chọn đường dẫn mới: {output}")
    output.mkdir(parents=True)
    documents = expected_documents(
        args.library_root.resolve(), args.course_root.resolve(), args.scope
    )
    results: list[RenderResult] = []
    thumbnails: list[Path] = []
    for index, (label, source) in enumerate(documents, start=1):
        result, thumbnail = render_one(index, label, source, output)
        results.append(result)
        thumbnails.append(thumbnail)
        print(f"PASS {index:02d}/{len(documents)} {source.name} ({result.pages} trang)")
    sheets = contact_sheets(results, thumbnails, output)
    write_report(results, sheets, output)
    print(f"RENDER QA PASSED: {len(results)}/{len(results)}; output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
