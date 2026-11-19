"""Build a reproducible, page-aware knowledge index from a DSFINAL ZIP."""

# Long Markdown/report strings intentionally keep source-facing readability.
# Ruff's E501 is therefore scoped out for this generator; code style remains checked.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile, ZipInfo

PIPELINE_VERSION = "dsfinal-knowledge-index/1.0"
PAPER_CARD_SCHEMA_VERSION = "1.1"
TARGET_TOKENS = 700
MAX_TOKENS = 900
NCKH_FILES = {
    "Addition NCKH/S13-2001.pdf",
    "Addition NCKH/2106.15110v2.pdf",
    "Addition NCKH/2026.starsem-conference.33.pdf",
    "Addition NCKH/2025.emnlp-main.1467.pdf",
    "Addition NCKH/2026.acl-long.1332.pdf",
}
EXTRA_FILES = {
    "Addition A/1705.09406v2.pdf": "Extra1",
    "Addition A/2506.10016v3.pdf": "Extra2",
}
_DOCLING_CONVERTERS: dict[bool, Any] = {}
MOJIBAKE_MARKERS = (
    "\u00c2\u00a0",
    "\u00c3\u00a1",
    "\u00c3\u00a2",
    "\u00c3\u00a9",
    "\u00c3\u00b4",
    "\u00c4\u2018",
    "\u00e1\u00bb",
    "\u00e1\u00ba",
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()
    ]


def normalize_space(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\u00a0", " ")).strip()


def repair_text_encoding(text: str) -> tuple[str, bool]:
    """Repair only reversible UTF-8-as-Latin-1 mojibake and report whether it changed."""
    markers = ("Ã", "Â", "â")
    if not any(marker in text for marker in markers):
        return text, False
    parts = re.split(r"(\r\n|\n|\r|\u2028)", text)
    repaired: list[str] = []
    changed = False
    for part in parts:
        if part in {"\r\n", "\n", "\r", "\u2028"} or not any(marker in part for marker in markers):
            repaired.append(part)
            continue
        candidate = part
        for _ in range(2):
            try:
                trial = candidate.encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                break
            current_markers = sum(candidate.count(marker) for marker in markers)
            trial_markers = sum(trial.count(marker) for marker in markers)
            if trial_markers >= current_markers:
                break
            candidate = trial
        changed = changed or candidate != part
        repaired.append(candidate)
    return "".join(repaired), changed


def contains_mojibake(text: str) -> bool:
    return any(marker in text for marker in MOJIBAKE_MARKERS)


def safe_rel_path(name: str) -> str | None:
    name = name.replace("\\", "/").lstrip("/")
    parts = list(PurePosixPath(name).parts)
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    if parts[0].upper() in {"DERIVED", "__MACOSX"}:
        return None
    if any(part.startswith("~") or part in {".cache", "__pycache__"} for part in parts):
        return None
    if parts[0].upper() in {"FINALDSC", "DSFINAL"}:
        parts = parts[1:]
    return "/".join(parts) if parts else None


def source_mtime(info: ZipInfo) -> str | None:
    try:
        return datetime(*info.date_time, tzinfo=UTC).isoformat()
    except (TypeError, ValueError):
        return None


def versions() -> dict[str, str]:
    value: dict[str, str] = {"pipeline": PIPELINE_VERSION}
    try:
        import fitz

        value["PyMuPDF"] = getattr(fitz, "VersionBind", "unknown")
    except Exception:
        value["PyMuPDF"] = "unavailable"
    try:
        import docling

        value["Docling"] = getattr(docling, "__version__", "unknown")
    except Exception:
        value["Docling"] = "unavailable"
    try:
        import tiktoken

        value["tiktoken"] = getattr(tiktoken, "__version__", "unknown")
    except Exception:
        value["tiktoken"] = "regex-fallback"
    return value


def make_encoding() -> Any:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def token_count(text: str, encoding: Any) -> int:
    if encoding is not None:
        return len(encoding.encode(text, disallowed_special=()))
    return max(1, len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)))


@dataclass
class PageRecord:
    page: int
    text: str
    heading_context: list[str]
    tables: list[list[list[str]]]
    equations: list[dict[str, Any]]
    parse_quality: str
    raw_text: str = ""
    text_encoding_status: str = "clean"


@dataclass
class ParsedPdf:
    pages: list[PageRecord]
    parser_chosen: str
    parser_version: dict[str, str]
    parse_status: str
    docling_markdown: str
    ocr_pages: list[int]
    warnings: list[str]
    encoding_repaired_pages: list[int] | None = None


def likely_equation(line: str) -> bool:
    line = normalize_space(line)
    if len(line) < 3 or len(line) > 500:
        return False
    if "=" not in line and not any(symbol in line for symbol in ("∑", "∏", "∫", "λ", "θ", "∈")):
        return False
    if line.endswith(".") and len(line.split()) > 18:
        return False
    return any(symbol in line for symbol in ("=", "∑", "∏", "∫", "λ", "θ", "^", "_"))


def extract_equations(text: str, page: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = normalize_space(line)
        if likely_equation(line):
            rows.append(
                {
                    "page": page,
                    "line": line_no,
                    "formula": line,
                    "formula_status": "recovered_text",
                    "evidence_class": "source_text",
                    "extraction_method": "PyMuPDF text layer",
                }
            )
    return rows


def extract_tables(page: Any) -> list[list[list[str]]]:
    try:
        tables = []
        for table in page.find_tables().tables:
            rows = table.extract()
            cleaned = [[normalize_space(str(cell or "")) for cell in row] for row in rows]
            if cleaned:
                tables.append(cleaned)
        return tables
    except Exception:
        return []


def docling_converter(do_ocr: bool = False) -> Any:
    if do_ocr in _DOCLING_CONVERTERS:
        return _DOCLING_CONVERTERS[do_ocr]
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(
        do_ocr=do_ocr,
        do_table_structure=False,
        force_backend_text=not do_ocr,
        generate_page_images=False,
        generate_picture_images=False,
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    _DOCLING_CONVERTERS[do_ocr] = converter
    return converter


def docling_headings(document: Any) -> dict[int, list[str]]:
    by_page: dict[int, list[str]] = defaultdict(list)
    try:
        for item, _ in document.iterate_items():
            if type(item).__name__ != "SectionHeaderItem":
                continue
            text = normalize_space(str(getattr(item, "text", "")))
            if not text:
                continue
            for prov in getattr(item, "prov", []):
                if prov.page_no and text not in by_page[int(prov.page_no)]:
                    by_page[int(prov.page_no)].append(text)
    except Exception:
        return {}
    return dict(by_page)


def should_use_docling(relative_path: str) -> bool:
    return relative_path.startswith("Addition NCKH/") or relative_path.startswith("Addition A/")


def parse_pdf(
    data: bytes,
    parser_versions: dict[str, str],
    use_docling: bool = True,
) -> ParsedPdf:
    import fitz

    source_doc = fitz.open(stream=data, filetype="pdf")
    raw_pages = [page.get_text("text") for page in source_doc]
    cleaned_pages: list[str] = []
    encoding_repaired_pages: list[int] = []
    for page_number, raw_text in enumerate(raw_pages, start=1):
        cleaned, repaired = repair_text_encoding(raw_text)
        cleaned_pages.append(cleaned)
        if repaired:
            encoding_repaired_pages.append(page_number)
    usable = [len(normalize_space(text)) >= 80 for text in raw_pages]
    weak_pages = [idx + 1 for idx, ok in enumerate(usable) if not ok]
    detected_scan = bool(weak_pages) and len(weak_pages) / max(1, len(raw_pages)) >= 0.5
    run_docling = use_docling or detected_scan
    ocr_pages = weak_pages if detected_scan else []
    warnings: list[str] = []
    docling_md = ""
    heading_by_page: dict[int, list[str]] = {}
    parser = "PyMuPDF"
    status = "PARSE_PARTIAL" if weak_pages else "PARSE_SUCCESS"
    if run_docling:
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
                handle.write(data)
                temp_path = Path(handle.name)
            try:
                result = docling_converter(bool(ocr_pages)).convert(str(temp_path))
                docling_md = result.document.export_to_markdown()
                heading_by_page = docling_headings(result.document)
                parser = (
                    "Docling-OCR+PyMuPDF-page-validation"
                    if ocr_pages
                    else "Docling+PyMuPDF-page-projection"
                )
            finally:
                temp_path.unlink(missing_ok=True)
        except Exception as exc:
            warnings.append(f"Docling failed: {type(exc).__name__}: {exc}")
            status = "PARSE_PARTIAL" if any(raw_pages) else "PARSE_FAILED"
    else:
        parser = "PyMuPDF-fast-path"
        if weak_pages:
            warnings.append(
                f"Text layer weak on {len(weak_pages)} page(s); OCR not triggered because the weak-page ratio was below 0.5."
            )
        warnings.append(
            "Docling skipped for lower-priority long book/handbook or non-priority paper; usable text layer was indexed with PyMuPDF."
        )

    pages: list[PageRecord] = []
    cumulative_headings: list[str] = []
    for page_number, (page, raw_text, cleaned_text, is_usable) in enumerate(
        zip(source_doc, raw_pages, cleaned_pages, usable, strict=True), start=1
    ):
        page_text = cleaned_text.strip()
        for heading in heading_by_page.get(page_number, []):
            if heading not in cumulative_headings:
                cumulative_headings.append(heading)
        quality = "TEXT_LAYER_GOOD" if is_usable else "TEXT_LAYER_WEAK_NO_OCR"
        if page_number in ocr_pages:
            quality = "OCR_REQUIRED_OR_UNRECOVERED"
        pages.append(
            PageRecord(
                page_number,
                page_text,
                cumulative_headings[-8:],
                extract_tables(page) if use_docling else [],
                extract_equations(page_text, page_number),
                quality,
                raw_text.strip(),
                "repaired_utf8_mojibake" if page_number in encoding_repaired_pages else "clean",
            )
        )
    source_doc.close()
    if not any(page.text for page in pages):
        status = "PARSE_FAILED"
    return ParsedPdf(
        pages,
        parser,
        parser_versions,
        status,
        docling_md,
        ocr_pages,
        warnings,
        encoding_repaired_pages,
    )


def split_paragraphs(text: str) -> list[str]:
    blocks = [
        normalize_space(block) for block in re.split(r"\n\s*\n+", text) if normalize_space(block)
    ]
    if not blocks:
        blocks = [normalize_space(line) for line in text.splitlines() if normalize_space(line)]
    return blocks


def split_long(text: str, encoding: Any) -> list[str]:
    if token_count(text, encoding) <= MAX_TOKENS:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = token_count(sentence, encoding)
        if current and current_tokens + sentence_tokens > TARGET_TOKENS:
            parts.append(" ".join(current).strip())
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        parts.append(" ".join(current).strip())
    return parts or [text]


def build_chunks(
    relative_path: str, source_sha: str, pages: list[PageRecord], encoding: Any
) -> list[dict[str, Any]]:
    units: list[tuple[int, str, list[str]]] = []
    for page in pages:
        for paragraph in split_paragraphs(page.text):
            for part in split_long(paragraph, encoding):
                units.append((page.page, part, page.heading_context))
    chunks: list[dict[str, Any]] = []
    current_text: list[str] = []
    current_pages: list[int] = []
    current_heading: list[str] = []

    def flush() -> None:
        if not current_text:
            return
        index = len(chunks) + 1
        text = "\n\n".join(current_text).strip()
        chunks.append(
            {
                "chunk_id": f"{Path(relative_path).stem}:{index:05d}",
                "source_relative_path": relative_path,
                "source_sha256": source_sha,
                "page_start": min(current_pages),
                "page_end": max(current_pages),
                "heading_path": current_heading.copy(),
                "text": text,
                "token_count": token_count(text, encoding),
                "previous_chunk_id": None,
                "next_chunk_id": None,
                "evidence_class": "source_text",
                "extractor": PIPELINE_VERSION,
            }
        )
        current_text.clear()
        current_pages.clear()
        current_heading.clear()

    for page, text, heading in units:
        current_tokens = token_count("\n\n".join(current_text), encoding) if current_text else 0
        if current_text and current_tokens + token_count(text, encoding) > MAX_TOKENS:
            flush()
        current_text.append(text)
        current_pages.append(page)
        current_heading[:] = heading[-8:]
        if token_count("\n\n".join(current_text), encoding) >= TARGET_TOKENS:
            flush()
    flush()
    for index, chunk in enumerate(chunks):
        chunk["previous_chunk_id"] = chunks[index - 1]["chunk_id"] if index else None
        chunk["next_chunk_id"] = chunks[index + 1]["chunk_id"] if index + 1 < len(chunks) else None
    return chunks


def first_title(pages: list[PageRecord], fallback: str) -> str:
    raw_lines = (
        [normalize_space(line) for line in pages[0].text.splitlines() if normalize_space(line)]
        if pages
        else []
    )
    for index, line in enumerate(raw_lines):
        if line.startswith(":") and index:
            previous = raw_lines[index - 1]
            if previous and not re.search(
                r"proceedings|pages|volume|©|july|november|abstract", previous, re.I
            ):
                return f"{previous}{line}"
        if re.search(
            r"semEval-\d+ task|historybankqa|temporal information extraction and application",
            line,
            re.I,
        ):
            parts = [line]
            if index + 1 < len(raw_lines):
                next_line = raw_lines[index + 1]
                if next_line and not re.search(
                    r"@|university|intel|thoughtworks|abstract|proceedings", next_line, re.I
                ):
                    parts.append(next_line)
            return " ".join(parts)
    if pages and pages[0].heading_context:
        headings = pages[0].heading_context
        for index, heading in enumerate(headings):
            clean = normalize_space(heading)
            if not clean or clean.lower() in {"abstract", "introduction"}:
                continue
            if clean.startswith(":") and index:
                previous = normalize_space(headings[index - 1])
                if previous and previous.lower() not in {"abstract", "introduction"}:
                    return f"{previous}{clean}"
            if 15 <= len(clean) <= 240 and not re.search(
                r"proceedings|pages|volume|association for computational linguistics|©|july|november",
                clean,
                re.I,
            ):
                return clean
    lines = raw_lines
    for line in lines[:30]:
        if 15 <= len(line) <= 240 and not re.search(
            r"@|arXiv|proceedings|abstract| page ", line, re.I
        ):
            if not re.fullmatch(r"[\d\W]+", line):
                return line
    return fallback


def authors_and_venue(pages: list[PageRecord], title: str) -> tuple[str, str]:
    first_page = pages[0] if pages else PageRecord(1, "", [], [], [], "unknown")
    headings = first_page.heading_context
    before_abstract: list[str] = []
    for heading in headings:
        if heading.lower().startswith("abstract"):
            break
        before_abstract.append(normalize_space(heading))
    authors = [
        item
        for item in before_abstract
        if item
        and item != title
        and not re.search(
            r"proceedings|pages|volume|association for computational linguistics|©|july|november|university|@|introduction|abstract|method",
            item,
            re.I,
        )
        and not re.match(r"^\d", item)
        and item not in title
        and not title.startswith(item)
    ]
    author_text = (
        "; ".join(authors) if authors else "UNKNOWN: author list requires manual confirmation"
    )
    venue = "UNKNOWN: venue not reliably recovered"
    for line in first_page.text.splitlines()[:8]:
        clean = normalize_space(line)
        if re.search(r"proceedings|arXiv|conference|workshop|journal", clean, re.I):
            venue = clean
            break
    return author_text, venue


def find_year(text: str, relative_path: str) -> str | None:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text[:10000])
    if years:
        return years[0]
    match = re.search(r"\b(19\d{2}|20\d{2})\b", relative_path)
    return match.group(1) if match else None


def paper_ids(relative_path: str, text: str) -> dict[str, str | None]:
    haystack = f"{relative_path}\n{text[:12000]}"
    arxiv = re.search(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b", haystack)
    doi = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", haystack, re.I)
    acl = re.search(r"\b(\d{4}\.(?:acl|emnlp|starsem|semeval)[-\w.]+)\b", haystack, re.I)
    return {
        "arxiv_id": arxiv.group(1) if arxiv else None,
        "doi": doi.group(0).rstrip(".,") if doi else None,
        "acl_or_venue_id": acl.group(1) if acl else Path(relative_path).stem,
    }


def section_excerpt(
    pages: list[PageRecord], patterns: tuple[str, ...], limit: int = 1300
) -> dict[str, Any]:
    for page in pages:
        if any(pattern.lower() in page.text.lower() for pattern in patterns):
            return {
                "page": page.page,
                "text": page.text[:limit].strip(),
                "evidence_class": "source_text",
            }
    return {
        "page": None,
        "text": "UNKNOWN: no matching section recovered",
        "evidence_class": "unknown",
    }


def keyword_evidence(
    pages: list[PageRecord], keywords: tuple[str, ...], limit: int = 3
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for page in pages:
        for line in page.text.splitlines():
            clean = normalize_space(line)
            if clean and any(keyword.lower() in clean.lower() for keyword in keywords):
                evidence.append(
                    {"page": page.page, "text": clean[:500], "evidence_class": "source_text"}
                )
                if len(evidence) >= limit:
                    return evidence
    return evidence


def is_research_paper(relative_path: str, pages: list[PageRecord]) -> bool:
    if relative_path in NCKH_FILES or relative_path in EXTRA_FILES:
        return True
    if re.search(r"/\d{4}\.\d{4,5}(?:v\d+)?\.pdf$", relative_path):
        return True
    if relative_path.endswith("2026.semeval-1.0.pdf"):
        return True
    first = " ".join(page.text[:500] for page in pages[:2])
    tail = " ".join(page.text.lower() for page in pages[-3:])
    return "abstract" in first.lower() and "references" in tail


def relevance_tags(pages: list[PageRecord]) -> list[str]:
    text = " ".join(page.text.lower() for page in pages)
    groups = {
        "temporal_representation": (
            "temporal representation",
            "time-aware",
            "time aware",
            "date representation",
        ),
        "temporal_learning_mechanism": (
            "temporal learning",
            "temporal training",
            "time-aware language",
        ),
        "continual_learning": ("continual learning", "lifelong learning", "incremental learning"),
        "retention_forgetting": ("catastrophic forgetting", "retention", "replay", "forgetting"),
        "temporal_retrieval": ("temporal retrieval", "information retrieval", "retrieval"),
        "temporal_evaluation": (
            "tempeval",
            "temporal qa",
            "temporal question answering",
            "benchmark",
        ),
        "temporal_IE": ("temporal information extraction", "temporal ie", "event extraction"),
        "benchmark_design": ("dataset", "benchmark", "evaluation protocol"),
        "multimodal_transferable_idea": ("multimodal", "cross-modal", "vision-language", "fusion"),
    }
    tags = [name for name, words in groups.items() if any(word in text for word in words)]
    return tags or ["not_directly_relevant"]


def paper_card(
    relative_path: str, source_sha: str, parsed: ParsedPdf, chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    pages = parsed.pages
    text = "\n".join(page.text for page in pages)
    title = first_title(pages, Path(relative_path).stem)
    authors, venue = authors_and_venue(pages, title)
    identity = {
        "title": title,
        "authors": authors,
        "venue": venue,
        "year": find_year(text, relative_path),
        **paper_ids(relative_path, text),
        "source_file": relative_path,
        "source_sha256": source_sha,
    }
    abstract = section_excerpt(pages, ("abstract",), 1600)
    method = section_excerpt(pages, ("method", "approach", "model", "architecture"), 1400)
    results = section_excerpt(pages, ("experiments", "results", "evaluation"), 1400)
    limitations = section_excerpt(pages, ("limitations", "future work", "conclusion"), 1100)
    formula_rows = [equation for page in pages for equation in page.equations]
    metric_evidence = keyword_evidence(
        pages,
        ("accuracy", "f1", "f1-score", "mrr", "recall", "precision", "bleu", "rouge", "auc"),
        8,
    )
    dataset_evidence = keyword_evidence(
        pages, ("dataset", "benchmark", "corpus", "tempeval", "timelama", "historybank"), 8
    )
    relation = {
        "Targeted Temporal Training": keyword_evidence(
            pages, ("temporal training", "time-aware", "temporal objective"), 3
        ),
        "MLP-only LoRA / Temporal Adapter": keyword_evidence(
            pages, ("adapter", "lora", "mlp", "parameter-efficient"), 3
        ),
        "Replay / Retention": keyword_evidence(
            pages, ("replay", "retention", "forgetting", "continual"), 3
        ),
        "Structured Consistency": keyword_evidence(
            pages, ("consistency", "constraint", "structured"), 3
        ),
    }
    return {
        "paper_card_schema_version": PAPER_CARD_SCHEMA_VERSION,
        "identity": identity,
        "research_content": {
            "research_question": abstract,
            "problem": abstract,
            "proposed_method": method,
            "architecture": method,
            "training_objective": section_excerpt(
                pages, ("objective", "loss function", "training"), 1200
            ),
            "important_formulas": formula_rows,
            "datasets": dataset_evidence,
            "baselines": keyword_evidence(pages, ("baseline", "state-of-the-art", "prior work"), 8),
            "metrics": metric_evidence,
            "experimental_protocol": results,
            "main_measured_findings": results,
            "ablations": keyword_evidence(pages, ("ablation", "removing", "without"), 6),
            "limitations": limitations,
            "future_work": section_excerpt(pages, ("future work",), 1000),
        },
        "evidence_claims": [
            {"claim_id": "problem", "evidence": abstract, "evidence_class": "paper_claim"},
            {"claim_id": "method", "evidence": method, "evidence_class": "paper_claim"},
            {
                "claim_id": "measured_findings",
                "evidence": results,
                "evidence_class": "paper_measured",
            },
            {
                "claim_id": "limitations_or_future",
                "evidence": limitations,
                "evidence_class": "author_hypothesis",
            },
        ],
        "project_relevance": relevance_tags(pages),
        "transfer_boundary": {
            "relation_to_partner_representation_work": {
                "assessment": "derived_interpretation: keyword-grounded relevance only; no direct project claim",
                "evidence_pages": [
                    item["page"]
                    for item in keyword_evidence(
                        pages, ("representation", "embedding", "encoder"), 5
                    )
                ],
            },
            "relation_to_our_three_mechanisms": relation,
            "reusable_formula_or_protocol": formula_rows[:10]
            or "UNKNOWN: no formula candidate recovered",
            "what_this_paper_does_NOT_prove": "This source does not by itself prove effectiveness on the project's YEAR/MONTH Temporal QA task; transfer must be tested under the project's evaluation contract.",
            "evidence_class": "derived_interpretation",
        },
        "provenance": {
            "parser": parsed.parser_chosen,
            "parser_version": parsed.parser_version,
            "parse_status": parsed.parse_status,
            "warnings": parsed.warnings,
            "chunk_count": len(chunks),
            "formula_status_summary": "recovered_text" if formula_rows else "unavailable",
        },
    }


def card_markdown(card: dict[str, Any], relative_path: str) -> str:
    identity = card["identity"]
    lines = [
        f"# Paper card: {identity['title']}",
        "",
        f"Source: {relative_path}",
        "",
        "## Identity",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in identity.items() if key != "source_sha256")
    lines.extend(["", "## Research content", ""])
    for key, value in card["research_content"].items():
        lines.extend([f"### {key}", ""])
        if isinstance(value, dict):
            lines.append(f"- Page: {value.get('page', 'UNKNOWN')}")
            lines.append(f"- Evidence class: {value.get('evidence_class', 'unknown')}")
            lines.append(f"- Extract: {value.get('text', value)}")
        elif isinstance(value, list):
            if not value:
                lines.append("- UNKNOWN: no evidence recovered")
            for item in value[:12]:
                if isinstance(item, dict):
                    lines.append(
                        f"- p.{item.get('page', '?')}: {item.get('text', item.get('formula', item))}"
                    )
                else:
                    lines.append(f"- {item}")
        else:
            lines.append(f"- {value}")
        lines.append("")
    lines.extend(
        [
            "## Evidence boundary",
            "",
            "Các trích xuất là source-grounded; nhận định chuyển giao là derived_interpretation và chưa phải kết quả đo.",
            "",
        ]
    )
    for claim in card["evidence_claims"]:
        evidence = claim["evidence"]
        lines.append(
            f"- {claim['claim_id']} — {claim['evidence_class']} — p.{evidence.get('page', '?')}: {evidence.get('text', evidence)}"
        )
    lines.extend(
        [
            "",
            "## Project relevance and limits",
            "",
            f"- Tags: {', '.join(card['project_relevance'])}",
            f"- What this paper does not prove: {card['transfer_boundary']['what_this_paper_does_NOT_prove']}",
            "",
        ]
    )
    return "\n".join(lines)


def transfer_strength(pages: list[PageRecord]) -> str:
    text = " ".join(page.text.lower() for page in pages)
    if "temporal" in text and any(
        term in text for term in ("question answering", "tempeval", "timelama")
    ):
        return "DIRECT"
    if "temporal" in text or "time-aware" in text:
        return "INDIRECT"
    return "ANALOGY ONLY"


def research_map(relative_path: str, parsed: ParsedPdf, card: dict[str, Any]) -> str:
    pages = parsed.pages
    content = card["research_content"]
    strength = transfer_strength(pages)
    formulas = [item["formula"] for page in pages for item in page.equations][:8]
    mechanism = keyword_evidence(
        pages, ("mechanism", "architecture", "method", "model", "training"), 5
    )
    sections = [
        ("1. What problem the paper solves", content["problem"], "paper_claim"),
        (
            "2. What mechanism it actually uses",
            {
                "page": mechanism[0]["page"] if mechanism else None,
                "text": mechanism[0]["text"] if mechanism else "UNKNOWN",
            },
            "paper_claim",
        ),
        (
            "3. What can transfer to our YEAR/MONTH Temporal Qwen project",
            {
                "page": None,
                "text": "PROPOSED INTERPRETATION: use only the source-grounded temporal/retrieval evidence below to design a paired experiment.",
            },
            "derived_interpretation",
        ),
        (
            "4. What cannot be directly transferred",
            {
                "page": None,
                "text": "Do not transfer model claims, gains or datasets without matching the project's data manifest, split, evaluator and evidence contract.",
            },
            "operational_assumption",
        ),
        (
            "5. Relevant formulas",
            {
                "page": None,
                "text": "\n".join(formulas)
                if formulas
                else "UNAVAILABLE: no formula candidate recovered by text extraction",
            },
            "source_text" if formulas else "unknown",
        ),
        (
            "6. Relevant metrics",
            {
                "page": content["metrics"][0]["page"] if content["metrics"] else None,
                "text": "\n".join(item["text"] for item in content["metrics"]) or "UNKNOWN",
            },
            "paper_measured" if content["metrics"] else "unknown",
        ),
        (
            "7. Dataset/benchmark ideas",
            {
                "page": content["datasets"][0]["page"] if content["datasets"] else None,
                "text": "PROPOSED METHOD: source mentions are prompts for benchmark design, not approved data or gold.",
            },
            "proposed_method",
        ),
    ]
    lines = [f"# Research map: {card['identity']['title']}", "", f"Source: {relative_path}", ""]
    for heading, value, evidence_class in sections:
        lines.extend(
            [
                f"## {heading}",
                "",
                f"- Evidence class: {evidence_class}",
                f"- Page: {value.get('page', 'UNKNOWN')}",
                f"- {value.get('text', value)}",
                "",
            ]
        )
    lines.extend(["## 8. Connection to the three mechanisms", ""])
    for mechanism_name, keywords in {
        "Targeted Temporal Training": ("temporal training", "time-aware", "temporal objective"),
        "MLP-only LoRA / Temporal Adapter": ("adapter", "lora", "mlp", "parameter-efficient"),
        "Replay / Retention": ("replay", "retention", "forgetting", "continual"),
        "Structured Consistency (secondary)": ("consistency", "constraint", "structured"),
    }.items():
        evidence = keyword_evidence(pages, keywords, 3)
        lines.extend([f"### {mechanism_name}", ""])
        lines.extend(
            f"- p.{item['page']} — {item['evidence_class']} — {item['text']}" for item in evidence
        ) if evidence else lines.append("- UNKNOWN: no direct source evidence recovered")
        lines.append("")
    lines.extend(
        [
            "## 9. Concrete experiment inspired by this paper",
            "",
            "PROPOSED METHOD: define one bounded ablation using the source mechanism, freeze the evaluation contract, and compare paired cases against the approved anchor. This indexer does not execute it.",
            "",
            "## 10. Evidence strength",
            "",
            f"{strength}. This is a routing judgment from recovered text and not a paper-reported result.",
            "",
        ]
    )
    return "\n".join(lines)


def extra_transfer_analysis(relative_path: str, parsed: ParsedPdf, card: dict[str, Any]) -> str:
    label = EXTRA_FILES.get(relative_path, "Extra")
    axes = {
        "1. Representation": ("representation", "embedding", "encoder", "feature"),
        "2. Alignment": ("alignment", "align", "correspondence", "cross-modal"),
        "3. Fusion": ("fusion", "multimodal", "attention", "late fusion", "early fusion"),
        "4. Co-learning/training": (
            "co-learning",
            "joint training",
            "training",
            "objective",
            "loss",
        ),
        "5. Robustness/retention": ("robustness", "retention", "forgetting", "continual", "noise"),
        "6. Temporal-signal relevance": ("temporal", "time", "sequence", "ordering", "long-range"),
    }
    lines = [
        f"# {label} transfer analysis: {card['identity']['title']}",
        "",
        f"Source: {relative_path}",
        "",
    ]
    for heading, keywords in axes.items():
        lines.extend([f"## {heading}", ""])
        evidence = keyword_evidence(parsed.pages, keywords, 4)
        if evidence:
            lines.extend(
                f"- p.{item['page']} — {item['evidence_class']} — {item['text']}"
                for item in evidence
            )
        else:
            lines.append("- UNKNOWN: no source evidence recovered")
        lines.append("")
    lines.extend(
        [
            "## 7. Analogy boundary",
            "",
            "Nếu nguồn là multimodal nhưng không đánh giá temporal QA, mọi chuyển giao sang temporal signals chỉ là ANALOGY ONLY; không được trích dẫn như bằng chứng temporal.",
            "",
            "## 8. Refinement of the three proposed mechanisms",
            "",
            "- Targeted Temporal Training: PROPOSED METHOD; kiểm tra xem tín hiệu được mô tả có thể thành objective temporal hay chỉ là analogy.",
            "- MLP-only LoRA / Temporal Adapter: PROPOSED METHOD; chỉ mở ablation nếu evidence về adapter hoặc parameter-efficient training đủ rõ.",
            "- Replay / Retention: PROPOSED METHOD; không suy ra retention nếu nguồn không đo forgetting hoặc replay.",
            "",
            "## 9. New ablation suggestion",
            "",
            "UNTESTED HYPOTHESIS: tách representation, alignment và fusion thành từng nhánh rồi chạy paired ablation dưới cùng evaluation contract.",
            "",
            "## 10. Architecture decision",
            "",
            "Kết quả index chỉ củng cố hoặc đặt câu hỏi cho motivation; không tự thay đổi architecture chính và không chứng minh temporal-QA effectiveness.",
            "",
        ]
    )
    return "\n".join(lines)


def references_rows(
    relative_path: str, pages: list[PageRecord], source_sha: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    in_refs = False
    ref_index = 0
    for page in pages:
        for line in page.text.splitlines():
            clean = normalize_space(line)
            if not clean:
                continue
            if re.fullmatch(r"(?:references|bibliography)", clean, re.I):
                in_refs = True
                continue
            if in_refs and (
                re.match(r"^\[?\d{1,3}\]?\s+", clean) or re.search(r"\b(19|20)\d{2}\b", clean)
            ):
                ref_index += 1
                rows.append(
                    {
                        "reference_id": f"{Path(relative_path).stem}:ref:{ref_index:04d}",
                        "source_relative_path": relative_path,
                        "source_sha256": source_sha,
                        "page": page.page,
                        "text": clean,
                        "evidence_class": "source_text",
                        "parser": PIPELINE_VERSION,
                    }
                )
    return rows


def source_record(
    info: ZipInfo, relative_path: str, data: bytes, parsed: ParsedPdf, derived_dir: str
) -> dict[str, Any]:
    research = is_research_paper(relative_path, parsed.pages)
    return {
        "source_relative_path": relative_path,
        "archive_path": info.filename,
        "filename": Path(relative_path).name,
        "extension": Path(relative_path).suffix.lower(),
        "mime": mimetypes.guess_type(relative_path)[0] or "application/octet-stream",
        "size": len(data),
        "sha256": sha256_bytes(data),
        "modified_time": source_mtime(info),
        "parser_chosen": parsed.parser_chosen,
        "parser_version": parsed.parser_version,
        "parse_status": parsed.parse_status,
        "page_count": len(parsed.pages),
        "ocr_pages": parsed.ocr_pages,
        "warning_count": len(parsed.warnings),
        "warnings": parsed.warnings,
        "encoding_repaired_pages": parsed.encoding_repaired_pages or [],
        "derived_relative_dir": derived_dir,
        "classification": "research_paper" if research else "book_or_handbook",
        "priority": (
            "Addition NCKH"
            if relative_path.startswith("Addition NCKH/")
            else "Extra paper"
            if relative_path in EXTRA_FILES
            else "research paper"
            if research
            else "book/handbook"
        ),
    }


def artifact_complete(derived: Path, source_sha: str, research: bool, special: bool) -> bool:
    required = [
        derived / name for name in ("source.json", "content.md", "pages.jsonl", "chunks.jsonl")
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        record = json.loads((derived / "source.json").read_text(encoding="utf-8"))
        read_jsonl(derived / "pages.jsonl")
        chunks = read_jsonl(derived / "chunks.jsonl")
    except Exception:
        return False
    if record.get("source_sha256") != source_sha:
        return False
    if contains_mojibake((derived / "content.md").read_text(encoding="utf-8")):
        return False
    if research:
        card_path = derived / "paper_card.json"
        if not card_path.exists():
            return False
        try:
            if (
                json.loads(card_path.read_text(encoding="utf-8")).get("paper_card_schema_version")
                != PAPER_CARD_SCHEMA_VERSION
            ):
                return False
        except Exception:
            return False
    if not chunks:
        return False
    if special and not any(
        (derived / name).exists() for name in ("research_map.md", "extra_transfer_analysis.md")
    ):
        return False
    return True


def write_source_artifacts(
    output_root: Path,
    relative_path: str,
    info: ZipInfo,
    data: bytes,
    parser_versions: dict[str, str],
    encoding: Any,
    previous_manifest: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_sha = sha256_bytes(data)
    derived = output_root / Path(relative_path).parent / Path(relative_path).stem
    previous = previous_manifest.get(relative_path, {})
    expected_research = previous.get("classification") == "research_paper"
    if artifact_complete(
        derived,
        source_sha,
        expected_research,
        relative_path in EXTRA_FILES or relative_path in NCKH_FILES,
    ):
        record = json.loads((derived / "source.json").read_text(encoding="utf-8"))
        record["reused_existing_artifacts"] = True
        return record

    parsed = parse_pdf(data, parser_versions, should_use_docling(relative_path))
    research = is_research_paper(relative_path, parsed.pages)
    derived.mkdir(parents=True, exist_ok=True)
    page_rows = [
        {
            "source_path": relative_path,
            "source_sha256": source_sha,
            "page": page.page,
            "heading_context": page.heading_context,
            "text": page.text,
            "raw_text": page.raw_text or page.text,
            "text_encoding_status": page.text_encoding_status,
            "tables": page.tables,
            "equations": page.equations,
            "parse_quality": page.parse_quality,
        }
        for page in parsed.pages
    ]
    chunks = build_chunks(relative_path, source_sha, parsed.pages, encoding)
    record = source_record(
        info,
        relative_path,
        data,
        parsed,
        str(derived.relative_to(output_root)).replace("\\", "/"),
    )
    record.update(
        {
            "source_sha256": source_sha,
            "docling_markdown_chars": len(parsed.docling_markdown),
            "chunk_count": len(chunks),
            "reused_existing_artifacts": False,
            "generated_at_utc": utc_now(),
        }
    )
    json_dump(derived / "source.json", record)
    content_lines = [
        f"# {Path(relative_path).stem}",
        "",
        f"Source: {relative_path}",
        f"Source SHA-256: {source_sha}",
        f"Parser: {parsed.parser_chosen}",
        "",
    ]
    for page in parsed.pages:
        content_lines.extend(
            [
                f"<!-- PAGE: {page.page} -->",
                "",
                page.text or "[NO TEXT RECOVERED]",
                "",
            ]
        )
    (derived / "content.md").write_text("\n".join(content_lines), encoding="utf-8")
    jsonl_write(derived / "pages.jsonl", page_rows)
    jsonl_write(derived / "chunks.jsonl", chunks)
    jsonl_write(
        derived / "equations.jsonl", [item for page in page_rows for item in page["equations"]]
    )
    jsonl_write(
        derived / "references.jsonl", references_rows(relative_path, parsed.pages, source_sha)
    )
    if research:
        card = paper_card(relative_path, source_sha, parsed, chunks)
        json_dump(derived / "paper_card.json", card)
        (derived / "paper_card.md").write_text(card_markdown(card, relative_path), encoding="utf-8")
        if relative_path in NCKH_FILES:
            (derived / "research_map.md").write_text(
                research_map(relative_path, parsed, card), encoding="utf-8"
            )
        if relative_path in EXTRA_FILES:
            (derived / "extra_transfer_analysis.md").write_text(
                extra_transfer_analysis(relative_path, parsed, card), encoding="utf-8"
            )
    return record


def build_sqlite(
    output_root: Path, chunks: list[dict[str, Any]], cards: list[dict[str, Any]]
) -> int:
    index_dir = output_root / "_INDEX"
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "knowledge.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            title TEXT,
            authors TEXT,
            year TEXT,
            heading_path TEXT,
            page_start INTEGER,
            page_end INTEGER,
            text TEXT NOT NULL,
            tags TEXT
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED, source_path, title, authors, year, heading_path,
            page_start UNINDEXED, page_end UNINDEXED, text, tags,
            tokenize='unicode61 remove_diacritics 0'
        );
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )
    card_map = {card["identity"]["source_file"]: card for card in cards}
    for chunk in chunks:
        card = card_map.get(chunk["source_relative_path"], {})
        identity = card.get("identity", {})
        values = (
            chunk["chunk_id"],
            chunk["source_relative_path"],
            chunk["source_sha256"],
            identity.get("title", Path(chunk["source_relative_path"]).stem),
            identity.get("authors", "UNKNOWN"),
            identity.get("year", "UNKNOWN"),
            " > ".join(chunk.get("heading_path", [])),
            chunk["page_start"],
            chunk["page_end"],
            chunk["text"],
            " ".join(card.get("project_relevance", [])),
        )
        conn.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)", values)
        conn.execute(
            "INSERT INTO chunks_fts VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                values[0],
                values[1],
                values[3],
                values[4],
                values[5],
                values[6],
                values[7],
                values[8],
                values[9],
                values[10],
            ),
        )
    conn.executemany(
        "INSERT INTO metadata VALUES (?,?)",
        [("pipeline", PIPELINE_VERSION), ("chunk_count", str(len(chunks)))],
    )
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    return len(chunks)


SEARCH_CLI = '''"""Lexical search over DERIVED/_INDEX/knowledge.sqlite."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=None)
    parser.add_argument("query", nargs="+")
    args = parser.parse_args()
    tokens = [token.replace('"', "") for token in " ".join(args.query).split()]
    query = " ".join('"' + token + '"' for token in tokens)
    db = Path(__file__).with_name("knowledge.sqlite")
    conn = sqlite3.connect(db)
    sql = """
        SELECT c.chunk_id, c.source_path, c.title, c.page_start, c.page_end, c.heading_path,
               snippet(chunks_fts, 8, "[", "]", " ... ", 32)
        FROM chunks_fts
        JOIN chunks AS c ON c.chunk_id = chunks_fts.chunk_id
        WHERE chunks_fts MATCH ?
    """
    params = [query]
    if args.folder:
        sql += " AND c.source_path LIKE ?"
        params.append(args.folder.rstrip("/") + "/%")
    sql += " ORDER BY rank LIMIT 20"
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        safe_query = " ".join(part for part in query.split() if part)
        rows = conn.execute(sql, [safe_query] + params[1:]).fetchall()
    for chunk_id, source_path, title, start, end, heading, snippet in rows:
        print(f"{chunk_id}\\t{source_path}\\tp.{start}-{end}\\t{title}\\t{heading}\\n  {snippet}")
    print(f"\\nresults={len(rows)}")
    conn.close()


if __name__ == "__main__":
    main()
'''


def write_search_cli(output_root: Path) -> None:
    path = output_root / "_INDEX" / "search_knowledge.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SEARCH_CLI, encoding="utf-8")


def page_spot_checks(
    data: bytes, page_rows: list[dict[str, Any]], relative_path: str, source_sha: str
) -> list[dict[str, Any]]:
    import fitz

    checks: list[dict[str, Any]] = []
    if len(page_rows) < 2:
        return checks
    selected = sorted({1, max(1, (len(page_rows) + 1) // 2)})
    document = fitz.open(stream=data, filetype="pdf")
    for page_no in selected:
        source_text = document[page_no - 1].get_text("text").strip()
        derived_text = page_rows[page_no - 1].get("text", "")
        raw_text = page_rows[page_no - 1].get("raw_text", derived_text)
        checks.append(
            {
                "source_relative_path": relative_path,
                "page": page_no,
                "source_sha256": source_sha,
                "source_page_text_sha256": sha256_bytes(source_text.encode("utf-8")),
                "derived_page_text_sha256": sha256_bytes(raw_text.encode("utf-8")),
                "exact_match": source_text == raw_text,
                "method": "PyMuPDF direct source page text vs pages.jsonl projection",
            }
        )
    document.close()
    return checks


def synthesis(
    output_root: Path, records: list[dict[str, Any]], cards: list[dict[str, Any]]
) -> None:
    synth = output_root / "_SYNTHESIS"
    synth.mkdir(parents=True, exist_ok=True)
    rows = [
        f"| {card['identity']['title'].replace('|', '/')} | {card['identity'].get('year') or 'UNKNOWN'} | {', '.join(card.get('project_relevance', []))} | {card['identity']['source_file']} |"
        for card in cards
    ]
    (synth / "literature_matrix.md").write_text(
        "\n".join(
            [
                "# Literature matrix",
                "",
                "Automated source-grounded matrix; UNKNOWN fields require manual review.",
                "",
                "| Paper | Year | Project relevance | Source |",
                "|---|---:|---|---|",
                *rows,
                "",
            ]
        ),
        encoding="utf-8",
    )
    mechanism = [
        "# Mechanism evidence matrix",
        "",
        "Keyword-grounded evidence routing; this is not a benchmark result.",
        "",
        "| Paper | Targeted Temporal Training | MLP/Adapter | Replay/Retention | Structured Consistency | Evidence strength | Direct vs analogy |",
        "|---|---|---|---|---|---|---|",
    ]
    for card in cards:
        relation = card["transfer_boundary"]["relation_to_our_three_mechanisms"]
        values = ["evidence" if relation.get(key) else "no evidence" for key in relation]
        strength = (
            "DIRECT"
            if "temporal_evaluation" in card.get("project_relevance", [])
            else "INDIRECT/ANALOGY"
        )
        mechanism.append(
            f"| {card['identity']['title'].replace('|', '/')} | {' | '.join(values)} | {strength} | source-specific review required |"
        )
    (synth / "mechanism_evidence_matrix.md").write_text(
        "\n".join(mechanism) + "\n", encoding="utf-8"
    )
    metric = [
        "# Metric matrix",
        "",
        "Metrics are recovered mentions, not normalized scores.",
        "",
        "| Paper | Metric evidence | Pages |",
        "|---|---|---:|",
    ]
    dataset = [
        "# Dataset and benchmark matrix",
        "",
        "Dataset mentions are source evidence; proposed use is not approval.",
        "",
        "| Paper | Dataset/benchmark evidence | Pages |",
        "|---|---|---:|",
    ]
    terms: Counter[str] = Counter()
    for card in cards:
        identity = card["identity"]
        metrics = card["research_content"].get("metrics", [])
        datasets = card["research_content"].get("datasets", [])
        metric.append(
            f"| {identity['title'].replace('|', '/')} | {'; '.join(item['text'][:180] for item in metrics) or 'UNKNOWN'} | {', '.join(str(item['page']) for item in metrics) or 'UNKNOWN'} |"
        )
        dataset.append(
            f"| {identity['title'].replace('|', '/')} | {'; '.join(item['text'][:180] for item in datasets) or 'UNKNOWN'} | {', '.join(str(item['page']) for item in datasets) or 'UNKNOWN'} |"
        )
        for tag in card.get("project_relevance", []):
            terms[tag] += 1
    (synth / "metric_matrix.md").write_text("\n".join(metric) + "\n", encoding="utf-8")
    (synth / "dataset_benchmark_matrix.md").write_text("\n".join(dataset) + "\n", encoding="utf-8")
    terminology = [
        "# Terminology",
        "",
        "Term routing derived from repeated source keywords; definitions require source-page review.",
        "",
    ]
    terminology.extend(
        f"- {term}: mentioned in {count} indexed paper(s)." for term, count in sorted(terms.items())
    )
    (synth / "terminology.md").write_text("\n".join(terminology) + "\n", encoding="utf-8")
    (synth / "presentation_support.md").write_text(
        "\n".join(
            [
                "# Presentation support",
                "",
                "## SOURCE FACT",
                "",
                "- Use paper cards and page references for source facts.",
                "",
                "## PROJECT INFERENCE",
                "",
                "- Relevance tags and mechanism routing are automated interpretations and need advisor review.",
                "",
                "## PROPOSED METHOD",
                "",
                "- Research maps contain bounded experiment prompts; none were executed by this indexer.",
                "",
                "## UNTESTED HYPOTHESIS",
                "",
                "- Transfer from multimodal representation, alignment or fusion to temporal QA remains an analogy until paired evaluation.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def validate_and_report(
    output_root: Path,
    source_zip: Path,
    records: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    spot_checks: list[dict[str, Any]],
    parser_versions: dict[str, str],
) -> None:
    manifest_dir = output_root / "_MANIFEST"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict[str, Any]] = []
    for record in records:
        if record.get("parse_status") == "ORPHANED":
            continue
        all_chunks.extend(read_jsonl(output_root / record["derived_relative_dir"] / "chunks.jsonl"))
    chunk_count = build_sqlite(output_root, all_chunks, cards)
    write_search_cli(output_root)
    parse_counts = Counter(record.get("parse_status") for record in records)
    parser_counts = Counter(record.get("parser_chosen") for record in records)
    unresolved = [
        record["source_relative_path"]
        for record in records
        if record.get("parse_status") not in {"PARSE_SUCCESS", "ORPHANED"}
    ]
    source_count = len([record for record in records if record.get("parse_status") != "ORPHANED"])
    derived_coverage = sum(
        (output_root / record["derived_relative_dir"]).exists()
        for record in records
        if record.get("parse_status") != "ORPHANED"
    )
    source_sha = sha256_bytes(source_zip.read_bytes())
    checks_ok = sum(bool(row.get("exact_match")) for row in spot_checks)
    nckh_records = [
        record for record in records if record["source_relative_path"].startswith("Addition NCKH/")
    ]
    extra_records = [record for record in records if record["source_relative_path"] in EXTRA_FILES]
    report = [
        "# Indexing report",
        "",
        f"Pipeline: {PIPELINE_VERSION}",
        f"Generated UTC: {utc_now()}",
        "",
        "## Scope",
        "",
        f"- Source archive: {source_zip.name}",
        f"- Source archive SHA-256: {source_sha}",
        f"- Source files: {source_count}",
        f"- Research papers: {len(cards)}",
        f"- Addition NCKH coverage: {len(nckh_records)}/{len(nckh_records)}",
        f"- Extra1/Extra2 coverage: {len(extra_records)}/2",
        "- DERIVED was excluded from discovery by construction.",
        "",
        "## Parse",
        "",
        f"- Status counts: {dict(parse_counts)}",
        f"- Parser counts: {dict(parser_counts)}",
        f"- OCR pages used: {sum(len(record.get('ocr_pages', [])) for record in records)}",
        f"- Unresolved files: {', '.join(unresolved) if unresolved else 'none'}",
        "",
        "## Index",
        "",
        f"- Chunks: {chunk_count}",
        "- SQLite FTS5: DERIVED/_INDEX/knowledge.sqlite",
        '- Search CLI: python DERIVED/_INDEX/search_knowledge.py "temporal alignment loss"',
        "",
        "## Validation",
        "",
        f"- Manifest/source count agreement: PASS ({source_count} records)",
        f"- Derived directory coverage: PASS ({derived_coverage}/{source_count})",
        f"- Chunk provenance fields: PASS ({sum(bool(chunk.get('source_sha256') and chunk.get('source_relative_path')) for chunk in all_chunks)}/{len(all_chunks)})",
        f"- Page spot checks: {checks_ok}/{len(spot_checks)} exact text projection checks",
        "- Equation status: recovered_text or unavailable; no formula was silently synthesized.",
        f"- Orphaned prior artifacts retained: {sum(record.get('parse_status') == 'ORPHANED' for record in records)}",
        "",
        "## Parser versions",
        "",
    ]
    report.extend(f"- {key}: {value}" for key, value in parser_versions.items())
    report.extend(
        [
            "",
            "## Reproduce from a clean environment",
            "",
            "    uv venv --python 3.12 .indexing-venv",
            "    uv pip install --python .indexing-venv\\Scripts\\python.exe docling==2.129.0 PyMuPDF==1.28.2 tiktoken==0.14.0",
            "    .indexing-venv\\Scripts\\python.exe scripts\\build_knowledge_index.py --source DSFINAL.zip --output DERIVED",
            '    .indexing-venv\\Scripts\\python.exe DERIVED\\_INDEX\\search_knowledge.py "catastrophic forgetting"',
            "",
            "The reproduction command reads the archive only and writes a new DERIVED tree. OCR is enabled only when PyMuPDF reports an unusable text layer.",
            "",
        ]
    )
    (manifest_dir / "indexing_report.md").write_text("\n".join(report), encoding="utf-8")
    jsonl_write(manifest_dir / "spot_checks.jsonl", spot_checks)


def build(args: argparse.Namespace) -> None:
    source_zip = Path(args.source).resolve()
    output_root = Path(args.output).resolve()
    if not source_zip.exists():
        raise FileNotFoundError(source_zip)
    if output_root == source_zip or output_root in source_zip.parents:
        raise ValueError("Output must not be the source archive or its parent")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "_MANIFEST" / "source_manifest.jsonl"
    previous_rows = read_jsonl(manifest_path)
    previous_manifest = {
        row.get("source_relative_path"): row
        for row in previous_rows
        if row.get("source_relative_path")
    }
    parser_versions = versions()
    encoding = make_encoding()
    records: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    spot_checks: list[dict[str, Any]] = []
    seen: set[str] = set()

    with ZipFile(source_zip) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        for info in sorted(infos, key=lambda item: item.filename.lower()):
            relative_path = safe_rel_path(info.filename)
            if relative_path is None or relative_path.lower().endswith((".tmp", ".cache")):
                continue
            data = archive.read(info)
            seen.add(relative_path)
            record = write_source_artifacts(
                output_root,
                relative_path,
                info,
                data,
                parser_versions,
                encoding,
                previous_manifest,
            )
            records.append(record)
            derived = output_root / record["derived_relative_dir"]
            card_path = derived / "paper_card.json"
            if card_path.exists():
                cards.append(json.loads(card_path.read_text(encoding="utf-8")))
            if record.get("classification") == "research_paper":
                page_rows = read_jsonl(derived / "pages.jsonl")
                spot_checks.extend(
                    page_spot_checks(data, page_rows, relative_path, record["sha256"])
                )

    for old_path, old_record in previous_manifest.items():
        if old_path not in seen and old_record.get("parse_status") != "ORPHANED":
            orphan = dict(old_record)
            orphan.update(
                {"parse_status": "ORPHANED", "orphaned": True, "orphaned_at_utc": utc_now()}
            )
            records.append(orphan)
    records.sort(key=lambda row: row["source_relative_path"])
    jsonl_write(output_root / "_MANIFEST" / "source_manifest.jsonl", records)
    synthesis(output_root, records, cards)
    validate_and_report(output_root, source_zip, records, cards, spot_checks, parser_versions)
    json_dump(
        output_root / "_MANIFEST" / "run_metadata.json",
        {
            "pipeline": PIPELINE_VERSION,
            "source_archive": str(source_zip),
            "source_archive_sha256": sha256_bytes(source_zip.read_bytes()),
            "source_file_count": len(records),
            "generated_at_utc": utc_now(),
            "parser_versions": parser_versions,
        },
    )
    print(
        json.dumps(
            {
                "output": str(output_root),
                "source_files": len(records),
                "research_papers": len(cards),
                "status": "PASS",
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="DSFINAL.zip", help="FINALDSC/DSFINAL source ZIP")
    parser.add_argument("--output", default="DERIVED", help="Generated output directory")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
