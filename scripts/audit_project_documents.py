"""Audit Vietnamese project documents without modifying repositories.

The audit treats historical checkpoint/error logs as immutable evidence.  It
checks their readability and filename only; compactness warnings target active
Markdown/HTML sources and DOCX packages.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

LOG_NAME = re.compile(r".+_(?:Checkpoints|Error)_\d{2}-\d{2}-\d{2}_V[1-9]\d*\.md$")
MOJIBAKE = tuple(
    "".join(chr(codepoint) for codepoint in sequence)
    for sequence in (
        (0x00C3,),
        (0x00C6,),
        (0x00E1, 0x00BA),
        (0x00E1, 0x00BB),
        (0x00EF, 0x00BF, 0x00BD),
        (0xFFFD,),
    )
)
DEBUG_MARKERS = (
    "Traceback (most recent call last)",
    "DEBUG:",
    "pdb.set_trace(",
    "console.log(",
)
IGNORED_PARTS = {
    ".git",
    ".migration-backups",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
WORD_RE = re.compile(r"\b[^\W\d_][\w'-]*\b", re.UNICODE)
VIETNAMESE_MARKS = set("ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
ENGLISH_FUNCTION_WORDS = {
    "and",
    "before",
    "do",
    "every",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "must",
    "not",
    "of",
    "on",
    "only",
    "or",
    "the",
    "this",
    "to",
    "when",
    "with",
    "without",
}


@dataclass(frozen=True, slots=True)
class Finding:
    """Describe one actionable document finding."""

    level: str
    path: Path
    message: str


class TextCollector(HTMLParser):
    """Collect visible text while validating basic HTML syntax."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._suppressed_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Ignore non-visible style and script payloads."""
        del attrs
        if tag.casefold() in {"style", "script"}:
            self._suppressed_depth += 1

    def handle_endtag(self, tag: str) -> None:
        """Leave a suppressed style/script block without underflow."""
        if tag.casefold() in {"style", "script"} and self._suppressed_depth:
            self._suppressed_depth -= 1

    def handle_data(self, data: str) -> None:
        """Collect only text a reader can see."""
        if not self._suppressed_depth:
            self.parts.append(data)


def is_immutable_log(path: Path) -> bool:
    """Return whether ``path`` is an immutable General run log."""

    return path.parent.name.casefold() == "general" and bool(LOG_NAME.fullmatch(path.name))


def active_paragraphs(text: str, suffix: str) -> list[str]:
    """Extract prose paragraphs for compactness checks."""

    if suffix == ".html":
        parser = TextCollector()
        parser.feed(text)
        text = "\n\n".join(parser.parts)
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        compact = " ".join(line.strip() for line in block.splitlines())
        if not compact or compact.startswith(("#", "|", "```", "<!--")):
            continue
        if compact.startswith(("- ", "* ")):
            continue
        paragraphs.append(compact)
    return paragraphs


def audit_text(path: Path) -> list[Finding]:
    """Audit one UTF-8 Markdown or HTML file."""

    findings: list[Finding] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [Finding("ERROR", path, f"không giải mã được UTF-8: {exc}")]
    if not text.strip():
        findings.append(Finding("ERROR", path, "file rỗng"))
        return findings
    for marker in MOJIBAKE:
        if marker in text:
            findings.append(Finding("ERROR", path, f"nghi ngờ lỗi mã hóa: {marker!r}"))
            break
    for marker in DEBUG_MARKERS:
        if marker in text:
            findings.append(Finding("WARN", path, f"có dấu vết debug: {marker}"))
    if path.suffix.casefold() == ".html":
        parser = TextCollector()
        try:
            parser.feed(text)
            parser.close()
        except Exception as exc:  # HTMLParser exposes several syntax exceptions.
            findings.append(Finding("ERROR", path, f"HTML không đọc được: {exc}"))
        if "<html" not in text.casefold() or "</html>" not in text.casefold():
            findings.append(Finding("ERROR", path, "thiếu phần tử html đóng/mở"))
    if is_immutable_log(path):
        return findings
    seen: set[str] = set()
    for index, paragraph in enumerate(active_paragraphs(text, path.suffix.casefold()), start=1):
        words = WORD_RE.findall(paragraph)
        if len(words) > 130:
            findings.append(
                Finding(
                    "WARN",
                    path,
                    f"đoạn {index} dài {len(words)} từ; nên tách hoặc rút gọn",
                )
            )
        lowered_words = [word.casefold() for word in words]
        english_hits = sum(word in ENGLISH_FUNCTION_WORDS for word in lowered_words)
        if (
            len(words) >= 35
            and english_hits >= 6
            and not any(character in VIETNAMESE_MARKS for character in paragraph)
        ):
            findings.append(
                Finding(
                    "WARN",
                    path,
                    f"đoạn {index} chủ yếu là tiếng Anh; cần xác nhận ngoại lệ",
                )
            )
        normalized = re.sub(r"\W+", " ", paragraph.casefold()).strip()
        if len(normalized) >= 120 and normalized in seen:
            findings.append(Finding("WARN", path, f"đoạn {index} lặp nguyên văn"))
        seen.add(normalized)
    return findings


def audit_docx(path: Path) -> list[Finding]:
    """Validate the package and readable text of one DOCX file."""

    findings: list[Finding] = []
    try:
        with zipfile.ZipFile(path) as package:
            names = set(package.namelist())
            if "word/document.xml" not in names or "[Content_Types].xml" not in names:
                return [Finding("ERROR", path, "gói DOCX thiếu thành phần bắt buộc")]
            root = ElementTree.fromstring(package.read("word/document.xml"))
    except (OSError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        return [Finding("ERROR", path, f"DOCX không hợp lệ: {exc}")]
    text = " ".join(node.text or "" for node in root.iter() if node.tag.endswith("}t"))
    if len(WORD_RE.findall(text)) < 20:
        findings.append(Finding("ERROR", path, "DOCX có quá ít nội dung đọc được"))
    for marker in MOJIBAKE:
        if marker in text:
            findings.append(Finding("ERROR", path, f"DOCX nghi ngờ lỗi mã hóa: {marker!r}"))
            break
    return findings


def audit_repository(root: Path) -> list[Finding]:
    """Audit the Documents tree and local project instructions."""

    findings: list[Finding] = []
    candidates: set[Path] = set()
    document_roots = [
        path
        for path in root.rglob("Documents")
        if path.is_dir() and not IGNORED_PARTS.intersection(path.parts)
    ]
    if not document_roots:
        documents = root / "Documents"
        return [Finding("ERROR", documents, "không có thư mục Documents")]
    for documents in document_roots:
        for suffix in ("*.md", "*.html", "*.docx"):
            candidates.update(documents.rglob(suffix))
    candidates.update(
        path for path in root.rglob("SKILL.md") if not IGNORED_PARTS.intersection(path.parts)
    )
    if (root / "AGENTS.md").is_file():
        candidates.add(root / "AGENTS.md")
    for path in sorted(candidates):
        if path.suffix.casefold() == ".docx":
            findings.extend(audit_docx(path))
        else:
            findings.extend(audit_text(path))
    return findings


def parse_args() -> argparse.Namespace:
    """Parse repository roots supplied by the caller."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--strict-warnings", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Print a compact audit summary and return a gate-compatible status."""

    args = parse_args()
    all_findings: list[Finding] = []
    for root in args.roots:
        resolved = root.resolve()
        repo_findings = audit_repository(resolved)
        all_findings.extend(repo_findings)
        errors = sum(item.level == "ERROR" for item in repo_findings)
        warnings = sum(item.level == "WARN" for item in repo_findings)
        print(f"{resolved.name}: errors={errors}, warnings={warnings}")
    for finding in all_findings:
        print(f"{finding.level}: {finding.path}: {finding.message}")
    has_errors = any(item.level == "ERROR" for item in all_findings)
    has_warnings = any(item.level == "WARN" for item in all_findings)
    return 1 if has_errors or (args.strict_warnings and has_warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
