"""Regression tests for document-audit paragraph extraction."""

from scripts.audit_project_documents import active_paragraphs


def test_html_paragraphs_ignore_style_and_script_payloads() -> None:
    """Do not report CSS or JavaScript as long report prose."""
    hidden = " ".join(["hidden"] * 180)
    html = (
        f"<html><head><style>{hidden}</style></head>"
        f"<body><script>{hidden}</script><p>Nội dung báo cáo ngắn.</p></body></html>"
    )

    assert active_paragraphs(html, ".html") == ["Nội dung báo cáo ngắn."]
