"""Read-only Streamlit scaffold for project lineage and gate status."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from temporal_alignment.ui import build_dashboard_view


def main() -> None:
    """Render status only; never train, download, or call external services."""
    view = build_dashboard_view()
    st.set_page_config(page_title=view.title, layout="wide")
    st.title(view.title)
    st.caption(view.scope)
    st.warning(view.warning)
    st.subheader("Lineage checkpoint")
    st.dataframe(view.lineage.rows(), width="stretch", hide_index=True)
    st.subheader("Gates và mức bằng chứng")
    st.dataframe(view.rows(), width="stretch", hide_index=True)
    st.info("Nguồn trạng thái: checkpoint mới nhất theo Documents/General/RULE.md.")


if __name__ == "__main__":
    main()
