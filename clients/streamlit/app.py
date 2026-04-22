"""Minimal Streamlit entry point for the bootstrap foundation."""

from __future__ import annotations

from creative_coding_assistant.core import load_settings


def main() -> None:
    """Render a placeholder that keeps V1 UI separate from backend logic."""

    import streamlit as st

    settings = load_settings()
    st.set_page_config(page_title=settings.app_name, page_icon=":art:")
    st.title(settings.app_name)
    st.info("Bootstrap foundation is ready. Product features are not enabled yet.")


if __name__ == "__main__":
    main()
