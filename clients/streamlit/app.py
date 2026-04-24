"""Streamlit V1 chat client for the Creative Coding Assistant."""

from __future__ import annotations

from functools import lru_cache
from uuid import uuid4

from creative_coding_assistant.app import build_assistant_service
from creative_coding_assistant.clients import (
    ChatHistoryEntry,
    StreamRenderState,
    assistant_history_entry,
    build_chat_request,
    build_provider_warning,
    default_domain,
    default_mode,
    reduce_stream_event,
)
from creative_coding_assistant.contracts import AssistantMode, CreativeCodingDomain
from creative_coding_assistant.core import load_settings

_CHAT_HISTORY_KEY = "chat_history"
_CONVERSATION_ID_KEY = "conversation_id"

_DOMAIN_LABELS = {
    CreativeCodingDomain.THREE_JS: "Three.js",
    CreativeCodingDomain.REACT_THREE_FIBER: "React Three Fiber",
    CreativeCodingDomain.P5_JS: "p5.js",
    CreativeCodingDomain.GLSL: "GLSL",
}

_MODE_LABELS = {
    AssistantMode.GENERATE: "Generate",
    AssistantMode.EXPLAIN: "Explain",
    AssistantMode.DEBUG: "Debug",
    AssistantMode.DESIGN: "Design",
    AssistantMode.REVIEW: "Review",
    AssistantMode.PREVIEW: "Preview",
}


def main() -> None:
    """Render the thin Streamlit client over the composed backend service."""

    import streamlit as st

    settings = load_settings()
    st.set_page_config(page_title=settings.app_name, page_icon=":art:")
    st.title(settings.app_name)
    st.caption("Three.js, React Three Fiber, p5.js, and GLSL chat support.")

    _ensure_session_state()

    with st.sidebar:
        selected_domain = st.selectbox(
            "Domain",
            options=list(CreativeCodingDomain),
            index=list(CreativeCodingDomain).index(default_domain(settings)),
            format_func=_format_domain,
        )
        selected_mode = st.selectbox(
            "Mode",
            options=list(AssistantMode),
            index=list(AssistantMode).index(default_mode(settings)),
            format_func=_format_mode,
        )
        if st.button("Clear chat", use_container_width=True):
            _reset_chat_state()
            st.rerun()

    provider_warning = build_provider_warning(settings)
    if provider_warning is not None:
        st.warning(provider_warning)

    _render_history()

    prompt = st.chat_input(
        "Ask about creative coding",
        disabled=provider_warning is not None,
    )
    if prompt:
        _run_chat_turn(
            prompt=prompt,
            domain=selected_domain,
            mode=selected_mode,
        )


@lru_cache(maxsize=1)
def _get_assistant_service():
    return build_assistant_service()


def _render_history() -> None:
    import streamlit as st

    for raw_entry in st.session_state[_CHAT_HISTORY_KEY]:
        entry = ChatHistoryEntry.model_validate(raw_entry)
        with st.chat_message(entry.role):
            st.markdown(entry.content)


def _run_chat_turn(
    *,
    prompt: str,
    domain: CreativeCodingDomain,
    mode: AssistantMode,
) -> None:
    import streamlit as st

    user_entry = ChatHistoryEntry(role="user", content=prompt)
    st.session_state[_CHAT_HISTORY_KEY].append(user_entry.model_dump(mode="json"))
    with st.chat_message(user_entry.role):
        st.markdown(user_entry.content)

    request = build_chat_request(
        query=prompt,
        conversation_id=st.session_state[_CONVERSATION_ID_KEY],
        settings=load_settings(),
        domain=domain,
        mode=mode,
    )

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        answer_placeholder = st.empty()
        error_placeholder = st.empty()
        state = StreamRenderState()

        try:
            for event in _get_assistant_service().stream(request):
                state = reduce_stream_event(state, event)
                if state.status_message is not None:
                    status_placeholder.caption(state.status_message)
                else:
                    status_placeholder.empty()

                if state.answer_text:
                    answer_placeholder.markdown(state.answer_text)

                if state.error_message is not None:
                    error_placeholder.error(state.error_message)
        except Exception:
            state = state.model_copy(
                update={"error_message": "Assistant request failed unexpectedly."}
            )
            error_placeholder.error(state.error_message)

        assistant_entry = assistant_history_entry(state)
        if assistant_entry.content and not state.answer_text:
            answer_placeholder.markdown(assistant_entry.content)
        st.session_state[_CHAT_HISTORY_KEY].append(
            assistant_entry.model_dump(mode="json")
        )


def _ensure_session_state() -> None:
    import streamlit as st

    if _CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[_CHAT_HISTORY_KEY] = []
    if _CONVERSATION_ID_KEY not in st.session_state:
        st.session_state[_CONVERSATION_ID_KEY] = uuid4().hex


def _reset_chat_state() -> None:
    import streamlit as st

    st.session_state[_CHAT_HISTORY_KEY] = []
    st.session_state[_CONVERSATION_ID_KEY] = uuid4().hex


def _format_domain(domain: CreativeCodingDomain) -> str:
    return _DOMAIN_LABELS[domain]


def _format_mode(mode: AssistantMode) -> str:
    return _MODE_LABELS[mode]


if __name__ == "__main__":
    main()
