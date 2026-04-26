"""Streamlit V1 chat client for the Creative Coding Assistant."""

from __future__ import annotations

from functools import lru_cache
from uuid import uuid4

from creative_coding_assistant.app import build_assistant_service
from creative_coding_assistant.clients import (
    ChatHistoryEntry,
    ContextDisplayItem,
    PromptVisibilitySummary,
    RetrievalDisplayItem,
    StreamRenderState,
    assistant_history_entry,
    build_chat_request,
    build_provider_warning,
    context_empty_message,
    context_expander_label,
    default_domain_selection,
    default_mode,
    prompt_visibility_empty_message,
    prompt_visibility_expander_label,
    prompt_visibility_meta,
    reduce_stream_event,
    retrieval_empty_message,
    retrieval_expander_label,
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
        selected_domains = st.multiselect(
            "Domains",
            options=list(CreativeCodingDomain),
            default=list(default_domain_selection()),
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
            domains=selected_domains,
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
            _render_context_visibility(
                kind="memory",
                context_items=entry.memory_items,
                context_state=entry.memory_state,
            )
            _render_retrieval_context(
                retrieval_items=entry.retrieval_items,
                retrieval_state=entry.retrieval_state,
            )
            _render_context_visibility(
                kind="context",
                context_items=entry.context_items,
                context_state=entry.context_state,
            )
            _render_prompt_visibility(
                kind="prompt_input",
                summary=entry.prompt_input_summary,
                visibility_state=entry.prompt_input_state,
            )
            _render_prompt_visibility(
                kind="rendered_prompt",
                summary=entry.rendered_prompt_summary,
                visibility_state=entry.rendered_prompt_state,
            )


def _run_chat_turn(
    *,
    prompt: str,
    domains: list[CreativeCodingDomain],
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
        domains=domains,
        mode=mode,
    )

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        answer_placeholder = st.empty()
        error_placeholder = st.empty()
        memory_placeholder = st.empty()
        retrieval_placeholder = st.empty()
        context_placeholder = st.empty()
        prompt_input_placeholder = st.empty()
        rendered_prompt_placeholder = st.empty()
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

                _render_context_visibility(
                    kind="memory",
                    context_items=state.memory_items,
                    context_state=state.memory_state,
                    placeholder=memory_placeholder,
                )
                _render_retrieval_context(
                    retrieval_items=state.retrieval_items,
                    retrieval_state=state.retrieval_state,
                    placeholder=retrieval_placeholder,
                )
                _render_context_visibility(
                    kind="context",
                    context_items=state.context_items,
                    context_state=state.context_state,
                    placeholder=context_placeholder,
                )
                _render_prompt_visibility(
                    kind="prompt_input",
                    summary=state.prompt_input_summary,
                    visibility_state=state.prompt_input_state,
                    placeholder=prompt_input_placeholder,
                )
                _render_prompt_visibility(
                    kind="rendered_prompt",
                    summary=state.rendered_prompt_summary,
                    visibility_state=state.rendered_prompt_state,
                    placeholder=rendered_prompt_placeholder,
                )
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


def _render_retrieval_context(
    *,
    retrieval_items: tuple[RetrievalDisplayItem, ...],
    retrieval_state: str,
    placeholder=None,
) -> None:
    import streamlit as st

    if retrieval_state == "unknown":
        if placeholder is not None:
            placeholder.empty()
        return

    container = placeholder.container() if placeholder is not None else st.container()
    with container:
        with st.expander(
            retrieval_expander_label(
                retrieval_items,
                retrieval_state=retrieval_state,
            ),
            expanded=False,
        ):
            empty_message = retrieval_empty_message(retrieval_state)
            if empty_message is not None:
                st.caption(empty_message)
                return

            for item in retrieval_items:
                label = item.title
                meta_parts = [item.source_id, _format_domain(item.domain)]
                if item.score is not None:
                    meta_parts.append(f"score {item.score:.3f}")
                elif item.distance is not None:
                    meta_parts.append(f"distance {item.distance:.3f}")

                st.markdown(f"**{label}**")
                st.caption(" | ".join(meta_parts))
                st.markdown(item.snippet)


def _render_context_visibility(
    *,
    kind: str,
    context_items: tuple[ContextDisplayItem, ...],
    context_state: str,
    placeholder=None,
) -> None:
    import streamlit as st

    if context_state == "unknown":
        if placeholder is not None:
            placeholder.empty()
        return

    container = placeholder.container() if placeholder is not None else st.container()
    with container:
        with st.expander(
            context_expander_label(
                kind=kind,
                items=context_items,
                visibility_state=context_state,
            ),
            expanded=False,
        ):
            empty_message = context_empty_message(
                kind=kind,
                visibility_state=context_state,
            )
            if empty_message is not None:
                st.caption(empty_message)
                return

            for item in context_items:
                meta_parts = []
                if item.source_id is not None:
                    meta_parts.append(item.source_id)
                if item.domain is not None:
                    meta_parts.append(_format_domain(item.domain))

                st.markdown(f"**{item.label}**")
                if meta_parts:
                    st.caption(" | ".join(meta_parts))
                st.markdown(item.snippet)


def _render_prompt_visibility(
    *,
    kind: str,
    summary: PromptVisibilitySummary | None,
    visibility_state: str,
    placeholder=None,
) -> None:
    import streamlit as st

    if visibility_state == "unknown":
        if placeholder is not None:
            placeholder.empty()
        return

    container = placeholder.container() if placeholder is not None else st.container()
    with container:
        with st.expander(
            prompt_visibility_expander_label(
                kind=kind,
                visibility_state=visibility_state,
                summary=summary,
            ),
            expanded=False,
        ):
            empty_message = prompt_visibility_empty_message(
                kind=kind,
                visibility_state=visibility_state,
            )
            if empty_message is not None:
                st.caption(empty_message)
                return

            meta = prompt_visibility_meta(summary)
            if meta is not None:
                st.caption(meta)

            assert summary is not None
            for item in summary.items:
                meta_parts = []
                if item.source_id is not None:
                    meta_parts.append(item.source_id)
                if item.domain is not None:
                    meta_parts.append(_format_domain(item.domain))

                st.markdown(f"**{item.label}**")
                if meta_parts:
                    st.caption(" | ".join(meta_parts))
                st.markdown(item.snippet)


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
