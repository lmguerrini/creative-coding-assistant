"""Streamlit V1 chat client for the Creative Coding Assistant."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from uuid import uuid4

from creative_coding_assistant.app import build_assistant_service
from creative_coding_assistant.clients import (
    ChatHistoryEntry,
    ContextDisplayItem,
    GenerationInputVisibilitySummary,
    PromptDisplayItem,
    PromptVisibilitySummary,
    RetrievalDisplayItem,
    StreamRenderState,
    TraceVisibilityLevel,
    answer_working_message,
    assistant_history_entry,
    build_chat_request,
    build_provider_warning,
    context_empty_message,
    context_expander_label,
    domain_selection_summary,
    generation_input_empty_message,
    generation_input_expander_label,
    generation_input_meta,
    mode_selection_summary,
    prompt_visibility_empty_message,
    prompt_visibility_expander_label,
    prompt_visibility_meta,
    reduce_stream_event,
    resolve_session_domain_selection,
    resolve_session_mode,
    resolve_session_trace_visibility,
    retrieval_empty_message,
    retrieval_expander_label,
    split_answer_segments,
    trace_sections_for_level,
    trace_visibility_summary,
)
from creative_coding_assistant.contracts import AssistantMode, CreativeCodingDomain
from creative_coding_assistant.core import load_settings

_CHAT_HISTORY_KEY = "chat_history"
_CONVERSATION_ID_KEY = "conversation_id"
_DOMAIN_SELECTION_KEY = "selected_domains"
_MODE_SELECTION_KEY = "selected_mode"
_TRACE_VISIBILITY_KEY = "trace_visibility"
_ANSWER_PHASE_STATUSES = {
    "Preparing response...",
    "Receiving response...",
}

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

    _ensure_session_state(settings)

    with st.sidebar:
        st.markdown("**Chat session**")
        with _section_container():
            st.markdown("**Domains**")
            selected_domains = st.multiselect(
                "Domains",
                options=list(CreativeCodingDomain),
                format_func=_format_domain,
                label_visibility="collapsed",
                placeholder="Select one or more domains",
                key=_DOMAIN_SELECTION_KEY,
            )
            st.caption(domain_selection_summary(selected_domains))
        with _section_container():
            st.markdown("**Mode**")
            selected_mode = st.selectbox(
                "Mode",
                options=list(AssistantMode),
                format_func=_format_mode,
                label_visibility="collapsed",
                key=_MODE_SELECTION_KEY,
            )
            st.caption(f"Mode: {mode_selection_summary(selected_mode)}")
        with _section_container():
            st.markdown("**Trace detail**")
            trace_visibility = st.selectbox(
                "Trace detail",
                options=list(TraceVisibilityLevel),
                format_func=_format_trace_visibility,
                label_visibility="collapsed",
                key=_TRACE_VISIBILITY_KEY,
            )
            st.caption(f"Trace detail: {trace_visibility_summary(trace_visibility)}")
        st.divider()
        if st.button("Clear chat", use_container_width=True):
            _reset_chat_state()
            st.rerun()

    provider_warning = build_provider_warning(settings)
    if provider_warning is not None:
        st.warning(provider_warning)

    _render_history(trace_visibility=trace_visibility)

    prompt = st.chat_input(
        "Ask about creative coding",
        disabled=provider_warning is not None,
    )
    if prompt:
        _run_chat_turn(
            prompt=prompt,
            domains=selected_domains,
            mode=selected_mode,
            trace_visibility=trace_visibility,
        )


@lru_cache(maxsize=1)
def _get_assistant_service():
    return build_assistant_service()


def _render_history(*, trace_visibility: TraceVisibilityLevel) -> None:
    import streamlit as st

    for index, raw_entry in enumerate(st.session_state[_CHAT_HISTORY_KEY]):
        entry = ChatHistoryEntry.model_validate(raw_entry)
        with st.chat_message(entry.role):
            _render_answer_body(
                text=entry.content,
                allow_unclosed_code_block=False,
                show_downloads=entry.role == "assistant",
                download_key_prefix=f"history-{index}",
            )
            _render_visibility_trace(
                ui_domains=entry.ui_domains,
                detected_domains=entry.detected_domains,
                retrieval_domains=entry.retrieval_domains,
                memory_items=entry.memory_items,
                memory_state=entry.memory_state,
                retrieval_items=entry.retrieval_items,
                retrieval_state=entry.retrieval_state,
                context_items=entry.context_items,
                context_state=entry.context_state,
                prompt_input_summary=entry.prompt_input_summary,
                prompt_input_state=entry.prompt_input_state,
                rendered_prompt_summary=entry.rendered_prompt_summary,
                rendered_prompt_state=entry.rendered_prompt_state,
                generation_input_summary=entry.generation_input_summary,
                generation_input_state=entry.generation_input_state,
                trace_visibility=trace_visibility,
            )


def _run_chat_turn(
    *,
    prompt: str,
    domains: list[CreativeCodingDomain],
    mode: AssistantMode,
    trace_visibility: TraceVisibilityLevel,
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
        visibility_placeholder = st.empty()
        state = StreamRenderState()

        try:
            for event in _get_assistant_service().stream(request):
                state = reduce_stream_event(state, event)
                if (
                    state.status_message is not None
                    and state.status_message not in _ANSWER_PHASE_STATUSES
                ):
                    status_placeholder.caption(state.status_message)
                else:
                    status_placeholder.empty()

                _render_answer_area(
                    placeholder=answer_placeholder,
                    text=state.answer_text,
                    status_message=state.status_message,
                    allow_unclosed_code_block=state.is_streaming_answer,
                )

                if state.error_message is not None:
                    error_placeholder.error(state.error_message)

                _render_visibility_trace(
                    ui_domains=state.ui_domains,
                    detected_domains=state.detected_domains,
                    retrieval_domains=state.retrieval_domains,
                    memory_items=state.memory_items,
                    memory_state=state.memory_state,
                    retrieval_items=state.retrieval_items,
                    retrieval_state=state.retrieval_state,
                    context_items=state.context_items,
                    context_state=state.context_state,
                    prompt_input_summary=state.prompt_input_summary,
                    prompt_input_state=state.prompt_input_state,
                    rendered_prompt_summary=state.rendered_prompt_summary,
                    rendered_prompt_state=state.rendered_prompt_state,
                    generation_input_summary=state.generation_input_summary,
                    generation_input_state=state.generation_input_state,
                    trace_visibility=trace_visibility,
                    placeholder=visibility_placeholder,
                )
        except Exception:
            state = state.model_copy(
                update={"error_message": "Assistant request failed unexpectedly."}
            )
            error_placeholder.error(state.error_message)

        assistant_entry = assistant_history_entry(state)
        if assistant_entry.content:
            _render_answer_area(
                placeholder=answer_placeholder,
                text=assistant_entry.content,
                status_message=None,
                allow_unclosed_code_block=False,
                show_downloads=True,
                download_key_prefix=(
                    f"live-{len(st.session_state[_CHAT_HISTORY_KEY])}"
                ),
            )
        st.session_state[_CHAT_HISTORY_KEY].append(
            assistant_entry.model_dump(mode="json")
        )


def _render_visibility_trace(
    *,
    ui_domains: tuple[CreativeCodingDomain, ...],
    detected_domains: tuple[CreativeCodingDomain, ...],
    retrieval_domains: tuple[CreativeCodingDomain, ...],
    memory_items: tuple[ContextDisplayItem, ...],
    memory_state: str,
    retrieval_items: tuple[RetrievalDisplayItem, ...],
    retrieval_state: str,
    context_items: tuple[ContextDisplayItem, ...],
    context_state: str,
    prompt_input_summary: PromptVisibilitySummary | None,
    prompt_input_state: str,
    rendered_prompt_summary: PromptVisibilitySummary | None,
    rendered_prompt_state: str,
    generation_input_summary: GenerationInputVisibilitySummary | None,
    generation_input_state: str,
    trace_visibility: TraceVisibilityLevel,
    placeholder=None,
) -> None:
    import streamlit as st

    visible_sections = trace_sections_for_level(trace_visibility)
    state_by_section = {
        "memory": memory_state,
        "retrieval": retrieval_state,
        "context": context_state,
        "prompt_input": prompt_input_state,
        "rendered_prompt": rendered_prompt_state,
        "generation_input": generation_input_state,
    }
    if all(state_by_section[section] == "unknown" for section in visible_sections):
        if placeholder is not None:
            placeholder.empty()
        return

    outer = placeholder.container() if placeholder is not None else st.container()
    with outer:
        with _section_container():
            st.caption(f"Trace: {trace_visibility_summary(trace_visibility)}")
            _render_debug_trace(
                ui_domains=ui_domains,
                detected_domains=detected_domains,
                retrieval_domains=retrieval_domains,
                retrieval_items=retrieval_items,
                context_items=context_items,
            )
            if "memory" in visible_sections:
                _render_context_visibility(
                    kind="memory",
                    context_items=memory_items,
                    context_state=memory_state,
                )
            if "retrieval" in visible_sections:
                _render_retrieval_context(
                    retrieval_items=retrieval_items,
                    retrieval_state=retrieval_state,
                )
            if "context" in visible_sections:
                _render_context_visibility(
                    kind="context",
                    context_items=context_items,
                    context_state=context_state,
                )
            if "prompt_input" in visible_sections:
                _render_prompt_visibility(
                    kind="prompt_input",
                    summary=prompt_input_summary,
                    visibility_state=prompt_input_state,
                )
            if "rendered_prompt" in visible_sections:
                _render_prompt_visibility(
                    kind="rendered_prompt",
                    summary=rendered_prompt_summary,
                    visibility_state=rendered_prompt_state,
                )
            if "generation_input" in visible_sections:
                _render_generation_input_visibility(
                    summary=generation_input_summary,
                    visibility_state=generation_input_state,
                )


def _render_debug_trace(
    *,
    ui_domains: tuple[CreativeCodingDomain, ...],
    detected_domains: tuple[CreativeCodingDomain, ...],
    retrieval_domains: tuple[CreativeCodingDomain, ...],
    retrieval_items: tuple[RetrievalDisplayItem, ...],
    context_items: tuple[ContextDisplayItem, ...],
) -> None:
    import streamlit as st

    if not (
        ui_domains
        or detected_domains
        or retrieval_domains
        or retrieval_items
        or context_items
    ):
        return

    with st.expander("Trace", expanded=False):
        st.markdown("**Domains**")
        st.caption(f"UI selected: {_format_domain_list(ui_domains)}")
        st.caption(f"Detected from query: {_format_domain_list(detected_domains)}")
        st.caption(f"Used for retrieval: {_format_domain_list(retrieval_domains)}")

        st.markdown("**Retrieval results**")
        if retrieval_items:
            _render_retrieval_items(retrieval_items, limit=5)
        else:
            st.caption("No retrieval results were available.")

        st.markdown("**Assembled context**")
        if context_items:
            _render_display_items(context_items, limit=5)
        else:
            st.caption("No assembled context items were available.")


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

            _render_retrieval_items(retrieval_items)


def _format_domain_list(domains: tuple[CreativeCodingDomain, ...]) -> str:
    if not domains:
        return "None"
    return ", ".join(_format_domain(domain) for domain in domains)


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

            _render_display_items(context_items)


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
            _render_display_items(summary.items)


def _render_generation_input_visibility(
    *,
    summary: GenerationInputVisibilitySummary | None,
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
            generation_input_expander_label(
                visibility_state=visibility_state,
                summary=summary,
            ),
            expanded=False,
        ):
            empty_message = generation_input_empty_message(
                visibility_state=visibility_state,
            )
            if empty_message is not None:
                st.caption(empty_message)
                return

            meta = generation_input_meta(summary)
            if meta is not None:
                st.caption(meta)

            assert summary is not None
            for item in summary.items:
                st.markdown(f"**{item.label}**")
                if item.role is not None:
                    st.caption(item.role)
                _render_trace_text(item.snippet)


def _render_retrieval_items(
    retrieval_items: Sequence[RetrievalDisplayItem],
    *,
    limit: int | None = None,
) -> None:
    import streamlit as st

    for item in retrieval_items[:limit]:
        meta_parts = [item.source_id, _format_domain(item.domain)]
        if item.score is not None:
            meta_parts.append(f"score {item.score:.3f}")
        elif item.distance is not None:
            meta_parts.append(f"distance {item.distance:.3f}")

        st.markdown(f"**{item.title}**")
        st.caption(" | ".join(meta_parts))
        _render_trace_text(item.snippet)


def _render_display_items(
    items: Sequence[ContextDisplayItem | PromptDisplayItem],
    *,
    limit: int | None = None,
) -> None:
    import streamlit as st

    for item in items[:limit]:
        meta_parts = _display_item_meta_parts(item)
        st.markdown(f"**{item.label}**")
        if meta_parts:
            st.caption(" | ".join(meta_parts))
        _render_trace_text(item.snippet)


def _display_item_meta_parts(
    item: ContextDisplayItem | PromptDisplayItem,
) -> list[str]:
    meta_parts = []
    if item.source_id is not None:
        meta_parts.append(item.source_id)
    if item.domain is not None:
        meta_parts.append(_format_domain(item.domain))
    return meta_parts


def _render_answer_area(
    *,
    placeholder,
    text: str,
    status_message: str | None,
    allow_unclosed_code_block: bool,
    show_downloads: bool = False,
    download_key_prefix: str | None = None,
) -> None:
    placeholder.empty()
    working_message = answer_working_message(
        status_message=status_message,
        has_content=bool(text),
    )
    if not text:
        if working_message is not None:
            placeholder.markdown(f"_{working_message}_")
        return

    with placeholder.container():
        _render_answer_body(
            text=text,
            allow_unclosed_code_block=allow_unclosed_code_block,
            show_downloads=show_downloads,
            download_key_prefix=download_key_prefix,
        )


def _render_answer_body(
    *,
    text: str,
    allow_unclosed_code_block: bool,
    show_downloads: bool = False,
    download_key_prefix: str | None = None,
) -> None:
    import streamlit as st

    segments = split_answer_segments(
        text,
        allow_unclosed_code_block=allow_unclosed_code_block,
    )
    if not segments:
        st.markdown(text)
        return

    code_index = 0
    for segment in segments:
        if segment.kind == "code":
            code_index += 1
            st.code(
                segment.content,
                language=segment.language,
            )
            if show_downloads and segment.suggested_filename is not None:
                st.download_button(
                    label=f"Download {segment.suggested_filename}",
                    data=segment.content,
                    file_name=segment.suggested_filename,
                    mime=segment.mime_type or "text/plain",
                    key=f"{download_key_prefix or 'answer'}-{code_index}",
                )
            continue

        prose = segment.content
        st.markdown(prose)


def _render_trace_text(text: str) -> None:
    _render_answer_body(
        text=text,
        allow_unclosed_code_block=False,
        show_downloads=False,
    )


def _ensure_session_state(settings) -> None:
    import streamlit as st

    if _CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[_CHAT_HISTORY_KEY] = []
    if _CONVERSATION_ID_KEY not in st.session_state:
        st.session_state[_CONVERSATION_ID_KEY] = uuid4().hex
    st.session_state[_DOMAIN_SELECTION_KEY] = list(
        resolve_session_domain_selection(
            st.session_state.get(_DOMAIN_SELECTION_KEY)
        )
    )
    st.session_state[_MODE_SELECTION_KEY] = resolve_session_mode(
        st.session_state.get(_MODE_SELECTION_KEY),
        settings=settings,
    )
    st.session_state[_TRACE_VISIBILITY_KEY] = resolve_session_trace_visibility(
        st.session_state.get(_TRACE_VISIBILITY_KEY)
    )


def _reset_chat_state() -> None:
    import streamlit as st

    st.session_state[_CHAT_HISTORY_KEY] = []
    st.session_state[_CONVERSATION_ID_KEY] = uuid4().hex


def _section_container():
    import streamlit as st

    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def _format_domain(domain: CreativeCodingDomain) -> str:
    return _DOMAIN_LABELS[domain]


def _format_mode(mode: AssistantMode) -> str:
    return _MODE_LABELS[mode]


def _format_trace_visibility(level: TraceVisibilityLevel) -> str:
    return level.value.title()


if __name__ == "__main__":
    main()
