import {
  buildConversationContextModel,
  type ConversationContextModel
} from "@/lib/conversation-context";

export function ConversationContextInspector({
  context = buildConversationContextModel({
    traceEvents: [],
    visibleEntryCount: 0
  })
}: {
  context?: ConversationContextModel;
}) {
  return (
    <article
      aria-label="Conversation context diagnostics"
      className="conversationContextInspector"
      data-source={context.source}
      role="group"
    >
      <header>
        <div>
          <span>Conversation context</span>
          <strong>Privacy-safe diagnostics</strong>
          <p>{context.summary}</p>
        </div>
        <small>{context.source === "stream" ? "Stream evidence" : "Awaiting evidence"}</small>
      </header>
      <dl>
        {context.diagnostics.map((diagnostic) => (
          <div data-state={diagnostic.state} key={diagnostic.id}>
            <dt>{diagnostic.label}</dt>
            <dd>{diagnostic.value}</dd>
            <p>{diagnostic.detail}</p>
          </div>
        ))}
      </dl>
    </article>
  );
}
