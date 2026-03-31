import type { ChatMessage } from "../../lib/api";

type ChatMessageListProps = {
  messages: ChatMessage[];
  streamingText: string;
};

export function ChatMessageList({ messages, streamingText }: ChatMessageListProps) {
  return (
    <div className="chat-messages" role="log" aria-live="polite">
      {messages.map((message) => (
        <article key={message.id} className={`chat-bubble ${message.role}`}>
          <header>
            <strong>{message.role === "user" ? "You" : "GAIA"}</strong>
            {message.runId ? <span className="run-id">run: {message.runId.slice(0, 12)}…</span> : null}
          </header>
          <p>{message.content}</p>
        </article>
      ))}

      {streamingText ? (
        <article className="chat-bubble assistant streaming">
          <header>
            <strong>GAIA</strong>
            <span className="streaming-dot" aria-label="Streaming response" />
          </header>
          <p>{streamingText}</p>
        </article>
      ) : null}
    </div>
  );
}
