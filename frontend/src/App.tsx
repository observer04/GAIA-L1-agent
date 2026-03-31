import { useMemo, useRef, useState } from "react";

import { ChatComposer } from "./features/chat/ChatComposer";
import { ChatMessageList } from "./features/chat/ChatMessageList";
import { TracePanel } from "./features/traces/TracePanel";
import { API_BASE, type ChatMessage, type StreamEventEnvelope, streamChat } from "./lib/api";

const SESSION_STORAGE_KEY = "gaia-agent-session-id";

function getInitialSessionId(): string {
  const existing = localStorage.getItem(SESSION_STORAGE_KEY);
  if (existing && existing.trim()) {
    return existing;
  }

  const generated = `sess-${crypto.randomUUID()}`;
  localStorage.setItem(SESSION_STORAGE_KEY, generated);
  return generated;
}

function asRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object") {
    return value as Record<string, unknown>;
  }
  return {};
}

export function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "intro-assistant",
      role: "assistant",
      content:
        "Hey — I’m your GAIA agent. Ask me a question and I’ll stream my answer. Open the trace panel to inspect tool calls and transitions.",
    },
  ]);
  const [traceEvents, setTraceEvents] = useState<StreamEventEnvelope[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [sessionId] = useState<string>(() => getInitialSessionId());

  const answerBufferRef = useRef("");
  const latestRunIdRef = useRef("");

  const statusText = useMemo(() => {
    if (isStreaming) {
      return "Streaming response…";
    }
    return `Ready • API: ${API_BASE}`;
  }, [isStreaming]);

  const handleSend = async () => {
    const prompt = input.trim();
    if (!prompt || isStreaming) {
      return;
    }

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: prompt,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsStreaming(true);
    setStreamingText("");
    setTraceEvents([]);
    answerBufferRef.current = "";
    latestRunIdRef.current = "";

    try {
      await streamChat(
        {
          message: prompt,
          session_id: sessionId,
          run_label: "web-chat",
        },
        (event) => {
          if (event.run_id) {
            latestRunIdRef.current = event.run_id;
          }

          setTraceEvents((prev) => [...prev, event]);

          if (event.event === "answer_delta") {
            const payload = asRecord(event.payload);
            const delta = String(payload.delta || "");
            answerBufferRef.current += delta;
            setStreamingText(answerBufferRef.current);
            return;
          }

          if (event.event === "run_completed") {
            const payload = asRecord(event.payload);
            const response = asRecord(payload.response);
            const resolvedAnswer = String(response.answer || "").trim() || answerBufferRef.current || "I don't know";

            const assistantMessage: ChatMessage = {
              id: `assistant-${Date.now()}`,
              role: "assistant",
              content: resolvedAnswer,
              runId: event.run_id || latestRunIdRef.current,
            };

            setMessages((prev) => [...prev, assistantMessage]);
            setStreamingText("");
            answerBufferRef.current = "";
            return;
          }

          if (event.event === "error") {
            const payload = asRecord(event.payload);
            const detail = String(payload.detail || "Unknown stream error");
            const assistantMessage: ChatMessage = {
              id: `assistant-error-${Date.now()}`,
              role: "assistant",
              content: `Error: ${detail}`,
              runId: event.run_id || latestRunIdRef.current,
            };
            setMessages((prev) => [...prev, assistantMessage]);
            setStreamingText("");
            answerBufferRef.current = "";
          }
        }
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setMessages((prev) => [
        ...prev,
        {
          id: `assistant-failure-${Date.now()}`,
          role: "assistant",
          content: `Error: ${message}`,
          runId: latestRunIdRef.current || undefined,
        },
      ]);
      setStreamingText("");
      answerBufferRef.current = "";
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>GAIA Agent</h1>
          <p>V1 chat UX + V2 trace inspector foundation</p>
        </div>
        <span className={`status-badge ${isStreaming ? "busy" : "ready"}`}>{statusText}</span>
      </header>

      <main className="app-main">
        <section className="chat-panel">
          <ChatMessageList messages={messages} streamingText={streamingText} />
          <ChatComposer value={input} disabled={isStreaming} onChange={setInput} onSubmit={handleSend} />
        </section>

        <TracePanel events={traceEvents} />
      </main>
    </div>
  );
}
