export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  runId?: string;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  run_label?: string;
  task_id?: string;
  level?: string;
  file_name?: string;
}

export interface ChatResponse {
  run_id: string;
  session_id: string;
  status: string;
  answer: string;
  submitted_answer: string;
  stop_reason: string;
  latency_seconds: number;
  attempt_count: number;
  tool_trace: Array<Record<string, unknown>>;
  created_at: string;
}

export interface StreamEventEnvelope {
  event: string;
  run_id: string;
  session_id: string;
  timestamp: string;
  payload: Record<string, unknown>;
}

export interface TraceRecord {
  run_id: string;
  session_id: string;
  question: string;
  created_at: string;
  status: string;
  stop_reason: string;
  latency_seconds: number;
  attempt_count: number;
  tool_trace: Array<Record<string, unknown>>;
  state_transitions: Array<{ from_node: string; to_node: string }>;
}

function resolveApiBase(): string {
  const explicit = String(import.meta.env.VITE_API_BASE_URL || "").trim();
  if (explicit) {
    return explicit.replace(/\/$/, "");
  }

  const configuredAppBase = String(import.meta.env.VITE_APP_BASE_PATH || "").trim();
  if (configuredAppBase) {
    const normalized = configuredAppBase.startsWith("/") ? configuredAppBase : `/${configuredAppBase}`;
    return `${normalized.replace(/\/$/, "")}/api`;
  }

  if (typeof window === "undefined") {
    return "/api";
  }

  const path = window.location.pathname || "/";
  const gaiaPrefix = path.startsWith("/gaia_agent") ? "/gaia_agent" : "";
  return `${gaiaPrefix}/api`;
}

export const API_BASE = resolveApiBase();

function asRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object") {
    return value as Record<string, unknown>;
  }
  return {};
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => {
    throw new Error(`Expected JSON response but received non-JSON payload (status ${response.status}).`);
  });

  if (!response.ok) {
    const details = asRecord(payload);
    const detailMessage = String(details.detail || response.statusText || "request failed");
    throw new Error(detailMessage);
  }

  return payload as T;
}

export async function sendChat(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  return parseJsonResponse<ChatResponse>(response);
}

function parseSseBlock(block: string): StreamEventEnvelope | null {
  const lines = block.split("\n");
  let eventName = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim() || "message";
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }

  if (dataLines.length === 0) {
    return null;
  }

  const joined = dataLines.join("\n");
  const parsed = JSON.parse(joined);
  const payload = asRecord(parsed);

  return {
    event: eventName,
    run_id: String(payload.run_id || ""),
    session_id: String(payload.session_id || ""),
    timestamp: String(payload.timestamp || ""),
    payload: asRecord(payload.payload),
  };
}

export async function streamChat(
  request: ChatRequest,
  onEvent: (event: StreamEventEnvelope) => void
): Promise<void> {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok || !response.body) {
    const message = `Streaming request failed with status ${response.status}`;
    throw new Error(message);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawBlock = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);

      if (rawBlock) {
        try {
          const event = parseSseBlock(rawBlock);
          if (event) {
            onEvent(event);
          }
        } catch (err) {
          console.error("Failed to parse SSE block", rawBlock, err);
        }
      }

      boundary = buffer.indexOf("\n\n");
    }
  }

  const tail = buffer.trim();
  if (tail) {
    const event = parseSseBlock(tail);
    if (event) {
      onEvent(event);
    }
  }
}

export async function fetchTrace(runId: string): Promise<TraceRecord> {
  const response = await fetch(`${API_BASE}/traces/${encodeURIComponent(runId)}`);
  return parseJsonResponse<TraceRecord>(response);
}
