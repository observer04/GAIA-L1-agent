import type { StreamEventEnvelope } from "../../lib/api";

type TracePanelProps = {
  events: StreamEventEnvelope[];
};

function formatPayload(payload: Record<string, unknown>): string {
  try {
    return JSON.stringify(payload, null, 2);
  } catch {
    return "{}";
  }
}

export function TracePanel({ events }: TracePanelProps) {
  return (
    <aside className="trace-panel" aria-label="Trace panel">
      <h2>Run Trace</h2>
      <p className="trace-subtitle">Tool calls and state transitions stream here in real time.</p>

      {events.length === 0 ? (
        <div className="trace-empty">No trace yet — send a message to start a run.</div>
      ) : (
        <ul>
          {events.map((event, index) => (
            <li key={`${event.run_id}-${event.event}-${index}`} className="trace-item">
              <header>
                <span className="event-name">{event.event}</span>
                <span className="event-time">{event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : ""}</span>
              </header>
              <pre>{formatPayload(event.payload)}</pre>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
