type ChatComposerProps = {
  value: string;
  disabled?: boolean;
  onChange: (next: string) => void;
  onSubmit: () => void;
};

export function ChatComposer({ value, disabled = false, onChange, onSubmit }: ChatComposerProps) {
  return (
    <form
      className="chat-composer"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <textarea
        value={value}
        disabled={disabled}
        placeholder="Ask GAIA Agent anything..."
        onChange={(event) => onChange(event.target.value)}
      />
      <button type="submit" disabled={disabled || !value.trim()}>
        {disabled ? "Sending..." : "Send"}
      </button>
    </form>
  );
}
