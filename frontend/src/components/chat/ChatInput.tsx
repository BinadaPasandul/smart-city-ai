import { useEffect, useId, useRef } from "react";
import type { KeyboardEvent } from "react";

import { Button } from "@/components/ui/Button";

const MAX_HEIGHT_PX = 160;

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  placeholder?: string;
  autoFocus?: boolean;
  className?: string;
  /** Locks the composer (textarea + send button) while a request is in flight. */
  disabled?: boolean;
}

/**
 * Reusable message composer: a textarea (not a single-line input) so
 * Shift+Enter can insert a newline, plus a send button. Used both on the
 * landing Hero and inside the dedicated chat layout so behavior stays
 * identical in both places.
 */
export function ChatInput({
  value,
  onChange,
  onSubmit,
  placeholder = "Ask anything about your city…",
  autoFocus = false,
  className,
  disabled = false,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const inputId = useId();
  const canSubmit = value.trim().length > 0 && !disabled;

  // Auto-grow with content, capped so the composer can't take over the screen.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT_PX)}px`;
  }, [value]);

  // autoFocus places the caret at the start of any pre-filled value (e.g.
  // when the Hero composer hands off to this dedicated chat layout's own
  // textarea mid-word). Move it to the end once, on mount, to match.
  useEffect(() => {
    if (!autoFocus) return;
    const el = textareaRef.current;
    if (!el) return;
    const end = el.value.length;
    el.setSelectionRange(end, end);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const submit = () => {
    if (canSubmit) {
      onSubmit();
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <form
      className={className}
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      <div className="glass-surface flex items-end gap-3 rounded-xl p-2 shadow-lg shadow-black/40 transition-colors duration-300 focus-within:border-gold-400/50">
        <label htmlFor={inputId} className="sr-only">
          Message
        </label>
        <textarea
          id={inputId}
          ref={textareaRef}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={1}
          autoFocus={autoFocus}
          disabled={disabled}
          className="max-h-40 flex-1 resize-none overflow-y-auto bg-transparent px-2 py-2 text-sm text-ink-50 placeholder:text-ink-500 focus:outline-none disabled:cursor-not-allowed"
        />
        <Button type="submit" disabled={!canSubmit}>
          Ask
        </Button>
      </div>
    </form>
  );
}
