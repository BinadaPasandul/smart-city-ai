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
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const inputId = useId();
  const canSubmit = value.trim().length > 0;

  // Auto-grow with content, capped so the composer can't take over the screen.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT_PX)}px`;
  }, [value]);

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
      <div className="flex items-end gap-3 rounded-xl border border-ink-200 bg-white p-2 shadow-sm shadow-ink-900/5 focus-within:border-civic-300">
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
          className="max-h-40 flex-1 resize-none overflow-y-auto bg-transparent px-2 py-2 text-sm text-ink-900 placeholder:text-ink-400 focus:outline-none"
        />
        <Button type="submit" disabled={!canSubmit}>
          Ask
        </Button>
      </div>
    </form>
  );
}
