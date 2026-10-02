import { CivaMark } from "@/components/icons/CivaMark";
import type { ChatMessage } from "@/types/chat";

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

interface AssistantMessageProps {
  message: ChatMessage;
}

/**
 * For Phase 3A this only renders plain text. Sources and agent/execution
 * metadata are intentionally not rendered yet — they depend on a real
 * backend response and arrive once that integration lands.
 */
export function AssistantMessage({ message }: AssistantMessageProps) {
  return (
    <li className="flex items-start gap-3">
      <span
        className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gold-500/15 text-gold-400 ring-1 ring-gold-400/30"
        aria-hidden="true"
      >
        <CivaMark className="h-5 w-5" />
      </span>
      <div className="flex max-w-[85%] flex-col items-start gap-1 sm:max-w-[70%]">
        <p className="glass-surface whitespace-pre-wrap break-words rounded-2xl rounded-tl-sm px-4 py-2.5 text-sm leading-relaxed text-ink-100">
          {message.content}
        </p>
        <time
          dateTime={message.timestamp.toISOString()}
          className="px-1 text-xs text-ink-400"
        >
          {formatTime(message.timestamp)}
        </time>
      </div>
    </li>
  );
}
