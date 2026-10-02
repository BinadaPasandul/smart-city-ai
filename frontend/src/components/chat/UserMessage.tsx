import type { ChatMessage } from "@/types/chat";

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

interface UserMessageProps {
  message: ChatMessage;
}

export function UserMessage({ message }: UserMessageProps) {
  return (
    <li className="flex justify-end">
      <div className="flex max-w-[85%] flex-col items-end gap-1 sm:max-w-[70%]">
        <p className="whitespace-pre-wrap break-words rounded-2xl rounded-tr-sm border border-gold-500/25 bg-ink-800 px-4 py-2.5 text-sm leading-relaxed text-ink-50 shadow-md shadow-black/30">
          {message.content}
        </p>
        <time
          dateTime={message.timestamp.toISOString()}
          className="px-1 text-xs text-ink-500"
        >
          {formatTime(message.timestamp)}
        </time>
      </div>
    </li>
  );
}
