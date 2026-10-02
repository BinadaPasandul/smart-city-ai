import { useEffect, useRef } from "react";

import { AssistantMessage } from "@/components/chat/AssistantMessage";
import { UserMessage } from "@/components/chat/UserMessage";
import { CivaMark } from "@/components/icons/CivaMark";
import type { ChatMessage } from "@/types/chat";

interface MessageListProps {
  messages: ChatMessage[];
  isAssistantTyping: boolean;
}

export function MessageList({ messages, isAssistantTyping }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isAssistantTyping]);

  return (
    <div
      className="min-h-0 flex-1 overflow-y-auto"
      role="log"
      aria-live="polite"
      aria-label="Conversation"
    >
      <ol className="mx-auto flex max-w-3xl flex-col gap-5 px-4 py-6 sm:px-6">
        {messages.map((message) =>
          message.role === "user" ? (
            <UserMessage key={message.id} message={message} />
          ) : (
            <AssistantMessage key={message.id} message={message} />
          ),
        )}

        {isAssistantTyping && (
          <li className="flex items-start gap-3" aria-label="Assistant is typing">
            <span
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gold-500/15 text-gold-400 ring-1 ring-gold-400/30"
              aria-hidden="true"
            >
              <CivaMark className="h-5 w-5" />
            </span>
            <div className="glass-surface flex items-center gap-1 rounded-2xl rounded-tl-sm px-4 py-3">
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-400 [animation-delay:-0.3s]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-400 [animation-delay:-0.15s]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-400" />
            </div>
          </li>
        )}
      </ol>
      <div ref={bottomRef} />
    </div>
  );
}
