import { ChatInput } from "@/components/chat/ChatInput";
import { ChatWelcome } from "@/components/chat/ChatWelcome";
import { MessageList } from "@/components/chat/MessageList";
import type { ChatMessage } from "@/types/chat";

interface ChatContainerProps {
  messages: ChatMessage[];
  draft: string;
  onDraftChange: (value: string) => void;
  onSubmit: () => void;
  isAssistantTyping: boolean;
}

/**
 * The dedicated chat layout: a scrollable conversation area (or the empty
 * welcome state) plus a composer pinned to the bottom. Fills whatever
 * height its parent gives it rather than growing the page.
 */
export function ChatContainer({
  messages,
  draft,
  onDraftChange,
  onSubmit,
  isAssistantTyping,
}: ChatContainerProps) {
  const hasMessages = messages.length > 0;

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col">
      {hasMessages ? (
        <MessageList messages={messages} isAssistantTyping={isAssistantTyping} />
      ) : (
        <ChatWelcome onSelectSuggestion={onDraftChange} />
      )}

      <div className="border-t border-ink-200/80 bg-ink-50/80 px-4 py-4 sm:px-6">
        <ChatInput
          value={draft}
          onChange={onDraftChange}
          onSubmit={onSubmit}
          autoFocus
          className="mx-auto max-w-3xl"
        />
        <p className="mx-auto mt-2 max-w-3xl text-center text-xs text-ink-400">
          Preview only — CIVA isn&apos;t connected to live city data yet.
        </p>
      </div>
    </div>
  );
}
