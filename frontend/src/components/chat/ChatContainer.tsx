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

  if (!hasMessages) {
    return (
      <div className="flex h-full min-h-0 flex-1 flex-col items-center justify-center px-4 py-10 sm:px-6">
        <ChatWelcome onSelectSuggestion={onDraftChange} />
        <ChatInput
          value={draft}
          onChange={onDraftChange}
          onSubmit={onSubmit}
          disabled={isAssistantTyping}
          autoFocus
          className="mt-8 w-full max-w-3xl"
        />
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col">
      <MessageList messages={messages} isAssistantTyping={isAssistantTyping} />

      <div className="border-t border-white/10 bg-ink-950/60 px-4 py-4 backdrop-blur-xl sm:px-6">
        <ChatInput
          value={draft}
          onChange={onDraftChange}
          onSubmit={onSubmit}
          disabled={isAssistantTyping}
          autoFocus
          className="mx-auto max-w-3xl"
        />
      </div>
    </div>
  );
}
