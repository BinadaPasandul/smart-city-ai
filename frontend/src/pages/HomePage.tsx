import { ChatContainer } from "@/components/chat/ChatContainer";
import { DomainCards } from "@/components/home/DomainCards";
import { Hero } from "@/components/home/Hero";
import { QuickActions } from "@/components/home/QuickActions";
import type { UseChatResult } from "@/hooks/useChat";

interface HomePageProps {
  chat: UseChatResult;
}

export function HomePage({ chat }: HomePageProps) {
  if (chat.hasStarted) {
    return (
      <div className="flex h-full min-h-0 flex-1 flex-col animate-fade-in">
        <ChatContainer
          messages={chat.messages}
          draft={chat.draft}
          onDraftChange={chat.setDraft}
          onSubmit={chat.sendMessage}
          isAssistantTyping={chat.isAssistantTyping}
        />
      </div>
    );
  }

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-20 px-4 py-16 animate-fade-in sm:gap-24 sm:px-6 sm:py-24 lg:px-8">
      <Hero
        draft={chat.draft}
        onDraftChange={chat.setDraft}
        onSubmit={chat.sendMessage}
        isAssistantTyping={chat.isAssistantTyping}
      />
      <QuickActions onSelectAction={chat.setDraft} />
      <DomainCards />
    </div>
  );
}
