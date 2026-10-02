import { ChatContainer } from "@/components/chat/ChatContainer";
import { DomainCards } from "@/components/home/DomainCards";
import { Hero } from "@/components/home/Hero";
import { OrchestratorSection } from "@/components/home/OrchestratorSection";
import { QuickActions } from "@/components/home/QuickActions";
import type { UseChatResult } from "@/hooks/useChat";

interface HomePageProps {
  chat: UseChatResult;
}

export function HomePage({ chat }: HomePageProps) {
  if (chat.hasStarted) {
    return (
      <ChatContainer
        messages={chat.messages}
        draft={chat.draft}
        onDraftChange={chat.setDraft}
        onSubmit={chat.sendMessage}
        isAssistantTyping={chat.isAssistantTyping}
      />
    );
  }

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-20 px-4 py-16 sm:gap-24 sm:px-6 sm:py-24 lg:px-8">
      <Hero
        draft={chat.draft}
        onDraftChange={chat.setDraft}
        onSubmit={chat.sendMessage}
      />
      <QuickActions onSelectAction={chat.setDraft} />
      <DomainCards />
      <OrchestratorSection />
    </div>
  );
}
