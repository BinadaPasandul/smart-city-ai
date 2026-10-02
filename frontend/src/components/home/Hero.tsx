import { ChatInput } from "@/components/chat/ChatInput";

interface HeroProps {
  draft: string;
  onDraftChange: (value: string) => void;
  onSubmit: () => void;
  isAssistantTyping: boolean;
}

/**
 * Primary introduction + the live chat entry point. Typing here (via the
 * shared ChatInput) hands off to the dedicated chat layout — see
 * useChat()'s `hasStarted` and HomePage's conditional render.
 */
export function Hero({ draft, onDraftChange, onSubmit, isAssistantTyping }: HeroProps) {
  return (
    <section className="relative mx-auto max-w-3xl text-center">
      <div
        aria-hidden="true"
        className="ambient-glow left-1/2 top-0 h-72 w-[36rem] -translate-x-1/2 -translate-y-1/3 bg-white/[0.05]"
      />

      <h1 className="relative text-4xl font-semibold tracking-tight text-ink-50 sm:text-5xl">
        Your city, connected through AI.
      </h1>

      <div className="gold-shine-line relative mx-auto mt-6 w-16" />

      <p className="relative mt-6 text-balance text-base leading-relaxed text-ink-400 sm:text-lg">
        CIVA brings a whole city together
      </p>

      <ChatInput
        value={draft}
        onChange={onDraftChange}
        onSubmit={onSubmit}
        disabled={isAssistantTyping}
        className="mx-auto mt-10 max-w-xl"
      />
    </section>
  );
}
