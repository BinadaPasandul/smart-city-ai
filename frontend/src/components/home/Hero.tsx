import { ChatInput } from "@/components/chat/ChatInput";

interface HeroProps {
  draft: string;
  onDraftChange: (value: string) => void;
  onSubmit: () => void;
}

/**
 * Primary introduction + the live chat entry point. Typing here (via the
 * shared ChatInput) hands off to the dedicated chat layout — see
 * useChat()'s `hasStarted` and HomePage's conditional render.
 */
export function Hero({ draft, onDraftChange, onSubmit }: HeroProps) {
  return (
    <section className="mx-auto max-w-3xl text-center">
      <span className="inline-flex items-center gap-1.5 rounded-full border border-civic-200 bg-civic-50 px-3 py-1 text-xs font-medium text-civic-700">
        <span className="h-1.5 w-1.5 rounded-full bg-civic-500" />
        Agentic Citizen Assistant
      </span>

      <h1 className="mt-6 text-4xl font-semibold tracking-tight text-ink-900 sm:text-5xl">
        Your city, connected through AI.
      </h1>

      <p className="mt-5 text-balance text-base leading-relaxed text-ink-600 sm:text-lg">
        CIVA brings together mobility, environmental, and public service
        information through one conversational assistant — so you
        don&apos;t have to search a dozen different places for an answer.
      </p>

      <ChatInput
        value={draft}
        onChange={onDraftChange}
        onSubmit={onSubmit}
        className="mx-auto mt-10 max-w-xl"
      />
      <p className="mt-3 text-xs text-ink-400">
        Preview only — CIVA isn&apos;t connected to live city data yet.
      </p>
    </section>
  );
}
