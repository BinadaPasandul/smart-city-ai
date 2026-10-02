import { Button } from "@/components/ui/Button";

/**
 * Primary introduction + the future chat entry point. The input/button are
 * intentionally inert (disabled, no state, no submit handler) — wiring them
 * to POST /api/v1/chat is a later phase.
 */
export function Hero() {
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

      <form
        className="mx-auto mt-10 flex max-w-xl flex-col gap-3 sm:flex-row"
        onSubmit={(event) => event.preventDefault()}
      >
        <label htmlFor="city-question" className="sr-only">
          Ask anything about your city
        </label>
        <input
          id="city-question"
          type="text"
          disabled
          placeholder="Ask anything about your city…"
          className="w-full flex-1 rounded-lg border border-ink-200 bg-white px-4 py-2.5 text-sm text-ink-400 placeholder:text-ink-400"
        />
        <Button type="submit" disabled className="sm:w-auto">
          Ask
        </Button>
      </form>
      <p className="mt-3 text-xs text-ink-400">
        The assistant isn&apos;t connected yet — this is a preview of the
        interface.
      </p>
    </section>
  );
}
