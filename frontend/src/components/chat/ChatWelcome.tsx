const SUGGESTIONS = [
  "Find hospitals near Colombo",
  "Check traffic conditions",
  "What's the air quality?",
  "Find government services",
];

interface ChatWelcomeProps {
  onSelectSuggestion: (text: string) => void;
}

/** Empty state for the dedicated chat layout, shown before the first message. */
export function ChatWelcome({ onSelectSuggestion }: ChatWelcomeProps) {
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center px-4 py-10 text-center">
      <h2 className="text-2xl font-semibold tracking-tight text-ink-900">
        How can I help you today?
      </h2>
      <p className="mt-2 max-w-sm text-sm leading-relaxed text-ink-500">
        Ask about mobility, the environment, or public services in your
        city.
      </p>

      <div className="mt-8 flex flex-wrap items-center justify-center gap-2">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            onClick={() => onSelectSuggestion(suggestion)}
            className="rounded-full border border-ink-200 bg-white px-4 py-2 text-sm font-medium text-ink-700 transition-colors duration-150 hover:border-civic-300 hover:bg-civic-50 hover:text-civic-700"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}
