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
    <div className="flex flex-col items-center px-4 text-center">
      <h2 className="text-2xl font-semibold tracking-tight text-ink-50">
        How can I help you today?
      </h2>
      <p className="mt-2 max-w-sm text-sm leading-relaxed text-ink-400">
        Ask about mobility, the environment, or public services in your
        city.
      </p>

      <div className="mt-8 flex flex-wrap items-center justify-center gap-2">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            onClick={() => onSelectSuggestion(suggestion)}
            className="glass-surface rounded-full px-4 py-2 text-sm font-medium text-ink-200 transition-all duration-300 hover:-translate-y-0.5 hover:border-gold-400/40 hover:bg-white/[0.07] hover:text-gold-300"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}
