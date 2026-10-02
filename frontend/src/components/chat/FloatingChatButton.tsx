interface FloatingChatButtonProps {
  onClick: () => void;
}

/**
 * Persistent bottom-left shortcut into the dedicated chat layout. Only
 * rendered on the landing page (see AppLayout) — once the chat is already
 * open, the composer itself is the entry point and this would be redundant.
 */
export function FloatingChatButton({ onClick }: FloatingChatButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label="Open chat with CIVA"
      className="group fixed bottom-6 left-6 z-30 flex h-14 w-14 animate-float items-center justify-center rounded-full border border-gold-300/40 bg-gold-500 text-ink-950 shadow-lg shadow-gold-900/40 backdrop-blur-xl transition-all duration-300 hover:scale-110 hover:bg-gold-400 hover:shadow-xl hover:shadow-gold-500/30 active:scale-95"
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        aria-hidden="true"
        className="h-6 w-6 transition-transform duration-300 group-hover:scale-110"
      >
        <path
          d="M4 5.5A1.5 1.5 0 0 1 5.5 4h13A1.5 1.5 0 0 1 20 5.5v10a1.5 1.5 0 0 1-1.5 1.5H9l-4.3 3.2A.5.5 0 0 1 4 19.8V5.5Z"
          stroke="currentColor"
          strokeWidth="1.7"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="8.5" cy="10.5" r="1" fill="currentColor" />
        <circle cx="12" cy="10.5" r="1" fill="currentColor" />
        <circle cx="15.5" cy="10.5" r="1" fill="currentColor" />
      </svg>
    </button>
  );
}
