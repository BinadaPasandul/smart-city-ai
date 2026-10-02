import type { ReactElement } from "react";

interface QuickAction {
  label: string;
  icon: ReactElement;
}

const QUICK_ACTIONS: QuickAction[] = [
  {
    label: "Find hospitals",
    icon: (
      <>
        <rect x="4" y="4" width="16" height="16" rx="3" stroke="currentColor" strokeWidth="1.6" />
        <path d="M12 8v8M8 12h8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </>
    ),
  },
  {
    label: "Find police stations",
    icon: (
      <path
        d="M12 3l7 3v5c0 4.6-3 7.9-7 10-4-2.1-7-5.4-7-10V6l7-3Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    ),
  },
  {
    label: "Check traffic",
    icon: (
      <>
        <rect x="9" y="3" width="6" height="17" rx="3" stroke="currentColor" strokeWidth="1.6" />
        <circle cx="12" cy="7.5" r="1" fill="currentColor" />
        <circle cx="12" cy="11.5" r="1" fill="currentColor" />
        <circle cx="12" cy="15.5" r="1" fill="currentColor" />
      </>
    ),
  },
  {
    label: "Find parking",
    icon: (
      <>
        <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.6" />
        <path
          d="M10 16V8h2.5a2.5 2.5 0 0 1 0 5H10"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </>
    ),
  },
  {
    label: "Air quality",
    icon: (
      <path
        d="M3 8h11a2.5 2.5 0 1 0-2.3-3.5M3 12.5h15a2.5 2.5 0 1 1-2.3 3.5M3 17h9"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    ),
  },
  {
    label: "Government services",
    icon: (
      <path
        d="M4 21h16M5 21V10l7-5 7 5v11M9 21v-6h6v6"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    ),
  },
];

interface QuickActionsProps {
  /** Places the clicked action's text into the chat draft — never submits it. */
  onSelectAction: (text: string) => void;
}

/**
 * Shortcut chips previewing common requests. Clicking one places its label
 * into the chat draft (via the same `setDraft` callback `ChatWelcome`'s
 * suggestions use) so the user can review or edit it before sending —
 * nothing here submits a message or calls the backend on its own.
 */
export function QuickActions({ onSelectAction }: QuickActionsProps) {
  return (
    <section aria-labelledby="quick-actions-heading">
      <h2 id="quick-actions-heading" className="sr-only">
        Quick actions
      </h2>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {QUICK_ACTIONS.map((action) => (
          <button
            key={action.label}
            type="button"
            onClick={() => onSelectAction(action.label)}
            className="flex flex-col items-center gap-2.5 rounded-xl border border-ink-200/80 bg-white px-3 py-5 text-center transition-colors duration-150 hover:border-civic-300 hover:bg-civic-50"
          >
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-ink-50 text-ink-600">
              <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5">
                {action.icon}
              </svg>
            </span>
            <span className="text-xs font-medium text-ink-700">
              {action.label}
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}
