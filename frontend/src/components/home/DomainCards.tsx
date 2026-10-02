import type { ReactElement } from "react";

import { Card } from "@/components/ui/Card";

interface Domain {
  name: string;
  description: string;
  icon: ReactElement;
}

const DOMAINS: Domain[] = [
  {
    name: "Mobility",
    description: "Traffic, public transport, routes, parking and EV charging.",
    icon: (
      <path
        d="M4 16h16M6 16l1.5-5a2 2 0 0 1 1.9-1.4h5.2a2 2 0 0 1 1.9 1.4L18 16M7 19a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Zm10 0a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    ),
  },
  {
    name: "Environment",
    description: "Air quality, pollution, weather and waste information.",
    icon: (
      <>
        <path
          d="M19 4c-7 0-13 4-13 11a7 7 0 0 0 7 7c7 0 11-6 11-13 0-1.7-.3-3.3-.9-4.8-1.2-.1-2.6-.2-4.1-.2Z"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path d="M9 19c0-7 3-11 10-14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </>
    ),
  },
  {
    name: "Public Services",
    description:
      "Hospitals, police stations, fire stations, government services and emergency information.",
    icon: (
      <path
        d="M12 21s-7-4.6-7-10a7 7 0 0 1 14 0c0 5.4-7 10-7 10Zm0-6a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    ),
  },
];

/** Previews the three specialist agents the Orchestrator can call on. */
export function DomainCards() {
  return (
    <section aria-labelledby="domains-heading">
      <div className="text-center">
        <h2
          id="domains-heading"
          className="text-sm font-semibold uppercase tracking-wide text-gold-400"
        >
          Specialist agents
        </h2>
        <p className="mt-2 text-2xl font-semibold tracking-tight text-ink-50">
          Three domains, one conversation
        </p>
      </div>

      <div className="mt-10 grid grid-cols-1 gap-5 sm:grid-cols-3">
        {DOMAINS.map((domain) => (
          <Card key={domain.name} className="text-left">
            <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-white/5 text-ink-200 ring-1 ring-white/10">
              <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5">
                {domain.icon}
              </svg>
            </span>
            <h3 className="mt-4 text-base font-semibold text-ink-50">
              {domain.name}
            </h3>
            <p className="mt-1.5 text-sm leading-relaxed text-ink-400">
              {domain.description}
            </p>
          </Card>
        ))}
      </div>
    </section>
  );
}
