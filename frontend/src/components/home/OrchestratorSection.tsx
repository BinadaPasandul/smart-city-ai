import { Card } from "@/components/ui/Card";

interface FlowStep {
  label: string;
}

const FLOW_STEPS: FlowStep[] = [
  { label: "Your question" },
  { label: "City Orchestrator" },
  { label: "Specialist agents" },
  { label: "One clear answer" },
];

/**
 * Plain-language explanation of how a request is handled — intentionally
 * free of implementation detail (no class names, routes, or model names).
 */
export function OrchestratorSection() {
  return (
    <section aria-labelledby="orchestrator-heading">
      <Card>
        <div className="text-center">
          <h2
            id="orchestrator-heading"
            className="text-2xl font-semibold tracking-tight text-ink-900"
          >
            One assistant. Multiple city specialists.
          </h2>
          <p className="mx-auto mt-3 max-w-2xl text-sm leading-relaxed text-ink-500">
            You ask one question, in plain language. The City Orchestrator
            works out what you need and brings in whichever specialist
            agents can answer it — then combines everything into a single,
            clear response.
          </p>
        </div>

        <ol className="mt-10 flex list-none flex-col items-stretch gap-3 sm:flex-row sm:items-center sm:justify-center sm:gap-2">
          {FLOW_STEPS.map((step, index) => (
            <li key={step.label} className="flex items-center gap-2">
              <div className="flex items-center gap-2.5 rounded-lg border border-ink-200/80 bg-ink-50 px-4 py-2.5">
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-civic-600 text-[11px] font-semibold text-white">
                  {index + 1}
                </span>
                <span className="text-sm font-medium text-ink-800">
                  {step.label}
                </span>
              </div>
              {index < FLOW_STEPS.length - 1 && (
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  aria-hidden="true"
                  className="hidden h-4 w-4 shrink-0 text-ink-300 sm:block"
                >
                  <path
                    d="M5 12h14m-5-5 5 5-5 5"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </li>
          ))}
        </ol>
      </Card>
    </section>
  );
}
