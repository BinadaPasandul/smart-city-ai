import { CivaMark } from "@/components/icons/CivaMark";

export function Footer() {
  return (
    <footer className="relative bg-ink-950/80 backdrop-blur-xl">
      <div className="gold-shine-line" />
      <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-8 sm:flex-row sm:items-start sm:justify-between">
          <div className="max-w-sm">
            <div className="flex items-center gap-2.5">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gold-500/10 text-gold-400 ring-1 ring-gold-400/30">
                <CivaMark className="h-5 w-5" />
              </span>
              <span className="text-base font-semibold tracking-tight text-ink-50">
                CIVA
              </span>
            </div>
            <p className="mt-3 text-sm leading-relaxed text-ink-400">
              <span className="font-medium text-ink-200">
                City Intelligence &amp; Virtual Assistant
              </span>{" "}
              , one conversational layer across mobility, the environment
              and public services.
            </p>
          </div>

          <div className="text-sm leading-relaxed text-ink-400 sm:text-right">
            <p>
              Developed by <span className="text-ink-200">DS Hustlers</span>
            </p>
            <p className="mt-1">
              Sri Lanka Institute of Information Technology{" "}
              <span className="text-ink-200">(SLIIT)</span>
            </p>
          </div>
        </div>

        <p className="mt-8 border-t border-white/5 pt-6 text-xs text-ink-500">
          © {new Date().getFullYear()} CIVA. Built for educational purposes.
        </p>
      </div>
    </footer>
  );
}
