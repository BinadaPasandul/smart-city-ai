import { CivaMark } from "@/components/icons/CivaMark";

const APP_NAME = import.meta.env.VITE_APP_NAME || "CIVA";

/**
 * Application shell header. Navigation is a single current-page link for
 * now — more entries get added once additional pages exist.
 */
export function Header() {
  return (
    <header className="sticky top-0 z-20 bg-ink-950/80 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <a href="/" className="group flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gold-500/10 text-gold-400 ring-1 ring-gold-400/30 transition-all duration-300 group-hover:bg-gold-500/20 group-hover:ring-gold-400/50">
            <CivaMark className="h-5 w-5" />
          </span>
          <span className="text-base font-semibold tracking-tight text-ink-50">
            {APP_NAME}
          </span>
        </a>

        <nav className="flex items-center gap-1" aria-label="Primary">
          <a
            href="/"
            aria-current="page"
            className="rounded-lg px-3 py-2 text-sm font-medium text-ink-100 transition-colors duration-200 hover:bg-white/5 hover:text-ink-50"
          >
            Home
          </a>
        </nav>
      </div>
      <div className="gold-shine-line" />
    </header>
  );
}
