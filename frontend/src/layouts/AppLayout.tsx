import type { ReactNode } from "react";

import { FloatingChatButton } from "@/components/chat/FloatingChatButton";
import { Footer } from "@/components/layout/Footer";
import { Header } from "@/components/layout/Header";

interface AppLayoutProps {
  children: ReactNode;
  /**
   * Hides the footer so `children` can use the full remaining viewport
   * height — used by the dedicated chat layout, which manages its own
   * internal scrolling instead of the whole page scrolling.
   */
  hideFooter?: boolean;
  /**
   * Opens the dedicated chat layout. When omitted, the floating chat
   * button is not rendered (used once the chat is already open, where the
   * composer itself is already the entry point).
   */
  onOpenChat?: () => void;
}

/** Large, low-opacity blurred color fields that give the dark theme depth. */
function AmbientBackground() {
  return (
    <div aria-hidden="true" className="pointer-events-none fixed inset-0 overflow-hidden">
      <div className="ambient-glow -top-40 -left-32 h-96 w-96 bg-white/[0.04]" />
      <div className="ambient-glow top-1/3 -right-40 h-[28rem] w-[28rem] bg-white/[0.03]" />
      <div className="ambient-glow bottom-0 left-1/4 h-80 w-80 bg-gold-900/10" />
    </div>
  );
}

/** The application shell: header + main content slot + footer. */
export function AppLayout({ children, hideFooter = false, onOpenChat }: AppLayoutProps) {
  // The dedicated chat layout manages its own internal scrolling and has no
  // footer, so it's locked to exactly one screen height. Everywhere else,
  // the page should scroll normally so the footer sits after the content
  // instead of staying pinned to the viewport's bottom edge.
  if (hideFooter) {
    return (
      <div className="relative flex h-screen flex-col overflow-hidden bg-ink-950">
        <AmbientBackground />
        <Header />
        <main className="relative flex-1 overflow-y-auto">{children}</main>
      </div>
    );
  }

  return (
    <div className="relative flex min-h-screen flex-col bg-ink-950">
      <AmbientBackground />
      <Header />
      <main className="relative flex-1">{children}</main>
      <Footer />
      {onOpenChat && <FloatingChatButton onClick={onOpenChat} />}
    </div>
  );
}
