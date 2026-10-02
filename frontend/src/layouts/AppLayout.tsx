import type { ReactNode } from "react";

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
}

/** The application shell: header + main content slot + footer. */
export function AppLayout({ children, hideFooter = false }: AppLayoutProps) {
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-ink-50">
      <Header />
      <main className="flex-1 overflow-y-auto">{children}</main>
      {!hideFooter && <Footer />}
    </div>
  );
}
