import type { ReactNode } from "react";

import { Footer } from "@/components/layout/Footer";
import { Header } from "@/components/layout/Header";

interface AppLayoutProps {
  children: ReactNode;
}

/** The application shell: header + main content slot + footer. */
export function AppLayout({ children }: AppLayoutProps) {
  return (
    <div className="flex min-h-screen flex-col bg-ink-50">
      <Header />
      <main className="flex-1">{children}</main>
      <Footer />
    </div>
  );
}
