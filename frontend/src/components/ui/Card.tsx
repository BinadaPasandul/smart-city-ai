import type { HTMLAttributes, ReactNode } from "react";

import { cn } from "@/utils/cn";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

/** Baseline surface primitive used for grouped content across the app. */
export function Card({ className, children, ...props }: CardProps) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-ink-200/80 bg-white p-6 shadow-sm shadow-ink-900/5",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}
