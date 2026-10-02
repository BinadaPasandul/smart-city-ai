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
        "glass-surface rounded-2xl p-6 shadow-lg shadow-black/40 transition-all duration-300 hover:-translate-y-1 hover:border-gold-500/30 hover:bg-white/[0.07] hover:shadow-xl hover:shadow-black/50",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}
