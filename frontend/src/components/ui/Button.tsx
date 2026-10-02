import type { ButtonHTMLAttributes, ReactNode } from "react";

import { cn } from "@/utils/cn";

type ButtonVariant = "primary" | "secondary" | "ghost";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  children: ReactNode;
}

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary:
    "bg-gold-500 text-ink-950 hover:bg-gold-400 active:bg-gold-600 shadow-md shadow-gold-900/30 hover:shadow-lg hover:shadow-gold-500/30",
  secondary:
    "border border-white/10 bg-white/5 text-ink-100 backdrop-blur-xl hover:border-gold-400/40 hover:bg-white/10 hover:text-gold-300",
  ghost: "text-ink-300 hover:bg-white/5 hover:text-ink-50",
};

/**
 * Baseline button primitive. First entry in the reusable-component
 * foundation — intentionally small; new variants/sizes get added as real
 * features need them rather than speculatively now.
 */
export function Button({
  variant = "primary",
  className,
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition-all duration-200 disabled:cursor-not-allowed disabled:opacity-50",
        VARIANT_CLASSES[variant],
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}
