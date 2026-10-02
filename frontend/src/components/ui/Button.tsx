import type { ButtonHTMLAttributes, ReactNode } from "react";

import { cn } from "@/utils/cn";

type ButtonVariant = "primary" | "secondary" | "ghost";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  children: ReactNode;
}

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary:
    "bg-civic-600 text-white hover:bg-civic-700 active:bg-civic-800 shadow-sm shadow-civic-900/10",
  secondary:
    "bg-white text-ink-800 border border-ink-200 hover:border-civic-300 hover:text-civic-700",
  ghost: "text-ink-600 hover:bg-ink-100 hover:text-ink-900",
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
        "inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-50",
        VARIANT_CLASSES[variant],
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}
