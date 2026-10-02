/**
 * Join conditional class names, skipping falsy values. A minimal local
 * stand-in for `clsx` so the project doesn't need that dependency just for
 * this one utility.
 */
export function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}
