import type { SVGProps } from "react";

/**
 * CIVA's icon mark: a minimal flat skyline with a small signal/node accent,
 * standing in for "city + AI". Solid `currentColor` fill — no gradients, no
 * second color — so it inherits whatever text color its container sets and
 * stays crisp at any size (header, favicon, or elsewhere) without a raster
 * asset.
 */
export function CivaMark(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 40 40" fill="none" aria-hidden="true" {...props}>
      <rect x="4" y="18" width="9" height="16" rx="1.5" fill="currentColor" />
      <rect x="15.5" y="8" width="9" height="26" rx="1.5" fill="currentColor" />
      <rect x="27" y="14" width="9" height="20" rx="1.5" fill="currentColor" />
      <path
        d="M20 8V6.4"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
      <circle cx="20" cy="4.6" r="1.6" fill="currentColor" />
    </svg>
  );
}
