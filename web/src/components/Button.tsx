import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "secondary" | "ink" | "ghost";
type Size = "md" | "lg" | "sm";

function cls(variant: Variant, size: Size, extra?: string) {
  return ["btn", `btn--${variant}`, `btn--${size}`, extra].filter(Boolean).join(" ");
}

export function Button({
  variant = "secondary",
  size = "md",
  icon,
  className,
  children,
  busy,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: Size; icon?: ReactNode; busy?: boolean }) {
  return (
    <button type="button" className={cls(variant, size, className)} aria-busy={busy || undefined} {...rest}>
      {busy ? <span className="spinner" aria-hidden="true" /> : icon}
      <span>{children}</span>
    </button>
  );
}

export function LinkButton({
  variant = "secondary",
  size = "md",
  icon,
  className,
  children,
  ...rest
}: AnchorHTMLAttributes<HTMLAnchorElement> & { variant?: Variant; size?: Size; icon?: ReactNode }) {
  return (
    <a className={cls(variant, size, className)} {...rest}>
      {icon}
      <span>{children}</span>
    </a>
  );
}
