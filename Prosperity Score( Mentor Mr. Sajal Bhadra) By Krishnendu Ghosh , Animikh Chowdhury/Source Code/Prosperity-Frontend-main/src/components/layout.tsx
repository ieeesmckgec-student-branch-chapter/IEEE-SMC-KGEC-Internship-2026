import { Link, useRouterState } from "@tanstack/react-router";
import type { ReactNode } from "react";
import { useApplication } from "@/lib/application";

const STEPS = [
  { to: "/", n: 1, title: "Identity & KYC", sub: "Who is applying" },
  {
    to: "/statement",
    n: 2,
    title: "Bank statement",
    sub: "Transaction evidence",
  },
  { to: "/profile", n: 3, title: "Financial profile", sub: "Declared signals" },
  { to: "/review", n: 4, title: "Review & generate", sub: "Consent and score" },
] as const;

export function SiteHeader() {
  const { reset } = useApplication();

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 py-4">
        {/* Keeping the logo clickable to go back home */}
        <Link to="/" className="flex items-center gap-3">
          <span className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-brand to-brand-2 text-lg">
            <svg
              viewBox="0 0 24 24"
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M4 15a8 8 0 1 1 16 0" strokeLinecap="round" />
              <path d="M12 15l4-5" strokeLinecap="round" />
            </svg>
          </span>
          <span>
            <span className="block font-display text-base font-bold">
              Prosperity Score
            </span>
            <span className="block text-xs text-muted-foreground">
              Alternative Credit Ledger
            </span>
          </span>
        </Link>
        <nav className="flex items-center gap-6 text-sm">
          <Link
            to="/"
            className="text-muted-foreground hover:text-foreground"
            activeProps={{ className: "text-foreground font-medium" }}
            activeOptions={{ exact: true }}
            onClick={() => {
              reset();
            }}
          >
            New applicant
          </Link>
        </nav>
      </div>
    </header>
  );
}

export function Stepper() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const currentIndex = Math.max(
    0,
    STEPS.findIndex((s) => s.to === (pathname === "/" ? "/" : pathname)),
  );

  return (
    <div className="grid grid-cols-2 divide-border border-b border-border md:grid-cols-4 md:divide-x">
      {STEPS.map((step, i) => {
        const done = i < currentIndex;
        const active = i === currentIndex;

        {
          /* Changed from <Link> to <div> to completely remove link navigation on click */
        }
        return (
          <div
            key={step.to}
            className={`flex items-center gap-3 px-5 py-4 select-none ${
              active ? "bg-accent/50" : ""
            }`}
          >
            <span
              className={`grid h-7 w-7 shrink-0 place-items-center rounded-full text-xs font-semibold ${
                done || active
                  ? "bg-gradient-to-br from-brand to-brand-2 text-primary-foreground"
                  : "bg-muted text-muted-foreground"
              }`}
            >
              {done ? "✓" : step.n}
            </span>
            <span className="min-w-0">
              <span className="block truncate text-sm font-semibold">
                {step.title}
              </span>
              <span className="block truncate text-xs text-muted-foreground">
                {step.sub}
              </span>
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function PageHero() {
  return (
    <div className="mx-auto max-w-6xl px-6 pt-10">
      <h1 className="max-w-3xl text-4xl font-bold leading-tight md:text-3xl">
        Turn a bank statement into a{" "}
        <span className="bg-gradient-to-r from-brand to-brand-2 bg-clip-text text-transparent">
          300–900 credit score
        </span>
      </h1>
      <p className="mt-4 max-w-2xl text-sm leading-relaxed text-muted-foreground">
        Just in Four steps: identity, statement, declared profile, review with a
        full explanation of the decision.
      </p>
    </div>
  );
}

export function WizardShell({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <>
      <PageHero />
      <main className="mx-auto max-w-6xl px-6 py-8">
        <section className="surface overflow-hidden">
          <Stepper />
          <div className="p-6 md:p-8">
            <h2 className="text-2xl font-bold">{title}</h2>
            <p className="mt-1 text-sm text-muted-foreground">{description}</p>
            <div className="mt-7">{children}</div>
          </div>
        </section>
      </main>
    </>
  );
}

export function SiteFooter() {
  return (
    <footer className="border-t border-border/70 py-6 text-center text-xs text-muted-foreground">
      Prosperity Score is an illustrative interface only. <br />
      Bank statement data is intended to be retrieved through an RBI-regulated
      Account Aggregator,
      <br />
      PAN / Aadhaar are collected under explicit consent per the Digital
      Personal Data Protection Act, 2023.
    </footer>
  );
}

export function Field({
  label,
  children,
  error,
}: {
  label: string;
  children: ReactNode;
  error?: string;
}) {
  return (
    <label className="block">
      <span className="field-label">{label}</span>
      {children}
      {error ? (
        <span className="mt-1 block text-xs text-destructive">{error}</span>
      ) : null}
    </label>
  );
}
