import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { getDisplayResult, maskPan, rupees, useApplication } from "@/lib/application";

export const Route = createFileRoute("/result")({
  head: () => ({
    meta: [
      { title: "Prosperity Score" },
      {
        name: "description",
        content:
          "The final 300–900 Prosperity Score with risk band, decision, eligible EMI and maximum loan offer for the applicant.",
      },
      { property: "og:title", content: "Score result — ProsperityScore" },
      {
        property: "og:description",
        content: "Final credit score, risk band and maximum loan offer for a thin-file borrower.",
      },
    ],
  }),
  component: ResultPage,
});

function Gauge({ score }: { score: number }) {
  const pct = Math.min(1, Math.max(0, (score - 300) / 600));
  const r = 88;
  const circumference = Math.PI * r;
  return (
    <div className="relative mx-auto w-full max-w-[240px]">
      <svg viewBox="0 0 200 120" className="w-full">
        <defs>
          <linearGradient id="gaugeGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="oklch(0.6 0.18 275)" />
            <stop offset="100%" stopColor="oklch(0.8 0.13 205)" />
          </linearGradient>
        </defs>
        <path
          d="M12 108 A 88 88 0 0 1 188 108"
          fill="none"
          stroke="oklch(0.32 0.02 258)"
          strokeWidth="16"
          strokeLinecap="round"
        />
        <path
          d="M12 108 A 88 88 0 0 1 188 108"
          fill="none"
          stroke="url(#gaugeGrad)"
          strokeWidth="16"
          strokeLinecap="round"
          strokeDasharray={`${circumference * pct} ${circumference}`}
        />
      </svg>
      <div className="pointer-events-none absolute inset-x-0 bottom-2 text-center">
        <p className="font-display text-5xl font-bold">{score}</p>
        <p className="text-xs text-muted-foreground">of 900</p>
      </div>
    </div>
  );
}

function ResultPage() {
  const { data,reset } = useApplication();
  const navigate = useNavigate();
  const result = getDisplayResult(data);

  if (!data.fullName) {
    return (
      <main className="mx-auto max-w-3xl px-6 py-24 text-center">
        <h1 className="text-2xl font-bold">No application yet</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Complete the four steps to generate a Prosperity Score.
        </p>
        <button className="btn-primary mt-6" onClick={() => navigate({ to: "/" })}>
          Start a new applicant
        </button>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <section className="surface p-6 md:p-8">
        <div className="grid gap-8 md:grid-cols-[280px_1fr]">
          <div className="rounded-2xl border border-border bg-panel/40 p-6 text-center">
            <Gauge score={result.score} />
            <div className="mt-4 flex items-center justify-center gap-3 text-xs text-muted-foreground">
              <span>300</span>
              <span className="rounded-full bg-muted px-3 py-1 text-xs font-medium text-foreground">
                {result.band}
              </span>
              <span>900</span>
            </div>
            <p className="field-label mt-6 mb-0">Prosperity score</p>
            <p className="text-sm text-muted-foreground">Composite {result.composite} / 100</p>
          </div>

          <div>
            <h1 className="text-3xl font-bold">{data.fullName}</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              {[data.occupation, data.city, `PAN ${maskPan(data.pan)}`].filter(Boolean).join(" · ")}
            </p>
            

            <div className="mt-6 grid gap-4 sm:grid-cols-2">
              <div className="rounded-xl border border-border bg-panel/40 p-4">
                <p className="field-label mb-1">Eligible EMI</p>
                <p className="font-display text-xl font-bold">{rupees(result.eligibleEmi)}</p>
              </div>
              <div className="rounded-xl border border-border bg-panel/40 p-4">
                <p className="field-label mb-1">Max loan Amount Permisable ≈</p>
                <p className="font-display text-xl font-bold">{rupees(result.maxLoan)}</p>
              </div>
            </div>

            <div className="mt-4 rounded-xl border border-border bg-panel/40 p-4">
              <p className="field-label mb-1">Risk action</p>
              <p className="text-sm">
                {result.band}
              </p>
            </div>
          </div>
        </div>
      </section>

      {result.reasons && result.reasons.length > 0 ? (
        <section className="mt-6 surface p-6">
          <p className="field-label mb-3">Why this score</p>
          <ul className="grid gap-2 sm:grid-cols-2">
            {result.reasons.map((reason, i) => (
              <li
                key={i}
                className="rounded-lg border border-border bg-panel/40 px-3 py-2 text-sm text-muted-foreground"
              >
                {reason}
              </li>
            ))}
          </ul>
        </section>
      ) : null}
      <div className="mt-8 flex justify-end">
        <button 
          className="btn-ghost" 
          onClick={() => {
            reset(); // Clears all form store states instantly
            navigate({ to: "/" });
          }}
        >
          New applicant
        </button>
      </div>
    </main>
  );
}

