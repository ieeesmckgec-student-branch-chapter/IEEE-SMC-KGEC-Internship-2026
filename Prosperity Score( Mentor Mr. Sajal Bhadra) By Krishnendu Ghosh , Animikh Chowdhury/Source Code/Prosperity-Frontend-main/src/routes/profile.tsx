import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Field, WizardShell } from "@/components/layout";
import { useApplication } from "@/lib/application";

export const Route = createFileRoute("/profile")({
  head: () => ({
    meta: [
      { title: "Prosperity Score" },
      {
        name: "description",
        content:
          "Declare your employment type and extra data points to sharpen the score.",
      },
    ],
  }),
  component: ProfilePage,
});

const EMPLOYMENT_TYPES = [
  "Salaried",
  "Self-employed",
  "Business owner",
  "Gig / freelance",
] as const;

function ProfilePage() {
  const { data, update, status } = useApplication();
  const [errors, setErrors] = useState<Record<string, string>>({});
  const navigate = useNavigate();

  // defaults (used only for rendering choices)
  const employmentType = data.employmentType || "Salaried";
  const utilityBillsOnTime = data.utilityBillsOnTime ?? true;

  // Autofill UX state (we keep a simple message for user feedback)
  const [autofillMessage, setAutofillMessage] = useState<string | null>(null);

  // Track whether the user has manually touched fields so autofill doesn't overwrite them
  const [touchedFields, setTouchedFields] = useState<{ employment?: boolean; utility?: boolean }>({});

  // Helpers
  function _n(val: any): number | null {
    if (val === null || val === undefined) return null;
    const n = Number(val);
    return Number.isFinite(n) ? n : null;
  }
  function _s(val: any): string | null {
    if (val === null || val === undefined) return null;
    return String(val);
  }
  function isEmptyField(val: any) {
    return val === undefined || val === null || val === "";
  }

  function parseBooleanLike(v: any): boolean | null {
    if (typeof v === "boolean") return v;
    if (v === null || v === undefined) return null;
    if (typeof v === "number") return v !== 0;
    const s = String(v).trim().toLowerCase();
    if (s === "true" || s === "yes" || s === "1") return true;
    if (s === "false" || s === "no" || s === "0") return false;
    return null;
  }

  // Employment normalization + tolerant matcher
  const EMPLOYMENT_NORMALIZATION: Record<string, string> = {
    salaried: "Salaried",
    salary: "Salaried",
    "self employed": "Self-employed",
    "self-employed": "Self-employed",
    selfemployed: "Self-employed",
    "business owner": "Business owner",
    business: "Business owner",
    "business-owner": "Business owner",
    gig: "Gig / freelance",
    freelance: "Gig / freelance",
    "gig / freelance": "Gig / freelance",
    "gig/freelance": "Gig / freelance",
    proprietor: "Business owner",
    entrepreneur: "Business owner",
  };

  function matchEmploymentCandidate(raw: any): string | null {
    const s = _s(raw)?.trim();
    if (!s) return null;
    // normalize: lowercase, remove non-alphanum, collapse whitespace
    const norm = s.toLowerCase().replace(/[^a-z0-9\s]/g, " ").replace(/\s+/g, " ").trim();
    if (EMPLOYMENT_NORMALIZATION[norm]) return EMPLOYMENT_NORMALIZATION[norm];
    // try substring matches against canonical types
    const found = EMPLOYMENT_TYPES.find(
      (t) =>
        t.toLowerCase() === norm ||
        t.toLowerCase().includes(norm) ||
        norm.includes(t.toLowerCase())
    );
    if (found) return found;
    return null;
  }

  // Map parsed-statement summary to form fields without overwriting valid user values
  function applySummaryToForm(summary: any) {
    if (!summary) return;
    const updates: Record<string, any> = {};

    // Numeric and simple fields (only set when empty)
    if (isEmptyField(data.monthlyIncome) && summary.monthlyIncome !== undefined && summary.monthlyIncome !== null) {
      const n = _n(summary.monthlyIncome);
      if (n !== null) updates.monthlyIncome = String(Math.round(n));
    }

    if (isEmptyField(data.avgBankBalance) && summary.avgBankBalance !== undefined && summary.avgBankBalance !== null) {
      const n = _n(summary.avgBankBalance);
      if (n !== null) updates.avgBankBalance = String(Math.round(n));
    }

    if (isEmptyField(data.monthlyRent) && summary.monthlyRent !== undefined && summary.monthlyRent !== null) {
      const n = _n(summary.monthlyRent);
      if (n !== null) updates.monthlyRent = String(Math.round(n));
    }

    if (isEmptyField(data.existingEmi) && summary.existingEmi !== undefined && summary.existingEmi !== null) {
      const n = _n(summary.existingEmi);
      if (n !== null) updates.existingEmi = String(Math.round(n));
    }

    if (isEmptyField(data.otherLoans) && summary.otherLoans !== undefined && summary.otherLoans !== null) {
      const n = _n(summary.otherLoans);
      if (n !== null) updates.otherLoans = String(Math.round(n));
    }

    // Utility bills: tolerant parsing of boolean-like values
    if (!touchedFields.utility && summary.utilityBillsOnTime !== undefined) {
      const parsed = parseBooleanLike(summary.utilityBillsOnTime);
      if (parsed !== null) updates.utilityBillsOnTime = parsed;
    }

    // Employment type: tolerant matching (case/whitespace-insensitive) - only if user hasn't touched employment
    if (!touchedFields.employment && summary.employmentType) {
      const matched = matchEmploymentCandidate(summary.employmentType);
      if (matched) {
        console.debug("Autofill: matched employmentType from summary.employmentType:", summary.employmentType, "=>", matched);
        updates.employmentType = matched;
      }
    }

    // Fallback candidates for older/alternate shapes (only fill when empty / not touched)
    if (isEmptyField(updates.monthlyIncome) && isEmptyField(data.monthlyIncome)) {
      const monthlyIncomeCandidates = [
        () => summary.avg_monthly_income,
        () => summary.features?.raw?.income?.avg_monthly_income,
        () => summary.features?.raw?.income_raw?.avg_monthly_income,
        () => summary.features?.raw?.income_raw?.effective_total_income,
        () => summary.income?.avg_monthly_income,
        () => summary.income_avg_monthly,
      ];
      for (const fn of monthlyIncomeCandidates) {
        const v = fn();
        const n = _n(v);
        if (n !== null) {
          updates.monthlyIncome = String(Math.round(n));
          break;
        }
      }
    }

    if (isEmptyField(updates.avgBankBalance) && isEmptyField(data.avgBankBalance)) {
      const avgBalanceCandidates = [
        () => summary.features?.raw?.cashflow?.avg_monthly_balance,
        () => summary.features?.raw?.cashflow?.avg_balance,
        () => summary.avg_monthly_balance,
        () => summary.avgBankBalance,
      ];
      for (const fn of avgBalanceCandidates) {
        const v = fn();
        const n = _n(v);
        if (n !== null) {
          updates.avgBankBalance = String(Math.round(n));
          break;
        }
      }
    }

    if (isEmptyField(updates.existingEmi) && isEmptyField(data.existingEmi)) {
      const existingEmiCandidates = [
        () => summary.features?.raw?.repayment?.emi_monthly_total,
        () => summary.features?.raw?.repayment?.existing_emi_amount,
        () => summary.existingEmi,
        () => summary.repayment?.emi_total_monthly,
      ];
      for (const fn of existingEmiCandidates) {
        const v = fn();
        const n = _n(v);
        if (n !== null) {
          updates.existingEmi = String(Math.round(n));
          break;
        }
      }
    }

    if (isEmptyField(updates.otherLoans) && isEmptyField(data.otherLoans)) {
      const otherLoansCandidates = [
        () => summary.features?.raw?.repayment?.emi_count,
        () => summary.repayment?.emi_count,
        () => summary.otherLoansCount,
        () => summary.otherLoans,
      ];
      for (const fn of otherLoansCandidates) {
        const v = fn();
        const n = _n(v);
        if (n !== null) {
          updates.otherLoans = String(Math.round(n));
          break;
        }
      }
    }

    if (isEmptyField(updates.monthlyRent) && isEmptyField(data.monthlyRent)) {
      const rentCandidates = [
        () => summary.features?.raw?.expense?.monthly_rent_estimate,
        () => summary.estimated_rent_monthly,
        () => summary.features?.raw?.expense?.rent_monthly,
        () => summary.monthlyRent,
      ];
      for (const fn of rentCandidates) {
        const v = fn();
        const n = _n(v);
        if (n !== null) {
          updates.monthlyRent = String(Math.round(n));
          break;
        }
      }
    }

    // Utility fallback detection (only if user hasn't touched utility)
    if (!touchedFields.utility && updates.utilityBillsOnTime === undefined) {
      const utilityCandidates = [
        () => summary.features?.raw?.behaviour?.utility_bills_on_time,
        () => summary.behaviour?.utility_bills_on_time,
        () => summary.utilityBillsOnTime,
      ];
      for (const fn of utilityCandidates) {
        const v = fn();
        const parsed = parseBooleanLike(v);
        if (parsed !== null) {
          updates.utilityBillsOnTime = parsed;
          break;
        }
      }
    }

    // Employment fallback detection (use tolerant matcher), only if user hasn't touched
    if (!touchedFields.employment && isEmptyField(updates.employmentType) && isEmptyField(data.employmentType)) {
      const empCandidates = [
        () => summary.profile?.employmentType,
        () => summary.features?.raw?.income?.inferred_employment_type,
        () => summary.employmentType,
      ];
      for (const fn of empCandidates) {
        const v = fn();
        const matched = matchEmploymentCandidate(v);
        if (matched) {
          console.debug("Autofill: matched employmentType from candidate:", v, "=>", matched);
          updates.employmentType = matched;
          break;
        }
      }
    }

    if (Object.keys(updates).length > 0) {
      update(updates);
      console.debug("Autofill applied updates:", updates);
      setAutofillMessage("Auto-filled available fields from statement");
      // clear message after a few seconds
      setTimeout(() => setAutofillMessage(null), 3500);
    } else {
      setAutofillMessage("No usable fields detected in statement summary to auto-fill");
      setTimeout(() => setAutofillMessage(null), 3500);
    }
  }

  // Fetch server summary (by id or cookie) and persist it into app state when found
  async function fetchServerSummary(id?: string) {
    let url = "/api/statement/summary";
    if (id) url += `?id=${encodeURIComponent(id)}`;
    try {
      const resp = await fetch(url, { method: "GET", credentials: "include" });
      if (!resp.ok) return null;
      const json = await resp.json();
      console.debug("Fetched statement summary (for autofill):", json);
      return json;
    } catch (err) {
      console.debug("Autofill fetch failed", err);
      return null;
    }
  }

  // Autofill on mount: check in-memory summary first, else try server (id -> cookie)
  useEffect(() => {
    let cancelled = false;
    async function tryAutofill() {
      // If in-memory summary exists, apply it (fast)
      if (data.statementSummary) {
        try {
          applySummaryToForm(data.statementSummary);
        } catch (e) {
          console.error("Autofill apply failed:", e);
        }
        // Still attempt server fetch by id to refresh the cached summary if id present
        if (data.statementSummaryId) {
          const fresh = await fetchServerSummary(data.statementSummaryId);
          if (cancelled) return;
          if (fresh) {
            update({ statementSummary: fresh });
            applySummaryToForm(fresh);
          }
        }
        return;
      }

      // If we have an id stored, try fetching by id first
      if (data.statementSummaryId) {
        const json = await fetchServerSummary(data.statementSummaryId);
        if (cancelled) return;
        if (json) {
          update({ statementSummary: json });
          applySummaryToForm(json);
          return;
        }
      }

      // Last resort: cookie-based fetch (server may use HttpOnly cookie to find summary)
      const json = await fetchServerSummary();
      if (cancelled) return;
      if (json) {
        update({ statementSummary: json });
        applySummaryToForm(json);
      }
    }

    tryAutofill();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // run once on mount

  function next() {
    const e: Record<string, string> = {};

    const numericFields = [
      "monthlyIncome",
      "avgBankBalance",
      "monthlyRent",
      "existingEmi",
      "otherLoans",
    ] as const;
    for (const key of numericFields) {
      if (data[key] === "") e[key] = "Enter 0 if not applicable";
    }

    if (Number(data.monthlyIncome) <= 0) e.monthlyIncome = "Income must be greater than 0";

    setErrors(e);
    if (Object.keys(e).length === 0) navigate({ to: "/review" });
  }

  return (
    <WizardShell
      title="Financial Profile"
      description="A few extra data points sharpen the score alongside what your statement already shows."
    >
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="text-sm text-muted-foreground">
            Fields below will auto-fill from your uploaded bank statement when possible.
          </div>
          <div className="flex items-center gap-2">
            {autofillMessage && <span className="text-xs text-muted-foreground">{autofillMessage}</span>}
          </div>
        </div>

        <div>
          <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground block mb-2">
            Employment Type
          </label>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {EMPLOYMENT_TYPES.map((type) => (
              <button
                key={type}
                type="button"
                onClick={() => {
                  update({ employmentType: type });
                  setTouchedFields((t) => ({ ...t, employment: true }));
                }}
                className={`py-3 px-4 border text-sm font-medium rounded-md transition-all ${
                  employmentType === type ? "bg-primary text-primary-foreground border-primary" : "bg-background text-foreground border-border hover:bg-muted"
                }`}
              >
                {type}
              </button>
            ))}
          </div>
        </div>

        <div className="grid gap-5 md:grid-cols-2">
          <Field label="Monthly Income (₹)" error={errors.monthlyIncome}>
            <input
              className="field-input"
              inputMode="numeric"
              placeholder="45000"
              value={data.monthlyIncome || ""}
              onChange={(e) => update({ monthlyIncome: e.target.value.replace(/\D/g, "") })}
            />
          </Field>

          <Field label="Avg. Bank Balance (₹)" error={errors.avgBankBalance}>
            <input
              className="field-input"
              inputMode="numeric"
              placeholder="12000"
              value={data.avgBankBalance || ""}
              onChange={(e) => update({ avgBankBalance: e.target.value.replace(/\D/g, "") })}
            />
          </Field>

          <Field label="Monthly Rent (₹, 0 if none)" error={errors.monthlyRent}>
            <input
              className="field-input"
              inputMode="numeric"
              placeholder="8000"
              value={data.monthlyRent || ""}
              onChange={(e) => update({ monthlyRent: e.target.value.replace(/\D/g, "") })}
            />
          </Field>

          <Field label="Existing EMI (₹, 0 if none)" error={errors.existingEmi}>
            <input
              className="field-input"
              inputMode="numeric"
              placeholder="0"
              value={data.existingEmi || ""}
              onChange={(e) => update({ existingEmi: e.target.value.replace(/\D/g, "") })}
            />
          </Field>

          <Field label="Other Open Loans (Count)" error={errors.otherLoans}>
            <input
              className="field-input"
              inputMode="numeric"
              placeholder="0"
              value={data.otherLoans || ""}
              onChange={(e) => update({ otherLoans: e.target.value.replace(/\D/g, "") })}
            />
          </Field>

          <div>
            <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground block mb-2">
              Utility Bills Paid on Time?
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => {
                  update({ utilityBillsOnTime: true });
                  setTouchedFields((t) => ({ ...t, utility: true }));
                }}
                className={`py-3 px-4 border text-sm font-medium rounded-md transition-all ${
                  utilityBillsOnTime === true ? "bg-primary text-primary-foreground border-primary" : "bg-background text-foreground border-border hover:bg-muted"
                }`}
              >
                Yes
              </button>
              <button
                type="button"
                onClick={() => {
                  update({ utilityBillsOnTime: false });
                  setTouchedFields((t) => ({ ...t, utility: true }));
                }}
                className={`py-3 px-4 border text-sm font-medium rounded-md transition-all ${
                  utilityBillsOnTime === false ? "bg-primary text-primary-foreground border-primary" : "bg-background text-foreground border-border hover:bg-muted"
                }`}
              >
                No
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-8 flex flex-wrap items-center justify-between gap-4 border-t border-border pt-6">
        <div className="flex gap-3">
          <button type="button" className="btn-ghost" onClick={() => navigate({ to: "/statement" })}>
            ← Back
          </button>
          <button type="button" className="btn-primary disabled:opacity-60" disabled={status === "uploading"} onClick={next}>
            Continue →
          </button>
        </div>
      </div>
    </WizardShell>
  );
}

export default ProfilePage;