import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { WizardShell } from "@/components/layout";
import { maskAadhaar, maskPan, rupees, useApplication } from "@/lib/application";

export const Route = createFileRoute("/review")({
  head: () => ({
    meta: [
      { title: "Prosperity Score" },
      {
        name: "description",
        content:
          "Masked summary of the captured application data with PAN and Aadhaar redacted, then generate the Prosperity Score.",
      },
      { property: "og:title", content: "Review & generate — ProsperityScore" },
      {
        property: "og:description",
        content: "Step 4: confirm masked applicant details and consent before scoring.",
      },
    ],
  }),
  component: ReviewPage,
});

function Cell({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-border bg-panel/40 p-4">
      <p className="field-label mb-1 text-xs text-muted-foreground font-semibold uppercase tracking-wider">{label}</p>
      <p className="truncate text-sm font-medium">{value || "—"}</p>
    </div>
  );
}

function ReviewPage() {
  const { data, update, loadSample } = useApplication();
  const navigate = useNavigate();
  const [error, setError] = useState("");

  return (
    <WizardShell
      title="Review & generate"
      description="Masked summary of captured data, then score generation."
    >
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {/* Core Profile Fields */}
        <Cell label="Full name" value={data.fullName} />
        <Cell label="Employment Type" value={data.employmentType} />
        <Cell label="PAN" value={maskPan(data.pan)} />
        <Cell label="Aadhaar" value={maskAadhaar(data.aadhaar)} />
        <Cell label="Mobile" value={data.mobile ? `+91 ${data.mobile}` : ""} />
        <Cell label="Email" value={data.email} />
        <Cell label="City / PIN" value={[data.city, data.pin].filter(Boolean).join(" ")} />
        
        {/* Updated Financial Fields */}
        <Cell label="Monthly Income" value={rupees(data.monthlyIncome)} />
        <Cell label="Avg. Bank Balance" value={rupees(data.avgBankBalance)} />
        <Cell label="Monthly Rent" value={Number(data.monthlyRent) > 0 ? rupees(data.monthlyRent) : "0 (None)"} />
        <Cell label="Existing EMI" value={Number(data.existingEmi) > 0 ? rupees(data.existingEmi) : "0 (None)"} />
        <Cell label="Other Open Loans" value={data.otherLoans ? `${data.otherLoans} Loans` : "0"} />
        <Cell label="Utility Bills On Time" value={data.utilityBillsOnTime ? "Yes" : "No"} />
        
        {/* Consent/Evidence Attachments */}
        <Cell label="Evidence" value={data.statementFileName} />
      </div>

      <label className="mt-6 flex items-start gap-3 rounded-xl border border-border bg-panel/40 p-5 text-sm cursor-pointer select-none">
        <input
          type="checkbox"
          className="mt-0.5 h-4 w-4 accent-[oklch(0.66_0.16_250)]"
          checked={data.consentDpdp}
          onChange={(e) => update({ consentDpdp: e.target.checked })}
        />
        <span className="text-muted-foreground">
          DPDP consent: the applicant permits processing of PAN and Aadhaar details for the purpose
          of generating this Prosperity Score.
        </span>
      </label>

      {error ? <p className="mt-4 text-sm text-destructive">{error}</p> : null}

      <div className="mt-8 flex flex-wrap items-center justify-between gap-4 border-t border-border pt-6">
        
        <div className="flex gap-3">
        
          <button
            className="btn-primary"
            onClick={() => {
              if (!data.consentDpdp) {
                setError("DPDP consent is required before generating a score.");
                return;
              }
              navigate({ to: "/result" });
            }}
          >
            ✦ Generate score
          </button>
        </div>
      </div>
    </WizardShell>
  );
}
