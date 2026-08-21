// src/routes/statement.tsx
import { Loader2, UploadCloud } from "lucide-react";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useRef, useState } from "react";
import { WizardShell } from "@/components/layout";
import { useApplication } from "@/lib/application";
import { scoreStatement, ScoringApiError } from "@/lib/scoring-api";

export const Route = createFileRoute("/statement")({
  head: () => ({
    meta: [
      { title: "Prosperity Score" },
      {
        name: "description",
        content:
          "Upload a CSV or XLSX bank statement. It is sent to the ProsperityScore pipeline API for scoring.",
      },
      {
        property: "og:title",
        content: "Bank statement upload — ProsperityScore",
      },
      {
        property: "og:description",
        content:
          "Step 2: provide the applicant's bank statement as transaction evidence.",
      },
    ],
  }),
  component: StatementPage,
});

type UploadStatus = "idle" | "uploading" | "success" | "error";

function StatementPage() {
  const { data, update, loadSample } = useApplication();
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState("");
  const [status, setStatus] = useState<UploadStatus>(
    data.apiResult ? "success" : "idle",
  );

  // Helper to detect "empty" fields in your app state (treat '' / undefined / null as empty)
  function isEmptyField(val: any) {
    return val === undefined || val === null || val === "";
  }

  async function handleFile(file: File | undefined) {
    if (!file) return;
    const ok = /\.(csv|xlsx|xls)$/i.test(file.name);
    if (!ok) {
      setError("Only CSV or XLSX files are accepted.");
      return;
    }
    setError("");

    let rows = 0;
    if (/\.csv$/i.test(file.name)) {
      const text = await file.text();
      rows = text.split(/\r?\n/).filter((l) => l.trim()).length - 1;
    }

    // Show the file immediately, then kick off the pipeline call.
    update({
      statementFileName: file.name,
      statementRows: Math.max(rows, 0),
      apiResult: null,
    });
    setStatus("uploading");

    try {
      const result = await scoreStatement(file);

      // Save the full API response for other UI (existing behaviour)
      update({ apiResult: result });

      // NEW: store parsed statement summary (returned by server as _statement_summary)
      const summary = (result as any)?._statement_summary;
      if (summary) {
        // Prepare updates: always store the summary object, and also fill individual form fields
        const derivedUpdates: Record<string, any> = { statementSummary: summary };

        // Only overwrite if fields are empty (so we don't stomp user input)
        if (isEmptyField(data.monthlyIncome) && summary.monthlyIncome != null) {
          // store as string (form expects strings)
          derivedUpdates.monthlyIncome = String(Math.round(Number(summary.monthlyIncome) || 0));
        }
        if (isEmptyField(data.avgBankBalance) && summary.avgBankBalance != null) {
          derivedUpdates.avgBankBalance = String(Math.round(Number(summary.avgBankBalance) || 0));
        }
        if (isEmptyField(data.monthlyRent) && summary.monthlyRent != null) {
          derivedUpdates.monthlyRent = String(Math.round(Number(summary.monthlyRent) || 0));
        }
        if (isEmptyField(data.existingEmi) && summary.existingEmi != null) {
          derivedUpdates.existingEmi = String(Math.round(Number(summary.existingEmi) || 0));
        }
        if (isEmptyField(data.otherLoans) && summary.otherLoans != null) {
          derivedUpdates.otherLoans = String(Math.round(Number(summary.otherLoans) || 0));
        }
        // utilityBillsOnTime: boolean (default false if missing)
        if ((data.utilityBillsOnTime === undefined || data.utilityBillsOnTime === null) && summary.utilityBillsOnTime != null) {
          derivedUpdates.utilityBillsOnTime = Boolean(summary.utilityBillsOnTime);
        }
        // employmentType: only if empty and provided by summary
        if (isEmptyField(data.employmentType) && summary.employmentType) {
          derivedUpdates.employmentType = String(summary.employmentType);
        }

        update(derivedUpdates);
      } else if (result && (result as any)._statement_id) {
        // fallback: store id so profile page can query by id if cookies/credentials are working
        update({ statementSummaryId: (result as any)._statement_id });
      }

      setStatus("success");
    } catch (err) {
      setStatus("error");
      setError(
        err instanceof ScoringApiError
          ? err.message
          : "Could not score this statement. Please try again.",
      );
    }
  }

  function next() {
    if (!data.statementFileName) {
      setError("Upload a bank statement to continue.");
      return;
    }
    if (!data.consentStatement) {
      setError("Applicant consent is required.");
      return;
    }
    if (status === "uploading") {
      setError("Please wait for the statement to finish scoring.");
      return;
    }
    if (status === "error" || !data.apiResult) {
      setError(
        "Scoring failed for this statement — retry the upload before continuing.",
      );
      return;
    }
    setError("");
    navigate({ to: "/profile" });
  }

  return (
    <WizardShell
      title="Bank statement"
      description="Upload the applicant's statement (CSV / XLSX)"
    >
      <div className="rounded-xl border border-border bg-panel/40 p-6">
        <p className="field-label">Bank statement (CSV / XLSX)</p>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.xlsx,.xls"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={status === "uploading"}
          className="btn-ghost w-full justify-center disabled:opacity-60"
        >
          {status === "uploading" ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
              Scoring statement…
            </>
          ) : (
            <>
              <UploadCloud className="h-4 w-4" aria-hidden />
              Choose file
            </>
          )}
        </button>

        <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
          <span>
            {data.statementFileName
              ? `${data.statementFileName}${data.statementRows ? ` · ${data.statementRows.toLocaleString("en-IN")} transactions detected` : ""}`
              : "No file selected"}
          </span>
          {status === "success" ? (
            <span className="rounded-full bg-muted px-2 py-0.5 font-medium">
              Upload Successful…
            </span>
          ) : null}
          {status === "uploading" ? (
            <span className="rounded-full bg-muted px-2 py-0.5 font-medium">
              Running pipeline…
            </span>
          ) : null}
        </div>
      </div>

      <label className="mt-6 flex items-start gap-3 rounded-xl border border-border bg-panel/40 p-5 text-sm">
        <input
          type="checkbox"
          className="mt-0.5 h-4 w-4 accent-[var(--primary)]"
          checked={data.consentStatement}
          onChange={(e) => update({ consentStatement: e.target.checked })}
        />
        <span className="text-muted-foreground">
          The applicant consents to their bank statement being read and analysed
          for credit assessment for this application only.
        </span>
      </label>

      {error ? <p className="mt-4 text-sm text-destructive">{error}</p> : null}

      <div className="mt-8 flex flex-wrap items-center justify-between gap-4 border-t border-border pt-6">
        <div className="flex gap-3">
          
          <button
            type="button"
            className="btn-primary disabled:opacity-60"
            disabled={status === "uploading"}
            onClick={next}
          >
            Continue →
          </button>
        </div>
      </div>
    </WizardShell>
  );
}

export default StatementPage;