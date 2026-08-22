import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { Field, WizardShell } from "@/components/layout";
import { useApplication } from "@/lib/application";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Prosperity Score" },
      {
        name: "description",
        content:
          "Start a ProsperityScore application: capture name, PAN, Aadhaar, mobile and address for identity verification.",
      },
      { property: "og:title", content: "Prosperity Score" },
      {
        property: "og:description",
        content: "Alternative Credit Ledger",
      },
    ],
  }),
  component: IdentityPage,
});

type Errors = Partial<Record<string, string>>;

function IdentityPage() {
  const { data, update, loadSample } = useApplication();
  const [errors, setErrors] = useState<Errors>({});
  const navigate = useNavigate();

  function validate(): boolean {
    const e: Errors = {};
    if (data.fullName.trim().length < 3) e.fullName = "Enter at least 3 characters";
    if (!data.occupation.trim()) e.occupation = "Required";
    if (!/^[A-Z]{5}[0-9]{4}[A-Z]$/.test(data.pan.toUpperCase())) e.pan = "Format: ABCDE1234F";
    if (!/^\d{12}$/.test(data.aadhaar)) e.aadhaar = "Aadhaar must be 12 digits";
    if (!/^[6-9]\d{9}$/.test(data.mobile)) e.mobile = "10 digits, starting 6–9";
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(data.email)) e.email = "Enter a valid email";
    if (!data.city.trim()) e.city = "Required";
    if (!/^\d{6}$/.test(data.pin)) e.pin = "PIN must be 6 digits";
    setErrors(e);
    return Object.keys(e).length === 0;
  }

  return (
    <WizardShell
      title="Identity & KYC"
      description="Capture name, PAN, Aadhaar, mobile and address for identity verification."
    >
      <form
        onSubmit={(ev) => {
          ev.preventDefault();
          if (validate()) navigate({ to: "/statement" });
        }}
      >
        <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-4">
          <Field label="Full name" error={errors.fullName}>
            <input
              className="field-input"
              placeholder="e.g. Rakesh Ghosh"
              value={data.fullName}
              onChange={(e) => update({ fullName: e.target.value })}
            />
          </Field>
          <Field label="Business / Occupation" error={errors.occupation}>
            <input
              className="field-input"
              placeholder="e.g. Tea stall owner"
              value={data.occupation}
              onChange={(e) => update({ occupation: e.target.value })}
            />
          </Field>
          <Field label="PAN" error={errors.pan}>
            <input
              className="field-input uppercase"
              placeholder="ABCDE1234F"
              maxLength={10}
              value={data.pan}
              onChange={(e) => update({ pan: e.target.value.toUpperCase() })}
            />
          </Field>
          <Field label="Aadhaar (12 digits)" error={errors.aadhaar}>
            <input
              className="field-input"
              placeholder="123412341234"
              inputMode="numeric"
              maxLength={12}
              value={data.aadhaar}
              onChange={(e) => update({ aadhaar: e.target.value.replace(/\D/g, "") })}
            />
          </Field>
          <Field label="Mobile (+91)" error={errors.mobile}>
            <input
              className="field-input"
              placeholder="9XXXXXXXXX"
              inputMode="numeric"
              maxLength={10}
              value={data.mobile}
              onChange={(e) => update({ mobile: e.target.value.replace(/\D/g, "") })}
            />
          </Field>
          <Field label="Email" error={errors.email}>
            <input
              className="field-input"
              placeholder="name@example.com"
              value={data.email}
              onChange={(e) => update({ email: e.target.value })}
            />
          </Field>
          <Field label="City" error={errors.city}>
            <input
              className="field-input"
              placeholder="Kolkata"
              value={data.city}
              onChange={(e) => update({ city: e.target.value })}
            />
          </Field>
          <Field label="PIN code" error={errors.pin}>
            <input
              className="field-input"
              placeholder="700001"
              inputMode="numeric"
              maxLength={6}
              value={data.pin}
              onChange={(e) => update({ pin: e.target.value.replace(/\D/g, "") })}
            />
          </Field>
        </div>

        <div className="mt-8 flex flex-wrap items-center justify-between gap-4 border-t border-border pt-6">
          
          <button type="submit" className="btn-primary">
            Continue → 
          </button>
        </div>
      </form>
    </WizardShell>
  );
}
