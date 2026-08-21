import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { ComponentScores, PipelineResult } from "@/lib/scoring-api";

export type ApplicationData = {
  fullName: string;
  occupation: string;
  pan: string;
  aadhaar: string;
  mobile: string;
  email: string;
  city: string;
  pin: string;
  statementFileName: string;
  statementRows: number;
  consentStatement: boolean;
  employmentType: string;
  monthlyIncome: string;
  avgBankBalance: string;
  utilityBillsOnTime: boolean;
  monthlyRent: string;
  existingEmi: string;
  otherLoans: string;
  loanAmount: string;
  tenure: string;
  consentDpdp: boolean;
  /** JSON result returned by the Python scoring pipeline API for the uploaded statement. */
  apiResult: PipelineResult | null;
};

export const emptyApplication: ApplicationData = {
  fullName: "",
  occupation: "",
  pan: "",
  aadhaar: "",
  mobile: "",
  email: "",
  city: "",
  pin: "",
  statementFileName: "",
  statementRows: 0,
  consentStatement: false,
  employmentType: "Salaried",
  monthlyIncome: "",
  avgBankBalance: "",
  utilityBillsOnTime: true,
  monthlyRent: "",
  existingEmi: "",
  otherLoans: "",
  loanAmount: "",
  tenure: "36",
  consentDpdp: false,
  apiResult: null,
};

export const sampleApplication: ApplicationData = {
  fullName: "Animikh Chowdhury",
  occupation: "Owner",
  pan: "ABKPC4127T",
  aadhaar: "482913451023",
  mobile: "9830233062",
  email: "animikh@example.com",
  city: "Howrah",
  pin: "700054",
  statementFileName: "Sanjay_Shah_BankStatement.xlsx",
  statementRows: 10788,
  consentStatement: true,
  employmentType: "Business owner",
  monthlyIncome: "78000",
  avgBankBalance: "34500",
  utilityBillsOnTime: true,
  monthlyRent: "8000",
  existingEmi: "15000",
  otherLoans: "0",
  loanAmount: "500000",
  tenure: "36",
  consentDpdp: true,
  apiResult: {
    component_scores: {
      income: 71.2,
      expense: 64.8,
      cashflow: 68.5,
      repayment: 77.4,
      behaviour: 62.1,
    },
    final_score: { final_score: 70.9, credit_score: 725.4 },
    eligibility: {
      decision: "APPROVE",
      reasons: ["Score in conditional range"],
      eligible_emi: 23400,
      approx_max_loan: 842400,
    },
    risk: { bucket: "Medium Risk", action: "Conditional Approval" },
    explanations: [
      "Stable and sufficient income",
      "Few bounced transactions",
      "Expense-to-income ratio is risky",
      "No glaring risk signals detected",
    ],
    raw: {
      income: {
        avg_monthly_income: 78000,
        income_stability_score: 0.74,
        income_consistency_score: 0.74,
        income_source_diversity_score: 0.4,
        income_growth_slope: 620.5,
        income_growth_score: 0.61,
        income_component_score: 71.2,
      },
      expense: {
        total_expenses: 54000,
        eir: 0.69,
        eir_score: 0.55,
        fixed_ratio: 0.42,
        variable_ratio: 0.33,
        discretionary_ratio: 0.25,
        discretionary_score: 0.6,
        expense_trend_slope: -120.3,
        expense_trend_score: 0.58,
        expense_component_score: 64.8,
      },
      cashflow: {
        avg_monthly_balance: 34500,
        min_monthly_balance: 6200,
        negative_balance_count: 0,
        min_balance_trend_slope: 45.1,
        cashflow_stability_score: 0.69,
        cashflow_component_score: 68.5,
      },
      repayment: {
        emi_count: 18,
        on_time_ratio: 0.94,
        bounce_count: 1,
        bounce_rate: 0.01,
        credit_card_min_ratio: 0.1,
        repayment_component_score: 77.4,
      },
      behaviour: {
        savings_rate: 0.18,
        financial_discipline_score: 0.62,
        spending_spike_score: 0.71,
        end_of_month_stress_score: 0.66,
        behaviour_component_score: 62.1,
      },
    },
    used_synthetic_data: false,
  },
};

const STORAGE_KEY = "prosperity-score-application";

type Ctx = {
  data: ApplicationData;
  update: (patch: Partial<ApplicationData>) => void;
  loadSample: () => void;
  reset: () => void;
  hydrated: boolean;
};

const ApplicationContext = createContext<Ctx | null>(null);

export function ApplicationProvider({ children }: { children: ReactNode }) {
  const [data, setData] = useState<ApplicationData>(emptyApplication);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (raw) setData({ ...emptyApplication, ...JSON.parse(raw) });
    } catch {
      /* ignore */
    }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch {
      /* ignore */
    }
  }, [data, hydrated]);

  const value = useMemo<Ctx>(
    () => ({
      data,
      hydrated,
      update: (patch) => setData((prev) => ({ ...prev, ...patch })),
      loadSample: () => setData(sampleApplication),
      reset: () => setData(emptyApplication),
    }),
    [data, hydrated],
  );

  return (
    <ApplicationContext.Provider value={value}>
      {children}
    </ApplicationContext.Provider>
  );
}

export function useApplication() {
  const ctx = useContext(ApplicationContext);
  if (!ctx)
    throw new Error("useApplication must be used inside ApplicationProvider");
  return ctx;
}

/* ---------- helpers ---------- */

export function maskPan(pan: string) {
  if (!pan) return "—";
  const value = pan.toUpperCase();
  if (value.length < 4) return value;
  return `${value.slice(0, 2)}•••••${value.slice(-1)}`;
}

export function maskAadhaar(aadhaar: string) {
  const digits = aadhaar.replace(/\D/g, "");
  if (!digits) return "—";
  return `•••• •••• ${digits.slice(-4)}`;
}

export function rupees(value: string | number) {
  const n = typeof value === "number" ? value : Number(value || 0);
  if (!Number.isFinite(n)) return "₹0";
  return `₹${n.toLocaleString("en-IN")}`;
}

export type ScoreResult = {
  score: number;
  band: string;
  decision: "APPROVE" | "REVIEW" | "DECLINE";
  composite: number;
  eligibleEmi: number;
  maxLoan: number;
  pillars: { key: string; label: string; value: number; weight: string }[];
  /** Present when this result came from the live scoring pipeline API. */
  source: "api" | "estimate";
  reasons?: string[];
};

const clamp = (n: number, min = 0, max = 100) =>
  Math.max(min, Math.min(max, n));

const PILLAR_META: Record<string, { label: string; weight: string }> = {
  income: { label: "Income", weight: "30% weight" },
  expense: { label: "Expense", weight: "20% weight" },
  cashflow: { label: "Cashflow", weight: "20% weight" },
  repayment: { label: "Repayment", weight: "30% weight" },
  behaviour: { label: "Behaviour", weight: "signal only" },
};

/** Map the eligibility decision string returned by the pipeline onto the
 * simplified APPROVE / REVIEW / DECLINE badge used by the UI. */
function mapDecision(decision: string): ScoreResult["decision"] {
  const d = decision.toUpperCase();
  if (d === "APPROVE") return "APPROVE";
  if (d === "CONDITIONAL") return "REVIEW";
  return "DECLINE";
}

/** Convert the JSON returned by the Python scoring pipeline API into the
 * ScoreResult shape the result page renders. */
export function mapPipelineResult(apiResult: PipelineResult): ScoreResult {
  const pillars = (
    Object.keys(apiResult.component_scores) as (keyof ComponentScores)[]
  ).map((key) => ({
    key,
    label: PILLAR_META[key]?.label ?? key,
    value: Math.round(apiResult.component_scores[key] * 10) / 10,
    weight: PILLAR_META[key]?.weight ?? "",
  }));

  return {
    score: Math.round(apiResult.final_score.credit_score),
    band: apiResult.risk.bucket,
    decision: mapDecision(apiResult.eligibility.decision),
    composite: Math.round(apiResult.final_score.final_score * 100) / 100,
    eligibleEmi: Math.round(apiResult.eligibility.eligible_emi),
    maxLoan: Math.max(0, Math.round(apiResult.eligibility.approx_max_loan)),
    pillars,
    source: "api",
    reasons: apiResult.explanations,
  };
}

/** Return the result to display: the real pipeline output when available,
 * otherwise a rough client-side estimate from the declared profile fields. */
export function getDisplayResult(data: ApplicationData): ScoreResult {
  if (data.apiResult) return mapPipelineResult(data.apiResult);
  return computeScore(data);
}

export function computeScore(data: ApplicationData): ScoreResult {
  const income = Number(data.monthlyIncome || 0);
  const rent = Number(data.monthlyRent || 0);
  const emi = Number(data.existingEmi || 0);
  const loans = Number(data.otherLoans || 0);
  const requested = Number(data.loanAmount || 0);
  const tenure = Math.max(6, Number(data.tenure || 36));

  const obligations = rent + emi;
  const surplus = Math.max(0, income - obligations);
  const hasStatement = Boolean(data.statementFileName);

  const incomeScore = clamp((income / 100000) * 100);
  const expenseScore = clamp(
    100 - (income ? (obligations / income) * 130 : 100),
  );
  const cashflowScore = clamp(income ? (surplus / income) * 120 : 0);
  const repaymentScore = clamp(
    100 - (income ? (emi / income) * 200 : 60) - (loans > 0 ? 12 : 0),
  );
  const behaviourScore = clamp(
    55 +
      (hasStatement ? 22 : 0) +
      (data.email ? 6 : 0) +
      (loans === 0 ? 10 : 0),
  );

  const composite =
    incomeScore * 0.3 +
    expenseScore * 0.2 +
    cashflowScore * 0.2 +
    repaymentScore * 0.3;

  const score = Math.round(300 + (composite / 100) * 600);

  const eligibleEmi = Math.round(Math.max(0, surplus * 0.55));
  const monthlyRate = 0.14 / 12;
  const maxLoan = Math.round(
    eligibleEmi > 0
      ? (eligibleEmi * (1 - Math.pow(1 + monthlyRate, -tenure))) / monthlyRate
      : 0,
  );

  const band =
    score >= 750 ? "Low Risk" : score >= 650 ? "Moderate Risk" : "High Risk";
  const decision: ScoreResult["decision"] =
    score >= 750 ? "APPROVE" : score >= 650 ? "REVIEW" : "DECLINE";

  return {
    score,
    band,
    decision,
    composite: Math.round(composite * 100) / 100,
    eligibleEmi,
    maxLoan: Math.max(maxLoan, 0) || 0,
    pillars: [
      {
        key: "income",
        label: "Income",
        value: Math.round(incomeScore * 10) / 10,
        weight: "30% weight",
      },
      {
        key: "expense",
        label: "Expense",
        value: Math.round(expenseScore * 10) / 10,
        weight: "20% weight",
      },
      {
        key: "cashflow",
        label: "Cashflow",
        value: Math.round(cashflowScore * 10) / 10,
        weight: "20% weight",
      },
      {
        key: "repayment",
        label: "Repayment",
        value: Math.round(repaymentScore * 10) / 10,
        weight: "30% weight",
      },
      {
        key: "behaviour",
        label: "Behaviour",
        value: Math.round(behaviourScore * 10) / 10,
        weight: "signal only",
      },
    ],
    source: "estimate",
    ...(requested ? {} : {}),
  };
}
