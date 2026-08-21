import {
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  ClipboardPlus,
  Database,
  History,
  ShieldCheck,
} from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { AppLayout } from "../components/AppLayout";
import { useAuth } from "../context/AuthContext";
import { getAuthenticatedUser } from "../services/api";

type BackendUser = {
  uid: string;
  email: string | null;
  email_verified: boolean;
  name: string | null;
};

export function DashboardPage() {
  const { user } = useAuth();

  const [backendUser, setBackendUser] =
    useState<BackendUser | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function verifyUserWithBackend() {
      try {
        const result =
          await getAuthenticatedUser();

        setBackendUser(result.user);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Authentication verification failed."
        );
      } finally {
        setLoading(false);
      }
    }

    verifyUserWithBackend();
  }, []);

  return (
    <AppLayout>
      <section className="dashboard-hero">
        <div className="dashboard-hero-content">
          <span className="eyebrow">
            <ShieldCheck size={17} />
            Secure AI-powered assessment
          </span>

          <h1>
            Welcome back,{" "}
            <span>
              {user?.displayName ||
                user?.email?.split("@")[0] ||
                "User"}
            </span>
          </h1>

          <p>
            Assess cardiovascular risk, understand the
            model’s decision through SHAP, and receive a
            readable Explainable Generative AI report.
          </p>

          <div className="hero-actions">
            <Link
              to="/assessment"
              className="primary-action"
            >
              Start assessment
              <ArrowRight size={18} />
            </Link>

            <Link
              to="/history"
              className="secondary-action"
            >
              View history
            </Link>
          </div>
        </div>

        <div className="hero-visual">
          <div className="pulse-ring pulse-ring-one" />
          <div className="pulse-ring pulse-ring-two" />

          <div className="heart-visual-card">
            <ClipboardPlus size={60} />
            <strong>Personalized risk insights</strong>
            <span>
              Ensemble ML + SHAP + X-GenAI
            </span>
          </div>
        </div>
      </section>

      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}

      <section className="dashboard-grid">
        <article className="dashboard-card feature-card">
          <span className="card-icon card-icon-green">
            <ClipboardPlus size={24} />
          </span>

          <h2>New assessment</h2>

          <p>
            Submit your health and lifestyle information
            to generate a cardiovascular risk estimate.
          </p>

          <Link to="/assessment">
            Begin assessment
            <ArrowRight size={17} />
          </Link>
        </article>

        <article className="dashboard-card feature-card">
          <span className="card-icon card-icon-blue">
            <BrainCircuit size={24} />
          </span>

          <h2>Explainable prediction</h2>

          <p>
            Review the factors that increased or decreased
            the model prediction using SHAP evidence.
          </p>

          <Link to="/assessment">
            Explore XAI
            <ArrowRight size={17} />
          </Link>
        </article>

        <article className="dashboard-card feature-card">
          <span className="card-icon card-icon-purple">
            <History size={24} />
          </span>

          <h2>Prediction history</h2>

          <p>
            Revisit your previous assessments and open
            complete saved reports at any time.
          </p>

          <Link to="/history">
            Open history
            <ArrowRight size={17} />
          </Link>
        </article>
      </section>

      <section className="dashboard-lower-grid">
        <article className="dashboard-card account-card">
          <div className="section-heading">
            <div>
              <span className="eyebrow">
                Account security
              </span>

              <h2>Connected account</h2>
            </div>

            <ShieldCheck size={26} />
          </div>

          {loading ? (
            <p>Verifying account with backend...</p>
          ) : backendUser ? (
            <div className="account-details">
              <div>
                <span>Email</span>
                <strong>
                  {backendUser.email ||
                    "Not available"}
                </strong>
              </div>

              <div>
                <span>Verification</span>
                <strong className="verified-status">
                  <CheckCircle2 size={17} />
                  Verified
                </strong>
              </div>

              <div>
                <span>Authentication</span>
                <strong>
                  Firebase + FastAPI
                </strong>
              </div>
            </div>
          ) : null}
        </article>

        <article className="dashboard-card system-card">
          <div className="section-heading">
            <div>
              <span className="eyebrow">
                System architecture
              </span>

              <h2>How your result is created</h2>
            </div>

            <Database size={26} />
          </div>

          <div className="system-flow">
            <span>User information</span>
            <ArrowRight size={16} />
            <span>ML ensemble</span>
            <ArrowRight size={16} />
            <span>SHAP</span>
            <ArrowRight size={16} />
            <span>X-GenAI report</span>
          </div>

          <p className="system-note">
            The generative model explains structured SHAP
            evidence. It does not independently calculate
            the medical-risk probability.
          </p>
        </article>
      </section>

      <div className="medical-disclaimer">
        <ShieldCheck size={21} />

        <p>
          This application is intended for research and
          preliminary screening only. It does not provide a
          medical diagnosis or replace consultation with a
          qualified healthcare professional.
        </p>
      </div>
    </AppLayout>
  );
}