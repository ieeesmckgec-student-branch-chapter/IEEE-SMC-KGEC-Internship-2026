import {
  ArrowRight,
  BrainCircuit,
  HeartPulse,
  LockKeyhole,
  Mail,
  ShieldCheck,
} from "lucide-react";
import { signInWithEmailAndPassword } from "firebase/auth";
import {
  useEffect,
  useState,
  type FormEvent,
} from "react";
import {
  Link,
  useNavigate,
} from "react-router-dom";

import { useAuth } from "../context/AuthContext";
import { auth } from "../lib/firebase";

function getFirebaseErrorMessage(errorCode?: string) {
  switch (errorCode) {
    case "auth/invalid-credential":
      return "The email address or password is incorrect.";

    case "auth/invalid-email":
      return "Enter a valid email address.";

    case "auth/user-disabled":
      return "This account has been disabled.";

    case "auth/too-many-requests":
      return (
        "Too many login attempts were made. " +
        "Please wait and try again."
      );

    case "auth/network-request-failed":
      return (
        "A network error occurred. Check your internet " +
        "connection and try again."
      );

    default:
      return "Login failed. Please check your details.";
  }
}

export function LoginPage() {
  const navigate = useNavigate();
  const { user, loading } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [submitting, setSubmitting] =
    useState(false);

  const [error, setError] = useState("");

  useEffect(() => {
    if (loading || !user) {
      return;
    }

    if (user.emailVerified) {
      navigate("/dashboard", {
        replace: true,
      });
    } else {
      navigate("/verify-email", {
        replace: true,
      });
    }
  }, [user, loading, navigate]);

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();

    setSubmitting(true);
    setError("");

    try {
      const credential =
        await signInWithEmailAndPassword(
          auth,
          email.trim(),
          password
        );

      await credential.user.reload();

      if (credential.user.emailVerified) {
        await credential.user.getIdToken(true);

        navigate("/dashboard", {
          replace: true,
        });
      } else {
        navigate("/verify-email", {
          replace: true,
        });
      }
    } catch (err) {
      const firebaseError = err as {
        code?: string;
      };

      setError(
        getFirebaseErrorMessage(
          firebaseError.code
        )
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-brand-panel">
        <div className="auth-brand-content">
          <Link
            to="/login"
            className="auth-brand"
          >
            <span>
              <HeartPulse size={29} />
            </span>

            <div>
              <strong>CardioInsight</strong>
              <small>
                Heart Risk X-GenAI
              </small>
            </div>
          </Link>

          <div className="auth-brand-message">
            <span className="auth-eyebrow">
              <ShieldCheck size={17} />
              Explainable cardiovascular screening
            </span>

            <h1>
              Understand the prediction, not just the score.
            </h1>

            <p>
              CardioInsight combines an ensemble machine-learning
              model, SHAP explanations, and generative AI to
              provide readable cardiovascular risk insights.
            </p>
          </div>

          <div className="auth-feature-list">
            <article>
              <BrainCircuit size={21} />

              <div>
                <strong>
                  Explainable predictions
                </strong>

                <span>
                  Understand which factors influenced the
                  XGBoost component.
                </span>
              </div>
            </article>

            <article>
              <LockKeyhole size={21} />

              <div>
                <strong>
                  Secure personal account
                </strong>

                <span>
                  Firebase authentication protects your
                  prediction history.
                </span>
              </div>
            </article>
          </div>
        </div>

        <div className="auth-decoration auth-decoration-one" />
        <div className="auth-decoration auth-decoration-two" />
      </section>

      <section className="auth-form-panel">
        <div className="auth-form-container">
          <div className="auth-form-heading">
            <span className="auth-mobile-logo">
              <HeartPulse size={24} />
            </span>

            <span className="eyebrow">
              Secure account access
            </span>

            <h2>Welcome back</h2>

            <p>
              Sign in to create an assessment or review your
              previous cardiovascular risk reports.
            </p>
          </div>

          <form
            className="auth-form"
            onSubmit={handleSubmit}
          >
            <label>
              Email address

              <div className="auth-input-wrapper">
                <Mail size={19} />

                <input
                  type="email"
                  placeholder="name@example.com"
                  autoComplete="email"
                  value={email}
                  onChange={(event) =>
                    setEmail(event.target.value)
                  }
                  required
                />
              </div>
            </label>

            <label>
              Password

              <div className="auth-input-wrapper">
                <LockKeyhole size={19} />

                <input
                  type="password"
                  placeholder="Enter your password"
                  autoComplete="current-password"
                  value={password}
                  onChange={(event) =>
                    setPassword(event.target.value)
                  }
                  required
                />
              </div>
              <Link
                to="/forgot-password"
                className="forgot-password-link"
              >
                Forgot password?
              </Link>
            </label>

            {error && (
              <div className="auth-error">
                {error}
              </div>
            )}

            <button
              className="auth-submit-button"
              type="submit"
              disabled={submitting}
            >
              {submitting
                ? "Signing in..."
                : (
                  <>
                    Sign in
                    <ArrowRight size={18} />
                  </>
                )}
            </button>
          </form>

          <p className="auth-switch-text">
            Do not have an account?{" "}
            <Link to="/register">
              Create an account
            </Link>
          </p>

          <div className="auth-disclaimer">
            <ShieldCheck size={18} />

            <p>
              This application provides research-based
              preliminary screening and does not provide a
              medical diagnosis.
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}