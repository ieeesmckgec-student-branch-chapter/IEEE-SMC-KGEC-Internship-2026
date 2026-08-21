import {
  ArrowRight,
  BrainCircuit,
  HeartPulse,
  LockKeyhole,
  Mail,
  ShieldCheck,
  UserRound,
} from "lucide-react";
import {
  createUserWithEmailAndPassword,
  sendEmailVerification,
  updateProfile,
} from "firebase/auth";
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
    case "auth/email-already-in-use":
      return "An account already exists with this email address.";

    case "auth/invalid-email":
      return "Enter a valid email address.";

    case "auth/weak-password":
      return "Use a stronger password with at least 6 characters.";

    case "auth/operation-not-allowed":
      return "Email and password registration is not enabled.";

    case "auth/network-request-failed":
      return (
        "A network error occurred. Check your internet " +
        "connection and try again."
      );

    case "auth/too-many-requests":
      return (
        "Too many requests were made. Please wait and try again."
      );

    default:
      return "Account creation failed. Please try again.";
  }
}

export function RegisterPage() {
  const navigate = useNavigate();
  const { user, loading } = useAuth();

  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] =
    useState("");

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

    setError("");

    const trimmedName = fullName.trim();
    const trimmedEmail = email.trim();

    if (trimmedName.length < 2) {
      setError(
        "Enter your full name using at least 2 characters."
      );
      return;
    }

    if (password.length < 6) {
      setError(
        "Password must contain at least 6 characters."
      );
      return;
    }

    if (password !== confirmPassword) {
      setError("The passwords do not match.");
      return;
    }

    setSubmitting(true);

    try {
      const credential =
        await createUserWithEmailAndPassword(
          auth,
          trimmedEmail,
          password
        );

      await updateProfile(
        credential.user,
        {
          displayName: trimmedName,
        }
      );

      await sendEmailVerification(
        credential.user
      );

      navigate("/verify-email", {
        replace: true,
      });
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
      <section className="auth-brand-panel register-brand-panel">
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
              Personalized cardiovascular insights
            </span>

            <h1>
              Create your secure health-screening account.
            </h1>

            <p>
              Register to generate cardiovascular risk
              assessments, understand model behaviour through
              SHAP, and access your saved prediction history.
            </p>
          </div>

          <div className="auth-feature-list">
            <article>
              <BrainCircuit size={21} />

              <div>
                <strong>
                  Explainable AI reports
                </strong>

                <span>
                  Receive readable explanations based on
                  structured model evidence.
                </span>
              </div>
            </article>

            <article>
              <ShieldCheck size={21} />

              <div>
                <strong>
                  Verified user access
                </strong>

                <span>
                  Email verification helps protect assessment
                  history and personal reports.
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
              New CardioInsight account
            </span>

            <h2>Create account</h2>

            <p>
              Register with your email address to begin using
              the cardiovascular risk-screening system.
            </p>
          </div>

          <form
            className="auth-form"
            onSubmit={handleSubmit}
          >
            <label>
              Full name

              <div className="auth-input-wrapper">
                <UserRound size={19} />

                <input
                  type="text"
                  placeholder="Enter your full name"
                  autoComplete="name"
                  value={fullName}
                  onChange={(event) =>
                    setFullName(
                      event.target.value
                    )
                  }
                  required
                />
              </div>
            </label>

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
                    setEmail(
                      event.target.value
                    )
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
                  placeholder="Minimum 6 characters"
                  autoComplete="new-password"
                  value={password}
                  onChange={(event) =>
                    setPassword(
                      event.target.value
                    )
                  }
                  minLength={6}
                  required
                />
              </div>
            </label>

            <label>
              Confirm password

              <div className="auth-input-wrapper">
                <LockKeyhole size={19} />

                <input
                  type="password"
                  placeholder="Enter the password again"
                  autoComplete="new-password"
                  value={confirmPassword}
                  onChange={(event) =>
                    setConfirmPassword(
                      event.target.value
                    )
                  }
                  minLength={6}
                  required
                />
              </div>
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
              {submitting ? (
                "Creating account..."
              ) : (
                <>
                  Create account
                  <ArrowRight size={18} />
                </>
              )}
            </button>
          </form>

          <p className="auth-switch-text">
            Already have an account?{" "}
            <Link to="/login">
              Sign in
            </Link>
          </p>

          <div className="auth-disclaimer">
            <ShieldCheck size={18} />

            <p>
              After registration, a verification link will be
              sent to your email address. You must verify your
              email before using assessments and reports.
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}