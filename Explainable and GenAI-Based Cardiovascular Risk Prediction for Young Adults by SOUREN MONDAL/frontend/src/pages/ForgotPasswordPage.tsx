import {
  ArrowLeft,
  HeartPulse,
  Mail,
  Send,
  ShieldCheck,
} from "lucide-react";
import {
  sendPasswordResetEmail,
} from "firebase/auth";
import {
  useState,
  type FormEvent,
} from "react";
import {
  Link,
} from "react-router-dom";

import { auth } from "../lib/firebase";

function getPasswordResetError(
  errorCode?: string
) {
  switch (errorCode) {
    case "auth/invalid-email":
      return "Enter a valid email address.";

    case "auth/user-not-found":
      return "No account was found with this email address.";

    case "auth/too-many-requests":
      return (
        "Too many requests were made. " +
        "Please wait before trying again."
      );

    case "auth/network-request-failed":
      return (
        "A network error occurred. Check your internet " +
        "connection and try again."
      );

    default:
      return "The password reset email could not be sent.";
  }
}

export function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [submitting, setSubmitting] =
    useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();

    setSubmitting(true);
    setMessage("");
    setError("");

    try {
      await sendPasswordResetEmail(
        auth,
        email.trim()
      );

      setMessage(
        "A password reset link has been sent. " +
          "Check your inbox and spam folder."
      );
    } catch (err) {
      const firebaseError = err as {
        code?: string;
      };

      setError(
        getPasswordResetError(
          firebaseError.code
        )
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="password-reset-page">
      <section className="password-reset-card">
        <Link
          to="/login"
          className="password-reset-brand"
        >
          <span>
            <HeartPulse size={27} />
          </span>

          <div>
            <strong>CardioInsight</strong>
            <small>Heart Risk X-GenAI</small>
          </div>
        </Link>

        <div className="password-reset-icon">
          <Mail size={38} />
        </div>

        <span className="eyebrow">
          <ShieldCheck size={17} />
          Secure password recovery
        </span>

        <h1>Reset your password</h1>

        <p className="password-reset-description">
          Enter the email address connected to your
          CardioInsight account. Firebase will send a secure
          password-reset link.
        </p>

        <form
          className="password-reset-form"
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

          {message && (
            <div className="verification-message">
              <Send size={18} />
              <span>{message}</span>
            </div>
          )}

          {error && (
            <div className="auth-error">
              {error}
            </div>
          )}

          <button
            type="submit"
            className="auth-submit-button"
            disabled={submitting}
          >
            {submitting
              ? "Sending reset link..."
              : (
                <>
                  Send reset link
                  <Send size={18} />
                </>
              )}
          </button>
        </form>

        <Link
          to="/login"
          className="password-reset-back-link"
        >
          <ArrowLeft size={17} />
          Back to login
        </Link>

        <div className="auth-disclaimer">
          <ShieldCheck size={18} />

          <p>
            For account security, never share password-reset
            links or verification emails with another person.
          </p>
        </div>
      </section>

      <div className="verification-background-orb verification-orb-one" />
      <div className="verification-background-orb verification-orb-two" />
    </main>
  );
}