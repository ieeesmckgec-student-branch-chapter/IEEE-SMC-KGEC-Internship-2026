import {
  ArrowRight,
  CheckCircle2,
  HeartPulse,
  LogOut,
  MailCheck,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import {
  sendEmailVerification,
  signOut,
} from "firebase/auth";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { useAuth } from "../context/AuthContext";
import { auth } from "../lib/firebase";

function getVerificationErrorMessage(errorCode?: string) {
  switch (errorCode) {
    case "auth/too-many-requests":
      return (
        "Too many verification emails were requested. " +
        "Please wait before trying again."
      );

    case "auth/network-request-failed":
      return (
        "A network error occurred. Check your internet " +
        "connection and try again."
      );

    case "auth/user-token-expired":
      return (
        "Your login session has expired. Please log in again."
      );

    default:
      return "The verification operation could not be completed.";
  }
}

export function VerifyEmailPage() {
  const navigate = useNavigate();
  const { user } = useAuth();

  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const [checking, setChecking] =
    useState(false);

  const [resending, setResending] =
    useState(false);

  async function handleCheckVerification() {
    const currentUser = auth.currentUser;

    if (!currentUser) {
      setError(
        "No signed-in user was found. Please log in again."
      );
      return;
    }

    setChecking(true);
    setError("");
    setMessage("");

    try {
      await currentUser.reload();

      if (currentUser.emailVerified) {
        await currentUser.getIdToken(true);

        setMessage(
          "Your email has been verified successfully."
        );

        window.setTimeout(() => {
          navigate("/dashboard", {
            replace: true,
          });
        }, 700);
      } else {
        setMessage(
          "Your email is not verified yet. Open the link in " +
            "your inbox, then return and check again."
        );
      }
    } catch (err) {
      const firebaseError = err as {
        code?: string;
      };

      setError(
        getVerificationErrorMessage(
          firebaseError.code
        )
      );
    } finally {
      setChecking(false);
    }
  }

  async function handleResend() {
    const currentUser = auth.currentUser;

    if (!currentUser) {
      setError(
        "No signed-in user was found. Please log in again."
      );
      return;
    }

    setResending(true);
    setError("");
    setMessage("");

    try {
      await sendEmailVerification(
        currentUser
      );

      setMessage(
        "A new verification email has been sent. " +
          "Check your inbox and spam folder."
      );
    } catch (err) {
      const firebaseError = err as {
        code?: string;
      };

      setError(
        getVerificationErrorMessage(
          firebaseError.code
        )
      );
    } finally {
      setResending(false);
    }
  }

  async function handleLogout() {
    await signOut(auth);

    navigate("/login", {
      replace: true,
    });
  }

  return (
    <main className="verification-page">
      <section className="verification-card">
        <div className="verification-brand">
          <span>
            <HeartPulse size={27} />
          </span>

          <div>
            <strong>CardioInsight</strong>
            <small>Heart Risk X-GenAI</small>
          </div>
        </div>

        <div className="verification-icon">
          <MailCheck size={43} />
        </div>

        <span className="eyebrow">
          <ShieldCheck size={17} />
          Secure account verification
        </span>

        <h1>Verify your email</h1>

        <p className="verification-description">
          We sent a verification link to your registered email
          address. Open the message and click the verification
          link before using assessments and saved reports.
        </p>

        <div className="verification-email-box">
          <span>Email address</span>

          <strong>
            {user?.email ||
              "Registered email address"}
          </strong>
        </div>

        <div className="verification-steps">
          <article>
            <span>1</span>

            <div>
              <strong>Check your inbox</strong>

              <p>
                Look for the Firebase verification email.
              </p>
            </div>
          </article>

          <article>
            <span>2</span>

            <div>
              <strong>Open the verification link</strong>

              <p>
                Click the link included in the email.
              </p>
            </div>
          </article>

          <article>
            <span>3</span>

            <div>
              <strong>Return to CardioInsight</strong>

              <p>
                Click the verification-status button below.
              </p>
            </div>
          </article>
        </div>

        {message && (
          <div className="verification-message">
            <CheckCircle2 size={19} />
            <span>{message}</span>
          </div>
        )}

        {error && (
          <div className="auth-error">
            {error}
          </div>
        )}

        <button
          type="button"
          className="verification-primary-button"
          onClick={handleCheckVerification}
          disabled={checking}
        >
          {checking ? (
            <>
              <RefreshCw
                className="verification-spin"
                size={18}
              />
              Checking verification...
            </>
          ) : (
            <>
              I have verified my email
              <ArrowRight size={18} />
            </>
          )}
        </button>

        <button
          type="button"
          className="verification-secondary-button"
          onClick={handleResend}
          disabled={resending}
        >
          {resending ? (
            <>
              <RefreshCw
                className="verification-spin"
                size={18}
              />
              Sending email...
            </>
          ) : (
            <>
              <MailCheck size={18} />
              Resend verification email
            </>
          )}
        </button>

        <button
          type="button"
          className="verification-logout-button"
          onClick={handleLogout}
        >
          <LogOut size={18} />
          Log out
        </button>

        <div className="verification-footer-note">
          <ShieldCheck size={17} />

          <p>
            Email verification helps protect your prediction
            history and personal assessment reports.
          </p>
        </div>
      </section>

      <div className="verification-background-orb verification-orb-one" />
      <div className="verification-background-orb verification-orb-two" />
    </main>
  );
}