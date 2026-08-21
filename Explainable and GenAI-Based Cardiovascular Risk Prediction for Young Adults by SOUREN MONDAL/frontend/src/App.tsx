import {
  Navigate,
  Route,
  Routes,
} from "react-router-dom";
import { AssessmentPage } from "./pages/AssessmentPage";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { DashboardPage } from "./pages/DashboardPage";
import { LoginPage } from "./pages/LoginPage";
import { RegisterPage } from "./pages/RegisterPage";
import { HistoryPage } from "./pages/HistoryPage";
import { PredictionDetailsPage } from "./pages/PredictionDetailsPage";
import { VerifiedRoute } from "./components/VerifiedRoute";
import { VerifyEmailPage } from "./pages/VerifyEmailPage";
import { ForgotPasswordPage } from "./pages/ForgotPasswordPage";

function App() {
  return (
    <Routes>
      <Route
        path="/"
        element={<Navigate to="/dashboard" replace />}
      />

      <Route
        path="/login"
        element={<LoginPage />}
      />

      <Route
        path="/register"
        element={<RegisterPage />}
      />

      <Route
        path="/dashboard"
        element={
          <VerifiedRoute>
            <DashboardPage />
          </VerifiedRoute>
        }
      />

      <Route
        path="/assessment"
        element={
          <VerifiedRoute>
            <AssessmentPage />
          </VerifiedRoute>
        }
      />
      <Route
        path="/history"
        element={
          <VerifiedRoute>
            <HistoryPage />
          </VerifiedRoute>
        }
      />
      <Route
        path="/history/:predictionId"
        element={
          <VerifiedRoute>
            <PredictionDetailsPage />
          </VerifiedRoute>
        }
      />
      <Route
        path="/verify-email"
        element={
          <ProtectedRoute>
            <VerifyEmailPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/forgot-password"
        element={<ForgotPasswordPage />}
      />
    </Routes>
  );
}

export default App;