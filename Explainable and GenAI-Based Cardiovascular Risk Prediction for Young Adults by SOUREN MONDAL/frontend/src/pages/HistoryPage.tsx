import {
  ArrowRight,
  CalendarDays,
  FileText,
  History,
  ShieldCheck,
} from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { AppLayout } from "../components/AppLayout";
import {
  getPredictionHistory,
  type PredictionHistoryItem,
} from "../services/api";

function getRiskClass(riskCategory: string) {
  const normalizedRisk = riskCategory.toLowerCase();

  if (normalizedRisk.includes("high")) {
    return "history-risk-high";
  }

  if (normalizedRisk.includes("elevated")) {
    return "history-risk-elevated";
  }

  return "history-risk-lower";
}

export function HistoryPage() {
  const [predictions, setPredictions] = useState<
    PredictionHistoryItem[]
  >([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadHistory() {
      try {
        setError("");

        const result = await getPredictionHistory();

        setPredictions(result.predictions);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Prediction history could not be loaded."
        );
      } finally {
        setLoading(false);
      }
    }

    loadHistory();
  }, []);

  return (
    <AppLayout>
      <section className="history-header">
        <span className="eyebrow">
          <History size={17} />
          Saved assessments
        </span>

        <div className="history-heading-row">
          <div>
            <h1>Prediction History</h1>

            <p>
              Review your previous cardiovascular risk
              assessments and open complete saved reports.
            </p>
          </div>

          <Link
            to="/assessment"
            className="primary-history-action"
          >
            New assessment
            <ArrowRight size={18} />
          </Link>
        </div>
      </section>

      {loading && (
        <div className="history-state-card">
          <div className="history-loader" />

          <p>Loading your saved predictions...</p>
        </div>
      )}

      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}

      {!loading &&
        !error &&
        predictions.length === 0 && (
          <section className="empty-history-card">
            <span>
              <FileText size={34} />
            </span>

            <h2>No saved predictions yet</h2>

            <p>
              Complete your first assessment to create a saved
              prediction report.
            </p>

            <Link
              to="/assessment"
              className="primary-action history-empty-action"
            >
              Start assessment
              <ArrowRight size={18} />
            </Link>
          </section>
        )}

      {!loading &&
        !error &&
        predictions.length > 0 && (
          <>
            <section className="history-summary">
              <article>
                <span>Total assessments</span>
                <strong>{predictions.length}</strong>
              </article>

              <article>
                <span>Latest probability</span>
                <strong>
                  {predictions[0].probability_percent}%
                </strong>
              </article>

              <article>
                <span>Latest category</span>
                <strong className="history-category-text">
                  {predictions[0].risk_category}
                </strong>
              </article>
            </section>

            <section className="history-list">
              {predictions.map((prediction, index) => (
                <article
                  key={prediction.prediction_id}
                  className="history-card"
                >
                  <div className="history-card-main">
                    <div
                      className={`history-score ${getRiskClass(
                        prediction.risk_category
                      )}`}
                    >
                      <strong>
                        {prediction.probability_percent}%
                      </strong>

                      <span>
                        Predicted probability
                      </span>
                    </div>

                    <div className="history-card-content">
                      <div className="history-card-title-row">
                        <div>
                          <span
                            className={`history-risk-badge ${getRiskClass(
                              prediction.risk_category
                            )}`}
                          >
                            {prediction.risk_category}
                          </span>

                          <h2>
                            Cardiovascular Risk Assessment
                          </h2>
                        </div>

                        {index === 0 && (
                          <span className="latest-badge">
                            Latest
                          </span>
                        )}
                      </div>

                      <div className="history-card-meta">
                        <span>
                          <CalendarDays size={16} />

                          {prediction.created_at
                            ? new Date(
                                prediction.created_at
                              ).toLocaleString()
                            : "Date not available"}
                        </span>

                        <span>
                          Model version{" "}
                          {prediction.model_version}
                        </span>
                      </div>
                    </div>
                  </div>

                  <Link
                    to={`/history/${prediction.prediction_id}`}
                    className="history-report-link"
                  >
                    View full report
                    <ArrowRight size={17} />
                  </Link>
                </article>
              ))}
            </section>
          </>
        )}

      <div className="medical-disclaimer">
        <ShieldCheck size={21} />

        <p>
          Saved reports explain model predictions and should not
          be treated as medical diagnoses.
        </p>
      </div>
    </AppLayout>
  );
}