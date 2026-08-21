import {
  ArrowLeft,
  BrainCircuit,
  CalendarDays,
  CheckCircle2,
  ShieldCheck,
  Trash2,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import {
  useEffect,
  useState,
} from "react";
import ReactMarkdown from "react-markdown";
import {
  Link,
  useNavigate,
  useParams,
} from "react-router-dom";

import { AppLayout } from "../components/AppLayout";
import {
  deletePrediction,
  getPredictionDetails,
  getPredictionShapImages,
  type PredictionDetails,
  type ShapVisualizations,
} from "../services/api";

function readableFeatureName(
  feature: string
) {
  return feature.replaceAll("_", " ");
}

function getRiskClass(
  riskCategory: string
) {
  const normalizedRisk =
    riskCategory.toLowerCase();

  if (
    normalizedRisk.includes("high")
  ) {
    return "report-risk-high";
  }

  if (
    normalizedRisk.includes("elevated")
  ) {
    return "report-risk-elevated";
  }

  return "report-risk-lower";
}

export function PredictionDetailsPage() {
  const navigate = useNavigate();
  const { predictionId } =
    useParams();

  const [
    prediction,
    setPrediction,
  ] =
    useState<PredictionDetails | null>(
      null
    );

  const [
    shapVisualizations,
    setShapVisualizations,
  ] =
    useState<ShapVisualizations | null>(
      null
    );

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  const [
    shapLoading,
    setShapLoading,
  ] = useState(true);

  const [
    shapError,
    setShapError,
  ] = useState("");

  const [deleting, setDeleting] =
    useState(false);

  const [
    deleteError,
    setDeleteError,
  ] = useState("");

  useEffect(() => {
    async function loadPrediction() {
      if (!predictionId) {
        setError(
          "Prediction ID is missing."
        );

        setLoading(false);
        setShapLoading(false);
        return;
      }

      try {
        setError("");

        const result =
          await getPredictionDetails(
            predictionId
          );

        setPrediction(
          result.prediction
        );
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Prediction could not be loaded."
        );

        setLoading(false);
        setShapLoading(false);
        return;
      } finally {
        setLoading(false);
      }

      try {
        setShapError("");

        const shapResult =
          await getPredictionShapImages(
            predictionId
          );

        setShapVisualizations(
          shapResult.visualizations
        );
      } catch (err) {
        setShapError(
          err instanceof Error
            ? err.message
            : "SHAP images could not be loaded."
        );
      } finally {
        setShapLoading(false);
      }
    }

    loadPrediction();
  }, [predictionId]);

  async function handleDeletePrediction() {
    if (
      !predictionId ||
      !prediction
    ) {
      return;
    }

    const confirmed =
      window.confirm(
        "Delete this saved prediction report? This action cannot be undone."
      );

    if (!confirmed) {
      return;
    }

    setDeleting(true);
    setDeleteError("");

    try {
      await deletePrediction(
        predictionId
      );

      navigate("/history", {
        replace: true,
      });
    } catch (err) {
      setDeleteError(
        err instanceof Error
          ? err.message
          : "Prediction could not be deleted."
      );
    } finally {
      setDeleting(false);
    }
  }

  return (
    <AppLayout>
      <Link
        to="/history"
        className="report-back-link"
      >
        <ArrowLeft size={18} />
        Back to prediction history
      </Link>

      {loading && (
        <section className="history-state-card">
          <div className="history-loader" />

          <p>
            Loading the saved prediction
            report...
          </p>
        </section>
      )}

      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}

      {!loading &&
        !error &&
        prediction && (
          <>
            <section className="report-header">
              <div>
                <span className="eyebrow">
                  Saved assessment report
                </span>

                <h1>
                  Cardiovascular Risk Report
                </h1>

                <p>
                  Review the saved ensemble
                  prediction, component model
                  probabilities, SHAP factors,
                  visual explanations, and
                  Explainable Generative AI
                  report.
                </p>
              </div>

              <div className="report-header-actions">
                <span
                  className={`report-risk-badge ${getRiskClass(
                    prediction
                      .risk_category
                  )}`}
                >
                  {
                    prediction
                      .risk_category
                  }
                </span>

                <button
                  type="button"
                  className="delete-report-button"
                  onClick={
                    handleDeletePrediction
                  }
                  disabled={deleting}
                >
                  <Trash2 size={17} />

                  {deleting
                    ? "Deleting..."
                    : "Delete report"}
                </button>
              </div>
            </section>

            {deleteError && (
              <div className="alert alert-error">
                {deleteError}
              </div>
            )}

            <section className="report-summary-card">
              <div
                className="probability-circle"
                style={{
                  background: `conic-gradient(
                    #0b8f78 ${
                      prediction
                        .probability_percent *
                      3.6
                    }deg,
                    #e7eeec 0deg
                  )`,
                }}
              >
                <div>
                  <strong>
                    {
                      prediction
                        .probability_percent
                    }
                    %
                  </strong>

                  <span>
                    Predicted probability
                  </span>
                </div>
              </div>

              <div className="report-summary-content">
                <div className="report-meta">
                  <span>
                    <CalendarDays
                      size={17}
                    />

                    {prediction.created_at
                      ? new Date(
                          prediction
                            .created_at
                        ).toLocaleString()
                      : "Date not available"}
                  </span>

                  <span>
                    Model version{" "}
                    {
                      prediction
                        .model_version
                    }
                  </span>
                </div>

                <h2>
                  Ensemble prediction summary
                </h2>

                <p>
                  The final probability combines
                  Logistic Regression, Random
                  Forest, and XGBoost using the
                  configured weighted ensemble.
                </p>

                <div className="report-threshold-grid">
                  <article>
                    <span>
                      Screening threshold
                    </span>

                    <strong>
                      {(
                        prediction
                          .thresholds
                          .screening *
                        100
                      ).toFixed(1)}
                      %
                    </strong>
                  </article>

                  <article>
                    <span>
                      High-specificity threshold
                    </span>

                    <strong>
                      {(
                        prediction
                          .thresholds
                          .high_specificity *
                        100
                      ).toFixed(1)}
                      %
                    </strong>
                  </article>

                  <article>
                    <span>
                      Screening result
                    </span>

                    <strong>
                      {prediction
                        .screening_prediction
                        ? "Positive"
                        : "Negative"}
                    </strong>
                  </article>

                  <article>
                    <span>
                      High-specificity result
                    </span>

                    <strong>
                      {prediction
                        .high_specificity_prediction
                        ? "Positive"
                        : "Negative"}
                    </strong>
                  </article>
                </div>
              </div>
            </section>

            <section className="component-section">
              <div className="report-section-heading">
                <div>
                  <span className="eyebrow">
                    Ensemble components
                  </span>

                  <h2>
                    Component model
                    probabilities
                  </h2>
                </div>

                <CheckCircle2
                  size={26}
                />
              </div>

              <div className="component-grid">
                <article>
                  <span>
                    Logistic Regression
                  </span>

                  <strong>
                    {(
                      prediction
                        .component_probabilities
                        .logistic_regression *
                      100
                    ).toFixed(1)}
                    %
                  </strong>
                </article>

                <article>
                  <span>
                    Random Forest
                  </span>

                  <strong>
                    {(
                      prediction
                        .component_probabilities
                        .random_forest *
                      100
                    ).toFixed(1)}
                    %
                  </strong>
                </article>

                <article>
                  <span>
                    XGBoost
                  </span>

                  <strong>
                    {(
                      prediction
                        .component_probabilities
                        .xgboost * 100
                    ).toFixed(1)}
                    %
                  </strong>
                </article>
              </div>
            </section>

            <section className="factor-grid">
              <article className="factor-card increasing-card">
                <div className="factor-heading">
                  <TrendingUp
                    size={23}
                  />

                  <div>
                    <h3>
                      Factors increasing the
                      prediction
                    </h3>

                    <p>
                      Positive SHAP
                      contributions
                    </p>
                  </div>
                </div>

                {prediction
                  .xai_explanation
                  .increasing_factors
                  .length > 0 ? (
                  <ul className="factor-list">
                    {prediction
                      .xai_explanation
                      .increasing_factors
                      .map((factor) => (
                        <li
                          key={
                            factor.feature
                          }
                        >
                          <span>
                            {readableFeatureName(
                              factor.feature
                            )}
                          </span>

                          <strong>
                            +
                            {factor.shap_value.toFixed(
                              3
                            )}
                          </strong>
                        </li>
                      ))}
                  </ul>
                ) : (
                  <p>
                    No major increasing
                    factors were identified.
                  </p>
                )}
              </article>

              <article className="factor-card decreasing-card">
                <div className="factor-heading">
                  <TrendingDown
                    size={23}
                  />

                  <div>
                    <h3>
                      Factors decreasing the
                      prediction
                    </h3>

                    <p>
                      Negative SHAP
                      contributions
                    </p>
                  </div>
                </div>

                {prediction
                  .xai_explanation
                  .decreasing_factors
                  .length > 0 ? (
                  <ul className="factor-list">
                    {prediction
                      .xai_explanation
                      .decreasing_factors
                      .map((factor) => (
                        <li
                          key={
                            factor.feature
                          }
                        >
                          <span>
                            {readableFeatureName(
                              factor.feature
                            )}
                          </span>

                          <strong>
                            {factor.shap_value.toFixed(
                              3
                            )}
                          </strong>
                        </li>
                      ))}
                  </ul>
                ) : (
                  <p>
                    No major decreasing
                    factors were identified.
                  </p>
                )}
              </article>
            </section>

            <section className="shap-visualization-section">
              <div className="report-section-heading">
                <div>
                  <span className="eyebrow">
                    Visual model explanation
                  </span>

                  <h2>
                    SHAP visualizations
                  </h2>
                </div>

                <BrainCircuit
                  size={27}
                />
              </div>

              {shapLoading && (
                <div className="history-state-card shap-loading-card">
                  <div className="history-loader" />

                  <p>
                    Generating SHAP
                    visualizations...
                  </p>
                </div>
              )}

              {shapError && (
                <div className="alert alert-error">
                  {shapError}
                </div>
              )}

              {!shapLoading &&
                !shapError &&
                shapVisualizations && (
                  <>
                    <p className="shap-visualization-note">
                      {
                        shapVisualizations
                          .important_note
                      }
                    </p>

                    <div className="shap-image-grid">
                      <article className="shap-image-card">
                        <div>
                          <h3>
                            Waterfall plot
                          </h3>

                          <p>
                            Shows how each
                            feature moved the
                            XGBoost output from
                            its baseline value.
                          </p>
                        </div>

                        <img
                          src={
                            shapVisualizations
                              .waterfall_image
                          }
                          alt="SHAP waterfall plot for the saved XGBoost prediction"
                        />
                      </article>

                      <article className="shap-image-card">
                        <div>
                          <h3>
                            Feature contribution
                            plot
                          </h3>

                          <p>
                            Shows the strongest
                            positive and negative
                            feature contributions.
                          </p>
                        </div>

                        <img
                          src={
                            shapVisualizations
                              .bar_image
                          }
                          alt="SHAP contribution bar plot for the saved prediction"
                        />
                      </article>
                    </div>
                  </>
                )}
            </section>

            <section className="report-ai-card">
              <div className="report-section-heading">
                <div>
                  <span className="eyebrow">
                    Explainable Generative AI
                  </span>

                  <h2>
                    Personalized explanation
                  </h2>
                </div>

                <BrainCircuit
                  size={28}
                />
              </div>

              <div className="markdown-report">
                <ReactMarkdown>
                  {
                    prediction
                      .xgenai_explanation
                  }
                </ReactMarkdown>
              </div>
            </section>

            <section className="submitted-data-card">
              <div className="report-section-heading">
                <div>
                  <span className="eyebrow">
                    Assessment input
                  </span>

                  <h2>
                    Submitted information
                  </h2>
                </div>
              </div>

              <div className="submitted-data-grid">
                {Object.entries(
                  prediction.input_data
                ).map(
                  ([key, value]) => (
                    <article key={key}>
                      <span>
                        {readableFeatureName(
                          key
                        )}
                      </span>

                      <strong>
                        {value}
                      </strong>
                    </article>
                  )
                )}
              </div>
            </section>

            <div className="medical-disclaimer">
              <ShieldCheck size={21} />

              <p>
                This saved report is intended
                for research and preliminary
                screening only. It does not
                provide a medical diagnosis or
                replace consultation with a
                qualified healthcare
                professional.
              </p>
            </div>
          </>
        )}
    </AppLayout>
  );
}